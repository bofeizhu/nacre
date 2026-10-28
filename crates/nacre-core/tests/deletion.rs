//! Forgetting: `delete_group` erases exactly one namespace, and
//! `delete_episode` erases exactly what one episode is the sole evidence
//! for. Both commit in one transaction: a group as a single `Purge`, an
//! episode as its `Purge` plus the reset and rebuild of the summaries of
//! the nodes it keeps.
//!
//! Two tiers, because the laws and the realism are separable:
//!
//! - **Hand-built grit state** for the mechanical laws: group isolation, the
//!   empty-`Purge` trap, the all-or-nothing batch, the enumeration
//!   boundaries a group scan cannot cross (an edge written into another
//!   group, a merge placeholder parked in the default group), the
//!   documented reach of the erasure into the op-log (the entries that
//!   carried a purged id go; the ones that merely name it stay), that a
//!   forgotten id is spent — re-using one is a silent no-op — and that a
//!   node an episode deletion keeps forgets what its summary learned from
//!   that episode, down to the prompts a later ingest sends.
//! - **A golden trace replayed into two sibling groups of one database**
//!   for the same laws over a graph the real pipeline wrote — merges,
//!   invalidations, summary updates, mentions and vectors included. The
//!   replay key carries no group id (recordings are keyed on the request,
//!   and no prompt context reads `group_id`), while every candidate pool the
//!   pipeline consults IS group-scoped, so the second ingest issues
//!   byte-identical requests and yields a structurally identical, id-
//!   disjoint sibling. That sibling is the control: deleting one group must
//!   leave it byte-identical.
//!
//! Each scenario gets its own database file — these tests run in parallel,
//! and a shared path surfaces as a SQLite "disk I/O error".

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, TimeZone, Utc};
use grit_core::{GraphOp, Grit, ManualClock, NodeField, Options, WriteItem};
use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::forget::{delete_episode, delete_group};
use nacre_core::model::{
    CompletionRequest, Embedder, EmbedderMeta, LanguageModel, ModelError, RecordingStore,
    ReplayEmbedder, ReplayModel,
};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, PipelineError, add_episode,
    retrieve_previous_episodes,
};
use nacre_core::search::search_edges;
use serde_json::{Value, json};
use uuid::Uuid;

// ---------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle")
}

/// A database of this scenario's own. Returns the handle and its path so
/// the test can clean up.
fn open_db(scenario: &str) -> (Grit, PathBuf) {
    let dir = std::env::temp_dir().join(format!("nacre-deletion-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{scenario}.db"));
    remove_db(&path);
    let grit = Grit::open(&path, Options::new("nacre-deletion")).unwrap();
    (grit, path)
}

fn remove_db(path: &Path) {
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
}

/// Everything about a group that a deletion could change: the rows
/// themselves (every field, expired ones included), provenance in both
/// directions, which ids hold a vector, the total oplog length (a
/// whole-database measure, so a purge elsewhere moves it), and the oplog
/// entries that carried *this* group's content — which a purge of another
/// group must leave verbatim.
#[derive(Debug, PartialEq)]
struct Snapshot {
    oplog: usize,
    creator_ops: Vec<String>,
    nodes: Vec<Value>,
    edges: Vec<Value>,
    episodes: Vec<Value>,
    /// `mentions_of` per node and edge — the episode → target direction.
    mentions: Vec<(Uuid, Vec<Uuid>)>,
    /// `episode_targets` per episode — the inverse.
    targets: Vec<(Uuid, Vec<Uuid>)>,
    embedded_nodes: Vec<Uuid>,
    embedded_edges: Vec<Uuid>,
}

fn snapshot(grit: &Grit, group_id: &str) -> Snapshot {
    let nodes = grit.nodes_in_group(group_id).unwrap();
    let edges = grit.edges_in_group(group_id).unwrap();
    let episodes = grit.episodes_in_group(group_id).unwrap();
    let mut mentions: Vec<(Uuid, Vec<Uuid>)> = nodes
        .iter()
        .map(|n| n.id)
        .chain(edges.iter().map(|e| e.id))
        .map(|id| (id, grit.mentions_of(id).unwrap()))
        .collect();
    mentions.sort();
    let mut targets: Vec<(Uuid, Vec<Uuid>)> = episodes
        .iter()
        .map(|e| (e.id, grit.episode_targets(e.id).unwrap()))
        .collect();
    targets.sort();
    let group_ids: Vec<Uuid> = nodes
        .iter()
        .map(|n| n.id)
        .chain(edges.iter().map(|e| e.id))
        .chain(episodes.iter().map(|e| e.id))
        .collect();
    Snapshot {
        oplog: grit.oplog_since(0).unwrap().len(),
        creator_ops: creator_ops(grit, &group_ids),
        mentions,
        targets,
        embedded_nodes: nodes
            .iter()
            .filter(|n| grit.get_node_embedding(n.id).unwrap().is_some())
            .map(|n| n.id)
            .collect(),
        embedded_edges: edges
            .iter()
            .filter(|e| grit.get_edge_embedding(e.id).unwrap().is_some())
            .map(|e| e.id)
            .collect(),
        nodes: nodes.iter().map(to_value).collect(),
        edges: edges.iter().map(to_value).collect(),
        episodes: episodes.iter().map(to_value).collect(),
    }
}

fn to_value<T: serde::Serialize>(row: &T) -> Value {
    serde_json::to_value(row).unwrap()
}

/// The oplog entries whose op **created** one of `ids` — the entries that
/// carry those ids' content, and exactly the ones grit's purge scrubs.
/// Serialized, in log order.
///
/// Ops that merely *reference* an id (`InvalidateEdge`, `MergeNodes`,
/// `Purge`, `PurgeNodeFields`, and the endpoint/mention fields of the ops
/// above) are not
/// creators: they are opaque uuids, they survive a purge by design, and
/// they are what keeps grit's merge laws exact. This mirrors grit's own
/// rule, which is not public API.
fn creator_ops(grit: &Grit, ids: &[Uuid]) -> Vec<String> {
    grit.oplog_since(0)
        .unwrap()
        .iter()
        .filter(|(_, entry)| match &entry.op {
            GraphOp::AddNode { id, .. }
            | GraphOp::UpdateNode { id, .. }
            | GraphOp::AddEdge { id, .. }
            | GraphOp::AddEpisode { id, .. } => ids.contains(id),
            GraphOp::InvalidateEdge { .. }
            | GraphOp::MergeNodes { .. }
            | GraphOp::Purge { .. }
            | GraphOp::PurgeNodeFields { .. } => false,
        })
        .map(|(_, entry)| serde_json::to_string(&entry.op).unwrap())
        .collect()
}

/// Every read API nacre or a binding exposes says the id is gone.
fn assert_unreachable(grit: &Grit, id: Uuid) {
    assert!(
        grit.node(id).unwrap().is_none(),
        "{id} still reads as a node"
    );
    assert!(
        grit.edge(id).unwrap().is_none(),
        "{id} still reads as an edge"
    );
    assert!(
        grit.episode(id).unwrap().is_none(),
        "{id} still reads as an episode"
    );
    assert!(
        grit.mentions_of(id).unwrap().is_empty(),
        "{id} still has provenance"
    );
    assert!(
        grit.episode_targets(id).unwrap().is_empty(),
        "{id} still points at what it wrote"
    );
    assert!(
        grit.get_node_embedding(id).unwrap().is_none()
            && grit.get_edge_embedding(id).unwrap().is_none(),
        "{id} still holds a vector"
    );
    assert!(
        grit.node_history(id).is_err(),
        "{id} still has a bi-temporal audit trail"
    );
}

/// One node pair, the fact between them, and the episode that evidences all
/// three — the smallest shape the pipeline ever writes.
struct Seed {
    src: Uuid,
    dst: Uuid,
    edge: Uuid,
    episode: Uuid,
}

/// Write that shape into `group`, with vectors, exactly as the pipeline
/// would: one batch, embeddings behind the rows they describe, the episode
/// mentioning everything it evidenced.
fn seed(grit: &Grit, group: &str, who: &str, fact: &str) -> Seed {
    grit.register_embedding_model("deletion-test", 3, "")
        .unwrap();
    let (src, dst, edge, episode) = (grit.new_id(), grit.new_id(), grit.new_id(), grit.new_id());
    let node = |id: Uuid, name: String| GraphOp::AddNode {
        id,
        kind: "Entity".into(),
        name,
        summary: String::new(),
        attrs: json!({"labels": ["Entity"]}),
        group_id: group.into(),
    };
    grit.apply_batch(vec![
        WriteItem::Op(node(src, who.to_owned())),
        WriteItem::NodeEmbedding {
            id: src,
            vector: vec![1.0, 0.0, 0.0],
        },
        WriteItem::Op(node(dst, format!("{who}'s employer"))),
        WriteItem::NodeEmbedding {
            id: dst,
            vector: vec![0.0, 1.0, 0.0],
        },
        WriteItem::Op(GraphOp::AddEdge {
            id: edge,
            src,
            dst,
            rel: "WORKS_AT".into(),
            fact: fact.into(),
            attrs: json!({}),
            group_id: group.into(),
            valid_at: None,
            invalid_at: None,
        }),
        WriteItem::EdgeEmbedding {
            id: edge,
            vector: vec![0.0, 0.0, 1.0],
        },
        WriteItem::Op(GraphOp::AddEpisode {
            id: episode,
            source: "chat".into(),
            kind: "message".into(),
            content: format!("{who}: {fact}"),
            occurred_at: 1_700_000_000_000,
            group_id: group.into(),
            metadata: Some(format!("host-row:{who}")),
            mentions: vec![src, dst, edge],
        }),
    ])
    .unwrap();
    Seed {
        src,
        dst,
        edge,
        episode,
    }
}

fn sorted(mut ids: Vec<Uuid>) -> Vec<Uuid> {
    ids.sort();
    ids
}

// ---------------------------------------------------------------------
// Tier A — the mechanical laws, over hand-built state
// ---------------------------------------------------------------------

/// `delete_group` erases every row of its group — rows, provenance in both
/// directions, vectors, FTS — and leaves a sibling group byte-identical.
#[test]
fn delete_group_erases_its_own_group_and_nothing_else() {
    let (grit, path) = open_db("group-isolation");
    let target = seed(&grit, "target", "Priya", "Priya works at Northwind Labs.");
    let sibling = seed(&grit, "sibling", "Jordan", "Jordan teaches at Belmont.");
    let before_sibling = snapshot(&grit, "sibling");
    let oplog_before = grit.oplog_since(0).unwrap().len();
    let carried = creator_ops(
        &grit,
        &[target.src, target.dst, target.edge, target.episode],
    )
    .len();

    let outcome = delete_group(&grit, "target").unwrap();
    assert_eq!(
        sorted(outcome.purged_node_ids.clone()),
        sorted(vec![target.src, target.dst])
    );
    assert_eq!(outcome.purged_edge_ids, vec![target.edge]);
    assert_eq!(outcome.purged_episode_ids, vec![target.episode]);

    // The group is empty by the only measure that counts: grit's group
    // scans are complete raw records (no temporal or tombstone filter), so
    // empty means gone, not merely retracted.
    assert!(grit.nodes_in_group("target").unwrap().is_empty());
    assert!(grit.edges_in_group("target").unwrap().is_empty());
    assert!(grit.episodes_in_group("target").unwrap().is_empty());
    for id in [target.src, target.dst, target.edge, target.episode] {
        assert_unreachable(&grit, id);
    }

    // One transaction, one audit record — and the log forgets with the
    // rows: the entries that carried the four purged ids go in the same
    // transaction, so the net change is one Purge minus what it scrubbed.
    assert_eq!(carried, 4, "the seed wrote four content-bearing entries");
    assert_eq!(
        grit.oplog_since(0).unwrap().len(),
        oplog_before + 1 - carried,
        "a whole-group deletion is a single op, and it takes the entries \
         that carried what it forgot with it"
    );

    assert_eq!(
        snapshot(&grit, "sibling"),
        Snapshot {
            // The sibling's own rows AND its own entries in the log are
            // untouched — the scrub is keyed by id, and this is what
            // proves it took only the right ones. All that moved is the
            // shared log's length: plus the Purge, minus the scrub.
            oplog: before_sibling.oplog + 1 - carried,
            ..before_sibling
        },
        "the sibling group must be byte-identical"
    );
    assert!(grit.node(sibling.src).unwrap().is_some());
    assert!(grit.get_edge_embedding(sibling.edge).unwrap().is_some());

    drop(grit);
    remove_db(&path);
}

/// Nothing to forget is not a failure. The op is skipped rather than
/// emitted empty — grit rejects an empty `Purge`, and a rejected op fails
/// the whole batch, so emitting one would turn "this group is already
/// gone" into an error.
#[test]
fn deleting_an_empty_group_writes_nothing() {
    let (grit, path) = open_db("empty-group");
    let sibling = seed(&grit, "sibling", "Jordan", "Jordan teaches at Belmont.");
    let before = snapshot(&grit, "sibling");

    let outcome = delete_group(&grit, "never-existed").unwrap();
    assert_eq!(outcome, Default::default(), "zero counts, no error");
    assert_eq!(snapshot(&grit, "sibling"), before, "and no write at all");

    // The trap the skip exists for.
    assert!(
        grit.apply_batch(vec![WriteItem::Op(GraphOp::Purge { ids: vec![] })])
            .is_err(),
        "an empty Purge is rejected — emitting one would fail the batch"
    );
    assert!(grit.node(sibling.src).unwrap().is_some());

    drop(grit);
    remove_db(&path);
}

/// A deletion batch is all-or-nothing — `tests/atomicity.rs`'s claim for
/// the ingest side, restated for this one. The lever is different: deletion
/// makes no model call, so there is no recording to withhold. What plays
/// the same role is an invalid op, because grit validates every item before
/// minting an id or an HLC: a batch carrying one writes nothing at all,
/// which is exactly what lets these entry points collect every id first and
/// commit once.
#[test]
fn a_rejected_deletion_batch_writes_nothing() {
    let (grit, path) = open_db("rejected-batch");
    let target = seed(&grit, "target", "Priya", "Priya works at Northwind Labs.");
    let before = snapshot(&grit, "target");

    let ids = vec![target.src, target.dst, target.edge, target.episode];
    let rejected = grit.apply_batch(vec![
        WriteItem::Op(GraphOp::Purge { ids: ids.clone() }),
        // Valid id, no fields: rejected by grit's validation pass.
        WriteItem::Op(GraphOp::UpdateNode {
            id: target.src,
            name: None,
            summary: None,
            kind: None,
            attrs: None,
        }),
    ]);
    assert!(
        rejected.is_err(),
        "the invalid companion op fails the batch"
    );
    assert_eq!(
        snapshot(&grit, "target"),
        before,
        "a rejected batch erases nothing — not even the valid Purge"
    );

    // And the same ids go cleanly when the batch is valid.
    delete_group(&grit, "target").unwrap();
    for id in ids {
        assert_unreachable(&grit, id);
    }

    drop(grit);
    remove_db(&path);
}

/// An edge's group is whatever its writer set, independent of its
/// endpoints'. Group deletion is complete only because nacre's pipeline
/// always writes an edge into its episode's group — this pins what happens
/// when that invariant is broken (`ExtractEdgesOptions::group_id` is the
/// supported way to break it), so the rustdoc's caveat is a tested claim.
#[test]
fn an_edge_written_into_another_group_survives_its_endpoints() {
    let (grit, path) = open_db("edge-group-independence");
    let target = seed(&grit, "target", "Priya", "Priya works at Northwind Labs.");
    let stray = grit.new_id();
    grit.apply(GraphOp::AddEdge {
        id: stray,
        src: target.src,
        dst: target.dst,
        rel: "MENTORS".into(),
        fact: "Priya mentors the new hires.".into(),
        attrs: json!({}),
        group_id: "elsewhere".into(),
        valid_at: None,
        invalid_at: None,
    })
    .unwrap();

    let outcome = delete_group(&grit, "target").unwrap();
    assert!(
        !outcome.purged_edge_ids.contains(&stray),
        "a group scan cannot see an edge tagged with another group"
    );
    assert!(
        grit.edge(stray).unwrap().is_some(),
        "the stray edge survives, fact sentence and all — sweeping by \
         incidence (node_history) is the caller's job when the pipeline \
         invariant is deliberately broken"
    );
    assert_eq!(grit.edges_in_group("elsewhere").unwrap().len(), 1);

    drop(grit);
    remove_db(&path);
}

/// A `MergeNodes` whose `from` has not arrived parks a placeholder in the
/// DEFAULT group `""`, not in the group the merge concerns — so a group
/// deletion does not see it, and `delete_group("")` may erase one belonging
/// to another namespace. Purging a placeholder tombstones the id forever:
/// the late `AddNode` becomes a no-op. Deletion wins over a write still in
/// flight, deliberately.
#[test]
fn merge_placeholders_park_in_the_default_group_and_purging_one_wins() {
    let (grit, path) = open_db("merge-placeholder");
    let canonical = grit.new_id();
    grit.apply(GraphOp::AddNode {
        id: canonical,
        kind: "Entity".into(),
        name: "Priya".into(),
        summary: String::new(),
        attrs: json!({"labels": ["Entity"]}),
        group_id: "target".into(),
    })
    .unwrap();

    // The absorbed node's own AddNode has not arrived yet.
    let absent = grit.new_id();
    grit.apply(GraphOp::MergeNodes {
        from: absent,
        into: canonical,
    })
    .unwrap();

    assert!(
        !grit
            .nodes_in_group("target")
            .unwrap()
            .iter()
            .any(|n| n.id == absent),
        "the placeholder is not in the merge's group"
    );
    assert!(
        grit.nodes_in_group("")
            .unwrap()
            .iter()
            .any(|n| n.id == absent),
        "it parks in the default group until the real AddNode lands"
    );

    // Forgetting the merge's group leaves it standing.
    delete_group(&grit, "target").unwrap();
    assert!(grit.node(absent).unwrap().is_some());

    // Forgetting the default group takes it, permanently.
    let outcome = delete_group(&grit, "").unwrap();
    assert_eq!(outcome.purged_node_ids, vec![absent]);
    assert!(grit.node(absent).unwrap().is_none());
    grit.apply(GraphOp::AddNode {
        id: absent,
        kind: "Entity".into(),
        name: "Priya Raman".into(),
        summary: String::new(),
        attrs: json!({"labels": ["Entity"]}),
        group_id: "target".into(),
    })
    .unwrap();
    assert!(
        grit.node(absent).unwrap().is_none(),
        "a tombstoned id stays dead: the late AddNode is a no-op"
    );

    drop(grit);
    remove_db(&path);
}

/// The erasure reaches the op-log, which is what lets nacre say "gone from
/// the live file" rather than only "unreachable through every read API".
/// grit deletes every entry whose op *created* a purged id in the purge's
/// own transaction, so neither `oplog_since` nor an export can ship the
/// content; what stays is the `Purge` itself, the audit record naming what
/// was forgotten. Pinned here so the claim nacre repeats to its callers is
/// tested, not inherited on trust.
#[test]
fn the_oplog_forgets_what_was_purged() {
    let (grit, path) = open_db("oplog-scrub");
    const FACT: &str = "Priya works at Northwind Labs.";
    let target = seed(&grit, "target", "Priya", FACT);
    let ids = vec![target.src, target.dst, target.edge, target.episode];
    assert_eq!(
        creator_ops(&grit, &ids).len(),
        4,
        "the seed wrote four content-bearing entries"
    );

    delete_group(&grit, "target").unwrap();
    assert_unreachable(&grit, target.edge);

    let log = grit.oplog_since(0).unwrap();
    assert!(
        !log.iter()
            .any(|(_, entry)| serde_json::to_string(&entry.op).unwrap().contains(FACT)),
        "no surviving entry carries the forgotten fact"
    );
    assert_eq!(log.len(), 1, "only the audit record is left");
    match &log[0].1.op {
        GraphOp::Purge { ids: purged } => assert_eq!(
            sorted(purged.clone()),
            sorted(ids),
            "the purge stays, and still names every id it forgot"
        ),
        other => panic!("expected the purge as the audit record, got {other:?}"),
    }

    // Not just the live tables: the export is the other way out of the
    // file, and it cannot ship what the log no longer holds.
    let mut export = Vec::new();
    grit.export_jsonl(&mut export).unwrap();
    assert!(
        !String::from_utf8(export).unwrap().contains(FACT),
        "and an export cannot carry it out either"
    );

    drop(grit);
    remove_db(&path);
}

/// The cascade, all four cases in one graph: a node only this episode
/// mentions goes; a node an earlier episode also mentions stays; an edge
/// this episode created goes; an edge it merely re-mentioned stays.
#[test]
fn delete_episode_erases_sole_evidence_and_keeps_the_corroborated() {
    let (grit, path) = open_db("episode-cascade");
    const GROUP: &str = "cascade";
    let first = seed(&grit, GROUP, "Priya", "Priya works at Northwind Labs.");

    // A second episode: re-states the first episode's fact (so it mentions
    // that edge and its endpoints without creating them) and adds one node
    // and one fact of its own.
    let (mentor, own_edge, second) = (grit.new_id(), grit.new_id(), grit.new_id());
    grit.apply_batch(vec![
        WriteItem::Op(GraphOp::AddNode {
            id: mentor,
            kind: "Entity".into(),
            name: "Sam Okafor".into(),
            summary: String::new(),
            attrs: json!({"labels": ["Entity"]}),
            group_id: GROUP.into(),
        }),
        WriteItem::Op(GraphOp::AddEdge {
            id: own_edge,
            src: first.src,
            dst: mentor,
            rel: "MENTORED_BY".into(),
            fact: "Priya is mentored by Sam Okafor.".into(),
            attrs: json!({}),
            group_id: GROUP.into(),
            valid_at: None,
            invalid_at: None,
        }),
        WriteItem::Op(GraphOp::AddEpisode {
            id: second,
            source: "chat".into(),
            kind: "message".into(),
            content: "Priya: still at Northwind; Sam mentors me.".into(),
            occurred_at: 1_700_000_100_000,
            group_id: GROUP.into(),
            metadata: None,
            // Endpoints of everything it evidences, as the pipeline does:
            // the re-stated edge and its src, plus its own node and edge.
            mentions: vec![first.src, first.dst, first.edge, mentor, own_edge],
        }),
    ])
    .unwrap();

    let outcome = delete_episode(&grit, second).unwrap();
    assert_eq!(outcome.episode_id, second);
    assert_eq!(
        outcome.purged_node_ids,
        vec![mentor],
        "only this episode mentions Sam Okafor"
    );
    assert_eq!(
        outcome.purged_edge_ids,
        vec![own_edge],
        "this episode is the mentoring fact's earliest evidence"
    );
    assert_eq!(
        sorted(outcome.kept_node_ids.clone()),
        sorted(vec![first.src, first.dst]),
        "the first episode still evidences both endpoints"
    );
    assert_eq!(
        outcome.kept_edge_ids,
        vec![first.edge],
        "an edge this episode only re-stated belongs to its creator"
    );

    for id in [mentor, own_edge, second] {
        assert_unreachable(&grit, id);
    }
    // What survived is intact, and now attributed to the first episode
    // alone — the re-statement's provenance went with it.
    for id in [first.src, first.dst, first.edge] {
        assert_eq!(grit.mentions_of(id).unwrap(), vec![first.episode]);
    }
    assert!(grit.get_edge_embedding(first.edge).unwrap().is_some());

    // Deleting the creator now takes the rest: nothing else evidences it.
    let outcome = delete_episode(&grit, first.episode).unwrap();
    assert_eq!(
        sorted(outcome.purged_node_ids.clone()),
        sorted(vec![first.src, first.dst])
    );
    assert_eq!(outcome.purged_edge_ids, vec![first.edge]);
    assert!(outcome.kept_node_ids.is_empty() && outcome.kept_edge_ids.is_empty());
    assert!(grit.nodes_in_group(GROUP).unwrap().is_empty());
    assert!(grit.edges_in_group(GROUP).unwrap().is_empty());
    assert!(grit.episodes_in_group(GROUP).unwrap().is_empty());

    drop(grit);
    remove_db(&path);
}

/// The other face of the raw-targets/alias-resolved-mentions asymmetry.
/// Resolving mentions through aliases is what makes the node rule right
/// ("would anything still evidence this entity?"), but `episode_targets`
/// returns the RAW id the episode wrote: a target merged away *after* the
/// mention was recorded is classified through its stub, and erased as
/// itself — leaving the canonical node that absorbed it standing with no
/// evidence at all. Unreachable through nacre's pipeline, which attributes
/// canonical ids in `mentions` and only merges fresh drafts into stored
/// nodes; pinned here as the hand-written/sync-era graph it belongs to,
/// like the edge-group case above.
#[test]
fn a_target_merged_away_after_the_mention_is_erased_through_its_stub() {
    let (grit, path) = open_db("merged-away-target");
    const GROUP: &str = "aliasing";
    let seeded = seed(&grit, GROUP, "Priya", "Priya works at Northwind Labs.");

    // Not something this pipeline writes: a merge of an already-mentioned
    // node into one drafted later.
    let canonical = grit.new_id();
    grit.apply_batch(vec![
        WriteItem::Op(GraphOp::AddNode {
            id: canonical,
            kind: "Entity".into(),
            name: "Priya Raman".into(),
            summary: String::new(),
            attrs: json!({"labels": ["Entity"]}),
            group_id: GROUP.into(),
        }),
        WriteItem::Op(GraphOp::MergeNodes {
            from: seeded.src,
            into: canonical,
        }),
    ])
    .unwrap();
    assert_eq!(
        grit.node(seeded.src).unwrap().unwrap().merged_into,
        Some(canonical),
        "the mentioned id is now an expired audit pointer, still a point-lookup hit"
    );
    assert_eq!(
        grit.mentions_of(canonical).unwrap(),
        vec![seeded.episode],
        "and the canonical node has inherited its evidence"
    );

    let outcome = delete_episode(&grit, seeded.episode).unwrap();
    assert!(
        outcome.purged_node_ids.contains(&seeded.src),
        "the stub is what the episode wrote, so the stub is what is weighed and erased"
    );
    assert!(
        !outcome.purged_node_ids.contains(&canonical)
            && !outcome.kept_node_ids.contains(&canonical),
        "the canonical node is not among this episode's targets at all"
    );
    assert!(
        grit.node(canonical).unwrap().is_some() && grit.mentions_of(canonical).unwrap().is_empty(),
        "so it survives with no evidence — the corner the rustdoc names"
    );

    drop(grit);
    remove_db(&path);
}

/// An episode that is not stored is reported, not silently ignored — the
/// mirror of `EpisodeExists`. A second delete of an episode already
/// deleted lands here too, and writes nothing.
#[test]
fn deleting_an_unknown_or_already_deleted_episode_is_reported_and_inert() {
    let (grit, path) = open_db("episode-not-found");
    let seeded = seed(&grit, "target", "Priya", "Priya works at Northwind Labs.");

    let unknown = grit.new_id();
    match delete_episode(&grit, unknown) {
        Err(PipelineError::EpisodeNotFound(id)) => assert_eq!(id, unknown),
        other => panic!("expected EpisodeNotFound, got {other:?}"),
    }

    delete_episode(&grit, seeded.episode).unwrap();
    let after = snapshot(&grit, "target");
    match delete_episode(&grit, seeded.episode) {
        Err(PipelineError::EpisodeNotFound(id)) => assert_eq!(id, seeded.episode),
        other => panic!("expected EpisodeNotFound on the second delete, got {other:?}"),
    }
    assert_eq!(
        snapshot(&grit, "target"),
        after,
        "the refused delete must not touch the graph"
    );

    drop(grit);
    remove_db(&path);
}

/// A forgotten id is spent. Re-using the deleted episode's id — the natural
/// reading of "`delete_episode` plus a fresh ingest", the documented recipe
/// for changing write-once `metadata` — is neither refused nor honoured,
/// and both halves of that are pinned here, because the rustdoc telling
/// callers to mint a NEW id is worth only as much as the mechanism under
/// it.
#[tokio::test]
async fn re_adding_a_purged_id_is_a_silent_no_op() {
    let (grit, path) = open_db("purged-id-reuse");
    const GROUP: &str = "reuse";
    let gone = seed(&grit, GROUP, "Priya", "Priya works at Northwind Labs.");
    delete_episode(&grit, gone.episode).unwrap();

    // Half one: the ingest guard cannot catch the reuse, and grit exposes
    // no per-id tombstone read that would let it. The episode really is
    // absent, so `add_episode` passes its idempotency check and walks on
    // into model I/O — refused here by an empty replay store, which stands
    // in for the real run that would have gone all the way to a commit.
    let meta = EmbedderMeta {
        model_id: "deletion-test".into(),
        dim: 3,
        model_version: String::new(),
    };
    let episode = EpisodeInput {
        id: Some(gone.episode),
        name: "the same memory, re-tagged".into(),
        content: "Priya: still at Northwind Labs.".into(),
        source: EpisodeSource::Message,
        source_description: "chat".into(),
        group_id: GROUP.into(),
        valid_at: None,
        metadata: Some("host-row:Priya/v2".into()),
    };
    let refused = add_episode(
        &grit,
        &ReplayModel::new(RecordingStore::new([])),
        &ReplayEmbedder::new(RecordingStore::new([]), meta),
        &episode,
        &[],
        &AddEpisodeOptions::default(),
        Utc.with_ymd_and_hms(2026, 7, 10, 0, 0, 0).unwrap(),
    )
    .await;
    assert!(
        matches!(refused, Err(PipelineError::Model(_))),
        "a purged id reads as absent, so the ingest guard lets it through \
         to model I/O — got {refused:?}"
    );

    // Half two: what that run would have committed. grit gates its inserts
    // on the tombstone table, so the episode row and every mention naming
    // it are dropped — while the node decided beside them lands, evidenced
    // by nothing.
    let orphan = grit.new_id();
    grit.apply_batch(vec![
        WriteItem::Op(GraphOp::AddNode {
            id: orphan,
            kind: "Entity".into(),
            name: "Sam Okafor".into(),
            summary: String::new(),
            attrs: json!({"labels": ["Entity"]}),
            group_id: GROUP.into(),
        }),
        WriteItem::Op(GraphOp::AddEpisode {
            id: gone.episode,
            source: "chat".into(),
            kind: "message".into(),
            content: episode.content.clone(),
            occurred_at: 1_700_000_200_000,
            group_id: GROUP.into(),
            metadata: episode.metadata.clone(),
            mentions: vec![orphan],
        }),
    ])
    .expect("the batch is accepted — that is the whole problem");

    assert!(
        grit.episode(gone.episode).unwrap().is_none(),
        "a tombstoned episode id stays dead: the re-ingest's row is dropped"
    );
    assert!(
        grit.episodes_in_group(GROUP).unwrap().is_empty(),
        "and it is dropped from the group's raw record too, not merely hidden"
    );
    assert!(
        grit.episode_targets(gone.episode).unwrap().is_empty(),
        "its mentions are dropped with it"
    );
    assert_eq!(
        grit.node(orphan).unwrap().map(|n| n.name),
        Some("Sam Okafor".to_owned()),
        "while the ops beside it land — this is the corruption the doc warns of"
    );
    assert!(
        grit.mentions_of(orphan).unwrap().is_empty(),
        "leaving a node no episode evidences"
    );

    drop(grit);
    remove_db(&path);
}

// ---------------------------------------------------------------------
// Tier A — kept nodes forget what their summaries learned
// ---------------------------------------------------------------------

/// The deleted episode's own words. High-entropy and unlike any prompt
/// text, so a substring hit anywhere is a leak, never a coincidence.
const FORGOTTEN: &str = "zephyrine-41c9";

/// A database with an injected clock, advanced one millisecond per write:
/// `node_history` orders a node's edges by creation time, so a test that
/// pins a rebuilt summary needs creation times that are distinct and known.
fn open_clocked(scenario: &str) -> (Grit, PathBuf, Arc<ManualClock>) {
    let dir = std::env::temp_dir().join(format!("nacre-deletion-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{scenario}.db"));
    remove_db(&path);
    let clock = Arc::new(ManualClock::new(1_700_000_000_000));
    let grit = Grit::open(&path, Options::new("nacre-deletion").clock(clock.clone())).unwrap();
    (grit, path, clock)
}

/// Writes a graph in the shape the pipeline leaves behind: nodes created
/// with an empty summary, each episode's new facts appended to both
/// endpoints' summaries by the fact-append shortcut and persisted as
/// summary-only `UpdateNode`s, and the episode mentioning every node and
/// edge it evidenced. One op per write, one millisecond apart.
struct Writer<'a> {
    grit: &'a Grit,
    clock: &'a ManualClock,
    group: String,
    summaries: HashMap<Uuid, String>,
}

impl Writer<'_> {
    fn op(&self, op: GraphOp) {
        self.clock.advance_ms(1);
        self.grit.apply(op).unwrap();
    }

    fn node(&mut self, name: &str) -> Uuid {
        self.node_in(name, &self.group.clone())
    }

    fn node_in(&mut self, name: &str, group: &str) -> Uuid {
        let id = self.grit.new_id();
        self.op(GraphOp::AddNode {
            id,
            kind: "Entity".into(),
            name: name.into(),
            summary: String::new(),
            attrs: json!({"labels": ["Entity"]}),
            group_id: group.into(),
        });
        self.summaries.insert(id, String::new());
        id
    }

    fn edge_in(&self, group: &str, src: Uuid, dst: Uuid, fact: &str) -> Uuid {
        let id = self.grit.new_id();
        self.op(GraphOp::AddEdge {
            id,
            src,
            dst,
            rel: "RELATES_TO".into(),
            fact: fact.into(),
            attrs: json!({}),
            group_id: group.into(),
            valid_at: None,
            invalid_at: None,
        });
        id
    }

    /// One episode: `facts` are the edges it creates (appended to both
    /// endpoints' summaries), `also` the existing nodes and edges it merely
    /// re-mentions.
    fn episode(
        &mut self,
        content: &str,
        facts: &[(Uuid, Uuid, &str)],
        also: &[Uuid],
    ) -> (Uuid, Vec<Uuid>) {
        let mut mentions: Vec<Uuid> = also.to_vec();
        let mut edges = Vec::new();
        let mut touched: Vec<Uuid> = Vec::new();
        for &(src, dst, fact) in facts {
            let group = self.group.clone();
            let edge = self.edge_in(&group, src, dst, fact);
            edges.push(edge);
            mentions.extend([src, dst, edge]);
            for end in [src, dst] {
                let summary = self.summaries.get_mut(&end).unwrap();
                *summary = format!("{summary}\n{fact}").trim().to_owned();
                if !touched.contains(&end) {
                    touched.push(end);
                }
            }
        }
        for node in touched {
            self.op(GraphOp::UpdateNode {
                id: node,
                name: None,
                summary: Some(self.summaries[&node].clone()),
                kind: None,
                attrs: None,
            });
        }
        mentions.sort();
        mentions.dedup();
        let id = self.grit.new_id();
        self.clock.advance_ms(1);
        let occurred_at = self.clock.now_ms_for_test();
        self.op(GraphOp::AddEpisode {
            id,
            source: "chat".into(),
            kind: "message".into(),
            content: content.into(),
            occurred_at,
            group_id: self.group.clone(),
            metadata: None,
            mentions,
        });
        (id, edges)
    }
}

/// `ManualClock` reads through grit's `Clock` trait.
trait NowForTest {
    fn now_ms_for_test(&self) -> i64;
}

impl NowForTest for ManualClock {
    fn now_ms_for_test(&self) -> i64 {
        grit_core::Clock::now_ms(self)
    }
}

/// A long, distinct fact from the first episode — long enough that ten of
/// them sit just under the fact-append bound, which is what lets the sweep
/// below push a kept node's rebuilt summary through a real LLM summary pass.
fn ledger_fact(i: usize) -> String {
    format!(
        "Priya shipped release {i:02} of the Northwind Labs ledger service, coordinating the \
         migration plan, the rollback drill, the on-call handoff and the customer notes."
    )
}

struct KeptGraph {
    priya: Uuid,
    northwind: Uuid,
    biscuit: Uuid,
    queens: Uuid,
    /// The episode the tests delete.
    deleted: Uuid,
    /// Priya's summary as the rebuild must leave it.
    priya_rebuilt: String,
    /// An edge incident to Priya in another group, and one whose other
    /// endpoint an earlier purge erased: neither may enter the rebuild.
    cross_group_fact: &'static str,
    ghost_fact: &'static str,
}

/// Three episodes about Priya. The second — the one deleted — adopts a dog
/// whose name is [`FORGOTTEN`], and the fact-append shortcut writes that
/// into Priya's and the dog's summaries; the third episode keeps both nodes
/// alive, so a correct deletion must keep them and still forget the dog's
/// name.
fn kept_graph(grit: &Grit, clock: &ManualClock) -> KeptGraph {
    let mut w = Writer {
        grit,
        clock,
        group: "kept".into(),
        summaries: HashMap::new(),
    };
    let priya = w.node("Priya");
    let northwind = w.node("Northwind Labs");
    let ledger: Vec<String> = (1..=10).map(ledger_fact).collect();
    let ledger_facts: Vec<(Uuid, Uuid, &str)> = ledger
        .iter()
        .map(|f| (priya, northwind, f.as_str()))
        .collect();
    let (_, ledger_edges) = w.episode(
        "Priya: ten releases of the ledger, all shipped.",
        &ledger_facts,
        &[],
    );

    let biscuit = w.node("Biscuit");
    let adopted = format!("Priya adopted a dog named {FORGOTTEN}.");
    let (deleted, _) = w.episode(
        &format!("Priya: we adopted {FORGOTTEN} today! Still shipping the ledger."),
        &[(priya, biscuit, &adopted)],
        // Re-states the first episode's fact: corroboration, not creation.
        &[ledger_edges[0], northwind],
    );

    let queens = w.node("Queens");
    w.episode(
        "Priya: walking Biscuit before the commute in from Queens.",
        &[
            (priya, queens, "Priya commutes from Queens."),
            (biscuit, priya, "Biscuit is Priya's dog."),
        ],
        &[],
    );

    // Hand-written residue a rebuild must not pick up.
    let cross_group_fact = "Priya mentors a colleague in another workspace.";
    w.edge_in("elsewhere", priya, northwind, cross_group_fact);
    let xavier = w.node("Xavier");
    let ghost_fact = "Priya met Xavier at the conference.";
    w.edge_in("kept", priya, xavier, ghost_fact);
    w.op(GraphOp::Purge { ids: vec![xavier] });

    let mut rebuilt = ledger;
    rebuilt.push("Priya commutes from Queens.".into());
    rebuilt.push("Biscuit is Priya's dog.".into());
    KeptGraph {
        priya,
        northwind,
        biscuit,
        queens,
        deleted,
        priya_rebuilt: rebuilt.join("\n"),
        cross_group_fact,
        ghost_fact,
    }
}

fn oplog_text(grit: &Grit) -> String {
    grit.oplog_since(0)
        .unwrap()
        .iter()
        .map(|(_, entry)| serde_json::to_string(&entry.op).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

fn export_text(grit: &Grit) -> String {
    let mut buf = Vec::new();
    grit.export_jsonl(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

/// The whole reset in one graph: every kept node's summary is rebuilt from
/// its surviving facts — oldest first, in its own group, both endpoints
/// live — and the deleted episode's words are gone from the summaries, the
/// op-log, the export and the search index. One batch: the `Purge`, one
/// `PurgeNodeFields`, then one rebuilt `UpdateNode` per kept node.
#[test]
fn delete_episode_resets_and_rebuilds_the_summaries_it_keeps() {
    let (grit, path, clock) = open_clocked("kept-summaries");
    let g = kept_graph(&grit, &clock);
    let queens_before = grit.node(g.queens).unwrap().unwrap().summary;
    assert!(
        grit.node(g.priya)
            .unwrap()
            .unwrap()
            .summary
            .contains(FORGOTTEN),
        "the setup must reproduce the leak: the summary learned the dog's name"
    );
    let log_before = grit.oplog_since(0).unwrap().len();

    let outcome = delete_episode(&grit, g.deleted).unwrap();
    assert_eq!(
        sorted(outcome.kept_node_ids.clone()),
        sorted(vec![g.priya, g.northwind, g.biscuit])
    );
    assert_eq!(outcome.purged_node_ids, Vec::<Uuid>::new());
    assert_eq!(outcome.purged_edge_ids.len(), 1, "the adoption fact");

    let summary = |id: Uuid| grit.node(id).unwrap().unwrap().summary;
    assert_eq!(summary(g.priya), g.priya_rebuilt);
    assert_eq!(summary(g.biscuit), "Biscuit is Priya's dog.");
    assert_eq!(
        summary(g.northwind),
        (1..=10).map(ledger_fact).collect::<Vec<_>>().join("\n"),
        "a node whose summary the episode never changed is rebuilt to the same facts"
    );
    assert_eq!(
        summary(g.queens),
        queens_before,
        "a node the episode never mentioned is untouched"
    );
    assert!(!summary(g.priya).contains(g.cross_group_fact));
    assert!(!summary(g.priya).contains(g.ghost_fact));

    for (place, text) in [
        ("op-log", oplog_text(&grit)),
        ("export", export_text(&grit)),
    ] {
        assert!(
            !text.contains(FORGOTTEN),
            "the {place} still carries the forgotten name"
        );
    }
    assert!(
        grit.search(grit_core::Query::text(FORGOTTEN).group("kept"))
            .unwrap()
            .is_empty()
    );

    // The batch, in order, at the tail of the log.
    let log = grit.oplog_since(0).unwrap();
    let tail: Vec<&GraphOp> = log.iter().map(|(_, e)| &e.op).rev().take(5).collect();
    let mut rebuilt: Vec<Uuid> = tail[..3]
        .iter()
        .map(|op| match op {
            GraphOp::UpdateNode {
                id,
                summary: Some(_),
                name: None,
                kind: None,
                attrs: None,
            } => *id,
            other => panic!("expected a rebuilt summary, got {other:?}"),
        })
        .collect();
    rebuilt.sort();
    assert_eq!(rebuilt, sorted(outcome.kept_node_ids.clone()));
    assert!(matches!(
        tail[3],
        GraphOp::PurgeNodeFields { ids, fields }
            if sorted(ids.clone()) == sorted(outcome.kept_node_ids.clone())
                && fields == &[NodeField::Summary]
    ));
    assert!(matches!(tail[4], GraphOp::Purge { .. }));
    // Five entries in; out went the episode, its fact, and every
    // summary-only update the three kept nodes ever had (one per episode
    // that appended to them: three for Priya, two for Biscuit, one for
    // Northwind).
    let removed = 1 + 1 + 6;
    assert_eq!(log.len(), log_before + 5 - removed);

    drop(grit);
    remove_db(&path);
}

/// The rebuild keeps the newest whole facts that fit in the fact-append
/// bound (`2 × MAX_SUMMARY_CHARS`), oldest first within them — the most
/// upstream ever stores without an LLM pass.
#[test]
fn a_rebuilt_summary_keeps_the_newest_facts_within_the_bound() {
    let (grit, path, clock) = open_clocked("kept-bound");
    let mut w = Writer {
        grit: &grit,
        clock: &clock,
        group: "bound".into(),
        summaries: HashMap::new(),
    };
    let hub = w.node("Hub");
    let other = w.node("Other");
    // Thirty facts of exactly 100 characters each.
    let facts: Vec<String> = (0..30)
        .map(|i| format!("Hub fact {i:02}: {}", "x".repeat(87)))
        .collect();
    assert!(facts.iter().all(|f| f.chars().count() == 100));
    for fact in &facts {
        w.episode("another day at the hub", &[(hub, other, fact)], &[]);
    }
    let (deleted, _) = w.episode(
        &format!("hub and {FORGOTTEN}"),
        &[(hub, other, FORGOTTEN)],
        &[],
    );

    delete_episode(&grit, deleted).unwrap();
    // 19 facts and 18 newlines are 1918 characters; a 20th would be 2019.
    let expected = facts[11..].join("\n");
    assert_eq!(expected.chars().count(), 1918);
    assert_eq!(grit.node(hub).unwrap().unwrap().summary, expected);

    drop(grit);
    remove_db(&path);
}

/// A language model that answers by schema name and records every request
/// — what a test needs to assert what the model was *not* shown. (The
/// pipeline's own test double is private to its unit tests.)
struct ScriptModel {
    responses: Vec<(&'static str, Value)>,
    seen: Mutex<Vec<Value>>,
}

impl LanguageModel for ScriptModel {
    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> impl std::future::Future<Output = Result<Value, ModelError>> + Send {
        self.seen
            .lock()
            .unwrap()
            .push(serde_json::to_value(request).unwrap());
        let result = self
            .responses
            .iter()
            .find(|(schema, _)| *schema == request.schema_name)
            .map(|(_, response)| response.clone())
            .ok_or_else(|| ModelError::ReplayMiss {
                canonical_request: request.schema_name.clone(),
            });
        async move { result }
    }
}

/// Every input embeds to the same vector: every stored node is a dedup
/// candidate for every draft, so the candidate pool is the whole group.
struct ConstEmbedder;

impl Embedder for ConstEmbedder {
    fn meta(&self) -> EmbedderMeta {
        EmbedderMeta {
            model_id: "const-test".into(),
            dim: 3,
            model_version: String::new(),
        }
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
        Ok(inputs.iter().map(|_| vec![1.0, 1.0, 1.0]).collect())
    }
}

/// The promise, end to end: after the deletion, the next ingest that
/// touches the kept nodes shows the model their rebuilt summaries — in the
/// dedupe prompt and in an LLM summary pass — and nothing the deleted
/// episode said, in any request it issues. Swept like the metadata
/// sentinel: every request, rendered whole.
#[tokio::test]
async fn a_later_ingest_never_sees_what_the_deleted_episode_taught() {
    let (grit, path, clock) = open_clocked("kept-sweep");
    let g = kept_graph(&grit, &clock);
    delete_episode(&grit, g.deleted).unwrap();

    // "Priya" resolves to the stored node by exact name (no model call), so
    // her rebuilt summary goes to the summarize step — and this fact pushes
    // it past the fact-append bound, forcing a real LLM summary pass.
    // "Biscuit the dog" is no exact match, so it goes to the dedupe prompt
    // with the whole group as candidates.
    let long_fact = format!(
        "Priya took Biscuit the dog to the vet in Queens, {}",
        "then to the park and back along the river path. ".repeat(10)
    );
    assert!(
        g.priya_rebuilt.chars().count() + 1 + long_fact.chars().count()
            > 2 * nacre_core::prompts::MAX_SUMMARY_CHARS,
        "the fixture must push Priya past the fact-append bound"
    );
    let model = ScriptModel {
        responses: vec![
            (
                "ExtractedEntities",
                json!({"extracted_entities": [
                    {"name": "Priya", "entity_type_id": 0, "episode_indices": [0]},
                    {"name": "Biscuit the dog", "entity_type_id": 0, "episode_indices": [0]},
                ]}),
            ),
            (
                "NodeResolutions",
                json!({"entity_resolutions": [
                    {"id": 0, "name": "Biscuit the dog", "duplicate_candidate_id": -1},
                ]}),
            ),
            (
                "ExtractedEdges",
                json!({"edges": [{
                    "source_entity_name": "Priya",
                    "target_entity_name": "Biscuit the dog",
                    "relation_type": "TOOK_TO_VET",
                    "fact": long_fact,
                    "episode_indices": [0],
                }]}),
            ),
            (
                "EdgeDuplicate",
                json!({"duplicate_facts": [], "contradicted_facts": []}),
            ),
            (
                "EdgeTimestamps",
                json!({"valid_at": null, "invalid_at": null}),
            ),
            (
                "SummarizedEntities",
                json!({"summaries": [
                    {"name": "Priya", "summary": "Priya ships the ledger and walks her dog."},
                ]}),
            ),
        ],
        seen: Mutex::new(Vec::new()),
    };
    let now = Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap();
    let episode = EpisodeInput {
        id: None,
        name: "later".into(),
        content: "Priya: took Biscuit the dog to the vet in Queens.".into(),
        source: EpisodeSource::Message,
        source_description: "chat".into(),
        group_id: "kept".into(),
        valid_at: Some("2030-01-01T00:00:00+00:00".into()),
        metadata: None,
    };
    let previous = retrieve_previous_episodes(
        &grit,
        "kept",
        EpisodeSource::Message,
        now,
        PREVIOUS_EPISODE_WINDOW,
    )
    .unwrap();
    add_episode(
        &grit,
        &model,
        &ConstEmbedder,
        &episode,
        &previous,
        &AddEpisodeOptions::default(),
        now,
    )
    .await
    .unwrap();

    let requests = model.seen.lock().unwrap().clone();
    let rendered = |schema: &str| -> Vec<String> {
        requests
            .iter()
            .filter(|r| r["schema_name"] == schema)
            .map(Value::to_string)
            .collect()
    };
    // Positive controls: the kept nodes really reached both prompts, so the
    // sweep below is checking something.
    assert!(
        rendered("NodeResolutions")
            .iter()
            .any(|r| r.contains("Biscuit is Priya's dog.")),
        "the dedupe prompt must show the kept dog's rebuilt summary"
    );
    assert!(
        rendered("SummarizedEntities")
            .iter()
            .any(|r| r.contains("Priya commutes from Queens.")),
        "an LLM summary pass must have seen Priya's rebuilt summary"
    );
    assert!(
        requests
            .iter()
            .any(|r| r.to_string().contains("walking Biscuit before the commute")),
        "the previous-episode window must reach a prompt"
    );
    for request in &requests {
        assert!(
            !request.to_string().contains(FORGOTTEN),
            "the deleted episode leaked into a {} request",
            request["schema_name"]
        );
    }

    drop(grit);
    remove_db(&path);
}

// ---------------------------------------------------------------------
// Tier B — the same laws over a graph the pipeline wrote
// ---------------------------------------------------------------------

fn trace_episode(spec: &Value, group_id: &str) -> EpisodeInput {
    EpisodeInput {
        id: None,
        name: spec["name"].as_str().unwrap().into(),
        content: spec["content"].as_str().unwrap().into(),
        source: match spec["source"].as_str().unwrap_or("message") {
            "text" => EpisodeSource::Text,
            "json" => EpisodeSource::Json,
            _ => EpisodeSource::Message,
        },
        source_description: spec["source_description"].as_str().unwrap().into(),
        group_id: group_id.to_owned(),
        valid_at: Some(spec["reference_time"].as_str().unwrap().into()),
        metadata: None,
    }
}

/// Replay every episode of `spec` into `group_id`.
async fn ingest_trace(
    grit: &Grit,
    spec: &Value,
    group_id: &str,
    model: &ReplayModel,
    embedder: &ReplayEmbedder,
) {
    for episode_spec in spec["episodes"].as_array().unwrap() {
        let episode = trace_episode(episode_spec, group_id);
        let now = episode
            .valid_at
            .as_deref()
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|t| t.with_timezone(&Utc))
            .unwrap_or_else(|| Utc.with_ymd_and_hms(2026, 7, 10, 0, 0, 0).unwrap());
        let previous = retrieve_previous_episodes(
            grit,
            group_id,
            episode.source,
            now,
            PREVIOUS_EPISODE_WINDOW,
        )
        .unwrap();
        add_episode(
            grit,
            model,
            embedder,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .unwrap_or_else(|e| panic!("add_episode({}) into {group_id} failed: {e}", episode.name));
    }
}

/// trace1 ingested twice into one database under two group ids, then
/// deleted one episode and one whole group. The sibling is the control: it
/// went through the identical pipeline, so any leakage across the group
/// boundary — a purged id shared, an index entry not group-scoped, a vector
/// dropped for the wrong row — shows up as a diff against its snapshot.
#[tokio::test]
async fn trace1_deletion_leaves_the_sibling_group_byte_identical() {
    let trace_dir = oracle_dir().join("fixtures").join("trace1");
    let episodes_path = trace_dir.join("episodes.json");
    if !episodes_path.exists() {
        eprintln!(
            "SKIP: golden trace \"trace1\" not captured yet ({}). Run oracle/capture.py first.",
            episodes_path.display()
        );
        return;
    }
    let spec: Value =
        serde_json::from_str(&std::fs::read_to_string(&episodes_path).unwrap()).unwrap();
    let retrieval: Value =
        serde_json::from_str(&std::fs::read_to_string(trace_dir.join("retrieval.json")).unwrap())
            .unwrap();
    let model =
        ReplayModel::new(RecordingStore::load(&trace_dir.join("llm_recordings.json")).unwrap());
    let embedder = ReplayEmbedder::new(
        RecordingStore::load(&trace_dir.join("embedder_recordings.json")).unwrap(),
        EmbedderMeta {
            model_id: "embedding-3".into(),
            dim: 1024,
            model_version: String::new(),
        },
    );

    const TARGET: &str = "trace1";
    const SIBLING: &str = "trace1-sibling";
    let (grit, path) = open_db("trace1-two-groups");
    ingest_trace(&grit, &spec, TARGET, &model, &embedder).await;
    ingest_trace(&grit, &spec, SIBLING, &model, &embedder).await;

    let target_before = snapshot(&grit, TARGET);
    let sibling_before = snapshot(&grit, SIBLING);
    assert_eq!(
        (
            target_before.nodes.len(),
            target_before.edges.len(),
            target_before.episodes.len()
        ),
        (
            sibling_before.nodes.len(),
            sibling_before.edges.len(),
            sibling_before.episodes.len()
        ),
        "replaying the same recordings under a second group must build the \
         same shape — otherwise the sibling is not a control"
    );
    assert!(
        !target_before.embedded_edges.is_empty(),
        "the trace must have persisted fact vectors for the checks below"
    );

    // --- One episode: the last one, whose residue is real pipeline output
    // (merges into existing nodes, new facts, invalidations). ---
    let last_episode = grit.episodes_in_group(TARGET).unwrap().pop().unwrap().id;
    let sole_evidence: Vec<Uuid> = grit
        .episode_targets(last_episode)
        .unwrap()
        .into_iter()
        .filter(|id| grit.mentions_of(*id).unwrap() == [last_episode])
        .collect();
    let carried = {
        let mut ids = sole_evidence.clone();
        ids.push(last_episode);
        creator_ops(&grit, &ids).len()
    };
    // The kept half, read before the delete: the facts the episode created
    // (erased with their edges), and the summary-only updates on the nodes
    // it keeps (every one of them predates the reset, so the field purge
    // deletes it whole — label promotions write kind/attrs and survive).
    let kept_nodes: Vec<Uuid> = grit
        .episode_targets(last_episode)
        .unwrap()
        .into_iter()
        .filter(|id| grit.node(*id).unwrap().is_some() && !sole_evidence.contains(id))
        .collect();
    let created_facts: Vec<String> = sole_evidence
        .iter()
        .filter_map(|id| grit.edge(*id).unwrap())
        .map(|edge| edge.fact)
        .collect();
    let summary_only_updates = grit
        .oplog_since(0)
        .unwrap()
        .iter()
        .filter(|(_, entry)| {
            matches!(&entry.op, GraphOp::UpdateNode { id, name: None, summary: Some(_), kind: None, attrs: None }
                if kept_nodes.contains(id))
        })
        .count();
    assert!(
        kept_nodes.iter().any(|id| {
            let summary = grit.node(*id).unwrap().unwrap().summary;
            created_facts
                .iter()
                .any(|fact| summary.contains(fact.as_str()))
        }),
        "the trace must exercise the leak being fixed: a kept summary carrying a created fact"
    );
    let summaries_before: Vec<String> = kept_nodes
        .iter()
        .map(|id| grit.node(*id).unwrap().unwrap().summary)
        .collect();
    let outcome = delete_episode(&grit, last_episode).unwrap();
    let purged: Vec<Uuid> = sorted(
        outcome
            .purged_node_ids
            .iter()
            .chain(&outcome.purged_edge_ids)
            .copied()
            .collect(),
    );
    assert_eq!(
        purged,
        sorted(sole_evidence),
        "exactly the ids this episode was the sole evidence for"
    );
    assert!(
        !purged.is_empty() && !outcome.kept_node_ids.is_empty(),
        "trace1's last episode must exercise both halves of the rule"
    );
    for id in &purged {
        assert_unreachable(&grit, *id);
    }
    assert_unreachable(&grit, last_episode);
    for id in outcome.kept_node_ids.iter().chain(&outcome.kept_edge_ids) {
        assert!(
            grit.node(*id).unwrap().is_some() || grit.edge(*id).unwrap().is_some(),
            "{id} was corroborated elsewhere and must survive"
        );
        assert!(
            !grit.mentions_of(*id).unwrap().contains(&last_episode),
            "{id} must lose the deleted episode's provenance"
        );
    }
    assert!(carried > 0, "the deleted episode's own entry alone carried");
    assert_eq!(
        sorted(outcome.kept_node_ids.clone()),
        sorted(kept_nodes.clone())
    );
    // Kept summaries: reset and rebuilt, carrying none of the facts the
    // episode created.
    let rebuilt = kept_nodes
        .iter()
        .filter(|id| !grit.node(**id).unwrap().unwrap().summary.is_empty())
        .count();
    for (id, before) in kept_nodes.iter().zip(&summaries_before) {
        let summary = grit.node(*id).unwrap().unwrap().summary;
        for fact in &created_facts {
            assert!(
                !summary.contains(fact.as_str()),
                "kept node {id} still carries {fact:?} in {summary:?}"
            );
        }
        // Fidelity: trace1's summaries are pure fact-append lists, so the
        // rebuild is exactly the old summary with the deleted episode's lines
        // taken out — save that a fact appended twice (upstream's quirk for a
        // node two drafts resolved to) is listed once.
        let mut expected: Vec<&str> = Vec::new();
        for line in before.lines() {
            if !created_facts.iter().any(|fact| fact == line) && !expected.contains(&line) {
                expected.push(line);
            }
        }
        assert_eq!(summary, expected.join("\n"), "kept node {id}'s rebuild");
    }
    let target_log: String = grit
        .oplog_since(0)
        .unwrap()
        .iter()
        .filter(|(_, entry)| match &entry.op {
            GraphOp::AddNode { id, .. } | GraphOp::UpdateNode { id, .. } => kept_nodes.contains(id),
            _ => false,
        })
        .map(|(_, entry)| serde_json::to_string(&entry.op).unwrap())
        .collect();
    for fact in &created_facts {
        assert!(
            !target_log.contains(fact.as_str()),
            "a kept node's op-log history still carries {fact:?}"
        );
    }
    assert_eq!(
        snapshot(&grit, SIBLING),
        Snapshot {
            // Only the shared log moved, and by exactly this much: the
            // Purge, minus the entries it scrubbed; the field purge, minus
            // the summary-only updates it emptied; one rebuilt summary per
            // kept node with a surviving fact. The sibling's own entries
            // are still there verbatim.
            oplog: sibling_before.oplog + 1 - carried + 1 - summary_only_updates + rebuilt,
            ..sibling_before
        },
        "an episode deletion must not touch the sibling group"
    );
    let sibling_before = snapshot(&grit, SIBLING);

    // --- Then the whole group. ---
    let carried = snapshot(&grit, TARGET).creator_ops.len();
    let outcome = delete_group(&grit, TARGET).unwrap();
    assert!(grit.nodes_in_group(TARGET).unwrap().is_empty());
    assert!(grit.edges_in_group(TARGET).unwrap().is_empty());
    assert!(grit.episodes_in_group(TARGET).unwrap().is_empty());
    for id in outcome
        .purged_node_ids
        .iter()
        .chain(&outcome.purged_edge_ids)
        .chain(&outcome.purged_episode_ids)
    {
        assert_unreachable(&grit, *id);
    }
    assert!(
        retrieve_previous_episodes(
            &grit,
            TARGET,
            EpisodeSource::Message,
            Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap(),
            PREVIOUS_EPISODE_WINDOW,
        )
        .unwrap()
        .is_empty(),
        "the previous-episode window over a forgotten group is empty"
    );
    assert_eq!(
        snapshot(&grit, SIBLING),
        Snapshot {
            oplog: sibling_before.oplog + 1 - carried,
            ..sibling_before
        },
        "the sibling group is byte-identical after both deletions"
    );

    // Retrieval: nothing left to find in the forgotten group, and the
    // sibling still answers every trace query (FTS mirrors and vectors are
    // shared tables — this is where a non-group-scoped delete would show).
    for query_spec in retrieval.as_array().unwrap() {
        let query = query_spec["query"].as_str().unwrap();
        assert!(
            search_edges(&grit, &embedder, query, TARGET, 10)
                .await
                .unwrap()
                .is_empty(),
            "the forgotten group still answers {query:?}"
        );
        assert!(
            !search_edges(&grit, &embedder, query, SIBLING, 10)
                .await
                .unwrap()
                .is_empty(),
            "the sibling group lost hits for {query:?}"
        );
    }

    drop(grit);
    remove_db(&path);
}
