//! Forgetting: `delete_group` erases exactly one namespace, and
//! `delete_episode` erases exactly what one episode is the sole evidence
//! for. Both commit as a single `Purge` — one transaction, one audit
//! record.
//!
//! Two tiers, because the laws and the realism are separable:
//!
//! - **Hand-built grit state** for the mechanical laws: group isolation, the
//!   empty-`Purge` trap, the all-or-nothing batch, the enumeration
//!   boundaries a group scan cannot cross (an edge written into another
//!   group, a merge placeholder parked in the default group), the
//!   documented non-guarantee that the op-log keeps what was purged, and
//!   that a forgotten id is spent — re-using one is a silent no-op.
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

use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use grit_core::{GraphOp, Grit, Options, WriteItem};
use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::forget::{delete_episode, delete_group};
use nacre_core::model::{EmbedderMeta, RecordingStore, ReplayEmbedder, ReplayModel};
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
/// directions, which ids hold a vector, and the oplog length.
#[derive(Debug, PartialEq)]
struct Snapshot {
    oplog: usize,
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
    Snapshot {
        oplog: grit.oplog_since(0).unwrap().len(),
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

    // One transaction, one audit record.
    assert_eq!(
        grit.oplog_since(0).unwrap().len(),
        oplog_before + 1,
        "a whole-group deletion is a single op"
    );

    assert_eq!(
        snapshot(&grit, "sibling"),
        Snapshot {
            // The sibling's own rows are untouched; only the shared oplog
            // grew, by the one Purge asserted above.
            oplog: before_sibling.oplog + 1,
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

/// The guarantee is "unreachable through every read API and every index",
/// not "gone from the file". grit keeps the original ops — the op-log is
/// the sync unit and rewriting history would break the merge laws — so an
/// application promising erasure to a person must promise it on its own
/// terms. Pinned here so the claim is tested, not assumed.
#[test]
fn the_oplog_keeps_what_was_purged() {
    let (grit, path) = open_db("oplog-retention");
    const FACT: &str = "Priya works at Northwind Labs.";
    let target = seed(&grit, "target", "Priya", FACT);

    delete_group(&grit, "target").unwrap();
    assert_unreachable(&grit, target.edge);

    let log = grit.oplog_since(0).unwrap();
    assert!(
        log.iter()
            .any(|(_, entry)| serde_json::to_string(&entry.op).unwrap().contains(FACT)),
        "the op that first carried the fact stays in the log, by design"
    );
    assert!(
        matches!(log.last().unwrap().1.op, GraphOp::Purge { .. }),
        "and the purge itself is the audit record"
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
    assert_eq!(
        snapshot(&grit, SIBLING),
        Snapshot {
            oplog: sibling_before.oplog + 1,
            ..sibling_before
        },
        "an episode deletion must not touch the sibling group"
    );
    let sibling_before = snapshot(&grit, SIBLING);

    // --- Then the whole group. ---
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
            oplog: sibling_before.oplog + 1,
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
