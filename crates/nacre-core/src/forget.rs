//! Deletion: the right-to-forget surface.
//!
//! Ports upstream's two removal entry points onto grit's purge family. Nacre
//! decides *what* to erase — which is a judgment about evidence, not a
//! storage question — and grit executes it:
//!
//! - [`delete_episode`] ports `graphiti_core/graphiti.py::remove_episode`:
//!   forget one episode and the graph that exists only because of it.
//! - [`delete_group`] ports the service-layer `delete_group`
//!   (`server/graph_service/zep_graphiti.py`); `graphiti_core` has no
//!   group-deletion entry point of its own, only `clear_data`, which is a
//!   test-fixture reset.
//!
//! **One batch, one transaction.** Each call decides everything first and
//! commits it through one `apply_batch`, so a deletion is as atomic as an
//! ingest: it happens whole or not at all. [`delete_group`]'s batch is a
//! single `GraphOp::Purge`, one oplog entry naming what was forgotten.
//! [`delete_episode`]'s batch grows beyond that whenever the episode leaves
//! nodes behind: the `Purge`, then one `GraphOp::PurgeNodeFields` resetting
//! the summaries of the nodes it keeps, then one `UpdateNode` per kept node
//! whose summary is rebuilt — so its audit record is the two content-free
//! purge entries, and the rebuilt summaries are ordinary, current content.
//! Upstream issues one un-transacted call per kind (edges, then nodes, then
//! the episode), so a failure between them leaves a half-forgotten graph —
//! the same class of defect the atomic ingest fixed (DEVIATIONS.md "Atomic
//! episode commit").
//!
//! **The write is atomic; the decision above it is not.** Which ids to erase
//! is settled by a sequence of reads, and grit's reader is a separate WAL
//! connection that never blocks behind the writer — so a `Grit` handle also
//! being written through can gain corroborating evidence after this call has
//! weighed it, and the purge, already decided, erases the id anyway. Nacre
//! cannot close that window: grit exposes no transaction to Layer 2 (nacre
//! decides, grit executes), deliberately. A caller that deletes while
//! ingesting into the same group should serialize the two. Upstream holds no
//! transaction at any point, so the gap is narrower here than in the code
//! being ported — but it is a gap, and the paragraph above it should not be
//! read as covering the decision.
//!
//! **What "forgotten" means.** grit's purge family is a tiered promise,
//! stated there in full on `GraphOp::Purge` and `GraphOp::PurgeNodeFields`,
//! and nacre inherits it whole — never more than it:
//!
//! 1. **Erased from the live file, the op-log included.** The derived state
//!    goes — the rows, their per-field update and invalidation records,
//!    mention links in both directions, every FTS mirror entry and every
//!    stored vector — *and* so do the op-log entries that **created** the
//!    purged ids, in the same transaction. Nothing reaches the content
//!    through a read, through `oplog_since`, or through an export, and a
//!    copy of a scrubbed entry re-delivered by a peer is refused at ingest
//!    rather than re-logged. Entries that merely *reference* a purged id —
//!    an edge's endpoints, an episode's mentions, a merge's pointers, the
//!    `Purge` itself, which stays as the audit record — are kept
//!    deliberately: opaque uuids, content-free, and what keeps grit's merge
//!    laws exact.
//! 2. **The pages the purge frees are zeroed** (`secure_delete` scoped to
//!    that transaction, then a WAL checkpoint). Two limits grit states and
//!    nacre cannot improve on: a checkpoint that finds a reader busy defers
//!    the write-back to the next one, and FTS5 records a deletion by
//!    writing a *new* segment that re-encodes the deleted terms — the index
//!    stops matching immediately, but those bytes live until the segments
//!    merge. `Grit::compact()` resolves both; it is maintenance, O(file
//!    size), and neither nacre nor grit ever calls it for you.
//! 3. **Out of scope, and it stays out of scope**: OS and platform backups
//!    or snapshots, copies taken before the delete, blocks retained beneath
//!    the filesystem, and devices that never receive the `Purge` op — one
//!    that *does* receive it erases on arrival. An application promising a
//!    person erasure still owns those; what it no longer has to own is the
//!    live file.
//!
//! Bounded, on top of all three, by what was listed: these two functions
//! decide the ids, and an id neither of them enumerates keeps its row and
//! its entry. Each documents its own boundary below.
//!
//! **Kept nodes forget what their summaries learned.** An entity another
//! episode also mentions outlives an episode deletion, but its summary does
//! not come through unchanged: the summarize pass rewrote it while the
//! deleted episode was ingested — in the fact-append path by literally
//! appending the facts of the edges that episode created, which the purge
//! has just erased — and every later rewrite descended from that one. So
//! [`delete_episode`] resets the summary of every node it keeps, erasing
//! every earlier value from the live file and the op-log at the tiers above
//! (`PurgeNodeFields` — the floor it leaves also strips a stale value that
//! arrives later), and writes in its place a fresh list of the node's
//! surviving facts. No model call is made. What this does **not** reach, and
//! nacre does not claim:
//!
//! - **Context.** A later episode saw the deleted one in its
//!   previous-episode window, and what its extraction (or an LLM summary
//!   pass) derived *through that context* is attributed to the later
//!   episode, not to this one. It stays. Upstream is the same.
//! - **Invalidations.** A kept edge this episode invalidated stays
//!   invalidated; grit's invalidations are a MIN register with no
//!   retraction.
//! - **Labels and names.** A label promotion this episode caused on a kept
//!   node, and a kept node's name (its creating episode's spelling), stay.
//!   The merged-away draft rows below are unreachable, as documented.
//! - **A concurrent writer.** A summary refresh computed elsewhere from the
//!   old value, with a later HLC, wins and can carry it back; one with an
//!   earlier HLC is dropped. Serialize deletes against ingests — the same
//!   advice as the decision gap above.
//!
//! **A forgotten id is spent.** Purge also tombstones every id it names,
//! permanently, and grit gates its inserts on that tombstone table: a later
//! op naming a purged id is accepted, applied as a no-op, and reported as
//! success. That is what makes purge commute with concurrent adds (a write
//! still in flight from another device cannot resurrect what was forgotten),
//! but it means an id is usable exactly once — here and on every device that
//! syncs the purge. [`delete_episode`] spells out what that costs at the
//! ingest seam, where the id is the caller's.

use std::collections::HashSet;

use grit_core::{GraphOp, Grit, NodeField, WriteItem};
use uuid::Uuid;

use crate::pipeline::PipelineError;
use crate::prompts::MAX_SUMMARY_CHARS;

/// What one [`delete_episode`] call erased, and what it deliberately left.
///
/// The `kept_*` lists are the interesting half: they are the nodes and facts
/// this episode evidenced that **other** episodes evidence too, and are
/// therefore still true after it is forgotten. A caller showing a user what
/// a deletion did should show both halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteEpisodeOutcome {
    /// The episode that was erased.
    pub episode_id: Uuid,
    /// Nodes erased: mentioned by this episode and no other.
    pub purged_node_ids: Vec<Uuid>,
    /// Edges erased: this episode was their earliest evidence.
    pub purged_edge_ids: Vec<Uuid>,
    /// Nodes kept: other episodes mention them too. Each one's summary was
    /// reset and rebuilt from its surviving facts — see [`delete_episode`],
    /// "Kept nodes' summaries".
    pub kept_node_ids: Vec<Uuid>,
    /// Edges kept: an earlier episode is their evidence.
    pub kept_edge_ids: Vec<Uuid>,
}

/// What one [`delete_group`] call erased. Empty everywhere when the group
/// held nothing — an unknown group is not an error, it is a group with no
/// rows (see [`delete_group`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeleteGroupOutcome {
    /// Node rows erased, id-ordered.
    pub purged_node_ids: Vec<Uuid>,
    /// Edge rows erased, id-ordered.
    pub purged_edge_ids: Vec<Uuid>,
    /// Episode rows erased, in the group scan's chronological order.
    pub purged_episode_ids: Vec<Uuid>,
}

/// Forget one episode, and the graph that exists only because of it.
///
/// Upstream's rule, restated on grit's data model:
///
/// - **Edges** this episode *created* go; edges it merely corroborated stay.
/// - **Nodes** mentioned by this episode *and no other* go; nodes another
///   episode also mentions stay.
/// - The episode row itself always goes.
/// - **Kept nodes' summaries are reset and rebuilt** (a nacre extension —
///   see below).
///
/// Everything is decided before anything is written, then committed in one
/// batch — one transaction.
///
/// # Deriving "created by"
///
/// Upstream reads `edge.episodes[0]`: its episode list is an append-ordered
/// array, so the first entry is the episode that first stated the fact.
/// grit's `mentions` is a **set** (primary key `(episode_id, target_id)`)
/// with no insertion order, so that index does not exist here. nacre derives
/// the creator instead: of the episodes mentioning the edge, the one
/// recorded earliest (`Episode::created_at`, ties broken by id). Both name
/// the episode that first evidenced the fact; see DEVIATIONS.md "Episode
/// deletion: earliest evidence stands in for `episodes[0]`" for why the
/// derivation is faithful and where it can differ.
///
/// # What this does not reach
///
/// - **Merged-away draft rows.** When a draft node resolves onto an existing
///   entity, the pipeline attributes the *canonical* id in `mentions`, and
///   grit keeps the absorbed draft row as an expired audit pointer. That row
///   is not provenance-linked to any episode, so no episode deletion reaches
///   it. Upstream has no such row at all (it never persists merged-away
///   drafts), so there is no upstream behavior to match. [`delete_group`]
///   does sweep them — it enumerates rows, not evidence.
/// - **The node behind a merged-away target**, in a graph nacre did not
///   write. `episode_targets` returns the raw ids the episode wrote while
///   `mentions_of` resolves merge aliases — the asymmetry that makes the
///   node rule right. Where a mentioned id was merged away *afterwards*,
///   the two meet: the absorbed stub is weighed as the episode's target,
///   found to be sole evidence, and erased as itself, while the node that
///   absorbed it — not one of this episode's targets — survives with no
///   evidence at all. The pipeline attributes canonical ids in `mentions`
///   and only ever merges fresh drafts into stored nodes, so it never
///   writes that shape.
/// - **Incident edges of a purged node**, in a graph nacre did not write.
///   grit's purge does not cascade (a cascade would make the op
///   order-dependent against concurrent writes), so an edge is erased only
///   by the rule above. In a graph this pipeline wrote, that is never a gap:
///   an episode mentions the endpoints of every edge it mentions, so a node
///   whose only evidence is this episode has no edge with other evidence,
///   and the two rules agree. A hand-written graph that breaks the invariant
///   can be left with an edge whose endpoint is gone — reads filter such an
///   edge out of traversal and search, but its fact sentence is still in the
///   group scan.
///
/// # Kept nodes' summaries
///
/// Every kept node gets a `GraphOp::PurgeNodeFields` on its summary, in the
/// same batch as the `Purge`: the summary resets, and every earlier value of
/// it leaves the live file and the op-log (see the module docs for what that
/// does and does not reach). In its place goes a **fresh fact list** built
/// the way upstream's fact-append shortcut builds summaries, from what the
/// graph still holds — not a restoration of what the summary would have been
/// without this episode, which no stored state can reproduce once an LLM
/// pass has compressed it:
///
/// - the node's incident edges in creation order (`node_history`'s: the
///   edges one episode wrote come back in the order it wrote them),
/// - in the node's own group (an edge written into another namespace stays
///   out of this one's prompts),
/// - with both endpoints still live — none purged by this call or an
///   earlier one,
/// - invalidated facts included, as the fact-append shortcut keeps them,
/// - joined with newlines, keeping the newest facts that fit within
///   `2 × MAX_SUMMARY_CHARS` — the most upstream ever stores without an LLM
///   pass. A node with no surviving fact keeps the empty summary.
///
/// Where every earlier pass took the fact-append shortcut — every summary
/// in both golden traces — the list is the old summary with this episode's
/// facts taken out, save that a fact upstream appended twice (a node two
/// drafts of one episode resolved to) is listed once. Where an LLM pass
/// wrote part of the summary, that part is gone, and the facts it covered
/// come back as list lines.
///
/// The next ingest that mentions the node appends to that list, or
/// compresses it with an LLM pass once it is over the bound — upstream's
/// ordinary behavior. Upstream itself leaves the summary unchanged; see
/// DEVIATIONS.md "Episode deletion resets kept nodes' summaries".
///
/// # Re-ingesting afterwards needs a NEW id
///
/// The deleted episode's id is tombstoned forever (see the module docs), so
/// **a fresh ingest after a delete must mint a new
/// [`crate::extract::EpisodeInput::id`]** — including the one documented
/// reason to delete and re-add, changing an episode's
/// [`metadata`](crate::extract::EpisodeInput::metadata), which is
/// write-once. Reusing the id is neither refused nor honoured:
/// [`crate::pipeline::add_episode`]'s idempotency check passes (the row
/// really is absent), every model call runs, and the batch commits — but
/// grit skips the tombstoned episode row and every mention naming it, while
/// the nodes, edges and merges decided beside them land. The call returns
/// `Ok` with an `episode_id` that reads back as nothing, over a graph whose
/// new rows no episode evidences. grit exposes no per-id tombstone read, so
/// nacre cannot catch the reuse for a caller; the id is the caller's to
/// retire. Pinned by
/// `tests/deletion.rs::re_adding_a_purged_id_is_a_silent_no_op`.
///
/// # Errors
///
/// [`PipelineError::EpisodeNotFound`] when the id is not stored — including
/// a second delete of an episode already deleted. Nothing is written in that
/// case, so a caller may treat it as "already gone" once it has decided that
/// is what it means; the error exists so the decision is the caller's.
/// [`PipelineError::Storage`] for a grit failure, in which case nothing was
/// erased.
///
/// What is erased is erased from the live file, op-log entries included,
/// within the limits the module docs state.
// ports: graphiti_core/graphiti.py::remove_episode (plus the summary reset,
// which has no upstream counterpart)
pub fn delete_episode(
    grit: &Grit,
    episode_id: Uuid,
) -> Result<DeleteEpisodeOutcome, PipelineError> {
    // Reported rather than silently doing nothing, for the same reason as
    // `PipelineError::EpisodeExists`: only the caller knows whether a
    // missing episode is a retry landing after the first delete succeeded
    // or a bug in its own bookkeeping.
    if grit.episode(episode_id)?.is_none() {
        return Err(PipelineError::EpisodeNotFound(episode_id));
    }

    let mut outcome = DeleteEpisodeOutcome {
        episode_id,
        purged_node_ids: Vec::new(),
        purged_edge_ids: Vec::new(),
        kept_node_ids: Vec::new(),
        kept_edge_ids: Vec::new(),
    };

    // `episode_targets` returns the ids this episode wrote, mixed and
    // undiscriminated (grit stores node and edge mentions in one column),
    // so classify each by point lookup. An id that is neither — a mention
    // whose row never arrived on this device — is skipped: there is nothing
    // to erase and no evidence to weigh.
    for target in grit.episode_targets(episode_id)? {
        if grit.edge(target)?.is_some() {
            if creator_of(grit, target)? == Some(episode_id) {
                outcome.purged_edge_ids.push(target);
            } else {
                outcome.kept_edge_ids.push(target);
            }
        } else if grit.node(target)?.is_some() {
            // Sole evidence. `mentions_of` resolves merge aliases and unions
            // across them, so a canonical node inherits the mentions of
            // everything folded into it — exactly the right count for
            // "would anything still evidence this entity?".
            if grit.mentions_of(target)?.as_slice() == [episode_id] {
                outcome.purged_node_ids.push(target);
            } else {
                outcome.kept_node_ids.push(target);
            }
        }
    }

    let mut ids = outcome.purged_edge_ids.clone();
    ids.extend(&outcome.purged_node_ids);
    ids.push(episode_id);

    // Decided before anything is written, like the ids: the rebuild reads
    // the edges the purge is about to erase, so it must exclude them itself.
    let purged: HashSet<Uuid> = ids.iter().copied().collect();
    let mut rebuilt: Vec<(Uuid, String)> = Vec::new();
    for &node in &outcome.kept_node_ids {
        let summary = surviving_facts(grit, node, &purged)?;
        if !summary.is_empty() {
            rebuilt.push((node, summary));
        }
    }

    // Never empty: the episode itself is always in `ids`.
    let mut batch = vec![WriteItem::Op(GraphOp::Purge { ids })];
    if !outcome.kept_node_ids.is_empty() {
        batch.push(WriteItem::Op(GraphOp::PurgeNodeFields {
            ids: outcome.kept_node_ids.clone(),
            fields: vec![NodeField::Summary],
        }));
        // Minted after the reset, so each outranks it and survives.
        for (id, summary) in rebuilt {
            batch.push(WriteItem::Op(GraphOp::UpdateNode {
                id,
                name: None,
                summary: Some(summary),
                kind: None,
                attrs: None,
            }));
        }
    }
    grit.apply_batch(batch)?;
    Ok(outcome)
}

/// A kept node's replacement summary: its surviving facts as a fresh list,
/// by the rule [`delete_episode`] states under "Kept nodes' summaries".
/// `purged` is everything the deletion is about to erase.
fn surviving_facts(
    grit: &Grit,
    node: Uuid,
    purged: &HashSet<Uuid>,
) -> Result<String, PipelineError> {
    let history = grit.node_history(node)?;
    let mut facts: Vec<String> = Vec::new();
    for edge in history.edges {
        if purged.contains(&edge.id)
            || edge.group_id != history.node.group_id
            || edge.fact.trim().is_empty()
        {
            continue;
        }
        // `node_history` keeps an edge whose other endpoint is gone; its fact
        // names a forgotten entity.
        let other = if edge.src == node { edge.dst } else { edge.src };
        if purged.contains(&other) || grit.node(other)?.is_none() {
            continue;
        }
        facts.push(edge.fact);
    }
    // The newest facts that fit, whole, in upstream's fact-append bound.
    let bound = MAX_SUMMARY_CHARS * 2;
    let mut kept: Vec<String> = Vec::new();
    let mut len = 0;
    for fact in facts.into_iter().rev() {
        let added = fact.chars().count() + usize::from(!kept.is_empty());
        if len + added > bound {
            break;
        }
        len += added;
        kept.push(fact);
    }
    kept.reverse();
    Ok(kept.join("\n").trim().to_owned())
}

/// Forget a whole namespace: every node, edge and episode row tagged with
/// `group_id`, erased in a single `Purge`. Every node the group holds goes,
/// so their summaries go whole — there is no kept node to reset.
///
/// This enumerates rows, not evidence — the full bi-temporal record, since
/// grit's three group scans apply no temporal or tombstone filter. Expired
/// nodes, merged-away drafts, invalidated edges and every episode go
/// together.
///
/// An unknown or empty group is **not** an error: it is a group with no
/// rows, and the honest answer is a zero-count outcome. Nothing is written
/// in that case — an empty `Purge` is rejected by grit and would fail the
/// whole batch, so the op is skipped rather than emitted empty.
///
/// # Completeness, and its four edges
///
/// 1. **Edges are covered because nacre's pipeline puts them in the
///    episode's group.** An edge's `group_id` is whatever its writer set and
///    is independent of its endpoints' groups, so grit cannot promise that a
///    group's edges are the edges incident to its nodes — this call inherits
///    a *pipeline* invariant, not a storage guarantee.
///    [`crate::extract::edges::ExtractEdgesOptions::group_id`] is the
///    supported way to break it: an edge written into another namespace
///    survives this call and is reachable in that namespace. A caller using
///    that override, or writing edges through grit directly, must sweep by
///    incidence (`Grit::node_history` over the group's nodes) as well. The
///    override cuts the other way too: deleting the namespace such an edge
///    was written *into* erases the edge, but the summarize pass appended its
///    fact to its endpoints' summaries, and those live in the group the
///    episode came from — which this call does not touch.
/// 2. **The default group `""` is a real group, and holds strays.** When a
///    `MergeNodes` arrives before the `AddNode` it absorbs, grit parks a
///    placeholder row for the absent node in `""` until the real add lands.
///    So `delete_group("")` can erase a placeholder belonging to another
///    namespace, and a group scan does not see its own pending placeholders.
///    Purging a placeholder tombstones that id permanently: a late `AddNode`
///    for it becomes a no-op. That is deliberate — deletion wins over a
///    write that has not arrived — but it means forgetting a group can
///    silently discard a row still in flight from another device.
/// 3. **Orphan update records are not enumerable by group.** An
///    `UpdateNode` or `InvalidateEdge` for an id whose base row never
///    arrived leaves a record carrying no `group_id`, so no group scan finds
///    it. It stays until its row arrives (whereupon the id becomes
///    enumerable) — a sync-era limitation, recorded rather than worked
///    around.
/// 4. **The op-log forgets with the rows, to exactly this boundary.** The
///    entries that created the enumerated ids are scrubbed in the purge's
///    own transaction, so nothing this call erases survives in
///    `oplog_since` or an export — but an id it never enumerated keeps its
///    entry along with its row. The edge that escapes by edge 1 escapes in
///    the log too, fact sentence and all. See the module docs for the
///    tiers and their limits.
///
/// # Errors
///
/// [`PipelineError::Storage`] for a grit failure, in which case nothing was
/// erased.
// ports: server/graph_service/zep_graphiti.py::delete_group
// (graphiti_core has no group-deletion entry point; its clear_data is a
// test-fixture reset that deletes by node label and lets DETACH take the
// edges)
pub fn delete_group(grit: &Grit, group_id: &str) -> Result<DeleteGroupOutcome, PipelineError> {
    let outcome = DeleteGroupOutcome {
        purged_node_ids: grit
            .nodes_in_group(group_id)?
            .iter()
            .map(|n| n.id)
            .collect(),
        purged_edge_ids: grit
            .edges_in_group(group_id)?
            .iter()
            .map(|e| e.id)
            .collect(),
        purged_episode_ids: grit
            .episodes_in_group(group_id)?
            .iter()
            .map(|e| e.id)
            .collect(),
    };
    let mut ids = outcome.purged_edge_ids.clone();
    ids.extend(&outcome.purged_node_ids);
    ids.extend(&outcome.purged_episode_ids);
    purge(grit, ids)?;
    Ok(outcome)
}

/// The episode that first evidenced `target`: the earliest-recorded of the
/// episodes mentioning it, ties broken by id. `None` when nothing mentions
/// it. See [`delete_episode`] for why this stands in for upstream's
/// `edge.episodes[0]`.
///
/// A mentioning episode whose row is not stored cannot be shown to be the
/// earliest, so it sorts last rather than winning by default.
fn creator_of(grit: &Grit, target: Uuid) -> Result<Option<Uuid>, PipelineError> {
    let mut recorded: Vec<(i64, Uuid)> = Vec::new();
    for id in grit.mentions_of(target)? {
        let created_at = grit.episode(id)?.map_or(i64::MAX, |e| e.created_at);
        recorded.push((created_at, id));
    }
    Ok(recorded.into_iter().min().map(|(_, id)| id))
}

/// Erase `ids` in one transaction. Skipped entirely when there is nothing to
/// erase: grit rejects a `Purge` with no ids, and a rejected op fails the
/// whole batch, so "nothing to forget" must not become "the call failed".
fn purge(grit: &Grit, ids: Vec<Uuid>) -> Result<(), PipelineError> {
    if ids.is_empty() {
        return Ok(());
    }
    grit.apply_batch(vec![WriteItem::Op(GraphOp::Purge { ids })])?;
    Ok(())
}
