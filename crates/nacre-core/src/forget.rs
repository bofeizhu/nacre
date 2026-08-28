//! Deletion: the right-to-forget surface.
//!
//! Ports upstream's two removal entry points onto grit's single destructive
//! op. Nacre decides *what* to erase — which is a judgment about evidence,
//! not a storage question — and grit executes one `Purge`:
//!
//! - [`delete_episode`] ports `graphiti_core/graphiti.py::remove_episode`:
//!   forget one episode and the graph that exists only because of it.
//! - [`delete_group`] ports the service-layer `delete_group`
//!   (`server/graph_service/zep_graphiti.py`); `graphiti_core` has no
//!   group-deletion entry point of its own, only `clear_data`, which is a
//!   test-fixture reset.
//!
//! **One batch, one op, one audit record.** Each call collects every id
//! first and commits a single `GraphOp::Purge` through `apply_batch`, so a
//! deletion is as atomic as an ingest: it happens whole or not at all, and
//! leaves exactly one oplog entry naming what was forgotten. Upstream issues
//! one un-transacted call per kind (edges, then nodes, then the episode), so
//! a failure between them leaves a half-forgotten graph — the same class of
//! defect the atomic ingest fixed (DEVIATIONS.md "Atomic episode commit").
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
//! **What "forgotten" means.** grit's purge erases the derived state — the
//! rows, their per-field update and invalidation records, mention links in
//! both directions, every FTS mirror entry and every stored vector — but
//! deliberately does NOT rewrite the oplog, which is the sync unit. The
//! guarantee is *unreachable through every read API and every index*, not
//! *gone from the file*. An application that promises erasure to a person
//! must make that promise on its own terms (a compacted export, a rekeyed
//! file); it cannot inherit it from here. Both functions below repeat this
//! where a caller will read it.
//!
//! **A forgotten id is spent.** Purge also tombstones every id it names,
//! permanently, and grit gates its inserts on that tombstone table: a later
//! op naming a purged id is accepted, applied as a no-op, and reported as
//! success. That is what makes purge commute with concurrent adds (a write
//! still in flight from another device cannot resurrect what was forgotten),
//! but it means an id is usable exactly once — here and on every device that
//! syncs the purge. [`delete_episode`] spells out what that costs at the
//! ingest seam, where the id is the caller's.

use grit_core::{GraphOp, Grit, WriteItem};
use uuid::Uuid;

use crate::pipeline::PipelineError;

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
    /// Nodes kept: other episodes mention them too.
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
///
/// Everything is decided before anything is written, then erased in a single
/// `Purge` — one transaction, one audit record.
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
/// The oplog keeps the purged content by design — see the module docs.
// ports: graphiti_core/graphiti.py::remove_episode
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
    purge(grit, ids)?;
    Ok(outcome)
}

/// Forget a whole namespace: every node, edge and episode row tagged with
/// `group_id`, erased in a single `Purge`.
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
///    incidence (`Grit::node_history` over the group's nodes) as well.
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
/// 4. **The oplog retains what was purged** — see the module docs.
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
