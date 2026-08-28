# Changelog

Notable changes to nacre, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before 0.2.0 predate this file; their notes live in the git tags
(`v0.1.0` … `v0.1.2`) and the matching GitHub releases.

## 0.3.0 — 2026-08-28

### Added

- **A deletion surface: `nacre_core::forget`.** `delete_episode(grit, id)`
  ports upstream's `remove_episode` — it erases the nodes this episode is
  the sole evidence for, the facts it created, and the episode row, while
  leaving anything another episode also evidences. `delete_group(grit,
  group_id)` erases a whole namespace: every node, edge and episode row
  tagged with it, the full bi-temporal record. Both collect every id first
  and commit a single `GraphOp::Purge` — one transaction, one audit record,
  and nothing at all written when there is nothing to erase. Both return a
  typed report naming what went and (for an episode) what stayed.
  grit's purge erases the rows, their update and invalidation records,
  provenance in both directions, every FTS mirror entry, every stored
  vector **and the op-log entries that created those ids** — so the
  content is gone from the live file, exports and re-delivery from a peer
  included, and the freed pages are zeroed. Stated in full on
  `nacre_core::forget` with the limits grit states and nacre cannot
  improve on: "no byte remains" needs grit's `compact()` maintenance call
  (FTS5 re-encodes deleted terms into a new segment until segments merge),
  and OS backups, copies taken before the delete and devices that never
  receive the `Purge` are out of scope — one that receives it erases on
  arrival. Ops that merely *reference* a purged id stay, deliberately:
  opaque uuids, content-free, and what keeps grit's merge laws exact. A
  forgotten id is also spent: purge tombstones it permanently, and a later
  op naming it is a silent no-op.
- **`PipelineError::EpisodeNotFound(id)`** — the mirror of `EpisodeExists`,
  returned when the episode to delete is not stored (including a second
  delete of one already deleted). Reported rather than silently ignored:
  only the caller knows whether that is the expected second attempt.
- **`EpisodeInput.metadata: Option<String>`** (`#[serde(default)]`) — an
  application payload stored verbatim on the episode row and handed back by
  every episode read path, including the previous-episode window. Opaque to
  the pipeline: never interpolated into a prompt, never embedded, never
  part of episode identity. Write-once — grit has no episode update op, so
  changing it means `delete_episode` plus a fresh ingest **under a new id**:
  the delete tombstones the old one, and re-ingesting under it returns `Ok`
  while silently storing neither the episode row nor its provenance.
- `tests/deletion.rs`: the mechanical laws over hand-built state (group
  isolation, the empty-`Purge` trap, all-or-nothing batches, the
  enumeration boundaries a group scan cannot cross, the op-log scrub and
  the export that cannot carry a purged fact out) plus
  a golden trace replayed into two sibling groups of one database — the
  sibling is the control that deleting one group must leave byte-identical.
- **Both binding surfaces carry the deletion surface and `metadata`.**
  Node gains `deleteEpisode(id)` / `deleteGroup(groupId)` returning typed
  outcome objects, `episode(id)` (the point lookup behind a provenance id;
  `null` when the graph has no such row), `metadata` on the `addEpisode`
  input and on every episode-bearing row, and the exported
  `EPISODE_NOT_FOUND_CODE` message prefix — the same napi limitation, and
  the same convention, as `EPISODE_EXISTS_CODE`. Swift gains
  `deleteEpisode(id:)` / `deleteGroup(groupId:)` with `DeleteEpisodeResult`
  / `DeleteGroupResult` records, `episode(id:) -> EpisodeRow?`, and
  `Episode.metadata` as a defaulted trailing field, so no existing
  `Episode(...)` call site changes; `NacreError.EpisodeNotFound` is now
  reachable. Deletion is synchronous on both surfaces, like the rest of
  the grit-only path — it makes no model call.
- **The sidecar forwards `id` and `metadata`** through `addEpisode` (the
  `id` gap carried over from 0.2.0) and gains `episode`, `deleteEpisode`
  and `deleteGroup` methods; its two stable error prefixes reach the
  protocol unchanged.

### Requires

- grit-core with schema v6: `AddEpisode.metadata`, `Episode.metadata` and
  `Grit::episode_targets`.
- grit-core 0.4.0's `Purge` for the deletion contract documented above:
  the op-log scrub, the ingest refusal for a re-delivered scrubbed entry,
  the zeroed pages and `Grit::compact()`. Against an older grit the code
  behaves identically but the promise is the weaker one it superseded
  (unreachable through every read API, the log retaining the original
  ops) — which is why the docs and the release move together.

### Not yet

- The viz read path is still absent from the sidecar protocol (`status`
  aggregates; `episode` exists only because `searchEdges` hands back
  provenance ids). The committed sample dumps under
  `crates/nacre-node/examples/viz/` predate `metadata` and gain the key
  whenever they are next regenerated.

## 0.2.0 — 2026-08-22

### Changed

- **`add_episode` is atomic.** Ingestion now runs in two phases: a planning
  phase does every model call and decides the whole write set without
  touching storage, then a single `grit-core` `apply_batch` commits it. An
  episode's nodes, merges, label promotions, edges, invalidations,
  summaries, embeddings and the episode row itself land in one transaction.
  A failure anywhere in the pipeline — a model error, a host that hung up —
  now leaves the graph byte-identical to what it was before the call,
  instead of stranding graph mutations that no episode evidences.
  Which ops are applied, and in what order, is unchanged.

### Added

- **`EpisodeInput.id: Option<Uuid>`** — the ingest idempotency key. `None`
  (the default, and how the field deserializes when absent) mints an id as
  before. A caller retrying a failed ingest passes the same id every time:
  `add_episode` either completes the ingest exactly once or returns the new
  `PipelineError::EpisodeExists(id)`, reported before any work is done. So
  a retrying caller observes either one clean ingest or "this already
  landed" — never duplicated, never orphaned, partial writes.
- **Both bindings carry the key.** `nacre-node`'s `addEpisode` episode
  object takes an optional `id` (UUID string) and rejects a second ingest
  under a stored id with a message led by the exported
  `EPISODE_EXISTS_CODE`. `nacre-swift`'s `Episode` takes `id: String?`
  (defaulted, so existing call sites are unchanged) and `NacreError` gains
  an `EpisodeExists(id:)` case. A malformed id is an argument error in
  both, raised before any model call.
- `tests/atomicity.rs`: a golden trace replayed with a truncated recording
  set, asserting that a failed ingest leaves rows, provenance links,
  vectors and the oplog untouched, and that the retry ingests exactly once.
- `tests/op_sequence.rs` + `oracle/fixtures/<trace>/op_sequence.json`: the
  ordered grit write sequence of both traces, frozen. It is what proves the
  two-phase rewrite changed only when writes commit, not what they are.

### Requires

- `grit-core` 0.3 (`apply_batch` / `WriteItem`).
