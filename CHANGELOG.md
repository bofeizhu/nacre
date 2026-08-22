# Changelog

Notable changes to nacre, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before 0.2.0 predate this file; their notes live in the git tags
(`v0.1.0` … `v0.1.2`) and the matching GitHub releases.

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
