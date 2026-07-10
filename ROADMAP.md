# ROADMAP.md — the build queue

Ordered task queue for building nacre. Worked top-to-bottom, one task per
increment. Conventions (binding for any agent working this file):

- Take the **first unchecked task that is not `BLOCKED(...)`**. Finish it
  completely (code + tests + docs) before touching the next.
- A task that turns out to need user input gets `BLOCKED(reason)` prepended,
  not silently skipped. Unblocking is the user's move.
- Discovered work is **inserted** where it belongs in the order (never
  appended blindly, never done "while I'm here" without a line item).
- Tasks are checked off only with the full gate green:
  `cargo fmt --all --check`, `clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo doc` (warnings deny). AGENTS.md wins over
  this file if they ever conflict.

## Milestone 2 groundwork + Rust replay infrastructure (offline, unblocked)

- [ ] `model/` module in nacre-core: `LanguageModel` + `Embedder` async traits
      (tokio), request/response types mirroring what Graphiti's pipeline needs
      (structured JSON output against a schema; batched embeddings). Include a
      `RecordingStore` (JSON files, keyed by request-content hash) and a
      `ReplayModel` that serves recordings and **fails loudly on a miss**.
      Unit tests with hand-written recordings.
- [ ] Port the prompt *output* models: serde structs for every response schema
      in `refs/graphiti/graphiti_core/prompts/models.py` and the per-prompt
      response models — field names byte-identical to the Python (that's what
      lands in recordings/traces). Round-trip serde tests.
- [ ] Verbatim prompt port: `extract_nodes` family → `nacre-core/src/extract/`
      prompt module(s) with upstream-path comments + PROMPTS.md ledger rows.
- [ ] Verbatim prompt port: `extract_edges` family.
- [ ] Verbatim prompt port: `dedupe_nodes` + `dedupe_edges` families.
- [ ] Verbatim prompt port: `summarize_nodes` family (skip sagas — deferred,
      see AGENTS.md open questions).
- [ ] oracle/ harness **code** (no networked run yet): uv project pinning
      `graphiti-core==0.29.2` + falkordb, docker-compose for FalkorDB, a
      recording LLM-client + embedder wrapper (capture/replay), a `capture`
      CLI that ingests an episode set and dumps: episode inputs, recordings,
      full graph state (all temporal fields), retrieval results for a fixed
      query list.
- [ ] Curated episode fixture set #1: a small multi-turn conversational
      scenario with entity overlap, a fact that later gets contradicted, and
      names that need dedup judgment. Committed under `oracle/episodes/`.
- [ ] BLOCKED(user: needs Docker running + an LLM API key for the one-time
      capture) First capture run → commit golden trace #1 + recordings.

## Milestone 3 — the pipeline port (each step: logic + replay tests green)

- [ ] `extract/nodes.rs`: episode → extracted entity nodes (ports
      `node_operations.py` extraction path). Replay tests w/ synthetic
      recordings until golden traces exist.
- [ ] `dedupe/nodes.rs`: candidate resolution using grit's
      `find_merge_candidates` + LLM judgment → `MergeNodes` ops (ports
      `dedup_helpers.py` + `node_operations.py` dedup path).
- [ ] `extract/edges.rs`: entity pairs → fat edges with fact sentences (ports
      `edge_operations.py` extraction path).
- [ ] `dedupe/edges.rs`: edge dedup judgment (ports `edge_operations.py`).
- [ ] `invalidate/`: temporal contradiction detection → `InvalidateEdge` ops
      with event-time reasoning (ports `edge_operations.py` invalidation +
      `temporal_operations`).
- [ ] `summarize/`: node summary refresh (ports `summarize_nodes` usage).
- [ ] `pipeline.rs`: `add_episode` seam stringing the steps, emitting +
      applying the `GraphOp` stream (ports `graphiti.py::add_episode`
      orchestration; date handling via injected Clock).
- [ ] `search/`: search orchestration over grit's legs — config recipes,
      filters, fusion parity (ports `search/` minus Cypher generation).
- [ ] Conformance harness: `tests/conformance.rs` loading `oracle/` fixtures,
      diffing graph state field-for-field and retrieval by rank.
      BLOCKED(until golden trace #1 exists) for the assertion half; the
      loader + differ can land first.
- [ ] Real Claude API `LanguageModel` client behind a `claude` feature flag
      (off by default; manual integration test only — never in `cargo test`).

## Later (do not start without a user decision)

- Golden traces #2+ (bulk-ish episode volume, group_id isolation, purge).
- Search cross-encoder reranking evaluation.
- Communities port. Saga summarization. crates.io publish of nacre-core.
