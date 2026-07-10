# AGENTS.md — nacre

The LLM extraction pipeline for agent memory: Graphiti's pipeline ported to Rust,
speaking grit's typed API. Nacre — mother-of-pearl — is the material an oyster
deposits, layer by layer, around a piece of grit. This is **Layer 2** of a
three-layer agent-memory stack:

- **Layer 1 (`grit`, separate repo):** deterministic graph storage + hybrid
  retrieval on SQLite. No LLM, no network.
- **Layer 2 (this repo):** entity/edge extraction, dedup judgment, temporal
  invalidation, summarization — every step that needs an LLM or an embedding.
- **Layer 3 (separate, future):** the agent harness (Rig / Claude API) in a Tauri app.

If you are an agent working in this repo: everything below is binding. When a
request conflicts with a **Design Invariant**, stop and surface the conflict
instead of coding around it. Cross-repo rules live in the umbrella `../CLAUDE.md`;
grit's own contract is `../grit/AGENTS.md`.

## Why this exists (decisions already made — do not relitigate)

Graphiti (getzep, Apache-2.0) is the best-designed episodic→semantic extraction
pipeline in the open: episodes in, a bi-temporal knowledge graph out, with LLM
judgment at exactly the right seams (what to extract, what is a duplicate, what
a new fact invalidates). But it is ~36.5k LOC of async Python built around
server-grade graph databases — unusable in-process on iOS, and unauditable as a
dependency for irreplaceable personal data.

The decision (July 2026): port the **pipeline**, not the codebase. Of those
36.5k LOC, ~15.5k is `driver/` (replaced wholesale by grit's typed API), ~3.3k
is an LLM/embedder provider zoo (replaced by two traits), and the real product —
prompts, extraction/dedup/invalidation logic, search orchestration — is ~12k LOC
of Python that becomes a small, deterministic-at-the-edges Rust crate.

The port is pinned to **Graphiti v0.29.2** (`../refs/graphiti`, read-only).
Rebasing to a newer Graphiti is a deliberate act: re-pin the reference clone,
re-capture golden traces, adapt. Never chase upstream continuously.

## Design invariants (non-negotiable)

1. **Behavioral port, not a transliteration.** Fidelity is defined by the golden
   traces (invariant 5), never by code resemblance. Rust idioms win wherever the
   traces cannot tell the difference. Prompts are the exception: prompt text is
   ported verbatim from `refs/graphiti/graphiti_core/prompts/` — every deliberate
   deviation is documented in `PROMPTS.md` with a reason.
2. **`grit-core` is the only storage dependency** — never grit-compat, grit-cli,
   or direct SQLite access. All graph mutations leave nacre as `GraphOp` values;
   nacre decides, grit executes (grit's `MergeNodes` rule). Steady state pins the
   published crate (`grit-core = "0.1"`); co-dev uses an UNCOMMITTED
   `[patch.crates-io]` path override (see `../CLAUDE.md`).
3. **LLM and embeddings live behind traits at the edges.** `LanguageModel` and
   `Embedder` are the only places I/O happens; pipeline steps between them are
   pure functions. Time is injected (grit's `Clock`) — never `SystemTime::now()`
   in library code. Rationale: the pipeline must be a replayable function of
   (episode stream, recorded model responses, clock).
4. **Every judgment is recorded and replayable.** All `LanguageModel`/`Embedder`
   calls flow through a recording layer keyed by request content. Tests replay
   recordings; `cargo test` never touches the network (enforced: the test
   LanguageModel fails loudly on a cache miss instead of calling out).
5. **The golden-trace oracle defines "correct".** Pinned Python Graphiti v0.29.2
   + FalkorDB, run on fixed episode inputs with recorded LLM responses, produces
   frozen graph-state and retrieval fixtures. The Rust stack (nacre + grit) must
   reproduce them — field-for-field on graph state, rank-order on retrieval.
   Capture harness and fixtures live in `oracle/`. Divergences are either bugs
   or documented, justified deviations in `DEVIATIONS.md` — never silent.
6. **License: Apache-2.0 only** (unlike grit's dual MIT/Apache). Nacre contains
   verbatim Apache-2.0 material from Graphiti (prompt text, ported logic);
   offering it under MIT would misrepresent what downstream may do. Attribution
   in NOTICE. No SSPL/GPL dependencies.
7. **Scale and latency are grit's envelope.** Nacre adds no storage and no
   indexes; anything that needs more than ≤100k nodes / ≤1M edges belongs
   elsewhere. Pipeline throughput is LLM-bound by design — do not add caching
   layers or batch heuristics that trade trace fidelity for speed before the
   port is complete and green.

## Architecture

Cargo workspace:

```
nacre/
  crates/
    nacre-core/       # the pipeline ← the product
      src/
        model/        # LanguageModel + Embedder traits, recording/replay layer,
                      # one real client (Claude API) behind a feature flag
        extract/      # episode → entity nodes + edges   (ports prompts/extract_*)
        dedupe/       # candidate resolution              (ports prompts/dedupe_*, dedup_helpers)
        invalidate/   # temporal contradiction handling   (ports edge_operations invalidation)
        summarize/    # node summaries                    (ports summarize_nodes)
        search/       # search orchestration over grit's legs (ports search/ configs + fusion)
        pipeline.rs   # add_episode: the seam that strings the steps together
  oracle/             # Python capture harness (uv project) + frozen golden traces
  AGENTS.md PROMPTS.md DEVIATIONS.md NOTICE
```

- **Async at the edges only.** `LanguageModel`/`Embedder` are async (tokio);
  every step between two model calls is a pure sync function, unit-testable
  without a runtime. grit calls are sync (it's an embedded library; its latency
  envelope makes blocking harmless).
- **The pipeline emits `GraphOp`s.** `add_episode` returns the ops it decided on
  (and applies them via the injected `grit_core::Grit` handle) so callers and
  tests can inspect exactly what the pipeline concluded.
- **Port order** (each step lands with its replay tests green against the
  corresponding golden-trace slice before the next begins):
  extract_nodes → dedupe_nodes → extract_edges → dedupe_edges →
  invalidation → summarize → search orchestration.

## What is deliberately NOT ported

- `driver/`, `graph_queries.py`, all Cypher — grit's typed API replaces them.
- The LLM/embedder provider zoo — two traits + one Claude client; more providers
  only if Layer 3 demands them.
- `server/`, `mcp_server/`, telemetry/tracer plumbing.
- Communities (`community_operations.py`) — optional in Graphiti, deferred here
  until a Layer 3 need appears.
- Cross-encoder reranking — grit's RRF fusion first; revisit with eval data.
- Bulk-ingest utilities (`bulk_utils.py`) — the agent use case is incremental.
  Revisit only with a proven need (grit has a batching seam if it comes).

## Testing (the trust budget)

- **Replay tests** (the workhorse): recorded LLM/embedder responses + fixed
  episodes → assert the exact `GraphOp` stream and resulting grit state.
  Deterministic, offline, fast.
- **Golden-trace conformance**: replay the oracle fixtures end-to-end through
  nacre + grit; diff graph state field-for-field (UUIDs mapped, timestamps from
  the injected clock) and retrieval results by rank. One conformance test per
  fixture; a red conformance test blocks merge unless the diff lands in
  `DEVIATIONS.md` the same commit.
- **Property tests** where judgment is NOT involved: chunking, prompt-context
  assembly, date resolution — pure functions get proptest coverage.
- **Zero-network suite**: `cargo test` runs offline (invariant 4 makes this
  free). The only networked code paths are the real Claude client (feature-
  gated, integration-tested manually) and the Python capture harness.
- **Capture is not CI.** `oracle/` capture runs are manual, local, documented in
  `oracle/README.md`, and produce committed fixtures. CI only replays.

## Explicit non-goals

- Being a general Graphiti alternative (multi-DB, multi-provider, server mode).
  Nacre exists to feed grit for Layer 3 — one path, done well.
- Prompt "improvements" during the port. First reproduce, then (post-parity,
  behind eval evidence) improve. Parity before creativity.
- Streaming/partial extraction, agentic tool-use extraction loops — Graphiti
  v0.29.2 doesn't do them; neither does the port.
- Embedding computation in-process (candle etc.) — `Embedder` is a trait;
  local-model backends are a Layer 3 decision.

## Open questions (decide before 0.2, fine to defer at 0.1)

- **Structured output mechanism**: Graphiti relies on provider JSON-schema
  modes. Claude API tool-use is the obvious equivalent — verify the recorded
  traces are representation-compatible (JSON out is JSON out) or record a
  deviation.
- **Token budgeting / context truncation**: Graphiti truncates prompt context
  by character count in places; port faithfully first, revisit with real
  tokenizer counts later.
- **`extract_nodes_and_edges` combined prompt** (v0.29.2 has both combined and
  split paths): capture traces for the path Graphiti actually defaults to;
  port that one first.
- **Saga summarization** (`summarize_sagas.py`): new in recent Graphiti;
  defer unless the default `add_episode` path exercises it.

## Commands

```bash
cargo build --workspace
cargo test  --workspace              # replay + conformance + property; offline
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt   --all
# capture (manual, networked, needs Docker + API keys — see oracle/README.md):
cd oracle && uv run capture …
```

## Style

- Edition 2024. `#![deny(unsafe_code)]` — nothing in this crate justifies unsafe.
- Errors: `thiserror` in the library; no `anyhow` in public API.
- Every public API doc-commented with an example; `cargo doc` must build clean.
- Prompt text lives in dedicated modules/files with the upstream path noted
  (`// ports: graphiti_core/prompts/extract_nodes.py::extract_message`) so
  diffs against the pinned reference stay mechanical.
- Match grit's tooling: rustfmt defaults, clippy `-D warnings`, CI on
  ubuntu + macos.
