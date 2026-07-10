# nacre

[![CI](https://github.com/bofeizhu/nacre/actions/workflows/ci.yml/badge.svg)](https://github.com/bofeizhu/nacre/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![grit-core](https://img.shields.io/crates/v/grit-core?label=grit-core)](https://crates.io/crates/grit-core)

**The LLM extraction pipeline for agent memory** — [Graphiti](https://github.com/getzep/graphiti)'s
pipeline ported to Rust, speaking [grit](https://github.com/bofeizhu/grit)'s typed API,
verified byte-for-byte against the original.

Nacre — mother-of-pearl — is the material an oyster deposits, layer by
layer, around a piece of grit. That is this architecture: raw episodes
(chat turns, document chunks, events) go in; nacre's LLM judgment calls —
*what to extract, what is a duplicate, what a new fact invalidates* —
deposit structured, bi-temporal memory into a grit graph.

```text
episode ──► extract entities ──► dedup (exact / fuzzy / LLM) ──► extract facts
                                                                      │
   grit (SQLite) ◄── summaries ◄── invalidate contradictions ◄── dedup facts
        │
        └──► hybrid retrieval: BM25 + vectors, RRF-fused, with provenance
```

## Why this exists

Agent memory has two hard problems living at different altitudes:
**storage** (bi-temporal facts, provenance, retrieval — deterministic,
testable) and **judgment** (what counts as an entity, when two names mean
one thing, when a new fact retires an old one — LLM territory). Most
stacks blur them. This one cuts them apart:

| Layer | What | Where |
|:-----:|------|-------|
| 3 | agent app (Electron + napi-rs) | *next* |
| **2** | **LLM extraction pipeline — this repo** | `nacre-core` |
| 1 | embedded bi-temporal graph on SQLite | [`grit-core`](https://crates.io/crates/grit-core) |

**Nacre decides; grit executes.** Nacre computes embeddings and makes every
LLM call; grit stores, retrieves, and applies the decisions — offline,
in-process, one SQLite file. Time is injected, every judgment is recorded
and replayable, and `cargo test` never touches the network.

## What works today

The full `add_episode` pipeline and hybrid search run end-to-end against
live APIs (see [`examples/live_smoke.rs`](crates/nacre-core/examples/live_smoke.rs)):
a real conversation ingested through DeepSeek + Zhipu into a fresh grit
file — extraction, dedup merges, a contradiction invalidated, summaries
refreshed, vectors persisted — then natural-language queries answered with
episode provenance.

```rust
let grit = Grit::open("memory.db", Options::new("my-device"))?;
let llm = ClaudeModel::new(ClaudeConfig::new(api_key));            // or ::deepseek(key)
let embedder = OpenAiEmbedder::new(OpenAiEmbedConfig::zhipu(key)); // or any /embeddings API

// Ingest: context windows come from the graph itself.
let previous = retrieve_previous_episodes(&grit, "chat", now, PREVIOUS_EPISODE_WINDOW)?;
let outcome = add_episode(&grit, &llm, &embedder, &episode, &previous,
                          &AddEpisodeOptions::default(), now).await?;
// outcome: node_ids, new_edge_ids, merges, invalidated_edge_ids — animate your UI with it.

// Recall: BM25 + vector legs, RRF-fused, provenance attached.
let hits = search_edges(&grit, &embedder, "Where does Priya work?", "chat", 10).await?;
```

### Model providers

| Feature flag | Client | Notes |
|---|---|---|
| `claude` | Anthropic Messages API | native structured outputs (`output_config` json_schema); **also drives Anthropic-compatible endpoints** — a `ClaudeConfig::deepseek()` preset uses DeepSeek's `/anthropic` endpoint with schema-in-prompt fallback |
| `openai-embed` | OpenAI-compatible `/embeddings` | Zhipu `embedding-3` preset; configurable base URL, model, dimension (MRL truncation), batch chunking |
| *(default)* | replay only | recordings in, decisions out — no network, ever |

## Correctness: the golden-trace oracle

“Ported to Rust” is an easy claim and a hard property. Nacre's is
falsifiable: **pinned Python Graphiti v0.29.2** runs on fixed episodes with
*recorded* LLM/embedder responses, and its graph state is frozen as a
fixture. The conformance test replays the same recordings through the Rust
stack and diffs the result — every prompt byte-identical, every node,
edge, fact, timestamp, and attribution equal, or the build is red.

Getting that trace *deterministic* surfaced real bugs on both sides — a
prompt suffix the port and its fixtures both missed, engine-nondeterministic
candidate ordering, an upstream save path whose result differs across
database backends. Each finding is either fixed or pinned as a documented
divergence:

- [`oracle/`](oracle/) — capture harness (Docker FalkorDB, digest-pinned) + trace fixtures
- [`DEVIATIONS.md`](DEVIATIONS.md) — every accepted divergence, with rationale and the test that asserts it
- [`PROMPTS.md`](PROMPTS.md) — the verbatim prompt-port ledger

## Coverage

Scope is a design decision here, not an accident — nacre ports the
pipeline pearl's product needs and keeps the conformance surface honest.
Four tiers:

### ✅ Ported & oracle-verified

| Capability | Notes |
|---|---|
| Entity extraction (message episodes) | multi-episode attribution, type mapping/exclusion |
| Node dedup | exact-name fast path, MinHash/LSH fuzzy (blake2b **bit-identical** to Python), batched LLM escalation with upstream's guardrails |
| Label promotion & summary refresh | fact-append shortcut + batched LLM flights; persisted via grit `UpdateNode` |
| Fact (edge) extraction | fat edges with fact sentences, endpoint validation, timestamp extraction |
| Edge dedup | verbatim fast path + LLM resolution, continuous candidate indexing |
| Temporal invalidation | extraction-time bounds, contradiction resolution, newer-information self-expiry — all three paths |
| Context windows | last-10 previous episodes from the graph, byte-equal to upstream's prompt windows |
| Embedding persistence | name/fact vectors stored at write time; warm graphs re-embed nothing |
| Hybrid retrieval | BM25 + query-vector legs, RRF-fused, provenance attached *(rank order advisory — see DEVIATIONS)* |

### 🧪 Ported, awaiting trace coverage

`text`/`json` episode sources, custom entity types, custom extraction
instructions, the episode-prompt summary path — all fixture-tested at the
prompt level; no golden trace exercises them yet.

### ⏳ Deferred until a proven need

Custom node/edge **attribute extraction** (typed ontologies); golden
traces #2+ (volume, multi-group isolation, purge).

### 🚫 Out of scope by design

| Feature | Why |
|---|---|
| Bulk ingestion API | upstream maintains a *second* ingestion implementation; two near-identical semantics is a bug factory — loop the single path |
| Communities | graph algorithms are a grit non-goal; topic rollups belong to the app layer, designed natively |
| Sagas | fast-moving upstream surface outside the core loop; porting it maximizes re-pin cost |
| Advanced search recipes (MMR, cross-encoder, node-distance, BFS) | cross-engine rank parity is untestable by construction; retrieval quality is improved natively in grit instead |

Exclusions are documented and reversible — "revisit with a proven need,
as a new decision" is the standing rule ([AGENTS.md](AGENTS.md)).

## Testing

```sh
cargo test --workspace     # offline: replay + property + prompt-fidelity + conformance
```

- **Prompt fidelity** — Rust prompt rendering asserted byte-identical to
  fixtures generated from the actual pinned Python.
- **Replay tests** — recorded LLM/embedder responses drive every judgment
  path deterministically.
- **Golden-trace conformance** — the end-to-end oracle described above.
- **Zero network** — the default build has no HTTP client compiled in.

## Repository map

| File | Role |
|---|---|
| [AGENTS.md](AGENTS.md) | the binding design contract (invariants, scope, testing) |
| [ROADMAP.md](ROADMAP.md) | ordered build queue with completion notes |
| [DEVIATIONS.md](DEVIATIONS.md) | accepted divergences from the golden traces |
| [PROMPTS.md](PROMPTS.md) | verbatim prompt-port ledger |
| [oracle/](oracle/) | capture harness + golden-trace fixtures |

## License

Apache-2.0. Contains material ported from
[Graphiti](https://github.com/getzep/graphiti) (Apache-2.0, Zep Software,
Inc.) — see [NOTICE](NOTICE). This project is not affiliated with or
endorsed by Zep.
