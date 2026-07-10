# nacre

**The LLM extraction pipeline for agent memory** — [Graphiti](https://github.com/getzep/graphiti)'s
pipeline ported to Rust, speaking [grit](https://github.com/bofeizhu/grit)'s typed API.

Nacre — mother-of-pearl — is the material an oyster deposits, layer by layer,
around a piece of grit. That is this architecture: raw episodes (chat turns,
document chunks, events) go in; nacre's LLM judgment calls — what to extract,
what is a duplicate, what a new fact invalidates — deposit structured,
bi-temporal memory into a grit graph.

**Status: pre-0.1, under construction.** The design contract is
[AGENTS.md](AGENTS.md); correctness is defined by golden traces captured from
pinned Python Graphiti (v0.29.2) in [oracle/](oracle/).

## The stack

| Layer | What | Where |
|---|---|---|
| 3 | agent harness (Tauri app) | future |
| **2** | **extraction pipeline — this repo** | `nacre-core` |
| 1 | embedded bi-temporal graph on SQLite | [`grit-core`](https://crates.io/crates/grit-core) |

The division of labor is sharp: nacre computes embeddings and makes LLM
judgment calls; grit stores, retrieves, and executes — deterministically,
offline, in-process.

## License

Apache-2.0. Contains material ported from Graphiti (Apache-2.0) — see
[NOTICE](NOTICE).
