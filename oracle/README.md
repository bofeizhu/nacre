# oracle — golden-trace capture harness

This directory holds **the definition of correct** for nacre: golden traces
captured from pinned Python Graphiti (v0.29.2, `../../refs/graphiti`) running
against FalkorDB with **recorded LLM responses**, frozen as fixtures that the
Rust stack (nacre + grit) must reproduce.

Status: **not yet built** — this is milestone 2 (see `../AGENTS.md`).

## Planned shape

- A `uv` Python project pinning `graphiti-core==0.29.2` + `falkordb` (Docker).
- A recording LLM/embedder client wrapper: capture mode hits the real API once
  and writes request→response recordings; replay mode is deterministic.
- Fixed episode input sets (small, curated, committed).
- Capture runs emit, per fixture: the episode inputs, the LLM recordings, the
  full resulting graph state (nodes/edges/episodes, all temporal fields), and
  retrieval results for a fixed query set.
- Fixtures are committed here; `cargo test` in the workspace replays them
  offline. Capture is manual and deliberate — never CI, never automatic.

## Rules (binding, from AGENTS.md)

- Fixture refresh happens only when deliberately re-pinning to a newer
  Graphiti release.
- Divergences between the Rust stack and a fixture are bugs or documented
  entries in `../DEVIATIONS.md` — never silently accepted.
