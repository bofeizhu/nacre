# oracle — golden-trace capture harness

This directory holds **the definition of correct** for nacre: golden traces
captured from pinned Python Graphiti (v0.29.2, `../../refs/graphiti`) running
against FalkorDB with **recorded LLM responses**, frozen as fixtures that the
Rust stack (nacre + grit) must reproduce.

## Layout

- `capture.py` — the capture CLI (see its docstring for usage)
- `recording_clients.py` — recording/replay LLM + embedder wrappers; the
  docstring defines THE RECORDING CONTRACT shared with the nacre crate's `model`
  module (pre-mutation messages are the request identity)
- `promptgen/` — renders prompt-fidelity fixtures from the pinned Python
  (offline, stock python3; feeds `tests/prompt_fidelity.rs`)
- `episodes/` — curated episode-set specs (inputs to capture)
- `fixtures/<trace>/` — frozen traces: `episodes.json`, `llm_recordings.json`,
  `embedder_recordings.json`, `graph_state.json`, `retrieval.json`, `meta.json`

## Running a capture (manual, networked — never CI)

```bash
cd oracle
docker compose up -d            # FalkorDB on :6379 (dedicated instance —
                                # capture CLEARS the database)
export OPENAI_API_KEY=...
uv run python capture.py episodes/trace1.json
uv run python capture.py episodes/trace1.json --replay   # offline determinism check
```

`--replay` reruns the whole pipeline from the saved recordings (no network)
and writes `graph_state.replay.json` / `retrieval.replay.json`; diffing them
against the originals proves the capture is deterministic.

## Determinism notes

- UUIDs are aliased deterministically (`n0…`, `e0…`, `ep0…`, sorted by
  content) so traces are comparable across runs and across languages.
- `created_at` fields are wall-clock on the Python side but injected-clock on
  the Rust side; the conformance harness treats them specially (exact-match
  everything else, documented mapping for created_at).
- Search uses Graphiti's default RRF hybrid path only; the cross-encoder is a
  fail-loud stub (reranking is deliberately not ported — see AGENTS.md).
- Embedding vectors live only in `embedder_recordings.json` (recomputable
  local state, grit invariant); `graph_state.json` carries no vectors.

## Rules (binding, from AGENTS.md)

- Fixture refresh happens only when deliberately re-pinning to a newer
  Graphiti release (re-pin `refs/graphiti`, re-capture, adapt).
- Divergences between the Rust stack and a fixture are bugs or documented
  entries in `../DEVIATIONS.md` — never silently accepted.
- Expect first-run friction (upstream API details verified only at capture
  time); fix the harness in place, keep the contract in
  `recording_clients.py` authoritative.
