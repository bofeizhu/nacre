# PROMPTS.md — prompt-port ledger

Prompt text is ported **verbatim** from `refs/graphiti/graphiti_core/prompts/`
(pinned v0.29.2). Every deliberate deviation from upstream prompt text is
recorded here with a reason; an empty table means "no deviations yet", not
"nobody checked".

Fidelity mechanism: fixtures under `crates/nacre-core/tests/fixtures/prompts/`
are rendered from the actual upstream Python by
`oracle/promptgen/gen_prompt_fixtures.py` (manual, offline, stub-imports);
`tests/prompt_fidelity.rs` asserts the Rust rendering is byte-identical.

| Prompt (upstream path) | Ported to | Deviation | Reason |
|---|---|---|---|
| `prompts/prompt_helpers.py::to_prompt_json` + `DO_NOT_ESCAPE_UNICODE` | `src/prompts/helpers.rs` | none | |
| `prompts/snippets.py::summary_instructions` | `src/prompts/snippets.rs` | none | |
| `utils/text_utils.py::MAX_SUMMARY_CHARS` | `src/prompts/mod.rs` | none | |
| `prompts/extract_nodes.py::extract_message` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_json` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_text` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::classify_nodes` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_attributes` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_summary` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_summaries_batch` | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_nodes.py::extract_entity_summaries_from_episodes` (+ system prompt) | `src/prompts/extract_nodes.rs` | none | |
| `prompts/extract_edges.py::edge` | `src/prompts/extract_edges.rs` | none | |
| `prompts/extract_edges.py::extract_attributes` | `src/prompts/extract_edges.rs` | none | |
| `prompts/extract_edges.py::extract_timestamps` | `src/prompts/extract_edges.rs` | none | |
| `prompts/extract_edges.py::extract_timestamps_batch` | `src/prompts/extract_edges.rs` | none | |
| `prompts/dedupe_nodes.py::node` | `src/prompts/dedupe_nodes.rs` | none | |
| `prompts/dedupe_nodes.py::nodes` | `src/prompts/dedupe_nodes.rs` | none | |
| `prompts/dedupe_nodes.py::node_list` | `src/prompts/dedupe_nodes.rs` | none | |
| `prompts/dedupe_edges.py::resolve_edge` | `src/prompts/dedupe_edges.rs` | none | |
| `prompts/summarize_nodes.py::summarize_pair` | `src/prompts/summarize_nodes.rs` | none | |
| `prompts/summarize_nodes.py::summarize_context` | `src/prompts/summarize_nodes.rs` | none | |
| `prompts/summarize_nodes.py::summary_description` | `src/prompts/summarize_nodes.rs` | none | |

Not ported (deliberately): `prompts/extract_nodes_and_edges.py` (combined
path — capture traces first to see which path Graphiti defaults to, per
AGENTS.md open questions), `prompts/summarize_sagas.py` (deferred),
`prompts/eval.py` (eval-only), `prompts/lib.py` (prompt registry — Rust
callers import functions directly).
