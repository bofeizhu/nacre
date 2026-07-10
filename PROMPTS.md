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
