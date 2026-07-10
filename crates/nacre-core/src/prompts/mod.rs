//! Verbatim ports of `graphiti_core/prompts/` (pinned v0.29.2).
//!
//! Prompt text is byte-identical to upstream (AGENTS.md invariant 1); every
//! deviation must be recorded in `PROMPTS.md`. Fidelity is pinned by
//! committed fixtures rendered from the *actual* upstream Python
//! (`oracle/promptgen/gen_prompt_fixtures.py`) and enforced by
//! `tests/prompt_fidelity.rs` — a transcription slip is a red test, not a
//! silent drift.
//!
//! Prompt functions mirror upstream's `dict[str, Any] -> list[Message]`
//! contract: they take a `serde_json::Value` context and reproduce Python's
//! two interpolation paths exactly — bare f-string interpolation
//! ([`py::py_interp`]: `str()` semantics, `repr` for containers) and
//! [`helpers::to_prompt_json`] (`json.dumps` with default separators).

pub mod extract_nodes;
pub mod helpers;
pub mod py;
pub mod snippets;

use crate::model::{Message, Role};

/// Maximum length for entity/community summaries.
// ports: graphiti_core/utils/text_utils.py::MAX_SUMMARY_CHARS
pub const MAX_SUMMARY_CHARS: usize = 1000;

pub(crate) fn msg(role: Role, content: impl Into<String>) -> Message {
    Message {
        role,
        content: content.into(),
    }
}
