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

pub mod dedupe_edges;
pub mod dedupe_nodes;
pub mod extract_edges;
pub mod extract_nodes;
pub mod helpers;
pub mod py;
pub mod snippets;
pub mod summarize_nodes;

use crate::model::{Message, Role};

/// Maximum length for entity/community summaries.
// ports: graphiti_core/utils/text_utils.py::MAX_SUMMARY_CHARS
pub const MAX_SUMMARY_CHARS: usize = 1000;

pub(crate) fn msg(role: Role, content: impl Into<String>) -> Message {
    let mut content = content.into();
    // ports: prompts/lib.py::VersionWrapper.__call__ — the prompt library
    // appends this suffix to every SYSTEM message at render time, so every
    // production prompt carries it. Discovered via golden trace #1: the
    // fixture generator originally called prompt functions directly and
    // missed the wrapper (it now applies it too).
    if role == Role::System {
        content.push_str(helpers::DO_NOT_ESCAPE_UNICODE);
    }
    Message { role, content }
}
