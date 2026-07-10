//! Node summary refresh.
//!
//! Ports the batched summarization path of
//! `graphiti_core/utils/maintenance/node_operations.py`
//! (`_extract_entity_summaries_batch` / `_process_summary_flight`) and the
//! sentence-aware truncation of `graphiti_core/utils/text_utils.py`
//! (pinned v0.29.2).

pub mod nodes;

use serde::{Deserialize, Serialize};

/// A node whose summary is being refreshed; mutated in place like
/// upstream's `EntityNode`s (the pipeline diffs before/after to emit ops).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizeNode {
    /// Storage identity (grit node id).
    pub id: String,
    /// Node name (summaries apply by case-insensitive name match).
    pub name: String,
    /// Current summary; updated in place.
    pub summary: String,
    /// Labels; always includes `Entity`.
    pub labels: Vec<String>,
    /// Custom attributes, shown to the summarization prompt.
    pub attributes: serde_json::Map<String, serde_json::Value>,
}
