//! Dedup judgment: extracted drafts vs existing graph nodes.
//!
//! Ports the resolution flow of
//! `graphiti_core/utils/maintenance/node_operations.py` +
//! `dedup_helpers.py` (pinned v0.29.2). Candidate *gathering* is the
//! pipeline seam's job (grit's `find_merge_candidates` needs persisted
//! nodes); this module owns the *judgment*: deterministic exact/fuzzy
//! resolution first, one batched LLM escalation for the rest. Nacre
//! decides; grit executes the resulting `MergeNodes` ops.

pub mod helpers;
pub mod nodes;

use serde::{Deserialize, Serialize};

/// An existing graph node offered as a dedup candidate. `id` is the
/// storage identity (grit node id) the pipeline uses to build merge ops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExistingNode {
    /// Storage identity (grit node id, stringly here to stay storage-thin).
    pub id: String,
    /// Node name.
    pub name: String,
    /// Labels; always includes `Entity`.
    pub labels: Vec<String>,
    /// Current summary (may be empty).
    pub summary: String,
    /// Custom attributes, flattened into the dedup prompt context.
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

/// The outcome for one extracted draft, parallel to the input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeResolution {
    /// `Some(existing)` when the draft duplicates an existing node — with
    /// label promotion already applied (a generic existing node inherits
    /// the draft's specific type). `None`: the draft stands as a new node.
    pub duplicate_of: Option<ExistingNode>,
}
