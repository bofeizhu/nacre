//! Extraction pipeline steps: episodes in, draft nodes/edges out.
//!
//! Ports the extraction paths of `graphiti_core/utils/maintenance/`
//! (pinned v0.29.2). Steps are pure between model calls: they build prompt
//! contexts, call the injected [`crate::model::LanguageModel`], decode the
//! typed response, and apply upstream's post-processing — no storage access
//! here (drafts flow to the dedupe step, then leave nacre as grit ops).

pub mod nodes;

use serde::{Deserialize, Serialize};

/// Where an episode came from — mirrors upstream `EpisodeType`, which picks
/// the extraction prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeSource {
    /// Conversational message(s) (`speaker: text` lines).
    Message,
    /// Unstructured text.
    Text,
    /// A JSON document.
    Json,
}

/// One episode as pipeline input.
///
/// Timestamps are ISO 8601 strings exactly as Python's
/// `datetime.isoformat()` renders them — they are interpolated into prompt
/// text, so their formatting is part of trace fidelity. The pipeline seam
/// (not this module) converts between grit's typed timestamps and these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeInput {
    /// Human-readable episode name.
    pub name: String,
    /// Raw episode body.
    pub content: String,
    /// Source kind; picks the extraction prompt.
    pub source: EpisodeSource,
    /// Free-text description of the source (used by the JSON prompt).
    pub source_description: String,
    /// Namespace, forwarded to created nodes.
    pub group_id: String,
    /// Event time, ISO 8601 (`datetime.isoformat()` form), if known.
    pub valid_at: Option<String>,
}

/// A caller-defined entity type: name plus the description shown to the
/// model (upstream uses the pydantic model docstring).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityTypeSpec {
    /// Type name (e.g. `Person`).
    pub name: String,
    /// Description used for classification guidance.
    pub description: String,
}

/// An extracted entity node before dedup/persistence. Identity is
/// positional at this stage — grit assigns durable ids when the ops are
/// applied (upstream creates `EntityNode` uuids here; nacre defers that to
/// the executor, per the decider/executor split).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftNode {
    /// Entity name as extracted.
    pub name: String,
    /// Labels: always contains `Entity`, plus the specific type when
    /// classified. Compare as sets — upstream materializes a Python set,
    /// so label ORDER is not part of the contract (golden-trace dumps
    /// sort labels).
    pub labels: Vec<String>,
    /// Empty at extraction; filled by the summarize step.
    pub summary: String,
    /// Namespace, from the primary episode.
    pub group_id: String,
    /// 0-indexed positions of the episodes this entity was extracted from.
    pub episode_indices: Vec<usize>,
}
