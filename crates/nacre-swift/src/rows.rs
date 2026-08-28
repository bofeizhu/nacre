//! Swift-facing row shapes for the graph view.
//!
//! Ported from the Node surface (`nacre-node/src/rows.rs`) so both hosts
//! read the same data contract: ids are strings, timestamps are RFC-3339
//! with milliseconds (`Z` form), `labels` is unpacked from nacre's
//! `attrs["labels"]` storage convention, and the remaining attributes ride
//! as a JSON *string* — UniFFI has no dynamic-value type, so hosts decode
//! `attrsJson` themselves (`JSONSerialization` / `JSONDecoder`).

use chrono::DateTime;

/// Render a millisecond epoch as RFC-3339 with millisecond precision
/// (`2026-07-14T00:00:00.000Z`). An out-of-range timestamp — impossible
/// for values grit produced — renders as the empty string rather than
/// panicking on the read path.
pub(crate) fn iso(ms: i64) -> String {
    DateTime::from_timestamp_millis(ms)
        .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

fn iso_opt(ms: Option<i64>) -> Option<String> {
    ms.map(iso)
}

/// An entity node row — the full bi-temporal record, merged-away rows
/// included (filter on `expired_at` for the live view).
#[derive(Debug, Clone, uniffi::Record)]
pub struct NodeRow {
    /// Storage identity (grit node id).
    pub id: String,
    /// The node's most specific entity label, e.g. `Person`.
    pub kind: String,
    /// Entity name.
    pub name: String,
    /// Rolling summary maintained by the pipeline.
    pub summary: String,
    /// Entity labels (always includes `Entity`); falls back to `[kind]`
    /// for rows written without the `labels` attribute.
    pub labels: Vec<String>,
    /// Remaining custom attributes as a JSON object string (`labels`
    /// removed — it is the field above).
    pub attrs_json: String,
    /// Namespace this node lives in.
    pub group_id: String,
    /// Belief time the row was written.
    pub created_at: String,
    /// Set when this row was merged away or otherwise retracted.
    pub expired_at: Option<String>,
    /// Canonical node this one folded into, if merged away.
    pub merged_into: Option<String>,
}

/// A fact edge row, on both timelines.
///
/// Edge attributes are deliberately not exposed (parity with the Node
/// surface): provenance is the useful drill-down and it comes from
/// [`crate::Memory::mentions_of`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct EdgeRow {
    /// Storage identity (grit edge id).
    pub id: String,
    /// Source node id.
    pub source_id: String,
    /// Target node id.
    pub target_id: String,
    /// Relation label, e.g. `WORKS_AT`.
    pub name: String,
    /// The fact sentence.
    pub fact: String,
    /// Namespace this edge lives in.
    pub group_id: String,
    /// Event time the fact became true.
    pub valid_at: Option<String>,
    /// Event time the fact stopped being true.
    pub invalid_at: Option<String>,
    /// Belief time the row was written.
    pub created_at: String,
    /// Belief time the fact was retracted.
    pub expired_at: Option<String>,
}

/// A provenance episode row.
#[derive(Debug, Clone, uniffi::Record)]
pub struct EpisodeRow {
    /// Storage identity (grit episode id).
    pub id: String,
    /// Source description as stored, e.g. `chat between friends` or
    /// `doc:notes.md`.
    pub source: String,
    /// Source-kind tag: `message`, `text`, or `json` (empty for episodes
    /// written before grit 0.2.2).
    pub kind: String,
    /// Raw episode body.
    pub content: String,
    /// Event time of the episode.
    pub occurred_at: String,
    /// Namespace this episode lives in.
    pub group_id: String,
    /// Belief time the row was written.
    pub created_at: String,
    /// Whatever the host stored on this episode at ingestion, verbatim —
    /// nacre never reads it. `nil` when nothing was stored (a different
    /// fact from `""`), and for episodes written before grit's schema v6.
    pub metadata: Option<String>,
}

/// A bounded neighborhood, as returned by [`crate::Memory::traverse`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct Subgraph {
    /// Nodes reached by the walk.
    pub nodes: Vec<NodeRow>,
    /// Edges between them.
    pub edges: Vec<EdgeRow>,
}

/// A node's bi-temporal audit trail.
#[derive(Debug, Clone, uniffi::Record)]
pub struct NodeHistory {
    /// The node row itself, even if expired.
    pub node: NodeRow,
    /// Every incident edge ever believed, oldest belief first.
    pub edges: Vec<EdgeRow>,
}

pub(crate) fn node_row(n: &grit_core::Node) -> NodeRow {
    let mut attrs = match &n.attrs {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    let labels = attrs
        .remove("labels")
        .and_then(|v| {
            v.as_array().map(|l| {
                l.iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_else(|| vec![n.kind.clone()]);
    NodeRow {
        id: n.id.to_string(),
        kind: n.kind.clone(),
        name: n.name.clone(),
        summary: n.summary.clone(),
        labels,
        attrs_json: serde_json::Value::Object(attrs).to_string(),
        group_id: n.group_id.clone(),
        created_at: iso(n.created_at),
        expired_at: iso_opt(n.expired_at),
        merged_into: n.merged_into.map(|id| id.to_string()),
    }
}

pub(crate) fn edge_row(e: &grit_core::Edge) -> EdgeRow {
    EdgeRow {
        id: e.id.to_string(),
        source_id: e.src.to_string(),
        target_id: e.dst.to_string(),
        name: e.rel.clone(),
        fact: e.fact.clone(),
        group_id: e.group_id.clone(),
        valid_at: iso_opt(e.valid_at),
        invalid_at: iso_opt(e.invalid_at),
        created_at: iso(e.created_at),
        expired_at: iso_opt(e.expired_at),
    }
}

pub(crate) fn episode_row(e: &grit_core::Episode) -> EpisodeRow {
    EpisodeRow {
        id: e.id.to_string(),
        source: e.source.clone(),
        kind: e.kind.clone(),
        content: e.content.clone(),
        occurred_at: iso(e.occurred_at),
        group_id: e.group_id.clone(),
        created_at: iso(e.created_at),
        metadata: e.metadata.clone(),
    }
}
