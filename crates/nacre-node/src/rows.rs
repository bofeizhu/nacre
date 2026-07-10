//! JS-facing row shapes for the graph view. Ids are strings, timestamps
//! are RFC-3339 with milliseconds, `labels` is unpacked from nacre's
//! `attrs["labels"]` storage convention, remaining attributes ride as
//! plain JSON.

use chrono::DateTime;
use napi_derive::napi;

pub fn iso(ms: i64) -> String {
    DateTime::from_timestamp_millis(ms)
        .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

fn iso_opt(ms: Option<i64>) -> Option<String> {
    ms.map(iso)
}

/// An entity node row (full bi-temporal record; filter `expiredAt` for the
/// live view).
#[napi(object)]
pub struct NodeRow {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub summary: String,
    /// Entity labels (always includes "Entity").
    pub labels: Vec<String>,
    /// Custom attributes (labels removed — they're the field above).
    pub attrs: serde_json::Value,
    pub group_id: String,
    pub created_at: String,
    /// Set when this row was merged away or otherwise retracted.
    pub expired_at: Option<String>,
    /// Canonical node this one folded into, if merged away.
    pub merged_into: Option<String>,
}

/// A fact edge row (both timelines).
#[napi(object)]
pub struct EdgeRow {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    /// Relation label, e.g. "WORKS_AT".
    pub name: String,
    /// The fact sentence.
    pub fact: String,
    pub group_id: String,
    /// Event time the fact became true.
    pub valid_at: Option<String>,
    /// Event time the fact stopped being true.
    pub invalid_at: Option<String>,
    pub created_at: String,
    pub expired_at: Option<String>,
}

/// A provenance episode row.
#[napi(object)]
pub struct EpisodeRow {
    pub id: String,
    /// Source description as stored (e.g. "chat between friends",
    /// "doc:notes.md").
    pub source: String,
    /// Source-kind tag: "message", "text", or "json" (empty for episodes
    /// written before grit 0.2.2).
    pub kind: String,
    pub content: String,
    /// Event time of the episode.
    pub occurred_at: String,
    pub group_id: String,
    pub created_at: String,
}

/// A bounded neighborhood, as returned by `traverse`.
#[napi(object)]
pub struct SubgraphJs {
    pub nodes: Vec<NodeRow>,
    pub edges: Vec<EdgeRow>,
}

/// A node's bi-temporal audit trail.
#[napi(object)]
pub struct NodeHistoryJs {
    pub node: NodeRow,
    /// Every incident edge ever believed, oldest belief first.
    pub edges: Vec<EdgeRow>,
}

pub fn node_row(n: &grit_core::Node) -> NodeRow {
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
        attrs: serde_json::Value::Object(attrs),
        group_id: n.group_id.clone(),
        created_at: iso(n.created_at),
        expired_at: iso_opt(n.expired_at),
        merged_into: n.merged_into.map(|id| id.to_string()),
    }
}

pub fn edge_row(e: &grit_core::Edge) -> EdgeRow {
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

pub fn episode_row(e: &grit_core::Episode) -> EpisodeRow {
    EpisodeRow {
        id: e.id.to_string(),
        source: e.source.clone(),
        kind: e.kind.clone(),
        content: e.content.clone(),
        occurred_at: iso(e.occurred_at),
        group_id: e.group_id.clone(),
        created_at: iso(e.created_at),
    }
}
