//! Relationship-edge extraction: episodes + resolved nodes → [`DraftEdge`]s.
//!
//! Ports the extraction path of
//! `graphiti_core/utils/maintenance/edge_operations.py` and the timestamp
//! handling of `graphiti_core/utils/datetime_utils.py` (pinned v0.29.2).

use std::collections::HashMap;

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde_json::{Value, json};

use super::nodes::concatenate_episodes;
use super::{DraftEdge, EdgeTypeSpec, EpisodeInput, NodeRef};
use crate::model::{CompletionRequest, LanguageModel, ModelError, ModelSize};
use crate::prompts::extract_edges as prompts;
use crate::schemas::{ExtractedEdges, ResponseSchema};

// ports: edge_operations.py::extract_edges (extract_edges_max_tokens)
const EXTRACT_EDGES_MAX_TOKENS: u32 = 16384;

/// Appended to custom instructions when extracting from multiple episodes.
// ports: edge_operations.py::extract_edges (episode_attribution)
const EPISODE_ATTRIBUTION: &str = "\n8. **Episode Attribution**: The CURRENT_MESSAGE contains multiple episodes labeled \
[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating \
when that episode occurred. Use the per-episode timestamp to resolve relative time \
mentions within each episode rather than relying solely on REFERENCE_TIME. \
For each extracted fact, set `episode_indices` \
to the 0-based list of episode numbers that the fact was derived from. \
A fact sourced from Episodes 0 and 1 should have `episode_indices: [0, 1]`.";

/// Render an ISO 8601 timestamp the way Python's `str(datetime)` does —
/// the `T` separator becomes a space. Upstream interpolates the raw
/// `datetime` into the edge prompt (`{context['reference_time']}`), so this
/// spelling is part of trace fidelity.
pub fn py_datetime_str(iso: &str) -> String {
    iso.replacen('T', " ", 1)
}

/// Parse an LLM-provided timestamp like upstream: `Z` normalized to
/// `+00:00`, `datetime.fromisoformat` leniency (date-only and naive forms
/// accepted), `ensure_utc` semantics (naive = UTC, aware converts).
/// Unparseable input is `None` (upstream logs and drops the bound).
// ports: edge_operations.py::extract_edges (date validation) + datetime_utils.py::ensure_utc
pub fn parse_llm_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    let s = raw.replace('Z', "+00:00");
    if let Ok(aware) = DateTime::parse_from_rfc3339(&s) {
        return Some(aware.with_timezone(&Utc));
    }
    // fromisoformat also accepts a space separator and offsets without
    // fractional seconds; try the common aware forms.
    for format in ["%Y-%m-%d %H:%M:%S%.f%:z", "%Y-%m-%dT%H:%M:%S%.f%:z"] {
        if let Ok(aware) = DateTime::parse_from_str(&s, format) {
            return Some(aware.with_timezone(&Utc));
        }
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&s, format) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
        return Some(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?));
    }
    None
}

/// Options for [`extract_edges`].
#[derive(Debug, Clone, Default)]
pub struct ExtractEdgesOptions {
    /// Caller-defined fact types.
    pub edge_types: Vec<EdgeTypeSpec>,
    /// (source type, target type) signature → fact type names valid for it.
    pub edge_type_map: Vec<((String, String), Vec<String>)>,
    /// Extra instructions appended to the extraction prompt.
    pub custom_extraction_instructions: String,
    /// Namespace override; empty uses the primary episode's group id.
    pub group_id: String,
}

/// Build the edge-extraction prompt context. Public so replay tests can
/// construct the exact request the step will issue.
// ports: edge_operations.py::extract_edges (context construction)
pub fn build_edge_extraction_context(
    episodes: &[EpisodeInput],
    nodes: &[NodeRef],
    previous_episodes: &[EpisodeInput],
    options: &ExtractEdgesOptions,
) -> Value {
    // Signature lookup per fact type name, insertion order preserved.
    let signatures_for = |type_name: &str| -> Vec<Value> {
        let found: Vec<Value> = options
            .edge_type_map
            .iter()
            .filter(|(_, names)| names.iter().any(|n| n == type_name))
            .map(|((src, dst), _)| json!([src, dst]))
            .collect();
        if found.is_empty() {
            vec![json!(["Entity", "Entity"])]
        } else {
            found
        }
    };
    let edge_types_context: Vec<Value> = options
        .edge_types
        .iter()
        .map(|spec| {
            json!({
                "fact_type_name": spec.name,
                "fact_type_signatures": signatures_for(&spec.name),
                "fact_type_description": spec.description,
            })
        })
        .collect();

    // Latest episode by event time is the reference time; upstream compares
    // datetimes — ISO strings with a uniform offset compare identically.
    // Python's max keeps the FIRST maximal element.
    let mut latest = &episodes[0];
    for episode in &episodes[1..] {
        if episode.valid_at.as_deref().unwrap_or("") > latest.valid_at.as_deref().unwrap_or("") {
            latest = episode;
        }
    }

    let mut custom = options.custom_extraction_instructions.clone();
    if episodes.len() > 1 {
        custom.push_str(EPISODE_ATTRIBUTION);
    }

    json!({
        "episode_content": concatenate_episodes(episodes),
        "nodes": nodes
            .iter()
            .map(|node| json!({"name": node.name, "entity_types": node.labels}))
            .collect::<Vec<_>>(),
        "previous_episodes": previous_episodes
            .iter()
            .map(|ep| json!({"content": ep.content, "timestamp": ep.valid_at}))
            .collect::<Vec<_>>(),
        // Upstream passes the raw datetime; bare f-string interpolation
        // renders it as str(datetime) — space-separated.
        "reference_time": latest.valid_at.as_deref().map(py_datetime_str),
        "edge_types": edge_types_context,
        "custom_extraction_instructions": custom,
    })
}

/// Extract relationship edges between resolved nodes from one or more
/// episodes. Endpoint names are validated against `nodes` (unknown names
/// and self-edges are dropped), timestamps are validated leniently, and
/// episode attribution is clamped with an all-episodes fallback.
///
/// # Panics
///
/// Panics if `episodes` is empty (upstream indexes `episodes[0]`).
// ports: edge_operations.py::extract_edges
pub async fn extract_edges<M: LanguageModel>(
    model: &M,
    episodes: &[EpisodeInput],
    nodes: &[NodeRef],
    previous_episodes: &[EpisodeInput],
    options: &ExtractEdgesOptions,
) -> Result<Vec<DraftEdge>, ModelError> {
    assert!(!episodes.is_empty(), "extract_edges requires >= 1 episode");
    let primary = &episodes[0];
    let group_id = if options.group_id.is_empty() {
        &primary.group_id
    } else {
        &options.group_id
    };

    let context = build_edge_extraction_context(episodes, nodes, previous_episodes, options);
    let request = CompletionRequest {
        messages: prompts::edge(&context),
        schema_name: ExtractedEdges::NAME.to_owned(),
        max_tokens: Some(EXTRACT_EDGES_MAX_TOKENS),
        model_size: ModelSize::Medium,
    };
    let response = model.complete(&request).await?;
    let extracted: ExtractedEdges = crate::model::decode_response(response, ExtractedEdges::NAME)?;

    // Name → node with upstream's last-wins semantics on duplicate names.
    let mut name_to_node: HashMap<&str, &NodeRef> = HashMap::new();
    for node in nodes {
        name_to_node.insert(node.name.as_str(), node);
    }

    let mut edges = Vec::new();
    for edge in extracted.edges {
        // Endpoint validation: unknown names and self-edges are dropped.
        let (Some(source), Some(target)) = (
            name_to_node.get(edge.source_entity_name.as_str()),
            name_to_node.get(edge.target_entity_name.as_str()),
        ) else {
            continue;
        };
        if source.id == target.id {
            continue;
        }
        // Empty facts are dropped.
        if edge.fact.trim().is_empty() {
            continue;
        }

        let valid_at = edge.valid_at.as_deref().and_then(parse_llm_timestamp);
        let invalid_at = edge.invalid_at.as_deref().and_then(parse_llm_timestamp);

        // Attribution: clamp; fall back to all episodes when empty.
        let mut indices: Vec<usize> = edge
            .episode_indices
            .iter()
            .filter_map(|&i| usize::try_from(i).ok())
            .filter(|&i| i < episodes.len())
            .collect();
        if indices.is_empty() {
            indices = (0..episodes.len()).collect();
        }

        // reference_time uses the FIRST RAW index when in range (upstream
        // checks episode_indices[0] before clamping), else the primary.
        let reference_episode = edge
            .episode_indices
            .first()
            .and_then(|&raw| usize::try_from(raw).ok())
            .filter(|&i| i < episodes.len())
            .map_or(primary, |i| &episodes[i]);

        edges.push(DraftEdge {
            source_id: source.id.clone(),
            target_id: target.id.clone(),
            name: edge.relation_type,
            fact: edge.fact,
            group_id: group_id.clone(),
            episode_indices: indices,
            valid_at,
            invalid_at,
            expired_at: None,
            reference_time: reference_episode.valid_at.clone(),
        });
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::EpisodeSource;
    use crate::model::{Recording, RecordingStore, ReplayModel};
    use serde_json::json;

    fn episode(name: &str, content: &str, valid_at: &str) -> EpisodeInput {
        EpisodeInput {
            name: name.into(),
            content: content.into(),
            source: EpisodeSource::Message,
            source_description: "chat".into(),
            group_id: "g".into(),
            valid_at: Some(valid_at.into()),
        }
    }

    fn node(id: &str, name: &str) -> NodeRef {
        NodeRef {
            id: id.into(),
            name: name.into(),
            labels: vec!["Entity".into()],
        }
    }

    #[test]
    fn timestamps_parse_with_upstream_leniency() {
        let utc = |s: &str| parse_llm_timestamp(s).unwrap().to_rfc3339();
        assert_eq!(utc("2025-04-30T00:00:00Z"), "2025-04-30T00:00:00+00:00");
        assert_eq!(utc("2025-04-30"), "2025-04-30T00:00:00+00:00");
        // Aware non-UTC converts (ensure_utc semantics).
        assert_eq!(
            utc("2025-04-30 10:00:00+02:00"),
            "2025-04-30T08:00:00+00:00"
        );
        // Naive assumes UTC.
        assert_eq!(utc("2025-04-30T10:00:00"), "2025-04-30T10:00:00+00:00");
        assert!(parse_llm_timestamp("not a date").is_none());
    }

    #[test]
    fn reference_time_renders_like_python_str_datetime() {
        let episodes = vec![
            episode("a", "x", "2026-03-02T18:30:00+00:00"),
            episode("b", "y", "2026-03-09T14:05:00+00:00"),
        ];
        let context =
            build_edge_extraction_context(&episodes, &[], &[], &ExtractEdgesOptions::default());
        // Latest episode wins; 'T' becomes a space, like str(datetime).
        assert_eq!(context["reference_time"], "2026-03-09 14:05:00+00:00");
    }

    #[test]
    fn edge_types_context_defaults_signature() {
        let options = ExtractEdgesOptions {
            edge_types: vec![
                EdgeTypeSpec {
                    name: "WORKS_AT".into(),
                    description: "Employment.".into(),
                },
                EdgeTypeSpec {
                    name: "LIVES_IN".into(),
                    description: "Residence.".into(),
                },
            ],
            edge_type_map: vec![(("Person".into(), "Entity".into()), vec!["WORKS_AT".into()])],
            ..Default::default()
        };
        let episodes = vec![episode("a", "x", "2026-01-01T00:00:00+00:00")];
        let context = build_edge_extraction_context(&episodes, &[], &[], &options);
        assert_eq!(
            context["edge_types"][0]["fact_type_signatures"],
            json!([["Person", "Entity"]])
        );
        // No signature registered -> upstream default.
        assert_eq!(
            context["edge_types"][1]["fact_type_signatures"],
            json!([["Entity", "Entity"]])
        );
    }

    #[tokio::test]
    async fn full_edge_path_validates_and_attributes() {
        let episodes = vec![
            episode(
                "ep-0",
                "Priya: I joined Northwind.",
                "2026-03-02T18:30:00+00:00",
            ),
            episode("ep-1", "Marco: nice!", "2026-03-09T14:05:00+00:00"),
        ];
        // "NYC" and "New York City" resolve to the same node post-dedup.
        let nodes = vec![
            node("id-p", "Priya"),
            node("id-n", "Northwind Labs"),
            node("id-c", "NYC"),
            node("id-c", "New York City"),
        ];
        let options = ExtractEdgesOptions::default();
        let context = build_edge_extraction_context(&episodes, &nodes, &[], &options);
        let request = CompletionRequest {
            messages: crate::prompts::extract_edges::edge(&context),
            schema_name: "ExtractedEdges".into(),
            max_tokens: Some(16384),
            model_size: ModelSize::Medium,
        };
        let response = json!({"edges": [
            {"source_entity_name": "Priya", "target_entity_name": "Northwind Labs",
             "relation_type": "WORKS_AT", "fact": "Priya joined Northwind Labs.",
             "valid_at": "2026-01-15T00:00:00Z", "invalid_at": "garbage",
             "episode_indices": [0]},
            {"source_entity_name": "Ghost", "target_entity_name": "Northwind Labs",
             "relation_type": "WORKS_AT", "fact": "dropped: unknown source",
             "episode_indices": [0]},
            {"source_entity_name": "NYC", "target_entity_name": "New York City",
             "relation_type": "IS", "fact": "dropped: self edge",
             "episode_indices": [0]},
            {"source_entity_name": "Priya", "target_entity_name": "NYC",
             "relation_type": "VISITED", "fact": "   ",
             "episode_indices": [0]},
            {"source_entity_name": "Priya", "target_entity_name": "NYC",
             "relation_type": "VISITED", "fact": "Priya visited NYC.",
             "episode_indices": [5]},
        ]});
        let model = ReplayModel::new(RecordingStore::new([Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }]));

        let edges = extract_edges(&model, &episodes, &nodes, &[], &options)
            .await
            .unwrap();
        assert_eq!(edges.len(), 2);

        let works_at = &edges[0];
        assert_eq!(
            (works_at.source_id.as_str(), works_at.target_id.as_str()),
            ("id-p", "id-n")
        );
        assert_eq!(
            works_at.valid_at.unwrap().to_rfc3339(),
            "2026-01-15T00:00:00+00:00"
        );
        assert!(works_at.invalid_at.is_none(), "garbage date -> None");
        assert_eq!(works_at.episode_indices, vec![0]);
        assert_eq!(
            works_at.reference_time.as_deref(),
            Some("2026-03-02T18:30:00+00:00")
        );
        assert_eq!(works_at.group_id, "g");

        let visited = &edges[1];
        // Raw index 5 is out of range: attribution falls back to all
        // episodes and reference_time to the primary.
        assert_eq!(visited.episode_indices, vec![0, 1]);
        assert_eq!(
            visited.reference_time.as_deref(),
            Some("2026-03-02T18:30:00+00:00")
        );
    }
}
