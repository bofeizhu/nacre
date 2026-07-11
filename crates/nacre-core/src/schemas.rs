//! Structured-output schemas for every pipeline prompt.
//!
//! Ports the pydantic response models from `graphiti_core/prompts/*.py`
//! (pinned v0.29.2). **Field names are byte-identical to the Python** —
//! they are what appears in recordings and golden traces — and defaults
//! match pydantic's (`episode_indices` defaults to `[0]`, optional
//! timestamps to `null`). Optional fields serialize as explicit `null`,
//! matching pydantic's `model_dump`, so round-trips are byte-stable.
//!
//! Struct names mirror the Python class names; [`ResponseSchema::NAME`]
//! carries that name into [`crate::model::CompletionRequest::schema_name`].

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// A typed prompt-response schema: its `NAME` is the Python class name and
/// the value of `CompletionRequest::schema_name` for requests expecting it.
pub trait ResponseSchema: DeserializeOwned {
    /// The upstream pydantic class name.
    const NAME: &'static str;
}

macro_rules! response_schema {
    ($ty:ty) => {
        impl ResponseSchema for $ty {
            const NAME: &'static str = stringify!($ty);
        }
    };
}

fn default_episode_indices() -> Vec<i64> {
    vec![0]
}

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/extract_nodes.py
// ---------------------------------------------------------------------------

/// One entity found in an episode.
// ports: graphiti_core/prompts/extract_nodes.py::ExtractedEntity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Name of the extracted entity.
    pub name: String,
    /// ID of the classified entity type; one of the provided
    /// `entity_type_id` integers.
    pub entity_type_id: i64,
    /// Episode numbers (0-indexed) this entity was extracted from;
    /// `[0]` when processing a single episode.
    #[serde(default = "default_episode_indices")]
    pub episode_indices: Vec<i64>,
}
response_schema!(ExtractedEntity);

/// Response of the entity-extraction prompts.
// ports: graphiti_core/prompts/extract_nodes.py::ExtractedEntities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEntities {
    /// List of extracted entities.
    pub extracted_entities: Vec<ExtractedEntity>,
}
response_schema!(ExtractedEntities);

/// Response carrying a single entity summary.
// ports: graphiti_core/prompts/extract_nodes.py::EntitySummary
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntitySummary {
    /// Summary of the entity.
    pub summary: String,
}
response_schema!(EntitySummary);

/// One updated entity summary in a batched summarization response.
// ports: graphiti_core/prompts/extract_nodes.py::SummarizedEntity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizedEntity {
    /// Name of the entity being summarized.
    pub name: String,
    /// Updated summary for the entity.
    pub summary: String,
}
response_schema!(SummarizedEntity);

/// Response of the batched summarization prompt; only entities needing
/// updates are included.
// ports: graphiti_core/prompts/extract_nodes.py::SummarizedEntities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizedEntities {
    /// List of entity summaries.
    pub summaries: Vec<SummarizedEntity>,
}
response_schema!(SummarizedEntities);

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/extract_edges.py
// ---------------------------------------------------------------------------

/// One relationship fact between two extracted entities.
// ports: graphiti_core/prompts/extract_edges.py::Edge
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// The name of the source entity from the ENTITIES list.
    pub source_entity_name: String,
    /// The name of the target entity from the ENTITIES list.
    pub target_entity_name: String,
    /// Relationship type in SCREAMING_SNAKE_CASE (e.g. `WORKS_AT`).
    pub relation_type: String,
    /// Natural-language description of the relationship, paraphrased from
    /// the source text.
    pub fact: String,
    /// When the relationship became true (ISO 8601), if stated.
    #[serde(default)]
    pub valid_at: Option<String>,
    /// When the relationship stopped being true (ISO 8601), if stated.
    #[serde(default)]
    pub invalid_at: Option<String>,
    /// Episode numbers (0-indexed) this fact was derived from; `[0]` when
    /// processing a single episode.
    #[serde(default = "default_episode_indices")]
    pub episode_indices: Vec<i64>,
}
response_schema!(Edge);

/// Response of the edge-extraction prompt.
// ports: graphiti_core/prompts/extract_edges.py::ExtractedEdges
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedEdges {
    /// The extracted relationship facts.
    pub edges: Vec<Edge>,
}
response_schema!(ExtractedEdges);

/// Temporal bounds extracted from a fact.
// ports: graphiti_core/prompts/extract_edges.py::EdgeTimestamps
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeTimestamps {
    /// When the fact became true (ISO 8601 with Z suffix).
    #[serde(default)]
    pub valid_at: Option<String>,
    /// When the fact stopped being true (ISO 8601 with Z suffix).
    #[serde(default)]
    pub invalid_at: Option<String>,
}
response_schema!(EdgeTimestamps);

/// Temporal bounds for a batch of facts, in input order.
// ports: graphiti_core/prompts/extract_edges.py::BatchEdgeTimestamps
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEdgeTimestamps {
    /// Timestamps for each fact, in the same order as the input facts.
    pub timestamps: Vec<EdgeTimestamps>,
}
response_schema!(BatchEdgeTimestamps);

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/dedupe_nodes.py
// ---------------------------------------------------------------------------

/// One dedup resolution: an extracted entity and the existing entity it
/// duplicates, if any.
// ports: graphiti_core/prompts/dedupe_nodes.py::NodeDuplicate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDuplicate {
    /// Integer id of the entity.
    pub id: i64,
    /// Most complete and descriptive name of the entity.
    pub name: String,
    /// `candidate_id` of the matching EXISTING ENTITY, or -1 if no
    /// duplicate exists.
    pub duplicate_candidate_id: i64,
}
response_schema!(NodeDuplicate);

/// Response of the node-dedup prompt.
// ports: graphiti_core/prompts/dedupe_nodes.py::NodeResolutions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeResolutions {
    /// List of resolved nodes.
    pub entity_resolutions: Vec<NodeDuplicate>,
}
response_schema!(NodeResolutions);

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/dedupe_edges.py
// ---------------------------------------------------------------------------

/// Response of the edge-dedup prompt: which existing facts the new fact
/// duplicates, and which facts it contradicts.
// ports: graphiti_core/prompts/dedupe_edges.py::EdgeDuplicate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeDuplicate {
    /// `idx` values of duplicate facts (only from the EXISTING FACTS
    /// range); empty if none.
    pub duplicate_facts: Vec<i64>,
    /// `idx` values of contradicted facts (from the full idx range);
    /// empty if none.
    pub contradicted_facts: Vec<i64>,
}
response_schema!(EdgeDuplicate);

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/extract_nodes_and_edges.py
// ---------------------------------------------------------------------------

/// Entity extracted by the combined node+edge extraction prompt.
// ports: graphiti_core/prompts/extract_nodes_and_edges.py::CombinedEntity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombinedEntity {
    /// Name of the extracted entity.
    pub name: String,
    /// ID of the classified entity type; one of the provided
    /// `entity_type_id` integers.
    pub entity_type_id: i64,
}
response_schema!(CombinedEntity);

/// Relationship fact extracted by the combined node+edge extraction prompt.
// ports: graphiti_core/prompts/extract_nodes_and_edges.py::CombinedFact
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombinedFact {
    /// The name of the source entity from the extracted entities list.
    pub source_entity_name: String,
    /// The name of the target entity from the extracted entities list.
    pub target_entity_name: String,
    /// Relationship type in SCREAMING_SNAKE_CASE (e.g. `WORKS_AT`).
    pub relation_type: String,
    /// Self-contained natural-language description of the relationship,
    /// with all specific details preserved.
    pub fact: String,
    /// Episode numbers (0-indexed) this fact was derived from; `[0]` when
    /// processing a single episode.
    #[serde(default = "default_episode_indices")]
    pub episode_indices: Vec<i64>,
}
response_schema!(CombinedFact);

/// Combined node and edge extraction response.
// ports: graphiti_core/prompts/extract_nodes_and_edges.py::CombinedExtraction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombinedExtraction {
    /// List of extracted entities.
    pub extracted_entities: Vec<CombinedEntity>,
    /// List of extracted relationship facts.
    pub edges: Vec<CombinedFact>,
}
response_schema!(CombinedExtraction);

// ---------------------------------------------------------------------------
// ports: graphiti_core/prompts/summarize_nodes.py
// ---------------------------------------------------------------------------

/// Response carrying an entity summary (bounded length; the bound lives in
/// the prompt text).
// ports: graphiti_core/prompts/summarize_nodes.py::Summary
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    /// Summary containing the important information about the entity.
    pub summary: String,
}
response_schema!(Summary);

/// Response carrying a one-sentence description of a summary.
// ports: graphiti_core/prompts/summarize_nodes.py::SummaryDescription
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryDescription {
    /// One sentence description of the provided summary.
    pub description: String,
}
response_schema!(SummaryDescription);

/// Validate a raw model response against the registered schema's Rust
/// model — a full serde decode, not a key-presence check. Prompt-only
/// providers (DeepSeek schema-in-prompt) sometimes return the right keys
/// with the wrong value types (observed live: `extracted_entities` as a
/// map instead of a sequence); the pipeline's later `decode_response`
/// would fail, so clients validate here and burn a retry instead.
/// Unknown schema names pass (mirrors the client registry's leniency).
// ports: the response-model validation both upstream clients and
// oracle/recording_clients.py perform before accepting a response
pub fn validate_response(schema_name: &str, value: &serde_json::Value) -> Result<(), String> {
    fn check<T: DeserializeOwned>(value: &serde_json::Value) -> Result<(), String> {
        serde_json::from_value::<T>(value.clone())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    match schema_name {
        "ExtractedEntity" => check::<ExtractedEntity>(value),
        "ExtractedEntities" => check::<ExtractedEntities>(value),
        "EntitySummary" => check::<EntitySummary>(value),
        "SummarizedEntity" => check::<SummarizedEntity>(value),
        "SummarizedEntities" => check::<SummarizedEntities>(value),
        "Edge" => check::<Edge>(value),
        "ExtractedEdges" => check::<ExtractedEdges>(value),
        "EdgeTimestamps" => check::<EdgeTimestamps>(value),
        "BatchEdgeTimestamps" => check::<BatchEdgeTimestamps>(value),
        "NodeDuplicate" => check::<NodeDuplicate>(value),
        "NodeResolutions" => check::<NodeResolutions>(value),
        "EdgeDuplicate" => check::<EdgeDuplicate>(value),
        "CombinedEntity" => check::<CombinedEntity>(value),
        "CombinedFact" => check::<CombinedFact>(value),
        "CombinedExtraction" => check::<CombinedExtraction>(value),
        "Summary" => check::<Summary>(value),
        "SummaryDescription" => check::<SummaryDescription>(value),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Deserialize `input` as `T`, re-serialize, and assert the result is
    /// exactly `expected` (or `input` itself when omitted) — the pydantic
    /// `model_dump` round-trip law.
    fn round_trip<T: ResponseSchema + Serialize>(input: Value, expected: Option<Value>) {
        let typed: T = serde_json::from_value(input.clone())
            .unwrap_or_else(|e| panic!("{} failed to deserialize: {e}", T::NAME));
        let back = serde_json::to_value(&typed).unwrap();
        assert_eq!(back, expected.unwrap_or(input), "{} round-trip", T::NAME);
    }

    #[test]
    fn extracted_entities_round_trips() {
        round_trip::<ExtractedEntities>(
            json!({"extracted_entities": [
                {"name": "Yoneda lemma", "entity_type_id": 0, "episode_indices": [0, 2]}
            ]}),
            None,
        );
    }

    #[test]
    fn extracted_entity_defaults_episode_indices_like_pydantic() {
        round_trip::<ExtractedEntity>(
            json!({"name": "Grothendieck", "entity_type_id": 1}),
            Some(json!({
                "name": "Grothendieck",
                "entity_type_id": 1,
                "episode_indices": [0]
            })),
        );
    }

    #[test]
    fn edge_defaults_and_nulls_match_pydantic() {
        // Missing optionals come back as explicit nulls, like model_dump.
        round_trip::<Edge>(
            json!({
                "source_entity_name": "Bofei",
                "target_entity_name": "pearl",
                "relation_type": "WORKS_ON",
                "fact": "Bofei works on pearl."
            }),
            Some(json!({
                "source_entity_name": "Bofei",
                "target_entity_name": "pearl",
                "relation_type": "WORKS_ON",
                "fact": "Bofei works on pearl.",
                "valid_at": null,
                "invalid_at": null,
                "episode_indices": [0]
            })),
        );
    }

    #[test]
    fn extracted_edges_full_round_trips() {
        round_trip::<ExtractedEdges>(
            json!({"edges": [{
                "source_entity_name": "a",
                "target_entity_name": "b",
                "relation_type": "KNOWS",
                "fact": "a knows b",
                "valid_at": "2026-07-01T00:00:00Z",
                "invalid_at": null,
                "episode_indices": [1]
            }]}),
            None,
        );
    }

    #[test]
    fn batch_edge_timestamps_round_trips() {
        round_trip::<BatchEdgeTimestamps>(
            json!({"timestamps": [
                {"valid_at": "2025-04-30T00:00:00Z", "invalid_at": null},
                {"valid_at": null, "invalid_at": null}
            ]}),
            None,
        );
    }

    #[test]
    fn node_resolutions_round_trips() {
        round_trip::<NodeResolutions>(
            json!({"entity_resolutions": [
                {"id": 0, "name": "Yoneda lemma", "duplicate_candidate_id": -1},
                {"id": 1, "name": "category theory", "duplicate_candidate_id": 3}
            ]}),
            None,
        );
    }

    #[test]
    fn edge_duplicate_round_trips() {
        round_trip::<EdgeDuplicate>(
            json!({"duplicate_facts": [], "contradicted_facts": [2, 5]}),
            None,
        );
    }

    #[test]
    fn combined_extraction_round_trips() {
        round_trip::<CombinedExtraction>(
            json!({
                "extracted_entities": [{"name": "oyster", "entity_type_id": 0}],
                "edges": [{
                    "source_entity_name": "oyster",
                    "target_entity_name": "grit",
                    "relation_type": "DEPOSITS_NACRE_AROUND",
                    "fact": "The oyster deposits nacre around grit.",
                    "episode_indices": [0]
                }]
            }),
            None,
        );
    }

    #[test]
    fn summaries_round_trip() {
        round_trip::<Summary>(json!({"summary": "An embedded graph store."}), None);
        round_trip::<SummaryDescription>(json!({"description": "One sentence."}), None);
        round_trip::<SummarizedEntities>(
            json!({"summaries": [{"name": "grit", "summary": "Layer 1."}]}),
            None,
        );
        round_trip::<EntitySummary>(json!({"summary": "Layer 2."}), None);
    }

    #[test]
    fn schema_names_match_python_class_names() {
        assert_eq!(ExtractedEntities::NAME, "ExtractedEntities");
        assert_eq!(Edge::NAME, "Edge");
        assert_eq!(BatchEdgeTimestamps::NAME, "BatchEdgeTimestamps");
        assert_eq!(NodeResolutions::NAME, "NodeResolutions");
        assert_eq!(EdgeDuplicate::NAME, "EdgeDuplicate");
        assert_eq!(CombinedExtraction::NAME, "CombinedExtraction");
        assert_eq!(SummaryDescription::NAME, "SummaryDescription");
    }

    #[test]
    fn decode_response_integrates_with_schemas() {
        let value = json!({"extracted_entities": []});
        let typed: ExtractedEntities =
            crate::model::decode_response(value, ExtractedEntities::NAME).unwrap();
        assert!(typed.extracted_entities.is_empty());
    }

    #[test]
    fn validate_response_catches_wrong_typed_values() {
        // The live-smoke failure shape: right key, map instead of sequence.
        let bad = json!({"extracted_entities": {"name": "Waffle", "entity_type_id": 0}});
        let err = validate_response("ExtractedEntities", &bad).unwrap_err();
        assert!(err.contains("invalid type"), "{err}");

        let good = json!({"extracted_entities": [
            {"name": "Waffle", "entity_type_id": 0, "episode_indices": [0]}
        ]});
        validate_response("ExtractedEntities", &good).unwrap();

        // Key-presence checks would pass this; full decode must not.
        let wrong_inner = json!({"duplicates": [{"id": "not-an-int"}]});
        assert!(validate_response("NodeResolutions", &wrong_inner).is_err());

        // Unregistered names stay lenient, mirroring the client registry.
        validate_response("SomeFutureSchema", &json!({"anything": 1})).unwrap();
    }
}
