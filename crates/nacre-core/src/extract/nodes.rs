//! Entity-node extraction: episodes → [`DraftNode`]s.
//!
//! Ports the extraction path of
//! `graphiti_core/utils/maintenance/node_operations.py` and the episode
//! concatenation of `graphiti_core/utils/text_utils.py` (pinned v0.29.2).

use serde_json::{Value, json};

use super::{DraftNode, EntityTypeSpec, EpisodeInput, EpisodeSource};
use crate::model::{CompletionRequest, LanguageModel, ModelError, ModelSize};
use crate::prompts::extract_nodes as prompts;
use crate::schemas::{ExtractedEntities, ResponseSchema};

/// The always-present fallback entity type.
// ports: node_operations.py::_build_entity_types_context (default entry)
const DEFAULT_ENTITY_TYPE_DESCRIPTION: &str = "A specific, identifiable entity that does not fit any of the other listed \
types. Must still be a concrete, meaningful thing — specific enough to be \
uniquely identifiable. GOOD: a named entity not covered by the other types. \
BAD: \"luck\", \"ideas\", \"tomorrow\", \"things\", \"them\", \"everybody\", \
\"a sense of wonder\", \"great times\". \
When in doubt, do not extract the entity.";

/// Appended to custom instructions when extracting from multiple episodes.
// ports: node_operations.py::extract_nodes (episode_attribution)
const EPISODE_ATTRIBUTION: &str = "\n7. **Episode Attribution**: The content contains multiple episodes labeled \
[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating \
when that episode occurred. For each extracted entity, set `episode_indices` \
to the 0-based list of episode numbers where that entity is mentioned. \
An entity appearing in Episodes 0 and 2 should have `episode_indices: [0, 2]`.";

pub use crate::dedupe::helpers::normalize_string_exact;

/// Concatenate episode contents with enumerated headers. A single episode
/// passes through as-is; multiple episodes get `[Episode N]` headers so the
/// model can attribute entities.
// ports: text_utils.py::concatenate_episodes
pub fn concatenate_episodes(episodes: &[EpisodeInput]) -> String {
    if episodes.len() == 1 {
        return episodes[0].content.clone();
    }
    let parts: Vec<String> = episodes
        .iter()
        .enumerate()
        .map(|(i, ep)| {
            let timestamp = ep.valid_at.as_deref().unwrap_or("unknown");
            format!("[Episode {i}] (timestamp: {timestamp})\n{}", ep.content)
        })
        .collect();
    parts.join("\n\n")
}

/// The entity-types prompt context: the default `Entity` entry followed by
/// caller-defined types, ids assigned by position.
// ports: node_operations.py::_build_entity_types_context
pub fn build_entity_types_context(entity_types: &[EntityTypeSpec]) -> Vec<Value> {
    let mut context = vec![json!({
        "entity_type_id": 0,
        "entity_type_name": "Entity",
        "entity_type_description": DEFAULT_ENTITY_TYPE_DESCRIPTION,
    })];
    context.extend(entity_types.iter().enumerate().map(|(i, spec)| {
        json!({
            "entity_type_id": i + 1,
            "entity_type_name": spec.name,
            "entity_type_description": spec.description,
        })
    }));
    context
}

/// Options for [`extract_nodes`].
#[derive(Debug, Clone, Default)]
pub struct ExtractNodesOptions {
    /// Caller-defined entity types (beyond the default `Entity`).
    pub entity_types: Vec<EntityTypeSpec>,
    /// Type names whose entities are dropped after classification.
    pub excluded_entity_types: Vec<String>,
    /// Extra instructions appended to the extraction prompt.
    pub custom_extraction_instructions: String,
}

/// Build the extraction prompt context.
///
/// Public so replay tests (and later the capture-diff tooling) can
/// construct the exact request the step will issue.
// ports: node_operations.py::extract_nodes (context construction)
pub fn build_extraction_context(
    episodes: &[EpisodeInput],
    previous_episodes: &[EpisodeInput],
    options: &ExtractNodesOptions,
) -> Value {
    let primary = &episodes[0];
    let mut custom = options.custom_extraction_instructions.clone();
    if episodes.len() > 1 {
        custom.push_str(EPISODE_ATTRIBUTION);
    }
    let mut context = json!({
        "episode_content": concatenate_episodes(episodes),
        "previous_episodes": previous_episodes
            .iter()
            .map(|ep| json!({"content": ep.content, "timestamp": ep.valid_at}))
            .collect::<Vec<_>>(),
        "custom_extraction_instructions": custom,
        "entity_types": build_entity_types_context(&options.entity_types),
        "source_description": primary.source_description,
    });
    // Upstream sets episode_timestamp unconditionally (and would crash on a
    // missing valid_at); no prompt interpolates it, so a None is simply
    // omitted here rather than crashing.
    if let Some(valid_at) = &primary.valid_at {
        context["episode_timestamp"] = json!(valid_at);
    }
    context
}

/// Extract entity nodes from one or more episodes.
///
/// The first episode is primary (chooses the prompt and provides metadata);
/// multi-episode extraction appends the episode-attribution instructions.
///
/// # Panics
///
/// Panics if `episodes` is empty (upstream indexes `episodes[0]`).
// ports: node_operations.py::extract_nodes
pub async fn extract_nodes<M: LanguageModel>(
    model: &M,
    episodes: &[EpisodeInput],
    previous_episodes: &[EpisodeInput],
    options: &ExtractNodesOptions,
) -> Result<Vec<DraftNode>, ModelError> {
    assert!(!episodes.is_empty(), "extract_nodes requires >= 1 episode");
    let primary = &episodes[0];
    let context = build_extraction_context(episodes, previous_episodes, options);

    // ports: node_operations.py::_call_extraction_llm (prompt routing)
    let messages = match primary.source {
        EpisodeSource::Message => prompts::extract_message(&context),
        EpisodeSource::Text => prompts::extract_text(&context),
        EpisodeSource::Json => prompts::extract_json(&context),
    };
    let request = CompletionRequest {
        messages,
        schema_name: ExtractedEntities::NAME.to_owned(),
        max_tokens: None,
        model_size: ModelSize::Medium,
    };
    let response = model.complete(&request).await?;
    let extracted: ExtractedEntities =
        crate::model::decode_response(response, ExtractedEntities::NAME)?;

    // ports: node_operations.py::extract_nodes (empty-name filter)
    let filtered = extracted
        .extracted_entities
        .into_iter()
        .filter(|e| !e.name.trim().is_empty());

    // ports: node_operations.py::_create_entity_nodes
    let entity_types_context = build_entity_types_context(&options.entity_types);
    let mut drafts: Vec<DraftNode> = Vec::new();
    for entity in filtered {
        let type_name = usize::try_from(entity.entity_type_id)
            .ok()
            .and_then(|id| entity_types_context.get(id))
            .and_then(|t| t["entity_type_name"].as_str())
            .unwrap_or("Entity");
        if options
            .excluded_entity_types
            .iter()
            .any(|excluded| excluded == type_name)
        {
            continue;
        }
        let mut labels = vec!["Entity".to_owned()];
        if type_name != "Entity" {
            labels.push(type_name.to_owned());
        }
        // Clamp to valid episode positions; fall back to all when empty.
        let mut indices: Vec<usize> = entity
            .episode_indices
            .iter()
            .filter_map(|&i| usize::try_from(i).ok())
            .filter(|&i| i < episodes.len())
            .collect();
        if indices.is_empty() {
            indices = (0..episodes.len()).collect();
        }
        drafts.push(DraftNode {
            name: entity.name,
            labels,
            summary: String::new(),
            group_id: primary.group_id.clone(),
            episode_indices: indices,
        });
    }

    Ok(collapse_exact_duplicates(drafts))
}

/// Collapse same-extraction duplicates with the same normalized name,
/// keeping the more specific node (more non-`Entity` labels; ties break to
/// the longer trimmed name) and merging episode attribution.
// ports: node_operations.py::_collapse_exact_duplicate_extracted_nodes
fn collapse_exact_duplicates(nodes: Vec<DraftNode>) -> Vec<DraftNode> {
    if nodes.len() < 2 {
        return nodes;
    }
    let mut canonical: Vec<(String, DraftNode)> = Vec::new();
    for node in nodes {
        let key = normalize_string_exact(&node.name);
        let Some(slot) = canonical.iter_mut().find(|(k, _)| *k == key) else {
            canonical.push((key, node));
            continue;
        };
        let existing = &slot.1;
        let specific = |n: &DraftNode| n.labels.iter().filter(|l| *l != "Entity").count();
        let name_len = |n: &DraftNode| n.name.trim().chars().count();
        let node_wins = specific(&node) > specific(existing)
            || (specific(&node) == specific(existing) && name_len(&node) > name_len(existing));

        let (mut winner, loser) = if node_wins {
            (
                node,
                std::mem::replace(&mut slot.1, DraftNode::default_placeholder()),
            )
        } else {
            (
                std::mem::replace(&mut slot.1, DraftNode::default_placeholder()),
                node,
            )
        };
        let mut merged: Vec<usize> = winner
            .episode_indices
            .iter()
            .chain(&loser.episode_indices)
            .copied()
            .collect();
        merged.sort_unstable();
        merged.dedup();
        winner.episode_indices = merged;
        slot.1 = winner;
    }
    canonical.into_iter().map(|(_, node)| node).collect()
}

impl DraftNode {
    fn default_placeholder() -> Self {
        DraftNode {
            name: String::new(),
            labels: Vec::new(),
            summary: String::new(),
            group_id: String::new(),
            episode_indices: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Recording, RecordingStore, ReplayModel};
    use serde_json::json;

    fn episode(name: &str, content: &str, source: EpisodeSource, valid_at: &str) -> EpisodeInput {
        EpisodeInput {
            name: name.into(),
            content: content.into(),
            source,
            source_description: "chat between friends".into(),
            group_id: "trace-test".into(),
            valid_at: Some(valid_at.into()),
        }
    }

    /// Build a replay model whose single recording answers the exact request
    /// this step will issue for (episodes, previous, options).
    fn replay_for(
        episodes: &[EpisodeInput],
        previous: &[EpisodeInput],
        options: &ExtractNodesOptions,
        response: Value,
    ) -> ReplayModel {
        let context = build_extraction_context(episodes, previous, options);
        let messages = match episodes[0].source {
            EpisodeSource::Message => crate::prompts::extract_nodes::extract_message(&context),
            EpisodeSource::Text => crate::prompts::extract_nodes::extract_text(&context),
            EpisodeSource::Json => crate::prompts::extract_nodes::extract_json(&context),
        };
        let request = CompletionRequest {
            messages,
            schema_name: "ExtractedEntities".into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        let recording = Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        };
        ReplayModel::new(RecordingStore::new([recording]))
    }

    #[tokio::test]
    async fn full_extraction_path_filters_maps_and_collapses() {
        let episodes = vec![
            episode(
                "ep-0",
                "Jordan: hi",
                EpisodeSource::Message,
                "2026-03-02T18:30:00+00:00",
            ),
            episode(
                "ep-1",
                "Marco: yo",
                EpisodeSource::Message,
                "2026-03-09T14:05:00+00:00",
            ),
        ];
        let previous = vec![episode(
            "prev-0",
            "Mina: earlier",
            EpisodeSource::Message,
            "2026-02-01T09:00:00+00:00",
        )];
        let options = ExtractNodesOptions {
            entity_types: vec![
                EntityTypeSpec {
                    name: "Person".into(),
                    description: "A human being mentioned by name.".into(),
                },
                EntityTypeSpec {
                    name: "Confidential".into(),
                    description: "Should never surface.".into(),
                },
            ],
            excluded_entity_types: vec!["Confidential".into()],
            custom_extraction_instructions: "Focus on people.".into(),
        };
        let response = json!({"extracted_entities": [
            {"name": "Jordan", "entity_type_id": 1, "episode_indices": [0]},
            {"name": "   ", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "jordan  ", "entity_type_id": 0, "episode_indices": [1, 7]},
            {"name": "Denver", "entity_type_id": 99, "episode_indices": []},
            {"name": "Secret Base", "entity_type_id": 2, "episode_indices": [0]},
        ]});
        let model = replay_for(&episodes, &previous, &options, response);

        let drafts = extract_nodes(&model, &episodes, &previous, &options)
            .await
            .unwrap();

        assert_eq!(
            drafts.len(),
            2,
            "blank filtered, Secret excluded, jordan collapsed"
        );
        // "Jordan" (Person) beats "jordan  " (Entity); indices merge with
        // out-of-range 7 clamped away.
        assert_eq!(drafts[0].name, "Jordan");
        assert_eq!(drafts[0].labels, vec!["Entity", "Person"]);
        assert_eq!(drafts[0].episode_indices, vec![0, 1]);
        assert_eq!(drafts[0].group_id, "trace-test");
        // Unknown type id 99 falls back to Entity; empty indices fall back
        // to all episodes.
        assert_eq!(drafts[1].name, "Denver");
        assert_eq!(drafts[1].labels, vec!["Entity"]);
        assert_eq!(drafts[1].episode_indices, vec![0, 1]);
    }

    #[tokio::test]
    async fn longer_name_wins_specificity_tie_and_keeps_first_seen_order() {
        let episodes = vec![episode(
            "ep-0",
            "Mary: hi",
            EpisodeSource::Message,
            "2026-03-02T18:30:00+00:00",
        )];
        let options = ExtractNodesOptions::default();
        let response = json!({"extracted_entities": [
            {"name": "NYC ", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "Trigger", "entity_type_id": 0, "episode_indices": [0]},
            {"name": "nyc", "entity_type_id": 0, "episode_indices": [0]},
        ]});
        let model = replay_for(&episodes, &[], &options, response);

        let drafts = extract_nodes(&model, &episodes, &[], &options)
            .await
            .unwrap();
        let names: Vec<&str> = drafts.iter().map(|d| d.name.as_str()).collect();
        // "NYC " trimmed is same length as "nyc" -> tie keeps existing;
        // collapsed entry stays in first-seen position.
        assert_eq!(names, vec!["NYC ", "Trigger"]);
    }

    #[tokio::test]
    async fn json_episodes_route_to_the_json_prompt() {
        let episodes = vec![EpisodeInput {
            name: "ep-0".into(),
            content: r#"{"user": "Jordan Lee"}"#.into(),
            source: EpisodeSource::Json,
            source_description: "CRM export".into(),
            group_id: "g".into(),
            valid_at: Some("2026-07-01T00:00:00+00:00".into()),
        }];
        let options = ExtractNodesOptions::default();
        let response = json!({"extracted_entities": [
            {"name": "Jordan Lee", "entity_type_id": 0, "episode_indices": [0]},
        ]});
        // replay_for keys the recording on the extract_json prompt; a hit
        // proves the routing (a miss would be a loud ReplayMiss error).
        let model = replay_for(&episodes, &[], &options, response);

        let drafts = extract_nodes(&model, &episodes, &[], &options)
            .await
            .unwrap();
        assert_eq!(drafts[0].name, "Jordan Lee");
    }

    #[test]
    fn concatenate_episodes_headers_only_for_multi() {
        let single = vec![episode("a", "solo content", EpisodeSource::Message, "t")];
        assert_eq!(concatenate_episodes(&single), "solo content");

        let mut multi = vec![
            episode(
                "a",
                "first",
                EpisodeSource::Message,
                "2026-01-01T00:00:00+00:00",
            ),
            episode(
                "b",
                "second",
                EpisodeSource::Message,
                "2026-01-02T00:00:00+00:00",
            ),
        ];
        multi[1].valid_at = None;
        assert_eq!(
            concatenate_episodes(&multi),
            "[Episode 0] (timestamp: 2026-01-01T00:00:00+00:00)\nfirst\n\n\
             [Episode 1] (timestamp: unknown)\nsecond"
        );
    }

    #[test]
    fn normalize_matches_python_semantics() {
        assert_eq!(normalize_string_exact("  Jordan\t LEE \n"), "jordan lee");
        assert_eq!(normalize_string_exact("NYC"), "nyc");
    }
}
