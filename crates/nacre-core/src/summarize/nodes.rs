//! Batched node summarization with the fact-append shortcut.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::SummarizeNode;
use crate::extract::nodes::concatenate_episodes;
use crate::extract::{EntityTypeSpec, EpisodeInput};
use crate::model::{CompletionRequest, LanguageModel, ModelError, ModelSize};
use crate::prompts::MAX_SUMMARY_CHARS;
use crate::prompts::extract_nodes as prompts;
use crate::schemas::{ResponseSchema, SummarizedEntities};

/// Maximum number of nodes summarized in a single LLM call.
// ports: node_operations.py::MAX_NODES
pub const MAX_NODES: usize = 30;

/// Truncate at or about `max_chars` while respecting sentence boundaries
/// (`.`/`!`/`?` followed by whitespace or end); hard-cuts when no boundary
/// exists before the limit.
// ports: text_utils.py::truncate_at_sentence
pub fn truncate_at_sentence(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() || chars.len() <= max_chars {
        return text.to_owned();
    }
    let truncated = &chars[..max_chars];
    // Find the last `[.!?](?:\s|$)` match; its end includes the whitespace.
    let mut last_end: Option<usize> = None;
    for (i, &c) in truncated.iter().enumerate() {
        if !matches!(c, '.' | '!' | '?') {
            continue;
        }
        if i + 1 == truncated.len() {
            last_end = Some(i + 1);
        } else if truncated[i + 1].is_whitespace() {
            last_end = Some(i + 2);
        }
    }
    match last_end {
        Some(end) => chars[..end]
            .iter()
            .collect::<String>()
            .trim_end()
            .to_owned(),
        None => truncated.iter().collect::<String>().trim_end().to_owned(),
    }
}

/// Index of the first sentence boundary: `.`/`!`/`?` at end-of-string or
/// followed by a space and an uppercase letter (avoids "e.g.", "Dr.", 2.0).
// ports: node_operations.py::_find_sentence_end
fn find_sentence_end(chars: &[char]) -> Option<usize> {
    let n = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        if !matches!(c, '.' | '!' | '?') {
            continue;
        }
        if i + 1 >= n {
            return Some(i);
        }
        if chars[i + 1] == ' ' && i + 2 < n && chars[i + 2].is_uppercase() {
            return Some(i);
        }
    }
    None
}

/// Concise type description for summary prompts: first paragraph only,
/// capped at 3 sentences (strips extraction-only GOOD/BAD guidance).
// ports: node_operations.py::_truncate_type_description
pub fn truncate_type_description(docstring: &str) -> String {
    let mut paragraph_lines: Vec<&str> = Vec::new();
    for line in docstring.lines() {
        if line.trim().is_empty() {
            if !paragraph_lines.is_empty() {
                break;
            }
            continue;
        }
        paragraph_lines.push(line);
    }
    let text: String = paragraph_lines
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join(" ");

    let mut sentences: Vec<String> = Vec::new();
    let mut remaining: Vec<char> = text.chars().collect();
    for _ in 0..3 {
        match find_sentence_end(&remaining) {
            None => {
                sentences.push(remaining.iter().collect());
                remaining.clear();
                break;
            }
            Some(idx) => {
                sentences.push(remaining[..=idx].iter().collect());
                let rest: Vec<char> = remaining[idx + 1..].to_vec();
                let skip = rest.iter().take_while(|c| c.is_whitespace()).count();
                remaining = rest[skip..].to_vec();
            }
        }
    }
    sentences.join(" ").trim().to_owned()
}

/// Node-id → connected edge facts, both endpoints attributed, edge order
/// preserved.
// ports: node_operations.py::_build_edges_by_node
pub fn build_edges_by_node(edges: &[(String, String, String)]) -> HashMap<String, Vec<String>> {
    let mut by_node: HashMap<String, Vec<String>> = HashMap::new();
    for (source_id, target_id, fact) in edges {
        by_node
            .entry(source_id.clone())
            .or_default()
            .push(fact.clone());
        by_node
            .entry(target_id.clone())
            .or_default()
            .push(fact.clone());
    }
    by_node
}

/// Options for [`extract_entity_summaries_batch`].
#[derive(Default)]
pub struct SummarizeOptions<'a> {
    /// Caller-defined entity types (descriptions get first-paragraph /
    /// 3-sentence truncation before entering the prompt).
    pub entity_types: &'a [EntityTypeSpec],
    /// When true, bypass the fact-append shortcut and route every node
    /// through the episode-based summary prompt (upstream's async-worker
    /// path).
    pub skip_fact_appending: bool,
    /// Optional per-node filter; nodes it rejects are left untouched.
    #[allow(clippy::type_complexity)]
    pub should_summarize_node: Option<&'a dyn Fn(&SummarizeNode) -> bool>,
}

/// Refresh summaries for `nodes` in place: fact-append shortcut for short
/// summaries, batched LLM flights of [`MAX_NODES`] for the rest, applied by
/// case-insensitive name with sentence-aware truncation.
// ports: node_operations.py::_extract_entity_summaries_batch + _process_summary_flight
pub async fn extract_entity_summaries_batch<M: LanguageModel>(
    model: &M,
    nodes: &mut [SummarizeNode],
    episodes: &[EpisodeInput],
    previous_episodes: &[EpisodeInput],
    edges_by_node: &HashMap<String, Vec<String>>,
    options: &SummarizeOptions<'_>,
) -> Result<(), ModelError> {
    let mut needs_llm: Vec<usize> = Vec::new();

    for (idx, node) in nodes.iter_mut().enumerate() {
        if let Some(filter) = options.should_summarize_node
            && !filter(node)
        {
            continue;
        }

        if options.skip_fact_appending {
            if !episodes.is_empty() || !node.summary.is_empty() {
                needs_llm.push(idx);
            }
            continue;
        }

        let node_facts = edges_by_node.get(&node.id);
        let mut summary_with_edges = node.summary.clone();
        if let Some(facts) = node_facts
            && !facts.is_empty()
        {
            let edge_facts = facts
                .iter()
                .filter(|fact| !fact.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            summary_with_edges = format!("{summary_with_edges}\n{edge_facts}")
                .trim()
                .to_owned();
        }

        // Close to the persisted limit: append edge facts directly, no LLM.
        if !summary_with_edges.is_empty()
            && summary_with_edges.chars().count() <= MAX_SUMMARY_CHARS * 2
        {
            node.summary = summary_with_edges;
            continue;
        }
        if summary_with_edges.is_empty() && episodes.is_empty() {
            continue;
        }
        needs_llm.push(idx);
    }

    if needs_llm.is_empty() {
        return Ok(());
    }

    // Entity-type descriptions, extraction-only guidance stripped.
    let mut entity_type_descriptions = serde_json::Map::new();
    for spec in options.entity_types {
        if !spec.description.is_empty() {
            entity_type_descriptions.insert(
                spec.name.clone(),
                json!(truncate_type_description(&spec.description)),
            );
        }
    }

    let episode_content = if episodes.is_empty() {
        String::new()
    } else {
        concatenate_episodes(episodes)
    };

    // Flights of MAX_NODES (upstream runs them concurrently; sequential
    // here — the recordings are identical either way).
    for flight in needs_llm.chunks(MAX_NODES) {
        let entities_context: Vec<Value> = flight
            .iter()
            .map(|&idx| {
                let node = &nodes[idx];
                json!({
                    "name": node.name,
                    "summary": node.summary,
                    "entity_types": node.labels,
                    "attributes": node.attributes,
                })
            })
            .collect();
        let context = json!({
            "entities": entities_context,
            "episode_content": episode_content,
            "previous_episodes": previous_episodes
                .iter()
                .map(|ep| json!({"content": ep.content, "timestamp": ep.valid_at}))
                .collect::<Vec<_>>(),
            "entity_type_descriptions": entity_type_descriptions,
        });
        let messages = if options.skip_fact_appending {
            prompts::extract_entity_summaries_from_episodes(&context)
        } else {
            prompts::extract_summaries_batch(&context)
        };
        let request = CompletionRequest {
            messages,
            schema_name: SummarizedEntities::NAME.to_owned(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        let response = model.complete(&request).await?;
        let summaries: SummarizedEntities =
            crate::model::decode_response(response, SummarizedEntities::NAME)?;

        // Case-insensitive name -> node indices (duplicates all update).
        let mut name_to_nodes: HashMap<String, Vec<usize>> = HashMap::new();
        for &idx in flight {
            name_to_nodes
                .entry(nodes[idx].name.to_lowercase())
                .or_default()
                .push(idx);
        }
        for summarized in summaries.summaries {
            let Some(matching) = name_to_nodes.get(&summarized.name.to_lowercase()) else {
                continue; // unknown entity name: ignored, like upstream
            };
            let truncated = truncate_at_sentence(&summarized.summary, MAX_SUMMARY_CHARS);
            for &idx in matching {
                nodes[idx].summary = truncated.clone();
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::EpisodeSource;
    use crate::model::{Recording, RecordingStore, ReplayModel};

    fn node(id: &str, name: &str, summary: &str) -> SummarizeNode {
        SummarizeNode {
            id: id.into(),
            name: name.into(),
            summary: summary.into(),
            labels: vec!["Entity".into()],
            attributes: serde_json::Map::new(),
        }
    }

    fn episode(content: &str) -> EpisodeInput {
        EpisodeInput {
            name: "ep".into(),
            content: content.into(),
            source: EpisodeSource::Message,
            source_description: "chat".into(),
            group_id: "g".into(),
            valid_at: Some("2026-07-01T00:00:00+00:00".into()),
        }
    }

    struct NoCallModel;
    impl LanguageModel for NoCallModel {
        fn complete(
            &self,
            request: &CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Value, ModelError>> + Send {
            panic!("unexpected model call: {}", request.schema_name);
            #[allow(unreachable_code)]
            async move {
                unreachable!()
            }
        }
    }

    #[test]
    fn truncate_at_sentence_respects_boundaries() {
        assert_eq!(truncate_at_sentence("short.", 100), "short.");
        assert_eq!(
            truncate_at_sentence("One. Two is long. Three continues here.", 20),
            "One. Two is long."
        );
        // No boundary before the limit -> hard cut.
        assert_eq!(truncate_at_sentence("abcdefghij", 5), "abcde");
        // Boundary exactly at end of the truncated window.
        assert_eq!(truncate_at_sentence("Hi. Yes", 3), "Hi.");
    }

    #[test]
    fn type_descriptions_truncate_to_first_paragraph_three_sentences() {
        let doc = "A human being. Known by name. Often a speaker. Fourth sentence dropped.\n\nGOOD: extraction examples stripped.";
        assert_eq!(
            truncate_type_description(doc),
            "A human being. Known by name. Often a speaker."
        );
        // Abbreviations don't split ("e.g." not followed by space+upper).
        assert_eq!(
            truncate_type_description("Things e.g. gadgets and gear."),
            "Things e.g. gadgets and gear."
        );
    }

    #[tokio::test]
    async fn fact_append_shortcut_skips_llm() {
        let mut nodes = vec![node("n1", "Priya", "Priya is a data engineer.")];
        let edges = vec![(
            "n1".to_owned(),
            "n2".to_owned(),
            "Priya works at Northwind Labs.".to_owned(),
        )];
        let edges_by_node = build_edges_by_node(&edges);
        extract_entity_summaries_batch(
            &NoCallModel,
            &mut nodes,
            &[episode("hi")],
            &[],
            &edges_by_node,
            &SummarizeOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            nodes[0].summary,
            "Priya is a data engineer.\nPriya works at Northwind Labs."
        );
    }

    #[tokio::test]
    async fn long_summaries_route_through_llm_and_apply_case_insensitively() {
        let long_summary = "x".repeat(MAX_SUMMARY_CHARS * 2 + 1);
        let mut nodes = vec![
            node("n1", "Priya Raman", &long_summary),
            node("n2", "priya raman", "short but same name"),
        ];
        // n2's short summary takes the shortcut; only n1 needs the LLM.
        let episodes = vec![episode("Priya: news!")];

        let entities_context = vec![json!({
            "name": "Priya Raman",
            "summary": long_summary,
            "entity_types": ["Entity"],
            "attributes": {},
        })];
        let context = json!({
            "entities": entities_context,
            "episode_content": "Priya: news!",
            "previous_episodes": [],
            "entity_type_descriptions": {},
        });
        let request = CompletionRequest {
            messages: prompts::extract_summaries_batch(&context),
            schema_name: "SummarizedEntities".into(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        let response = json!({"summaries": [
            {"name": "PRIYA RAMAN", "summary": "Priya Raman is a senior data engineer. Second sentence."},
            {"name": "Unknown Person", "summary": "ignored"},
        ]});
        let model = ReplayModel::new(RecordingStore::new([Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }]));

        extract_entity_summaries_batch(
            &model,
            &mut nodes,
            &episodes,
            &[],
            &HashMap::new(),
            &SummarizeOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            nodes[0].summary,
            "Priya Raman is a senior data engineer. Second sentence."
        );
        assert_eq!(nodes[1].summary, "short but same name");
    }

    #[tokio::test]
    async fn skip_fact_appending_routes_to_episode_prompt() {
        let mut nodes = vec![node("n1", "Jordan", "Jordan teaches ceramics.")];
        let context = json!({
            "entities": [{
                "name": "Jordan",
                "summary": "Jordan teaches ceramics.",
                "entity_types": ["Entity"],
                "attributes": {},
            }],
            "episode_content": "Jordan: kiln news",
            "previous_episodes": [],
            "entity_type_descriptions": {},
        });
        let request = CompletionRequest {
            messages: prompts::extract_entity_summaries_from_episodes(&context),
            schema_name: "SummarizedEntities".into(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        let model = ReplayModel::new(RecordingStore::new([Recording {
            request: serde_json::to_value(&request).unwrap(),
            response: json!({"summaries": [
                {"name": "Jordan", "summary": "Jordan teaches ceramics at Belmont."},
            ]}),
        }]));

        extract_entity_summaries_batch(
            &model,
            &mut nodes,
            &[episode("Jordan: kiln news")],
            &[],
            &HashMap::new(),
            &SummarizeOptions {
                skip_fact_appending: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(nodes[0].summary, "Jordan teaches ceramics at Belmont.");
    }

    #[tokio::test]
    async fn filter_excludes_nodes_entirely() {
        let mut nodes = vec![node("n1", "Skip Me", "unchanged")];
        let reject_all = |_: &SummarizeNode| false;
        extract_entity_summaries_batch(
            &NoCallModel,
            &mut nodes,
            &[episode("hi")],
            &[],
            &HashMap::new(),
            &SummarizeOptions {
                should_summarize_node: Some(&reject_all),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(nodes[0].summary, "unchanged");
    }
}
