//! Node dedup resolution: deterministic pass, then batched LLM escalation.
//!
//! Ports `resolve_extracted_nodes` / `_resolve_with_similarity` /
//! `_resolve_with_llm` from
//! `graphiti_core/utils/maintenance/node_operations.py` (pinned v0.29.2).

use serde_json::{Value, json};

use super::helpers::{
    CandidateIndexes, FUZZY_JACCARD_THRESHOLD, build_candidate_indexes, has_high_entropy,
    jaccard_similarity, lsh_bands, minhash_signature, normalize_name_for_fuzzy,
    normalize_string_exact, shingles,
};
use super::{ExistingNode, NodeResolution};
use crate::extract::{DraftNode, EntityTypeSpec, EpisodeInput};
use crate::model::{CompletionRequest, LanguageModel, ModelError, ModelSize};
use crate::prompts::dedupe_nodes as prompts;
use crate::schemas::{NodeResolutions, ResponseSchema};

/// Upgrade a generic canonical node when the duplicate carries a specific
/// type: the promoted label list is `Entity`, the existing labels, then the
/// draft's specific labels, first-seen order, deduplicated.
// ports: dedup_helpers.py::_promote_resolved_node
fn promote_resolved_node(extracted: &DraftNode, mut resolved: ExistingNode) -> ExistingNode {
    let resolved_specific = resolved.labels.iter().any(|l| l != "Entity");
    if resolved_specific {
        return resolved;
    }
    let extracted_specific: Vec<&String> =
        extracted.labels.iter().filter(|l| *l != "Entity").collect();
    if extracted_specific.is_empty() {
        return resolved;
    }
    let mut promoted: Vec<String> = Vec::new();
    for label in std::iter::once("Entity")
        .chain(resolved.labels.iter().map(String::as_str))
        .chain(extracted_specific.iter().map(|s| s.as_str()))
    {
        if !promoted.iter().any(|existing| existing == label) {
            promoted.push(label.to_owned());
        }
    }
    resolved.labels = promoted;
    resolved
}

/// Deterministic resolution for ONE draft against prebuilt indexes:
/// exact normalized-name hit (always attempted; ambiguity escalates),
/// then entropy-gated MinHash/LSH fuzzy matching.
// ports: dedup_helpers.py::_resolve_with_similarity (single-node use)
fn resolve_with_similarity(draft: &DraftNode, indexes: &CandidateIndexes) -> Option<ExistingNode> {
    let normalized_exact = normalize_string_exact(&draft.name);
    let normalized_fuzzy = normalize_name_for_fuzzy(&draft.name);

    if let Some(matches) = indexes.by_normalized_name.get(&normalized_exact) {
        if matches.len() == 1 {
            let candidate = indexes.existing_nodes[matches[0]].clone();
            return Some(promote_resolved_node(draft, candidate));
        }
        if matches.len() > 1 {
            // Ambiguous: escalate to the LLM.
            return None;
        }
    }

    if !has_high_entropy(&normalized_fuzzy) {
        return None;
    }

    let draft_shingles = shingles(&normalized_fuzzy);
    let signature = minhash_signature(&draft_shingles);
    let mut candidate_ids: Vec<usize> = Vec::new();
    for (band_index, band) in lsh_bands(&signature).into_iter().enumerate() {
        if let Some(bucket) = indexes.lsh_buckets.get(&(band_index, band)) {
            for &candidate in bucket {
                if !candidate_ids.contains(&candidate) {
                    candidate_ids.push(candidate);
                }
            }
        }
    }

    let mut best: Option<usize> = None;
    let mut best_score = 0.0;
    for candidate in candidate_ids {
        let score = jaccard_similarity(&draft_shingles, &indexes.shingles_by_candidate[candidate]);
        if score > best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    if let Some(candidate) = best
        && best_score >= FUZZY_JACCARD_THRESHOLD
    {
        let node = indexes.existing_nodes[candidate].clone();
        return Some(promote_resolved_node(draft, node));
    }
    None
}

/// First non-`Entity` label's description from the caller's type specs.
// ports: node_operations.py::_get_entity_type_description
fn entity_type_description(labels: &[String], entity_types: &[EntityTypeSpec]) -> String {
    let type_name = labels.iter().find(|l| *l != "Entity");
    type_name
        .and_then(|name| entity_types.iter().find(|spec| &spec.name == name))
        .map(|spec| spec.description.clone())
        .unwrap_or_else(|| "Default Entity Type".to_owned())
}

/// Merge per-draft candidate lists into one pool, preserving first-seen
/// order and dropping duplicate ids.
// ports: node_operations.py::_merge_candidate_nodes
fn merge_candidate_pools(pools: &[&[ExistingNode]]) -> Vec<ExistingNode> {
    let mut seen: Vec<&str> = Vec::new();
    let mut merged = Vec::new();
    for pool in pools {
        for candidate in *pool {
            if !seen.contains(&candidate.id.as_str()) {
                seen.push(&candidate.id);
                merged.push(candidate.clone());
            }
        }
    }
    merged
}

/// Build the batched dedup prompt context. Public so replay tests can
/// construct the exact request the step will issue.
// ports: node_operations.py::_resolve_with_llm (context construction)
pub fn build_dedupe_context(
    unresolved_drafts: &[&DraftNode],
    candidates: &[ExistingNode],
    entity_types: &[EntityTypeSpec],
    episode_content: &str,
    previous_episodes: &[EpisodeInput],
) -> Value {
    let extracted_nodes: Vec<Value> = unresolved_drafts
        .iter()
        .enumerate()
        .map(|(i, draft)| {
            json!({
                "id": i,
                "name": draft.name,
                "entity_type": draft.labels,
                "entity_type_description": entity_type_description(&draft.labels, entity_types),
            })
        })
        .collect();
    let existing_nodes: Vec<Value> = candidates
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            let mut entry = candidate.attributes.clone();
            entry.insert("candidate_id".into(), json!(i));
            entry.insert("name".into(), json!(candidate.name));
            entry.insert("entity_types".into(), json!(candidate.labels));
            let summary: String = candidate.summary.chars().take(120).collect();
            entry.insert("summary".into(), json!(summary));
            Value::Object(entry)
        })
        .collect();
    json!({
        "extracted_nodes": extracted_nodes,
        "existing_nodes": existing_nodes,
        "episode_content": episode_content,
        "previous_episodes": previous_episodes
            .iter()
            .map(|ep| json!({"content": ep.content, "timestamp": ep.valid_at}))
            .collect::<Vec<_>>(),
    })
}

/// Resolve extracted drafts against their candidate pools: deterministic
/// exact/fuzzy resolution per draft, then one batched LLM call for the
/// rest. `candidates[i]` is the pool for `drafts[i]` (from grit's
/// `find_merge_candidates`, gathered by the pipeline seam).
///
/// Returns one [`NodeResolution`] per draft, in input order. Upstream's
/// guardrails are preserved: out-of-range or repeated LLM ids are ignored,
/// invalid candidate ids and LLM omissions resolve to "new node".
///
/// # Panics
///
/// Panics if `candidates.len() != drafts.len()`.
// ports: node_operations.py::resolve_extracted_nodes + _resolve_with_llm
pub async fn resolve_extracted_nodes<M: LanguageModel>(
    model: &M,
    drafts: &[DraftNode],
    candidates: &[Vec<ExistingNode>],
    entity_types: &[EntityTypeSpec],
    episode_content: &str,
    previous_episodes: &[EpisodeInput],
) -> Result<Vec<NodeResolution>, ModelError> {
    assert_eq!(
        drafts.len(),
        candidates.len(),
        "one candidate pool per draft"
    );

    let mut resolutions: Vec<Option<NodeResolution>> = vec![None; drafts.len()];
    let mut unresolved: Vec<usize> = Vec::new();

    for (idx, (draft, pool)) in drafts.iter().zip(candidates).enumerate() {
        if pool.is_empty() {
            // No candidates: stands as new without LLM involvement.
            resolutions[idx] = Some(NodeResolution { duplicate_of: None });
            continue;
        }
        let indexes = build_candidate_indexes(pool.clone());
        match resolve_with_similarity(draft, &indexes) {
            Some(existing) => {
                resolutions[idx] = Some(NodeResolution {
                    duplicate_of: Some(existing),
                });
            }
            None => unresolved.push(idx),
        }
    }

    if !unresolved.is_empty() {
        let pools: Vec<&[ExistingNode]> = unresolved
            .iter()
            .map(|&idx| candidates[idx].as_slice())
            .collect();
        let pool = merge_candidate_pools(&pools);
        let unresolved_drafts: Vec<&DraftNode> = unresolved.iter().map(|&i| &drafts[i]).collect();
        let context = build_dedupe_context(
            &unresolved_drafts,
            &pool,
            entity_types,
            episode_content,
            previous_episodes,
        );
        let request = CompletionRequest {
            messages: prompts::nodes(&context),
            schema_name: NodeResolutions::NAME.to_owned(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        let response = model.complete(&request).await?;
        let parsed: NodeResolutions =
            crate::model::decode_response(response, NodeResolutions::NAME)?;

        let mut processed: Vec<i64> = Vec::new();
        for resolution in parsed.entity_resolutions {
            let relative_id = resolution.id;
            let Ok(relative) = usize::try_from(relative_id) else {
                continue; // negative id: out of range, ignored
            };
            if relative >= unresolved.len() || processed.contains(&relative_id) {
                continue; // out-of-range or duplicate id, ignored
            }
            processed.push(relative_id);
            let original_index = unresolved[relative];
            let draft = &drafts[original_index];

            let duplicate_of = usize::try_from(resolution.duplicate_candidate_id)
                .ok()
                .and_then(|cid| pool.get(cid))
                .map(|candidate| promote_resolved_node(draft, candidate.clone()));
            resolutions[original_index] = Some(NodeResolution { duplicate_of });
        }
    }

    // Anything still unresolved (LLM omissions) stands as new.
    Ok(resolutions
        .into_iter()
        .map(|r| r.unwrap_or(NodeResolution { duplicate_of: None }))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::EpisodeSource;
    use crate::model::{Recording, RecordingStore, ReplayModel};
    use serde_json::json;

    fn draft(name: &str, labels: &[&str]) -> DraftNode {
        DraftNode {
            name: name.into(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            summary: String::new(),
            group_id: "g".into(),
            episode_indices: vec![0],
        }
    }

    fn existing(id: &str, name: &str, labels: &[&str]) -> ExistingNode {
        ExistingNode {
            id: id.into(),
            name: name.into(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            summary: String::new(),
            attributes: serde_json::Map::new(),
        }
    }

    fn previous() -> Vec<EpisodeInput> {
        vec![EpisodeInput {
            name: "prev".into(),
            content: "Mina: earlier".into(),
            source: EpisodeSource::Message,
            source_description: "chat".into(),
            group_id: "g".into(),
            valid_at: Some("2026-01-01T00:00:00+00:00".into()),
        }]
    }

    /// A model that panics if called — for paths that must resolve
    /// deterministically.
    struct NoCallModel;
    impl LanguageModel for NoCallModel {
        fn complete(
            &self,
            request: &CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Value, ModelError>> + Send {
            panic!(
                "deterministic path must not call the model: {}",
                request.schema_name
            );
            #[allow(unreachable_code)]
            async move {
                unreachable!()
            }
        }
    }

    #[tokio::test]
    async fn exact_match_resolves_without_llm_and_promotes_labels() {
        let drafts = vec![draft("Priya Raman", &["Entity", "Person"])];
        let candidates = vec![vec![existing("id-1", "priya  raman", &["Entity"])]];
        let out = resolve_extracted_nodes(&NoCallModel, &drafts, &candidates, &[], "", &previous())
            .await
            .unwrap();
        let resolved = out[0].duplicate_of.as_ref().unwrap();
        assert_eq!(resolved.id, "id-1");
        // Generic existing node inherits the draft's specific type.
        assert_eq!(resolved.labels, vec!["Entity", "Person"]);
    }

    #[tokio::test]
    async fn fuzzy_match_resolves_when_only_punctuation_differs() {
        let drafts = vec![draft("Priya-Raman Engineer", &["Entity"])];
        let candidates = vec![vec![
            existing("id-1", "Priya Raman Engineer", &["Entity"]),
            existing("id-2", "Belmont Arts Center", &["Entity"]),
        ]];
        let out = resolve_extracted_nodes(&NoCallModel, &drafts, &candidates, &[], "", &[])
            .await
            .unwrap();
        assert_eq!(out[0].duplicate_of.as_ref().unwrap().id, "id-1");
    }

    #[tokio::test]
    async fn empty_pool_stands_new_without_llm() {
        let drafts = vec![draft("Trigger", &["Entity"])];
        let out = resolve_extracted_nodes(&NoCallModel, &drafts, &[vec![]], &[], "", &[])
            .await
            .unwrap();
        assert!(out[0].duplicate_of.is_none());
    }

    #[tokio::test]
    async fn llm_escalation_applies_guardrails() {
        // "NYC" is short/low-entropy -> escalates; so does the ambiguous
        // exact match ("Java" twice in the pool).
        let drafts = vec![draft("NYC", &["Entity"]), draft("Java", &["Entity"])];
        let nyc_pool = vec![
            existing("id-nyc", "New York City", &["Entity"]),
            existing("id-knicks", "New York Knicks", &["Entity"]),
        ];
        let java_pool = vec![
            existing("id-java1", "Java", &["Entity"]),
            existing("id-java2", "java", &["Entity"]),
        ];
        let candidates = vec![nyc_pool.clone(), java_pool.clone()];

        let entity_types = vec![EntityTypeSpec {
            name: "Person".into(),
            description: "A human being.".into(),
        }];
        let unresolved: Vec<&DraftNode> = vec![&drafts[0], &drafts[1]];
        let pools: Vec<&[ExistingNode]> = vec![&nyc_pool, &java_pool];
        let pool = merge_candidate_pools(&pools);
        let context = build_dedupe_context(&unresolved, &pool, &entity_types, "Marco: NYC!", &[]);
        let request = CompletionRequest {
            messages: prompts::nodes(&context),
            schema_name: "NodeResolutions".into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        let response = json!({"entity_resolutions": [
            {"id": 0, "name": "New York City", "duplicate_candidate_id": 0},
            {"id": 0, "name": "ignored duplicate", "duplicate_candidate_id": 1},
            {"id": 99, "name": "ignored out of range", "duplicate_candidate_id": 0},
            // id 1 deliberately gets an invalid candidate id -> stands new.
            {"id": 1, "name": "Java", "duplicate_candidate_id": 42},
        ]});
        let model = ReplayModel::new(RecordingStore::new([Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }]));

        let out = resolve_extracted_nodes(
            &model,
            &drafts,
            &candidates,
            &entity_types,
            "Marco: NYC!",
            &[],
        )
        .await
        .unwrap();
        assert_eq!(out[0].duplicate_of.as_ref().unwrap().id, "id-nyc");
        assert!(out[1].duplicate_of.is_none(), "invalid candidate id -> new");
    }

    #[tokio::test]
    async fn llm_omission_falls_back_to_new() {
        let drafts = vec![draft("NYC", &["Entity"])];
        let pool = vec![existing("id-nyc", "New York City", &["Entity"])];
        let unresolved: Vec<&DraftNode> = vec![&drafts[0]];
        let pools: Vec<&[ExistingNode]> = vec![&pool];
        let merged = merge_candidate_pools(&pools);
        let context = build_dedupe_context(&unresolved, &merged, &[], "", &[]);
        let request = CompletionRequest {
            messages: prompts::nodes(&context),
            schema_name: "NodeResolutions".into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        let model = ReplayModel::new(RecordingStore::new([Recording {
            request: serde_json::to_value(&request).unwrap(),
            response: json!({"entity_resolutions": []}),
        }]));

        let out = resolve_extracted_nodes(&model, &drafts, &[pool], &[], "", &[])
            .await
            .unwrap();
        assert!(out[0].duplicate_of.is_none());
    }
}
