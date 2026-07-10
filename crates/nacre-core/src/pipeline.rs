//! The `add_episode` seam: strings the pipeline steps together and applies
//! their decisions to grit.
//!
//! Ports the orchestration of `graphiti_core/graphiti.py::add_episode`
//! (pinned v0.29.2) onto grit's op vocabulary. Nacre decides; grit
//! executes:
//!
//! - extracted drafts → `AddNode` (kind = the specific label, full label
//!   list mirrored in `attrs["labels"]`)
//! - dedup resolutions → `MergeNodes` (new node folds into the existing)
//! - extracted facts → `AddEdge` (+ `InvalidateEdge` when resolution
//!   already bounded the fact's validity)
//! - invalidation decisions → `InvalidateEdge`
//! - the episode itself → `AddEpisode` last, with mentions of every node
//!   and edge it evidenced
//!
//! Known gaps, pending a grit op-vocabulary decision (see ROADMAP.md):
//! summary refresh and label promotion have no update op to persist
//! through — updated summaries are returned to the caller instead.
//! Candidate gathering also approximates upstream: node candidates come
//! from grit's `find_merge_candidates`, edge invalidation candidates from
//! 1-hop traversal around the endpoints (upstream uses hybrid search).
//! Golden-trace conformance decides whether these matter.

use chrono::{DateTime, Utc};
use grit_core::{GraphOp, Grit, Traversal};
use serde_json::json;
use uuid::Uuid;

use crate::dedupe::{self, EpisodeRef, ExistingEdge, ExistingNode};
use crate::extract::edges::{ExtractEdgesOptions, parse_llm_timestamp};
use crate::extract::nodes::ExtractNodesOptions;
use crate::extract::{self, EpisodeInput, NodeRef};
use crate::model::{LanguageModel, ModelError};

/// Upstream's candidate-pool bound.
// ports: node_operations.py::NODE_DEDUP_CANDIDATE_LIMIT
const NODE_DEDUP_CANDIDATE_LIMIT: usize = 15;
/// Upstream's similarity floor for dedup candidates.
// ports: node_operations.py::NODE_DEDUP_COSINE_MIN_SCORE
const NODE_DEDUP_MIN_SCORE: f64 = 0.6;

/// Pipeline errors: model-side or storage-side.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// LLM/embedding failure.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// grit failure.
    #[error(transparent)]
    Storage(#[from] grit_core::Error),
}

/// Options for [`add_episode`].
#[derive(Debug, Clone, Default)]
pub struct AddEpisodeOptions {
    /// Node extraction options (entity types, exclusions, instructions).
    pub extract_nodes: ExtractNodesOptions,
    /// Edge extraction options (fact types, signature map, instructions).
    pub extract_edges: ExtractEdgesOptions,
}

/// What one `add_episode` run did.
#[derive(Debug, Clone)]
pub struct AddEpisodeOutcome {
    /// The stored episode's id.
    pub episode_id: Uuid,
    /// Final node id per extracted draft (post-merge), in draft order.
    pub node_ids: Vec<Uuid>,
    /// Ids of edges created by this run.
    pub new_edge_ids: Vec<Uuid>,
    /// `(from, into)` node merges executed.
    pub merges: Vec<(Uuid, Uuid)>,
    /// Stored edges invalidated by this run.
    pub invalidated_edge_ids: Vec<Uuid>,
}

fn labels_of(node: &grit_core::Node) -> Vec<String> {
    node.attrs["labels"]
        .as_array()
        .map(|labels| {
            labels
                .iter()
                .filter_map(|l| l.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_else(|| vec![node.kind.clone()])
}

fn existing_node_from(node: &grit_core::Node) -> ExistingNode {
    let attributes = match &node.attrs {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    ExistingNode {
        id: node.id.to_string(),
        name: node.name.clone(),
        labels: labels_of(node),
        summary: node.summary.clone(),
        attributes,
    }
}

fn existing_edge_from(edge: &grit_core::Edge) -> ExistingEdge {
    let ms = |t: i64| DateTime::<Utc>::from_timestamp_millis(t);
    ExistingEdge {
        id: edge.id.to_string(),
        source_id: edge.src.to_string(),
        target_id: edge.dst.to_string(),
        name: edge.rel.clone(),
        fact: edge.fact.clone(),
        // Mentions live in grit's mentions table; edge-level episode lists
        // are only needed for the fast path's "already attributed" check,
        // which mentions_of covers at the call site.
        episodes: Vec::new(),
        valid_at: edge.valid_at.and_then(ms),
        invalid_at: edge.invalid_at.and_then(ms),
        expired_at: edge.expired_at.and_then(ms),
    }
}

/// Ingest one episode: extract nodes, dedup against the graph, extract and
/// resolve edges, apply the resulting ops, and record the episode with its
/// mentions. `now` is injected (grit's clock stays authoritative for
/// system-time columns it stamps itself).
pub async fn add_episode<M: LanguageModel>(
    grit: &Grit,
    model: &M,
    episode: &EpisodeInput,
    previous_episodes: &[EpisodeInput],
    options: &AddEpisodeOptions,
    now: DateTime<Utc>,
) -> Result<AddEpisodeOutcome, PipelineError> {
    let episodes = std::slice::from_ref(episode);

    // 1. Extract entity drafts.
    let drafts =
        extract::nodes::extract_nodes(model, episodes, previous_episodes, &options.extract_nodes)
            .await?;

    // 2. Persist drafts, then gather dedup candidates from grit. Drafts
    //    added in this batch are excluded from each other's pools (upstream
    //    dedups extractions against the EXISTING graph; within-batch dups
    //    were already collapsed at extraction).
    let mut draft_ids: Vec<Uuid> = Vec::with_capacity(drafts.len());
    for draft in &drafts {
        let id = grit.new_id();
        let kind = draft
            .labels
            .iter()
            .find(|l| *l != "Entity")
            .cloned()
            .unwrap_or_else(|| "Entity".to_owned());
        grit.apply(GraphOp::AddNode {
            id,
            kind,
            name: draft.name.clone(),
            summary: draft.summary.clone(),
            attrs: json!({"labels": draft.labels}),
            group_id: draft.group_id.clone(),
        })?;
        draft_ids.push(id);
    }

    let mut candidate_pools: Vec<Vec<ExistingNode>> = Vec::with_capacity(drafts.len());
    for &id in &draft_ids {
        let pool: Vec<ExistingNode> = grit
            .find_merge_candidates(id, NODE_DEDUP_MIN_SCORE)?
            .into_iter()
            .filter(|candidate| !draft_ids.contains(&candidate.node.id))
            .take(NODE_DEDUP_CANDIDATE_LIMIT)
            .map(|candidate| existing_node_from(&candidate.node))
            .collect();
        candidate_pools.push(pool);
    }

    // 3. Dedup judgment; duplicates fold into their existing node.
    let resolutions = dedupe::nodes::resolve_extracted_nodes(
        model,
        &drafts,
        &candidate_pools,
        &options.extract_nodes.entity_types,
        &episode.content,
        previous_episodes,
    )
    .await?;

    let mut merges: Vec<(Uuid, Uuid)> = Vec::new();
    let mut final_ids: Vec<Uuid> = Vec::with_capacity(drafts.len());
    let mut node_refs: Vec<NodeRef> = Vec::with_capacity(drafts.len());
    for ((draft, resolution), &draft_id) in drafts.iter().zip(&resolutions).zip(&draft_ids) {
        match &resolution.duplicate_of {
            Some(existing) => {
                let into: Uuid = existing.id.parse().expect("grit ids round-trip");
                grit.apply(GraphOp::MergeNodes {
                    from: draft_id,
                    into,
                })?;
                merges.push((draft_id, into));
                final_ids.push(into);
                node_refs.push(NodeRef {
                    id: existing.id.clone(),
                    name: existing.name.clone(),
                    labels: existing.labels.clone(),
                });
            }
            None => {
                final_ids.push(draft_id);
                node_refs.push(NodeRef {
                    id: draft_id.to_string(),
                    name: draft.name.clone(),
                    labels: draft.labels.clone(),
                });
            }
        }
    }

    // 4. Extract relationship edges between the resolved nodes.
    let draft_edges = extract::edges::extract_edges(
        model,
        episodes,
        &node_refs,
        previous_episodes,
        &options.extract_edges,
    )
    .await?;

    // 5. Resolve each edge against stored context and apply the decisions.
    let episode_id = grit.new_id();
    let episode_ref = EpisodeRef {
        id: episode_id.to_string(),
        valid_at: episode.valid_at.clone(),
    };
    let mut new_edge_ids: Vec<Uuid> = Vec::new();
    let mut invalidated_edge_ids: Vec<Uuid> = Vec::new();
    let mut mentioned_edges: Vec<Uuid> = Vec::new();

    for draft_edge in &draft_edges {
        let src: Uuid = draft_edge.source_id.parse().expect("grit ids round-trip");
        let dst: Uuid = draft_edge.target_id.parse().expect("grit ids round-trip");

        // Candidates from a 1-hop neighborhood of both endpoints:
        // same-endpoint edges are dedup candidates ("related"), the rest of
        // the neighborhood is the invalidation pool.
        let neighborhood = grit.traverse(&[src, dst], &Traversal::default().depth(1))?;
        let mut related: Vec<ExistingEdge> = Vec::new();
        let mut invalidation_pool: Vec<ExistingEdge> = Vec::new();
        for edge in &neighborhood.edges {
            if new_edge_ids.contains(&edge.id) {
                // Edges added earlier in this same run stay eligible, like
                // upstream's within-episode dedup.
            }
            let mut existing = existing_edge_from(edge);
            existing.episodes = grit
                .mentions_of(edge.id)?
                .iter()
                .map(Uuid::to_string)
                .collect();
            let same_endpoints =
                (edge.src == src && edge.dst == dst) || (edge.src == dst && edge.dst == src);
            if same_endpoints {
                related.push(existing);
            } else {
                invalidation_pool.push(existing);
            }
        }

        let resolution = dedupe::edges::resolve_extracted_edge(
            model,
            draft_edge,
            &related,
            &invalidation_pool,
            &episode_ref,
            now,
        )
        .await?;

        match &resolution.resolved.duplicate_of {
            Some(existing_id) => {
                // Reuse the stored edge; attribution flows through the
                // episode's mentions below.
                if resolution.resolved.append_episode {
                    mentioned_edges.push(existing_id.parse().expect("grit ids round-trip"));
                }
            }
            None => {
                let edge_id = grit.new_id();
                grit.apply(GraphOp::AddEdge {
                    id: edge_id,
                    src,
                    dst,
                    rel: draft_edge.name.clone(),
                    fact: draft_edge.fact.clone(),
                    attrs: json!({}),
                    group_id: draft_edge.group_id.clone(),
                    valid_at: resolution.resolved.valid_at.map(|t| t.timestamp_millis()),
                })?;
                if let Some(invalid_at) = resolution.resolved.invalid_at {
                    grit.apply(GraphOp::InvalidateEdge {
                        edge_id,
                        invalid_at: invalid_at.timestamp_millis(),
                    })?;
                }
                new_edge_ids.push(edge_id);
                mentioned_edges.push(edge_id);
            }
        }

        for invalidation in &resolution.invalidated {
            let edge_id: Uuid = invalidation.id.parse().expect("grit ids round-trip");
            grit.apply(GraphOp::InvalidateEdge {
                edge_id,
                invalid_at: invalidation.invalid_at.timestamp_millis(),
            })?;
            invalidated_edge_ids.push(edge_id);
        }
    }

    // 6. The episode itself, with provenance for everything it evidenced.
    let occurred_at = episode
        .valid_at
        .as_deref()
        .and_then(parse_llm_timestamp)
        .map_or_else(|| now.timestamp_millis(), |t| t.timestamp_millis());
    let mut mentions: Vec<Uuid> = final_ids.clone();
    mentions.extend(&mentioned_edges);
    mentions.dedup();
    grit.apply(GraphOp::AddEpisode {
        id: episode_id,
        source: episode.source_description.clone(),
        content: episode.content.clone(),
        occurred_at,
        group_id: episode.group_id.clone(),
        mentions,
    })?;

    Ok(AddEpisodeOutcome {
        episode_id,
        node_ids: final_ids,
        new_edge_ids,
        merges,
        invalidated_edge_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{EntityTypeSpec, EpisodeSource};
    use crate::model::{CompletionRequest, ModelSize, Recording, RecordingStore, ReplayModel};
    use crate::schemas::ResponseSchema;
    use chrono::TimeZone;
    use grit_core::Options;
    use serde_json::{Value, json};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 10, 12, 0, 0).unwrap()
    }

    fn episode(name: &str, content: &str, valid_at: &str) -> EpisodeInput {
        EpisodeInput {
            name: name.into(),
            content: content.into(),
            source: EpisodeSource::Message,
            source_description: "chat between friends".into(),
            group_id: "trace-test".into(),
            valid_at: Some(valid_at.into()),
        }
    }

    fn options() -> AddEpisodeOptions {
        AddEpisodeOptions {
            extract_nodes: ExtractNodesOptions {
                entity_types: vec![EntityTypeSpec {
                    name: "Person".into(),
                    description: "A human being mentioned by name.".into(),
                }],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn extraction_recording(
        episodes: &[EpisodeInput],
        previous: &[EpisodeInput],
        opts: &AddEpisodeOptions,
        response: Value,
    ) -> Recording {
        let context = crate::extract::nodes::build_extraction_context(
            episodes,
            previous,
            &opts.extract_nodes,
        );
        let request = CompletionRequest {
            messages: crate::prompts::extract_nodes::extract_message(&context),
            schema_name: crate::schemas::ExtractedEntities::NAME.into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }
    }

    fn edge_extraction_recording(
        episodes: &[EpisodeInput],
        nodes: &[NodeRef],
        previous: &[EpisodeInput],
        opts: &AddEpisodeOptions,
        response: Value,
    ) -> Recording {
        let context = crate::extract::edges::build_edge_extraction_context(
            episodes,
            nodes,
            previous,
            &opts.extract_edges,
        );
        let request = CompletionRequest {
            messages: crate::prompts::extract_edges::edge(&context),
            schema_name: crate::schemas::ExtractedEdges::NAME.into(),
            max_tokens: Some(16384),
            model_size: ModelSize::Medium,
        };
        Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }
    }

    fn resolve_edge_recording(
        facts: &[(usize, &str)],
        new_fact: &str,
        response: Value,
    ) -> Recording {
        let related: Vec<Value> = facts
            .iter()
            .map(|(idx, fact)| json!({"idx": idx, "fact": fact}))
            .collect();
        let context = json!({
            "existing_edges": related,
            "new_edge": new_fact,
            "edge_invalidation_candidates": [],
        });
        let request = CompletionRequest {
            messages: crate::prompts::dedupe_edges::resolve_edge(&context),
            schema_name: crate::schemas::EdgeDuplicate::NAME.into(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }
    }

    /// End-to-end against a real grit database: episode 1 builds the graph,
    /// episode 2 dedups deterministically against it (MergeNodes) and its
    /// contradicting fact invalidates the stored edge.
    #[tokio::test]
    async fn two_episode_flow_merges_and_invalidates() {
        let dir = std::env::temp_dir().join(format!("nacre-pipeline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pipeline-test.db");
        let _ = std::fs::remove_file(&path);
        let grit = Grit::open(&path, Options::new("nacre-test")).unwrap();
        let opts = options();

        // --- Episode 1: Priya joined Northwind Labs. ---
        let ep1 = episode(
            "ep-0",
            "Priya: I joined Northwind Labs in January as a data engineer.",
            "2026-03-02T18:30:00+00:00",
        );
        let ep1_nodes = json!({"extracted_entities": [
            {"name": "Priya", "entity_type_id": 1, "episode_indices": [0]},
            {"name": "Northwind Labs", "entity_type_id": 0, "episode_indices": [0]},
        ]});
        // Node refs the pipeline will build for ep1 (both new).
        let ep1_refs = |ids: &[Uuid]| {
            vec![
                NodeRef {
                    id: ids[0].to_string(),
                    name: "Priya".into(),
                    labels: vec!["Entity".into(), "Person".into()],
                },
                NodeRef {
                    id: ids[1].to_string(),
                    name: "Northwind Labs".into(),
                    labels: vec!["Entity".into()],
                },
            ]
        };
        // Edge extraction depends on runtime ids only through NodeRef.id,
        // which never enters the prompt — safe to prebuild with dummy ids.
        let dummy = [Uuid::nil(), Uuid::nil()];
        let ep1_edge_response = json!({"edges": [{
            "source_entity_name": "Priya",
            "target_entity_name": "Northwind Labs",
            "relation_type": "WORKS_AT",
            "fact": "Priya joined Northwind Labs in January as a data engineer.",
            "valid_at": "2026-01-15T00:00:00Z",
            "episode_indices": [0],
        }]});

        let model = ReplayModel::new(RecordingStore::new([
            extraction_recording(std::slice::from_ref(&ep1), &[], &opts, ep1_nodes),
            edge_extraction_recording(
                std::slice::from_ref(&ep1),
                &ep1_refs(&dummy),
                &[],
                &opts,
                ep1_edge_response,
            ),
        ]));
        let out1 = add_episode(&grit, &model, &ep1, &[], &opts, now())
            .await
            .unwrap();
        assert_eq!(out1.node_ids.len(), 2);
        assert_eq!(out1.new_edge_ids.len(), 1);
        assert!(out1.merges.is_empty());
        let works_at_id = out1.new_edge_ids[0];
        let stored = grit.edge(works_at_id).unwrap().unwrap();
        assert_eq!(stored.rel, "WORKS_AT");
        assert_eq!(
            stored.valid_at,
            Some(ts_ms("2026-01-15T00:00:00+00:00")),
            "valid_at survives into grit"
        );

        // --- Episode 2: Priya left. Dedup + invalidation. ---
        let ep2 = episode(
            "ep-1",
            "Priya: Big news — I left Northwind Labs last Friday.",
            "2026-06-08T20:45:00+00:00",
        );
        let previous = vec![ep1.clone()];
        let ep2_nodes = json!({"extracted_entities": [
            {"name": "Priya", "entity_type_id": 1, "episode_indices": [0]},
            {"name": "Northwind Labs", "entity_type_id": 0, "episode_indices": [0]},
        ]});
        // Post-dedup refs reuse the ep1 node names/labels (existing nodes).
        let resolved_refs = vec![
            NodeRef {
                id: out1.node_ids[0].to_string(),
                name: "Priya".into(),
                labels: vec!["Entity".into(), "Person".into()],
            },
            NodeRef {
                id: out1.node_ids[1].to_string(),
                name: "Northwind Labs".into(),
                labels: vec!["Entity".into()],
            },
        ];
        let left_fact = "Priya left Northwind Labs on the Friday before 2026-06-08.";
        let ep2_edge_response = json!({"edges": [{
            "source_entity_name": "Priya",
            "target_entity_name": "Northwind Labs",
            "relation_type": "NO_LONGER_WORKS_AT",
            "fact": left_fact,
            "valid_at": "2026-06-05T00:00:00Z",
            "episode_indices": [0],
        }]});

        let model = ReplayModel::new(RecordingStore::new([
            extraction_recording(std::slice::from_ref(&ep2), &previous, &opts, ep2_nodes),
            edge_extraction_recording(
                std::slice::from_ref(&ep2),
                &resolved_refs,
                &previous,
                &opts,
                ep2_edge_response,
            ),
            resolve_edge_recording(
                &[(
                    0,
                    "Priya joined Northwind Labs in January as a data engineer.",
                )],
                left_fact,
                json!({"duplicate_facts": [], "contradicted_facts": [0]}),
            ),
        ]));
        let out2 = add_episode(&grit, &model, &ep2, &previous, &opts, now())
            .await
            .unwrap();

        // Both extractions deduped deterministically onto the ep1 nodes.
        assert_eq!(out2.merges.len(), 2);
        assert_eq!(out2.node_ids, out1.node_ids);
        // Merged-away drafts are expired with an audit pointer.
        let merged = grit.node(out2.merges[0].0).unwrap().unwrap();
        assert_eq!(merged.merged_into, Some(out2.merges[0].1));

        // The January employment edge is invalidated at the June valid_at.
        assert_eq!(out2.invalidated_edge_ids, vec![works_at_id]);
        let old_edge = grit.edge(works_at_id).unwrap().unwrap();
        assert_eq!(
            old_edge.invalid_at,
            Some(ts_ms("2026-06-05T00:00:00+00:00"))
        );

        // The new fact landed as its own edge with June validity.
        assert_eq!(out2.new_edge_ids.len(), 1);
        let new_edge = grit.edge(out2.new_edge_ids[0]).unwrap().unwrap();
        assert_eq!(new_edge.rel, "NO_LONGER_WORKS_AT");
        assert_eq!(new_edge.valid_at, Some(ts_ms("2026-06-05T00:00:00+00:00")));

        // Episode mentions cover the nodes and the new edge.
        let mentions = grit.mentions_of(out2.node_ids[0]).unwrap();
        assert!(mentions.contains(&out2.episode_id));

        drop(grit);
        let _ = std::fs::remove_file(&path);
    }

    fn ts_ms(iso: &str) -> i64 {
        DateTime::parse_from_rfc3339(iso)
            .unwrap()
            .timestamp_millis()
    }
}
