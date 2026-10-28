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
//! Summary refresh and label promotion persist through grit's `UpdateNode`
//! (per-field LWW; grit ≥0.2) — the pipeline diffs refreshed metadata
//! against storage and emits ops only for real changes.
//! Candidate gathering is engine-free on both sides of the oracle (the
//! capture harness patches upstream identically — see DEVIATIONS.md):
//! node dedup candidates come from pinned-arithmetic cosine ranking of the
//! group's pre-episode nodes; edge dedup/invalidation pools are the
//! group's pre-episode edges, split same-pair vs rest and sorted by fact.
//!
//! **Ingestion is two-phase**: a planning phase does all model I/O and
//! decides the whole write set without touching storage, then one
//! `apply_batch` commits it. An episode's graph mutations and the episode
//! row that evidences them therefore land together or not at all — a
//! failed call leaves the graph exactly as it was, which is what makes a
//! retry safe (see [`EpisodeInput::id`]). See DEVIATIONS.md "Atomic
//! episode commit".

use chrono::{DateTime, Utc};
use grit_core::{GraphOp, Grit, WriteItem};
use serde_json::json;
use uuid::Uuid;

use crate::dedupe::{self, EpisodeRef, ExistingEdge, ExistingNode};
use crate::extract::edges::{ExtractEdgesOptions, parse_llm_timestamp};
use crate::extract::nodes::ExtractNodesOptions;
use crate::extract::{self, EpisodeInput, NodeRef};
use crate::model::{Embedder, LanguageModel, ModelError};
use crate::summarize::{self, SummarizeNode};

/// Upstream's candidate-pool bound.
// ports: node_operations.py::NODE_DEDUP_CANDIDATE_LIMIT
const NODE_DEDUP_CANDIDATE_LIMIT: usize = 15;
/// Upstream's similarity floor for dedup candidates.
// ports: node_operations.py::NODE_DEDUP_COSINE_MIN_SCORE
const NODE_DEDUP_MIN_SCORE: f64 = 0.6;

/// Previous-episode window size: upstream's add_episode fetches this many
/// prior episodes as extraction context (NOT the 3-episode
/// EPISODE_WINDOW_LEN default — verified against trace1's recorded prompts).
// ports: graphiti.py::add_episode (retrieve_episodes last_n=RELEVANT_SCHEMA_LIMIT)
pub const PREVIOUS_EPISODE_WINDOW: usize = 10;

/// Pipeline errors: model-side or storage-side.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// LLM/embedding failure.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// grit failure.
    #[error(transparent)]
    Storage(#[from] grit_core::Error),
    /// The caller-supplied [`EpisodeInput::id`] is already stored: this
    /// episode was ingested before. Reported instead of silently doing
    /// nothing — only the caller knows whether a second success is
    /// expected (a retry landing after all) or a bug in its own bookkeeping.
    #[error("episode {0} already exists")]
    EpisodeExists(Uuid),
    /// The episode asked for is not stored: nothing was deleted. The mirror
    /// of [`PipelineError::EpisodeExists`], reported for the same reason —
    /// a caller retrying a delete that already succeeded sees this, and only
    /// the caller knows whether that is the expected second attempt or a bug
    /// in its own bookkeeping.
    #[error("episode {0} not found")]
    EpisodeNotFound(Uuid),
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

/// The episode's write set while it is being decided: the batch that will
/// be committed, plus the node state it implies.
///
/// Planning reads back some of what it has just decided (a canonical node's
/// promoted labels, a draft node's name and summary). Committed storage
/// cannot answer those reads — nothing is written until the batch lands —
/// and grit is deliberately not asked to model an open transaction (nacre
/// decides, grit executes). So the overlay models it here: the node fields
/// the pipeline reads mid-flight, folded forward as ops are pushed, giving
/// exactly the answers the old op-at-a-time pipeline got from storage.
///
/// Only nodes are overlaid: edges, invalidations and the episode row are
/// never read back within the episode that writes them.
struct Pending {
    /// The batch, in the order the pipeline decided it.
    items: Vec<WriteItem>,
    /// Node rows as they will read once the batch commits.
    nodes: std::collections::HashMap<Uuid, grit_core::Node>,
    /// Injected time for synthetic rows; grit stamps its own at commit and
    /// no mid-flight read looks at timestamps.
    now_ms: i64,
}

impl Pending {
    fn new(now: DateTime<Utc>) -> Self {
        Self {
            items: Vec::new(),
            nodes: std::collections::HashMap::new(),
            now_ms: now.timestamp_millis(),
        }
    }

    /// Queue an op and fold its effect into the node overlay.
    fn push(&mut self, grit: &Grit, op: GraphOp) -> Result<(), PipelineError> {
        match &op {
            GraphOp::AddNode {
                id,
                kind,
                name,
                summary,
                attrs,
                group_id,
            } => {
                self.nodes.insert(
                    *id,
                    grit_core::Node {
                        id: *id,
                        kind: kind.clone(),
                        name: name.clone(),
                        summary: summary.clone(),
                        attrs: attrs.clone(),
                        group_id: group_id.clone(),
                        created_at: self.now_ms,
                        expired_at: None,
                        merged_into: None,
                    },
                );
            }
            GraphOp::MergeNodes { from, .. } => {
                // grit expires the absorbed row; `into` keeps whatever the
                // overlay (or storage) already says about it.
                self.nodes.remove(from);
            }
            GraphOp::UpdateNode {
                id,
                name,
                summary,
                kind,
                attrs,
            } => {
                // Per-field LWW, as grit folds it. Seed from storage when
                // the target is a committed node.
                if !self.nodes.contains_key(id)
                    && let Some(node) = grit.node(*id)?
                {
                    self.nodes.insert(*id, node);
                }
                if let Some(node) = self.nodes.get_mut(id) {
                    if let Some(name) = name {
                        node.name = name.clone();
                    }
                    if let Some(summary) = summary {
                        node.summary = summary.clone();
                    }
                    if let Some(kind) = kind {
                        node.kind = kind.clone();
                    }
                    if let Some(attrs) = attrs {
                        node.attrs = attrs.clone();
                    }
                }
            }
            // Never read back within this episode.
            GraphOp::AddEdge { .. }
            | GraphOp::InvalidateEdge { .. }
            | GraphOp::AddEpisode { .. } => {}
            // The purge family belongs to `forget`; an ingest never emits it.
            GraphOp::Purge { .. } | GraphOp::PurgeNodeFields { .. } => {}
        }
        self.items.push(WriteItem::Op(op));
        Ok(())
    }

    /// Read a node as it will be once the batch commits: the overlay when
    /// this episode has already decided something about it, storage
    /// otherwise.
    fn node(&self, grit: &Grit, id: Uuid) -> Result<Option<grit_core::Node>, PipelineError> {
        match self.nodes.get(&id) {
            Some(node) => Ok(Some(node.clone())),
            None => Ok(grit.node(id)?),
        }
    }

    /// Queue a node's name vector. Embeddings are recomputable local state,
    /// not oplog content — they ride in the batch only so they commit with
    /// the row they describe.
    fn push_node_embedding(&mut self, id: Uuid, vector: Vec<f32>) {
        self.items.push(WriteItem::NodeEmbedding { id, vector });
    }

    /// Queue an edge's fact vector (see [`Pending::push_node_embedding`]).
    fn push_edge_embedding(&mut self, id: Uuid, vector: Vec<f32>) {
        self.items.push(WriteItem::EdgeEmbedding { id, vector });
    }
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
    let mut attributes = match &node.attrs {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    // "labels" is nacre's storage convention inside grit attrs, not a
    // Graphiti node attribute — upstream candidates hydrated via
    // get_by_group_ids carry no labels key, and the dedupe context spreads
    // attributes verbatim into the prompt.
    attributes.remove("labels");
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

/// Live nodes in a group (grit's id-ordered scan, expired rows dropped):
/// upstream never persists merged-away drafts, so its candidate pools
/// cannot contain them — grit keeps the expired draft rows for audit, and
/// two same-name rows would wrongly turn exact-match resolutions ambiguous.
fn group_nodes_snapshot(
    grit: &Grit,
    group_id: &str,
) -> Result<Vec<grit_core::Node>, PipelineError> {
    Ok(grit
        .nodes_in_group(group_id)?
        .into_iter()
        .filter(|node| node.expired_at.is_none())
        .collect())
}

/// The previous-episode context window for the next `add_episode` call:
/// the last `last_n` stored episodes **of the given source kind** with
/// `occurred_at <= reference`, in chronological order — upstream's
/// `retrieve_episodes` (filter inclusive, `e.source = $source`,
/// ORDER BY valid_at DESC LIMIT n, reversed). `add_episode` passes the
/// CURRENT episode's source, so a text chunk only sees prior text chunks
/// in its prompt window. Timestamps render into prompts at second
/// precision, matching Python's `isoformat()` for the whole-second times
/// episodes carry.
/// Episodes written by grit < 0.2.2 carry an empty stored kind and match
/// no filter — re-ingest or accept the empty window for those.
// ports: graphiti_core/utils/maintenance/graph_data_operations.py::retrieve_episodes
// (with graphiti.py::add_episode's `source=source` argument)
pub fn retrieve_previous_episodes(
    grit: &Grit,
    group_id: &str,
    source: crate::extract::EpisodeSource,
    reference: DateTime<Utc>,
    last_n: usize,
) -> Result<Vec<EpisodeInput>, PipelineError> {
    let mut episodes = grit.episodes_in_group(group_id)?; // chronological
    episodes.retain(|e| e.kind == source.as_str() && e.occurred_at <= reference.timestamp_millis());
    let window_start = episodes.len().saturating_sub(last_n);
    Ok(episodes[window_start..]
        .iter()
        .map(|e| EpisodeInput {
            // These rows describe episodes already stored, so their
            // identity is known (prompt context reads only content and
            // timestamp; `name` is not stored at all, hence empty).
            id: Some(e.id),
            name: String::new(),
            content: e.content.clone(),
            source,
            source_description: e.source.clone(),
            group_id: group_id.to_owned(),
            valid_at: DateTime::<Utc>::from_timestamp_millis(e.occurred_at)
                .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, false)),
            // Carried verbatim: what grit stored is what a caller reading
            // this window back gets. Safe to carry precisely because no
            // context builder reads it — the exclusion is enforced at the
            // prompt boundary, not by dropping the value here.
            metadata: e.metadata.clone(),
        })
        .collect())
}

/// Cosine similarity with pinned arithmetic: components are f32 (both
/// sides of the oracle store vectors at f32), accumulation is sequential
/// f64 — byte-identical to the capture harness's Python loop, so scores
/// (and therefore candidate ranking) agree exactly across the oracle.
fn cosine_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Ingest one episode: extract nodes, dedup against the graph, extract and
/// resolve edges, and record the episode with its mentions. `now` is
/// injected (grit's clock stays authoritative for system-time columns it
/// stamps itself).
///
/// All model I/O happens first and decides the entire write set; a single
/// `apply_batch` then commits it. Nothing is stored until that commit, so
/// an `Err` — a model failure, a timeout, a host that hung up — leaves the
/// graph byte-identical to what it was before the call.
///
/// A caller that retries a failed ingest should pass the same
/// [`EpisodeInput::id`] every time: the retry either completes the ingest
/// exactly once or reports [`PipelineError::EpisodeExists`], which says the
/// earlier attempt did land after all. Never duplicated, never orphaned.
pub async fn add_episode<M: LanguageModel, E: Embedder>(
    grit: &Grit,
    model: &M,
    embedder: &E,
    episode: &EpisodeInput,
    previous_episodes: &[EpisodeInput],
    options: &AddEpisodeOptions,
    now: DateTime<Utc>,
) -> Result<AddEpisodeOutcome, PipelineError> {
    // Idempotency first, before any model or storage work: a caller
    // retrying a failed ingest passes the id of the attempt it is retrying.
    // The episode row is written in the same transaction as the graph
    // mutations it evidences, so its presence means the whole ingest
    // landed — there is nothing to complete and nothing to redo.
    if let Some(id) = episode.id
        && grit.episode(id)?.is_some()
    {
        return Err(PipelineError::EpisodeExists(id));
    }

    let episodes = std::slice::from_ref(episode);

    // Vectors persist under the embedder's identity (idempotent for a
    // same-dimension model; a dimension change fails loudly — grit has no
    // re-embedding flow yet). The one write outside this episode's batch:
    // it creates empty vector tables, holds no graph state, and costs
    // nothing if the episode then fails.
    let embedder_meta = embedder.meta();
    grit.register_embedding_model(
        &embedder_meta.model_id,
        embedder_meta.dim as usize,
        &embedder_meta.model_version,
    )?;

    // 1. Extract entity drafts.
    let drafts =
        extract::nodes::extract_nodes(model, episodes, previous_episodes, &options.extract_nodes)
            .await?;

    // Snapshot the group's nodes BEFORE this episode's drafts are planned:
    // upstream dedups extractions against the existing graph only
    // (within-batch dups were already collapsed at extraction). Committed
    // state IS the pre-episode graph here — this runs before the first
    // pending op — so it stays a direct read, no overlay.
    let group_nodes = group_nodes_snapshot(grit, &episode.group_id)?;

    // 2. Per-draft dedup candidates, mirroring upstream's
    //    `_semantic_candidate_search` as patched by the oracle harness
    //    (engine-free — see DEVIATIONS.md "Node dedup candidate search"):
    //    embed the draft names and every distinct existing name (sorted, so
    //    both sides issue identical embedder requests), then rank existing
    //    nodes by pinned f64 cosine, strict `score > min`, limit 15.
    let mut candidate_pools: Vec<Vec<ExistingNode>> = vec![Vec::new(); drafts.len()];
    if !drafts.is_empty() {
        let queries: Vec<String> = drafts.iter().map(|d| d.name.replace('\n', " ")).collect();
        let query_vectors = embedder.embed(&queries).await?;
        if !group_nodes.is_empty() {
            // Stored vectors first (written at persistence time from the
            // same raw name strings, so values are identical to a fresh
            // embed); only names with no vector yet go to the embedder —
            // sorted unique, mirroring the oracle harness's batch shape.
            // In a warm graph this loop makes zero embedder requests.
            // Same invariant as the snapshot above: these are pre-episode
            // rows, and no vector has been planned yet, so the committed
            // read is the whole truth.
            let mut name_vectors: std::collections::HashMap<&str, Vec<f32>> =
                std::collections::HashMap::new();
            for node in &group_nodes {
                if !name_vectors.contains_key(node.name.as_str())
                    && let Some(vector) = grit.get_node_embedding(node.id)?
                {
                    name_vectors.insert(node.name.as_str(), vector);
                }
            }
            let missing: Vec<String> = group_nodes
                .iter()
                .filter(|n| !name_vectors.contains_key(n.name.as_str()))
                .map(|n| n.name.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            if !missing.is_empty() {
                for (name, vector) in missing.iter().zip(embedder.embed(&missing).await?) {
                    name_vectors.insert(name.as_str(), vector);
                }
            }
            let name_vectors = name_vectors; // freeze
            for (pool, query_vector) in candidate_pools.iter_mut().zip(&query_vectors) {
                let mut scored: Vec<(f64, String, &grit_core::Node)> = group_nodes
                    .iter()
                    .filter_map(|n| {
                        let score = cosine_f64(query_vector, &name_vectors[n.name.as_str()]);
                        (score > NODE_DEDUP_MIN_SCORE).then(|| (score, n.id.to_string(), n))
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .expect("cosine is finite")
                        .then_with(|| a.1.cmp(&b.1))
                });
                *pool = scored
                    .into_iter()
                    .take(NODE_DEDUP_CANDIDATE_LIMIT)
                    .map(|(_, _, n)| existing_node_from(n))
                    .collect();
            }
        }
    }

    // Plan the draft nodes now that candidates are pinned. From here on
    // every write is queued in `pending` and committed in one batch at
    // step 8; reads that need to see queued writes go through the overlay.
    let mut pending = Pending::new(now);
    let mut draft_ids: Vec<Uuid> = Vec::with_capacity(drafts.len());
    for draft in &drafts {
        let id = grit.new_id();
        let kind = draft
            .labels
            .iter()
            .find(|l| *l != "Entity")
            .cloned()
            .unwrap_or_else(|| "Entity".to_owned());
        pending.push(
            grit,
            GraphOp::AddNode {
                id,
                kind,
                name: draft.name.clone(),
                summary: draft.summary.clone(),
                attrs: json!({"labels": draft.labels}),
                group_id: draft.group_id.clone(),
            },
        )?;
        draft_ids.push(id);
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
                pending.push(
                    grit,
                    GraphOp::MergeNodes {
                        from: draft_id,
                        into,
                    },
                )?;
                merges.push((draft_id, into));
                // Persist label promotion (the resolution's labels already
                // carry it): upstream re-saves the hydrated node; grit's
                // equivalent is an UpdateNode, skipped when nothing changed.
                // Overlay read: `into` is committed, but an EARLIER draft of
                // this same episode may already have promoted it — that
                // promotion is still pending, and re-emitting it would be a
                // duplicate op.
                let stored = pending.node(grit, into)?.expect("canonical node exists");
                if labels_of(&stored) != existing.labels {
                    let kind = existing
                        .labels
                        .iter()
                        .find(|l| *l != "Entity")
                        .cloned()
                        .unwrap_or_else(|| "Entity".to_owned());
                    let mut attrs = match &stored.attrs {
                        serde_json::Value::Object(map) => map.clone(),
                        _ => serde_json::Map::new(),
                    };
                    attrs.insert("labels".into(), json!(existing.labels));
                    pending.push(
                        grit,
                        GraphOp::UpdateNode {
                            id: into,
                            name: None,
                            summary: None,
                            kind: Some(kind),
                            attrs: Some(serde_json::Value::Object(attrs)),
                        },
                    )?;
                }
                final_ids.push(into);
                // Edge extraction sees the EXTRACTED name (upstream passes
                // extracted_nodes to the prompt and resolves endpoints via
                // uuid_map afterwards) — so a draft "Priya Raman" that
                // resolved onto "Priya" still renders as "Priya Raman", and
                // the endpoint lands on the canonical node.
                node_refs.push(NodeRef {
                    id: existing.id.clone(),
                    name: draft.name.clone(),
                    labels: draft.labels.clone(),
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

    // Collapse extracted duplicates upfront, keeping the first occurrence —
    // upstream dedups by (resolved source, resolved target, normalized
    // fact) before any resolution, so "Priya left" extracted via both the
    // "Priya" and "Priya Raman" drafts becomes ONE edge.
    // ports: edge_operations.py::resolve_extracted_edges (dedup prologue)
    let mut seen_draft_edges: std::collections::HashSet<(String, String, String)> =
        std::collections::HashSet::new();
    let draft_edges: Vec<_> = draft_edges
        .into_iter()
        .filter(|edge| {
            seen_draft_edges.insert((
                edge.source_id.clone(),
                edge.target_id.clone(),
                dedupe::helpers::normalize_string_exact(&edge.fact),
            ))
        })
        .collect();

    // Embed the extracted facts in one batch, mirroring upstream's
    // pre-resolution `create_entity_edge_embeddings(embedder, extracted_edges)`:
    // inputs are the RAW fact strings in draft order, empty facts filtered.
    // ports: graphiti_core/edges.py::create_entity_edge_embeddings
    // (verified against trace1 embedder_recordings.json — e.g. ep-3's batch
    // is the three post-dedup draft facts, unmodified)
    let fact_inputs: Vec<String> = draft_edges
        .iter()
        .filter(|edge| !edge.fact.is_empty())
        .map(|edge| edge.fact.clone())
        .collect();
    let mut fact_vectors: std::collections::HashMap<&str, Vec<f32>> =
        std::collections::HashMap::new();
    if !fact_inputs.is_empty() {
        for (input, vector) in fact_inputs.iter().zip(embedder.embed(&fact_inputs).await?) {
            fact_vectors.insert(input.as_str(), vector);
        }
    }

    // 5. Resolve each edge against stored context and apply the decisions.
    // The episode's identity is the caller's when it supplied one (checked
    // free above), minted here otherwise.
    let episode_id = episode.id.unwrap_or_else(|| grit.new_id());
    let episode_ref = EpisodeRef {
        id: episode_id.to_string(),
        valid_at: episode.valid_at.clone(),
    };
    let mut new_edge_ids: Vec<Uuid> = Vec::new();
    let mut invalidated_edge_ids: Vec<Uuid> = Vec::new();
    let mut mentioned_edges: Vec<Uuid> = Vec::new();
    // Edge temporal writes are applied AFTER the loop, one op per edge,
    // FIRST decision wins: upstream saves every mutated edge object in a
    // single bulk write whose duplicate-uuid semantics keep the first
    // occurrence — and the resolved (duplicate) entries precede the
    // invalidated ones in that list, so an unmutated duplicate resolution
    // blocks later contradictions of the same edge. `None` records such a
    // blocker without emitting an op.
    let mut pending_invalidations: Vec<(Uuid, Option<i64>)> = Vec::new();
    // (src, dst, fact) of edges created this episode — only these feed the
    // summary refresh, mirroring upstream's `edges=new_edges` ("to avoid
    // duplicating facts that already exist in the graph").
    let mut new_edge_tuples: Vec<(String, String, String)> = Vec::new();

    // Snapshot the group's edges once, before any of this episode's edge
    // ops: upstream resolves every extracted edge against the pre-episode
    // graph (it bulk-saves the batch only after resolving all of it), so
    // edges added earlier in this same episode are NOT candidates. Same-pair
    // edges are dedup candidates ("related", upstream's get_between_nodes);
    // the rest of the group is the invalidation pool (engine-free stand-in
    // for upstream's relevance-truncated hybrid search — see DEVIATIONS.md
    // "Edge dedup/invalidation candidate pools"). Direct committed read
    // for the same reason: no edge op has been queued yet, and the merges
    // queued above move no edges (a draft node has none).
    let group_edges = grit.edges_in_group(&episode.group_id)?;

    for draft_edge in &draft_edges {
        let src: Uuid = draft_edge.source_id.parse().expect("grit ids round-trip");
        let dst: Uuid = draft_edge.target_id.parse().expect("grit ids round-trip");

        let mut related: Vec<ExistingEdge> = Vec::new();
        let mut invalidation_pool: Vec<ExistingEdge> = Vec::new();
        for edge in &group_edges {
            let mut existing = existing_edge_from(edge);
            existing.episodes = grit
                .mentions_of(edge.id)?
                .iter()
                .map(Uuid::to_string)
                .collect();
            // DIRECTED, like upstream's get_between_nodes Cypher match
            // `(source)-[e]->(target)`: a stored Priya→Biscuit edge is NOT a
            // dedup candidate for a Biscuit→Priya draft (it lands in the
            // invalidation pool instead).
            let same_endpoints = edge.src == src && edge.dst == dst;
            if same_endpoints {
                related.push(existing);
            } else {
                invalidation_pool.push(existing);
            }
        }

        // ORACLE DEVIATION (DEVIATIONS.md): prompt-facing candidate lists
        // are sorted by fact text. Upstream's order is whatever FalkorDB
        // returns, which is not deterministic across processes — the oracle
        // capture harness applies this same sort (capture.py), making the
        // order well-defined on both sides. UTF-8 byte order equals unicode
        // code-point order, so Rust string sort matches Python's.
        related.sort_by(|a, b| (&a.fact, &a.id).cmp(&(&b.fact, &b.id)));
        invalidation_pool.sort_by(|a, b| (&a.fact, &a.id).cmp(&(&b.fact, &b.id)));

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
                // episode's mentions (grit's mentions table is a set, like
                // upstream's append-if-absent episodes list).
                let existing_uuid: Uuid = existing_id.parse().expect("grit ids round-trip");
                mentioned_edges.push(existing_uuid);
                // The duplicate resolution occupies the edge's first-wins
                // slot: a self-expiry mutation carries a value; an
                // unmutated resolution blocks later contradictions.
                if !pending_invalidations
                    .iter()
                    .any(|(id, _)| *id == existing_uuid)
                {
                    // `existing_uuid` comes from the pre-episode edge
                    // snapshot and this episode has queued no op against
                    // it (first-wins guard above), so committed is current.
                    let stored_invalid = grit
                        .edge(existing_uuid)?
                        .map(|e| e.invalid_at)
                        .unwrap_or_default();
                    let resolved_invalid =
                        resolution.resolved.invalid_at.map(|t| t.timestamp_millis());
                    let mutated = resolved_invalid != stored_invalid;
                    pending_invalidations
                        .push((existing_uuid, resolved_invalid.filter(|_| mutated)));
                }
            }
            None => {
                let edge_id = grit.new_id();
                // An extraction-time invalid_at rides on the edge itself —
                // a bounded fact, not a belief retraction; it leaves no
                // invalidation record (upstream: expired_at stays NULL).
                pending.push(
                    grit,
                    GraphOp::AddEdge {
                        id: edge_id,
                        src,
                        dst,
                        rel: draft_edge.name.clone(),
                        fact: draft_edge.fact.clone(),
                        attrs: json!({}),
                        group_id: draft_edge.group_id.clone(),
                        valid_at: resolution.resolved.valid_at.map(|t| t.timestamp_millis()),
                        invalid_at: resolution.resolved.invalid_at.map(|t| t.timestamp_millis()),
                    },
                )?;
                // Queued right behind its edge: the batch commits items in
                // order, so the row exists by the time the vector lands.
                if let Some(vector) = fact_vectors.get(draft_edge.fact.as_str()) {
                    pending.push_edge_embedding(edge_id, vector.clone());
                }
                new_edge_ids.push(edge_id);
                mentioned_edges.push(edge_id);
                new_edge_tuples.push((src.to_string(), dst.to_string(), draft_edge.fact.clone()));
            }
        }

        for invalidation in &resolution.invalidated {
            let edge_id: Uuid = invalidation.id.parse().expect("grit ids round-trip");
            let invalid_at = invalidation.invalid_at.timestamp_millis();
            // FIRST decision wins: upstream saves all mutated edge
            // objects in one FalkorDB bulk UNWIND, whose write semantics
            // keep the first occurrence of a duplicate uuid (verified
            // empirically against trace1: e8 mutated 07-01 then 06-08,
            // DB holds 07-01).
            if !pending_invalidations.iter().any(|(id, _)| *id == edge_id) {
                pending_invalidations.push((edge_id, Some(invalid_at)));
            }
        }
    }

    for &(edge_id, invalid_at) in &pending_invalidations {
        let Some(invalid_at) = invalid_at else {
            continue; // blocker only — no temporal change decided
        };
        pending.push(
            grit,
            GraphOp::InvalidateEdge {
                edge_id,
                invalid_at,
            },
        )?;
        invalidated_edge_ids.push(edge_id);
    }

    // 6. Refresh entity summaries — upstream's extract_attributes_from_nodes
    //    with edges=new_edges — and persist what changed via UpdateNode.
    //    Upstream operates on the per-draft resolved-node list, where
    //    repeated canonical nodes are ONE shared object — a node resolved
    //    from two drafts gets its facts appended twice (real upstream
    //    behavior). The summarize step emulates the shared object by
    //    propagating same-id summaries; entries here are per-draft.
    let mut summarize_nodes: Vec<SummarizeNode> = Vec::new();
    for &id in &final_ids {
        // Overlay read: most of these ids are drafts added moments ago
        // (they exist only in the batch), and a canonical one may carry a
        // pending label promotion.
        let stored = pending.node(grit, id)?.expect("resolved node exists");
        let mut attributes = match &stored.attrs {
            serde_json::Value::Object(map) => map.clone(),
            _ => serde_json::Map::new(),
        };
        attributes.remove("labels"); // storage convention, not a node attribute
        summarize_nodes.push(SummarizeNode {
            id: id.to_string(),
            name: stored.name.clone(),
            summary: stored.summary.clone(),
            labels: labels_of(&stored),
            attributes,
        });
    }
    let edges_by_node = summarize::nodes::build_edges_by_node(&new_edge_tuples);
    summarize::nodes::extract_entity_summaries_batch(
        model,
        &mut summarize_nodes,
        episodes,
        previous_episodes,
        &edges_by_node,
        &summarize::nodes::SummarizeOptions {
            entity_types: &options.extract_nodes.entity_types,
            ..Default::default()
        },
    )
    .await?;
    let mut updated_ids: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for node in &summarize_nodes {
        let id: Uuid = node.id.parse().expect("grit ids round-trip");
        if !updated_ids.insert(id) {
            continue; // repeated draft entries carry the same propagated summary
        }
        // Overlay read again: the summary being diffed is the one the
        // batch will store, not the one storage still shows.
        let stored = pending.node(grit, id)?.expect("resolved node exists");
        if stored.summary != node.summary {
            pending.push(
                grit,
                GraphOp::UpdateNode {
                    id,
                    name: None,
                    summary: Some(node.summary.clone()),
                    kind: None,
                    attrs: None,
                },
            )?;
        }
    }

    // 7. Persist node name embeddings, mirroring upstream's
    //    `create_entity_node_embeddings(embedder, hydrated_nodes)`: inputs
    //    are the RAW resolved node names, one per draft (a canonical node
    //    resolved from two drafts appears twice), empty names filtered.
    // ports: graphiti_core/nodes.py::create_entity_node_embeddings
    // (verified against trace1 embedder_recordings.json — ep-3's batch is
    // ["Priya", "Northwind Labs", "Meridian Health", "Marco", "Priya",
    //  "Sam Okafor"]: per-draft RESOLVED names, duplicates preserved)
    let mut name_inputs: Vec<String> = Vec::with_capacity(final_ids.len());
    let mut name_targets: Vec<Uuid> = Vec::with_capacity(final_ids.len());
    for &id in &final_ids {
        // Overlay read: a draft node's name lives in the batch only.
        let name = pending.node(grit, id)?.expect("resolved node exists").name;
        if !name.is_empty() {
            name_inputs.push(name);
            name_targets.push(id);
        }
    }
    if !name_inputs.is_empty() {
        for (&id, vector) in name_targets.iter().zip(embedder.embed(&name_inputs).await?) {
            pending.push_node_embedding(id, vector);
        }
    }

    // 8. The episode itself, with provenance for everything it evidenced,
    //    and then the commit: the whole episode in a single transaction.
    //    Everything above this line was a decision, not a write.
    let occurred_at = episode
        .valid_at
        .as_deref()
        .and_then(parse_llm_timestamp)
        .map_or_else(|| now.timestamp_millis(), |t| t.timestamp_millis());
    let mut mentions: Vec<Uuid> = final_ids.clone();
    mentions.extend(&mentioned_edges);
    mentions.dedup();
    pending.push(
        grit,
        GraphOp::AddEpisode {
            id: episode_id,
            source: episode.source_description.clone(),
            kind: episode.source.as_str().into(),
            content: episode.content.clone(),
            occurred_at,
            group_id: episode.group_id.clone(),
            // Verbatim, uninspected: grit stores it, no prompt sees it.
            metadata: episode.metadata.clone(),
            mentions,
        },
    )?;
    grit.apply_batch(pending.items)?;

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
    use crate::model::{
        CompletionRequest, EmbedderMeta, ModelSize, Recording, RecordingStore, ReplayModel,
    };
    use crate::schemas::ResponseSchema;
    use chrono::TimeZone;
    use grit_core::Options;
    use serde_json::{Value, json};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 10, 12, 0, 0).unwrap()
    }

    /// Deterministic offline embedder: same string, same vector. Exact
    /// values are irrelevant here — the flow assertions ride on the
    /// exact-name fast path, not on cosine ranking.
    struct HashEmbedder;

    impl Embedder for HashEmbedder {
        fn meta(&self) -> EmbedderMeta {
            EmbedderMeta {
                model_id: "hash-test".into(),
                dim: 3,
                model_version: String::new(),
            }
        }

        async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
            Ok(inputs
                .iter()
                .map(|s| {
                    let mut h: u64 = 1469598103934665603;
                    for b in s.bytes() {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(1099511628211);
                    }
                    vec![
                        (h % 97) as f32 + 1.0,
                        (h / 97 % 89) as f32 + 1.0,
                        (h / 8633 % 83) as f32 + 1.0,
                    ]
                })
                .collect())
        }
    }

    fn episode(name: &str, content: &str, valid_at: &str) -> EpisodeInput {
        EpisodeInput {
            id: None,
            name: name.into(),
            content: content.into(),
            source: EpisodeSource::Message,
            source_description: "chat between friends".into(),
            group_id: "trace-test".into(),
            valid_at: Some(valid_at.into()),
            metadata: None,
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
        let out1 = add_episode(&grit, &model, &HashEmbedder, &ep1, &[], &opts, now())
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
        let out2 = add_episode(&grit, &model, &HashEmbedder, &ep2, &previous, &opts, now())
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

    /// The previous-episode window filters by the CURRENT episode's source
    /// kind (upstream passes `source=source` to `retrieve_episodes`) — a
    /// text chunk must not see prior chat turns in its prompt window.
    #[test]
    fn previous_episodes_filter_by_source_kind() {
        let dir = std::env::temp_dir().join(format!("nacre-prevsrc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prev-source.db");
        let _ = std::fs::remove_file(&path);
        let grit = Grit::open(&path, Options::new("nacre-test")).unwrap();
        for (kind, content, at) in [
            ("message", "a chat turn", 1_000),
            ("text", "a document chunk", 2_000),
            ("message", "a later chat turn", 3_000),
        ] {
            grit.apply(GraphOp::AddEpisode {
                id: grit.new_id(),
                source: "test".into(),
                kind: kind.into(),
                content: content.into(),
                occurred_at: at,
                group_id: "g".into(),
                metadata: None,
                mentions: vec![],
            })
            .unwrap();
        }
        let at = |ms: i64| DateTime::<Utc>::from_timestamp_millis(ms).unwrap();
        let messages =
            retrieve_previous_episodes(&grit, "g", EpisodeSource::Message, at(5_000), 10).unwrap();
        assert_eq!(
            messages
                .iter()
                .map(|e| e.content.as_str())
                .collect::<Vec<_>>(),
            ["a chat turn", "a later chat turn"],
            "message window sees only message episodes, chronological"
        );
        assert!(messages.iter().all(|e| e.source == EpisodeSource::Message));
        let texts =
            retrieve_previous_episodes(&grit, "g", EpisodeSource::Text, at(5_000), 10).unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].content, "a document chunk");
        let json =
            retrieve_previous_episodes(&grit, "g", EpisodeSource::Json, at(5_000), 10).unwrap();
        assert!(json.is_empty(), "no json episodes stored");
        drop(grit);
        let _ = std::fs::remove_file(&path);
    }

    /// A model scripted by response-schema name, recording every request it
    /// is shown. The golden traces never exercise label promotion (it fires
    /// in neither trace), so the overlay needs a purpose-built driver;
    /// reconstructing exact prompts for a `ReplayModel` would only obscure
    /// what this test is about.
    struct ScriptModel {
        responses: Vec<(&'static str, Value)>,
        seen: std::sync::Mutex<Vec<Value>>,
    }

    impl ScriptModel {
        fn new(responses: Vec<(&'static str, Value)>) -> Self {
            Self {
                responses,
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// The rendered prompt of the single request issued for
        /// `schema_name` — what the pipeline actually showed the model.
        fn prompt_for(&self, schema_name: &str) -> String {
            let seen = self.seen.lock().expect("script log lock");
            let matching: Vec<&Value> = seen
                .iter()
                .filter(|request| request["schema_name"] == schema_name)
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "expected exactly one {schema_name} request, saw {}",
                matching.len()
            );
            matching[0]["messages"].to_string()
        }

        /// Every request the pipeline issued, whole and in order — for
        /// assertions about what the model was *not* shown.
        fn all_requests(&self) -> Vec<Value> {
            self.seen.lock().expect("script log lock").clone()
        }
    }

    impl LanguageModel for ScriptModel {
        fn complete(
            &self,
            request: &CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Value, ModelError>> + Send {
            self.seen
                .lock()
                .expect("script log lock")
                .push(serde_json::to_value(request).expect("request serializes"));
            let result = self
                .responses
                .iter()
                .find(|(schema, _)| *schema == request.schema_name)
                .map(|(_, response)| response.clone())
                .ok_or_else(|| ModelError::ReplayMiss {
                    canonical_request: request.schema_name.clone(),
                });
            async move { result }
        }
    }

    /// Every input embeds to the same vector, so cosine is 1.0 everywhere
    /// and every stored node is a dedup candidate for every draft: the
    /// candidate pool becomes a property of the fixture, not of a hash.
    struct ConstEmbedder;

    impl Embedder for ConstEmbedder {
        fn meta(&self) -> EmbedderMeta {
            EmbedderMeta {
                model_id: "const-test".into(),
                dim: 3,
                model_version: String::new(),
            }
        }

        async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
            Ok(inputs.iter().map(|_| vec![1.0, 1.0, 1.0]).collect())
        }
    }

    /// One readable line per op: the tag plus the fields this test pins.
    fn op_line(op: &GraphOp, alias: &dyn Fn(Uuid) -> String) -> String {
        let list = |ids: &[Uuid]| {
            ids.iter()
                .map(|id| alias(*id))
                .collect::<Vec<_>>()
                .join(",")
        };
        match op {
            GraphOp::AddNode { id, kind, name, .. } => {
                format!("AddNode {} kind={kind} name={name}", alias(*id))
            }
            GraphOp::MergeNodes { from, into } => {
                format!("MergeNodes {}->{}", alias(*from), alias(*into))
            }
            GraphOp::UpdateNode {
                id,
                summary,
                kind,
                attrs,
                ..
            } => format!(
                "UpdateNode {} kind={} attrs={} summary={}",
                alias(*id),
                kind.as_deref().unwrap_or("-"),
                attrs
                    .as_ref()
                    .map_or_else(|| "-".to_owned(), |a| a["labels"].to_string()),
                summary.as_deref().unwrap_or("-"),
            ),
            GraphOp::AddEdge { id, src, dst, .. } => {
                format!("AddEdge {} {}->{}", alias(*id), alias(*src), alias(*dst))
            }
            GraphOp::InvalidateEdge { edge_id, .. } => {
                format!("InvalidateEdge {}", alias(*edge_id))
            }
            GraphOp::AddEpisode { id, mentions, .. } => {
                format!("AddEpisode {} mentions=[{}]", alias(*id), list(mentions))
            }
            other => format!("{other:?}"),
        }
    }

    /// Two drafts resolve onto ONE stored node, the first promoting its
    /// labels: every read the rest of the pipeline makes about that node
    /// must see the promotion even though nothing is committed yet.
    ///
    /// This is the overlay's contract, and the golden traces cannot state
    /// it — label promotion fires in neither trace. What the assertions
    /// pin, beyond the exact op sequence: the SECOND resolution sees the
    /// promoted labels (no duplicate `UpdateNode`), the summarize inputs
    /// carry them (and the stored summary the promotion did not touch),
    /// and a node created earlier in the same episode is readable for its
    /// summary diff (it exists only in the overlay until the batch lands).
    #[tokio::test]
    async fn overlay_serves_pending_writes_within_one_episode() {
        const GROUP: &str = "overlay-test";
        // Long enough to defeat the fact-append shortcut (> 2 ×
        // MAX_SUMMARY_CHARS), so the summarize step really issues a
        // request we can inspect. The marker proves the overlay kept the
        // fields the label promotion did not write.
        let stored_summary = format!(
            "Prior summary about chartreuse kilns. {}",
            "x".repeat(2_100)
        );

        let dir = std::env::temp_dir().join(format!("nacre-overlay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overlay.db");
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
        let grit = Grit::open(&path, Options::new("nacre-test")).unwrap();

        // A node stored by an earlier episode: generic label, own summary.
        let canonical = grit.new_id();
        grit.apply(GraphOp::AddNode {
            id: canonical,
            kind: "Entity".into(),
            name: "Priya Raman".into(),
            summary: stored_summary,
            attrs: json!({"labels": ["Entity"]}),
            group_id: GROUP.into(),
        })
        .unwrap();
        let baseline_seq = grit.oplog_since(0).unwrap().last().unwrap().0;

        // Entity types with an EMPTY description keep "Person" out of the
        // summarize prompt's type-description block, so its appearance
        // there can only come from the promoted labels.
        let opts = AddEpisodeOptions {
            extract_nodes: ExtractNodesOptions {
                entity_types: vec![EntityTypeSpec {
                    name: "Person".into(),
                    description: String::new(),
                }],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut episode = episode(
            "ep-overlay",
            "Priya-Raman started at Northwind Labs today.",
            "2026-07-10T12:00:00+00:00",
        );
        episode.group_id = GROUP.into();

        let model = ScriptModel::new(vec![
            // Two spellings of the same person plus one new entity. The
            // spellings differ under exact normalization (the hyphen
            // survives) so extraction keeps both drafts; they converge on
            // the stored node through the exact and the fuzzy path.
            (
                "ExtractedEntities",
                json!({"extracted_entities": [
                    {"name": "Priya Raman", "entity_type_id": 1, "episode_indices": [0]},
                    {"name": "Priya-Raman", "entity_type_id": 1, "episode_indices": [0]},
                    {"name": "Northwind Labs", "entity_type_id": 0, "episode_indices": [0]},
                ]}),
            ),
            // Only "Northwind Labs" escalates; -1 means "no duplicate".
            (
                "NodeResolutions",
                json!({"entity_resolutions": [
                    {"id": 0, "name": "Northwind Labs", "duplicate_candidate_id": -1},
                ]}),
            ),
            (
                "ExtractedEdges",
                json!({"edges": [{
                    "source_entity_name": "Priya Raman",
                    "target_entity_name": "Northwind Labs",
                    "relation_type": "WORKS_AT",
                    "fact": "Priya Raman joined Northwind Labs.",
                    "valid_at": "2026-07-10T00:00:00Z",
                    "episode_indices": [0],
                }]}),
            ),
            (
                "SummarizedEntities",
                json!({"summaries": [
                    {"name": "Priya Raman", "summary": "Priya Raman works at Northwind Labs."},
                ]}),
            ),
        ]);

        let out = add_episode(&grit, &model, &ConstEmbedder, &episode, &[], &opts, now())
            .await
            .unwrap();

        // Both Priya drafts folded into the stored node; the third is new.
        assert_eq!(out.merges.len(), 2, "both drafts resolved onto one node");
        assert_eq!(out.node_ids, vec![canonical, canonical, out.node_ids[2]]);
        let (draft1, draft2) = (out.merges[0].0, out.merges[1].0);
        let northwind = out.node_ids[2];
        let edge = out.new_edge_ids[0];
        let episode_id = out.episode_id;
        let alias = move |id: Uuid| -> String {
            match id {
                _ if id == canonical => "canonical".to_owned(),
                _ if id == draft1 => "draft1".to_owned(),
                _ if id == draft2 => "draft2".to_owned(),
                _ if id == northwind => "northwind".to_owned(),
                _ if id == edge => "edge".to_owned(),
                _ if id == episode_id => "episode".to_owned(),
                other => other.to_string(),
            }
        };

        let ops: Vec<String> = grit
            .oplog_since(baseline_seq)
            .unwrap()
            .iter()
            .map(|(_, entry)| op_line(&entry.op, &alias))
            .collect();
        assert_eq!(
            ops,
            vec![
                "AddNode draft1 kind=Person name=Priya Raman",
                "AddNode draft2 kind=Person name=Priya-Raman",
                "AddNode northwind kind=Entity name=Northwind Labs",
                "MergeNodes draft1->canonical",
                // Promotion, exactly once: the second resolution reads the
                // overlay and finds the labels already promoted.
                r#"UpdateNode canonical kind=Person attrs=["Entity","Person"] summary=-"#,
                "MergeNodes draft2->canonical",
                "AddEdge edge canonical->northwind",
                "UpdateNode canonical kind=- attrs=- summary=Priya Raman works at Northwind Labs.",
                // Readable only through the overlay: this node's row does
                // not exist until the batch commits.
                "UpdateNode northwind kind=- attrs=- summary=Priya Raman joined Northwind Labs.",
                "AddEpisode episode mentions=[canonical,northwind,edge]",
            ],
            "the ops, and their order, are the same ones the sequential \
             pipeline applied"
        );

        // The summarize step saw the promoted labels, and the summary the
        // promotion left alone.
        let summarize_prompt = model.prompt_for("SummarizedEntities");
        assert!(
            summarize_prompt.contains("Person"),
            "summarize inputs must carry the pending label promotion"
        );
        assert!(
            summarize_prompt.contains("chartreuse"),
            "summarize inputs must carry the stored summary the promotion did not touch"
        );

        // Committed state matches what the reads were told mid-flight.
        let stored = grit.node(canonical).unwrap().unwrap();
        assert_eq!(stored.kind, "Person");
        assert_eq!(labels_of(&stored), ["Entity", "Person"]);
        assert_eq!(stored.summary, "Priya Raman works at Northwind Labs.");
        assert!(grit.get_node_embedding(canonical).unwrap().is_some());
        assert!(grit.get_node_embedding(northwind).unwrap().is_some());
        assert!(grit.get_edge_embedding(edge).unwrap().is_some());

        drop(grit);
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    /// `EpisodeInput.metadata` is stored, handed back by both episode read
    /// paths, and shown to no model.
    ///
    /// The exclusion is the load-bearing half. Nothing in the pipeline reads
    /// metadata, and every prompt context is built by cherry-picking episode
    /// fields — so a leak would take a deliberate code change, and no
    /// existing assertion would notice it: the golden traces carry no
    /// metadata, and `prompt_fidelity` renders from fixed fixture contexts
    /// that never see an `EpisodeInput`. Hence a sentinel swept across every
    /// request the run issues, with metadata set on BOTH the ingested
    /// episode and the previous-episode window that feeds the same prompts.
    #[tokio::test]
    async fn metadata_is_stored_read_back_and_never_reaches_a_prompt() {
        const GROUP: &str = "metadata-test";
        // High-entropy and unlike any prompt text, so a substring hit is a
        // leak and never a coincidence.
        const SENTINEL: &str = "NACRE-METADATA-SENTINEL-8f3a2c1d";
        const PRIOR_SENTINEL: &str = "NACRE-METADATA-SENTINEL-4b7e0a96";

        let dir = std::env::temp_dir().join(format!("nacre-metadata-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metadata.db");
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
        let grit = Grit::open(&path, Options::new("nacre-test")).unwrap();

        // A stored earlier episode carrying its own payload: the window
        // rebuild must hand it back, and it must reach no prompt either.
        let prior_id = grit.new_id();
        grit.apply(GraphOp::AddEpisode {
            id: prior_id,
            source: "chat between friends".into(),
            kind: "message".into(),
            content: "Priya: morning!".into(),
            occurred_at: now().timestamp_millis() - 60_000,
            group_id: GROUP.into(),
            metadata: Some(PRIOR_SENTINEL.into()),
            mentions: vec![],
        })
        .unwrap();

        let previous = retrieve_previous_episodes(
            &grit,
            GROUP,
            EpisodeSource::Message,
            now(),
            PREVIOUS_EPISODE_WINDOW,
        )
        .unwrap();
        assert_eq!(
            previous
                .iter()
                .map(|e| e.metadata.as_deref())
                .collect::<Vec<_>>(),
            [Some(PRIOR_SENTINEL)],
            "the previous-episode window hands stored metadata back verbatim"
        );

        let mut episode = episode(
            "ep-metadata",
            "Priya: I joined Northwind Labs today.",
            "2026-07-10T12:00:00+00:00",
        );
        episode.group_id = GROUP.into();
        episode.metadata = Some(SENTINEL.into());

        let model = ScriptModel::new(vec![
            (
                "ExtractedEntities",
                json!({"extracted_entities": [
                    {"name": "Priya", "entity_type_id": 1, "episode_indices": [0]},
                    {"name": "Northwind Labs", "entity_type_id": 0, "episode_indices": [0]},
                ]}),
            ),
            (
                "NodeResolutions",
                json!({"entity_resolutions": [
                    {"id": 0, "name": "Priya", "duplicate_candidate_id": -1},
                    {"id": 1, "name": "Northwind Labs", "duplicate_candidate_id": -1},
                ]}),
            ),
            (
                "ExtractedEdges",
                json!({"edges": [{
                    "source_entity_name": "Priya",
                    "target_entity_name": "Northwind Labs",
                    "relation_type": "WORKS_AT",
                    "fact": "Priya joined Northwind Labs.",
                    "valid_at": "2026-07-10T00:00:00Z",
                    "episode_indices": [0],
                }]}),
            ),
            (
                "SummarizedEntities",
                json!({"summaries": [
                    {"name": "Priya", "summary": "Priya works at Northwind Labs."},
                ]}),
            ),
        ]);

        let out = add_episode(
            &grit,
            &model,
            &ConstEmbedder,
            &episode,
            &previous,
            &options(),
            now(),
        )
        .await
        .unwrap();

        // No prompt, schema name or model-size field of any request the run
        // issued carries either payload.
        let requests = model.all_requests();
        assert!(
            !requests.is_empty(),
            "the run must actually have called the model, or this proves nothing"
        );
        assert!(
            requests
                .iter()
                .any(|r| r.to_string().contains("Priya: morning!")),
            "the previous-episode window must really reach a prompt, or the \
             sweep below is checking nothing about it"
        );
        for request in &requests {
            let rendered = request.to_string();
            for sentinel in [SENTINEL, PRIOR_SENTINEL] {
                assert!(
                    !rendered.contains(sentinel),
                    "episode metadata leaked into a {} request",
                    request["schema_name"]
                );
            }
        }

        // Stored verbatim, and readable through both episode read paths.
        let stored = grit.episode(out.episode_id).unwrap().unwrap();
        assert_eq!(stored.metadata.as_deref(), Some(SENTINEL));
        assert_eq!(
            grit.episodes_in_group(GROUP)
                .unwrap()
                .iter()
                .map(|e| e.metadata.clone())
                .collect::<Vec<_>>(),
            [Some(PRIOR_SENTINEL.to_owned()), Some(SENTINEL.to_owned())],
            "the group scan hands both payloads back, chronological"
        );

        drop(grit);
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
