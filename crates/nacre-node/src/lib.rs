//! Node.js bindings for nacre — the Layer 3 gateway.
//!
//! Thin `#[napi]` glue over `nacre-core` + `grit-core`; all logic lives in
//! those crates. Built into a loadable `.node` addon by `@napi-rs/cli`
//! (`npm run build` in this directory); plain `cargo build -p nacre-node`
//! type-checks and compiles the cdylib without any Node toolchain.

mod providers;
mod rows;

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use providers::{EmbedderConfig, LlmConfig, build_embedder, build_model};
use rows::{
    EdgeRow, EpisodeRow, NodeHistoryJs, NodeRow, SubgraphJs, edge_row, episode_row, node_row,
};

use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::forget::{delete_episode, delete_group};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, PipelineError, add_episode,
    retrieve_previous_episodes,
};
use nacre_core::search::search_edges;

/// The nacre-node crate version (addon load smoke-check).
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Marks the one `addEpisode` rejection a caller is expected to branch on:
/// the episode id it supplied is already stored, so that ingest already
/// landed in full. Rejected errors start their message with this token
/// followed by `": "` — match on it, never on the prose after it.
///
/// It rides in the message because napi derives a rejected error's `code`
/// from its own closed `Status` enum: an async binding cannot put a
/// domain-specific code there.
#[napi]
pub const EPISODE_EXISTS_CODE: &str = "EPISODE_EXISTS";

/// Marks the one `deleteEpisode` rejection a caller is expected to branch
/// on: the episode id it named is not stored, so nothing was deleted —
/// including a second delete of an episode already deleted. Rejected
/// errors start their message with this token followed by `": "` — match
/// on it, never on the prose after it.
///
/// Same mechanism, and the same napi limitation, as
/// [`EPISODE_EXISTS_CODE`].
#[napi]
pub const EPISODE_NOT_FOUND_CODE: &str = "EPISODE_NOT_FOUND";

/// One episode to ingest.
#[napi(object)]
pub struct EpisodeJs {
    /// Optional caller-supplied episode id (a UUID string) — the ingest
    /// idempotency key. Omit it and the pipeline mints one. A caller
    /// retrying a failed ingest passes the SAME id every time: the ingest
    /// either completes exactly once or rejects with EPISODE_EXISTS_CODE,
    /// so a retry can never duplicate a partially written episode.
    pub id: Option<String>,
    /// Short episode name (metadata only; not stored in the graph).
    pub name: String,
    /// Raw text of the episode.
    pub content: String,
    /// "message" (default) | "text" | "json"
    pub source: Option<String>,
    /// Where this came from, e.g. "chat" or "doc:notes.md".
    pub source_description: String,
    /// Namespace; every query filters on it.
    pub group_id: String,
    /// ISO-8601 event time. Also the pipeline's injected clock for this
    /// ingestion; defaults to the current system time.
    pub valid_at: Option<String>,
    /// Application payload stored verbatim on the episode row and handed
    /// back by every episode read path (`episode`, `episodesInGroup`,
    /// `previousEpisodes`). Opaque to nacre: never sent to a model, never
    /// embedded, never part of episode identity — it is where a caller
    /// records which of ITS OWN rows this memory came from.
    ///
    /// Write-once. grit has no episode update op, so changing it means
    /// `deleteEpisode` plus a fresh ingest **under a new id**: deletion
    /// tombstones the old id permanently, and re-ingesting under it
    /// resolves as success while storing neither the episode row nor its
    /// provenance.
    pub metadata: Option<String>,
}

/// What one ingestion changed — the UI animation feed.
#[napi(object)]
pub struct AddEpisodeResultJs {
    /// Stored episode id.
    pub episode_id: String,
    /// Final node id per extracted entity (post-merge), in draft order.
    pub node_ids: Vec<String>,
    /// Edges created by this ingestion.
    pub new_edge_ids: Vec<String>,
    /// `[from, into]` node merges executed.
    pub merges: Vec<Vec<String>>,
    /// Stored edges this ingestion invalidated.
    pub invalidated_edge_ids: Vec<String>,
}

/// What one `deleteEpisode` erased, and what it deliberately left.
///
/// The `kept*` lists are the interesting half: entities and facts this
/// episode evidenced that OTHER episodes evidence too, and which are
/// therefore still true after it is forgotten. A UI showing a user what a
/// deletion did should show both halves.
#[napi(object)]
pub struct DeleteEpisodeResultJs {
    /// The episode that was erased.
    pub episode_id: String,
    /// Nodes erased: mentioned by this episode and no other.
    pub purged_node_ids: Vec<String>,
    /// Edges erased: this episode was their earliest evidence.
    pub purged_edge_ids: Vec<String>,
    /// Nodes kept: other episodes mention them too.
    pub kept_node_ids: Vec<String>,
    /// Edges kept: an earlier episode is their evidence.
    pub kept_edge_ids: Vec<String>,
}

/// What one `deleteGroup` erased. Empty everywhere when the group held
/// nothing — an unknown group is not an error, it is a group with no rows.
#[napi(object)]
pub struct DeleteGroupResultJs {
    /// Node rows erased, id-ordered.
    pub purged_node_ids: Vec<String>,
    /// Edge rows erased, id-ordered.
    pub purged_edge_ids: Vec<String>,
    /// Episode rows erased, chronological.
    pub purged_episode_ids: Vec<String>,
}

fn ids(ids: &[uuid::Uuid]) -> Vec<String> {
    ids.iter().map(|id| id.to_string()).collect()
}

fn generic(e: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
}

/// Pipeline failures, as JS sees them. Model and storage failures render
/// as a `GenericFailure` carrying the cause. The two episode-identity
/// rejections are the ones a caller branches on, so their messages lead
/// with [`EPISODE_EXISTS_CODE`] / [`EPISODE_NOT_FOUND_CODE`] (see those
/// constants for why the code is not in `error.code`).
fn pipeline_error(e: PipelineError) -> Error {
    match e {
        PipelineError::EpisodeExists(id) => Error::new(
            Status::GenericFailure,
            format!("{EPISODE_EXISTS_CODE}: episode {id} already exists"),
        ),
        PipelineError::EpisodeNotFound(id) => Error::new(
            Status::GenericFailure,
            format!("{EPISODE_NOT_FOUND_CODE}: episode {id} not found"),
        ),
        e @ (PipelineError::Model(_) | PipelineError::Storage(_)) => generic(e),
    }
}

fn parse_time(iso: Option<&str>) -> Result<DateTime<Utc>> {
    match iso {
        None => Ok(Utc::now()),
        Some(t) => DateTime::parse_from_rfc3339(t)
            .map(|t| t.with_timezone(&Utc))
            .map_err(|e| Error::new(Status::InvalidArg, format!("validAt: {e}"))),
    }
}

/// A memory file: one grit graph plus the pipeline that writes it.
#[napi]
pub struct Memory {
    grit: grit_core::Grit,
}

#[napi]
impl Memory {
    /// Open (or create) a memory file. `deviceId` names this device in the
    /// op-log — keep it stable per installation.
    #[napi(factory)]
    pub fn open(path: String, device_id: String) -> Result<Memory> {
        let grit =
            grit_core::Grit::open(&path, grit_core::Options::new(device_id)).map_err(generic)?;
        Ok(Memory { grit })
    }

    /// Ingest one episode through the full pipeline (extraction, dedup,
    /// temporal invalidation, summaries, embeddings). Previous-episode
    /// context comes from the graph itself. Returns the change set.
    #[napi]
    pub async fn add_episode(
        &self,
        episode: EpisodeJs,
        llm: LlmConfig,
        embedder: EmbedderConfig,
    ) -> Result<AddEpisodeResultJs> {
        let source = match episode.source.as_deref() {
            None | Some("message") => EpisodeSource::Message,
            Some("text") => EpisodeSource::Text,
            Some("json") => EpisodeSource::Json,
            Some(other) => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("unknown episode source {other:?}"),
                ));
            }
        };
        let now = parse_time(episode.valid_at.as_deref())?;
        // Rejected before any model call or storage read, like every other
        // argument: a malformed idempotency key is a caller bug, and the
        // pipeline must not spend a request discovering it.
        let id = episode
            .id
            .as_deref()
            .map(|id| {
                id.parse::<uuid::Uuid>()
                    .map_err(|e| Error::new(Status::InvalidArg, format!("id {id:?}: {e}")))
            })
            .transpose()?;
        let input = EpisodeInput {
            id,
            name: episode.name,
            content: episode.content,
            source,
            source_description: episode.source_description,
            group_id: episode.group_id.clone(),
            valid_at: episode
                .valid_at
                .clone()
                .or_else(|| Some(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false))),
            metadata: episode.metadata,
        };
        let model = build_model(llm)?;
        let embedder = build_embedder(embedder)?;
        let previous = retrieve_previous_episodes(
            &self.grit,
            &episode.group_id,
            source,
            now,
            PREVIOUS_EPISODE_WINDOW,
        )
        .map_err(generic)?;
        let outcome = add_episode(
            &self.grit,
            &model,
            &embedder,
            &input,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .map_err(pipeline_error)?;
        Ok(AddEpisodeResultJs {
            episode_id: outcome.episode_id.to_string(),
            node_ids: outcome.node_ids.iter().map(|id| id.to_string()).collect(),
            new_edge_ids: outcome
                .new_edge_ids
                .iter()
                .map(|id| id.to_string())
                .collect(),
            merges: outcome
                .merges
                .iter()
                .map(|(from, into)| vec![from.to_string(), into.to_string()])
                .collect(),
            invalidated_edge_ids: outcome
                .invalidated_edge_ids
                .iter()
                .map(|id| id.to_string())
                .collect(),
        })
    }

    // ------------------------------------------------------------------
    // Read path. The calls below are the entire data contract the
    // memory-graph visualization needs: full dump (nodesInGroup /
    // edgesInGroup / episodesInGroup), focus+context (traverse), and
    // drill-down (nodeHistory + mentionsOf + episode).
    // ------------------------------------------------------------------

    /// Every node in a group, id-ordered — full bi-temporal record
    /// including merged-away rows (filter `expiredAt` for the live view).
    #[napi]
    pub fn nodes_in_group(&self, group_id: String) -> Result<Vec<NodeRow>> {
        Ok(self
            .grit
            .nodes_in_group(&group_id)
            .map_err(generic)?
            .iter()
            .map(node_row)
            .collect())
    }

    /// Every edge in a group, id-ordered — invalidated and expired rows
    /// included (that's the archaeology view).
    #[napi]
    pub fn edges_in_group(&self, group_id: String) -> Result<Vec<EdgeRow>> {
        Ok(self
            .grit
            .edges_in_group(&group_id)
            .map_err(generic)?
            .iter()
            .map(edge_row)
            .collect())
    }

    /// Every episode in a group, chronological.
    #[napi]
    pub fn episodes_in_group(&self, group_id: String) -> Result<Vec<EpisodeRow>> {
        Ok(self
            .grit
            .episodes_in_group(&group_id)
            .map_err(generic)?
            .iter()
            .map(episode_row)
            .collect())
    }

    /// One episode by id, or `null` when the graph does not hold it —
    /// the point lookup behind a provenance id from `searchEdges` or
    /// `mentionsOf`, and the way a caller reads its own `metadata` back
    /// without scanning the group.
    ///
    /// Absence is a `null`, not a rejection: a getter that answers "no
    /// such row" is not a failure, and a caller that wants the id's
    /// absence to be an error owns that decision. (`deleteEpisode` is the
    /// opposite case — deleting nothing is worth reporting.)
    #[napi]
    pub fn episode(&self, id: String) -> Result<Option<EpisodeRow>> {
        let id: uuid::Uuid = id
            .parse()
            .map_err(|e| Error::new(Status::InvalidArg, format!("episode id: {e}")))?;
        Ok(self
            .grit
            .episode(id)
            .map_err(generic)?
            .as_ref()
            .map(episode_row))
    }

    /// Bounded neighborhood around seed nodes. `asOf` filters by event
    /// time, `asAt` by belief time — both ISO-8601; omit for "now". The
    /// node budget keeps the walk viewport-sized (default 256).
    #[napi]
    pub fn traverse(
        &self,
        seeds: Vec<String>,
        options: Option<TraverseOptions>,
    ) -> Result<SubgraphJs> {
        let seeds: Vec<uuid::Uuid> = seeds
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| Error::new(Status::InvalidArg, format!("seed id {s:?}: {e}")))
            })
            .collect::<Result<_>>()?;
        let mut traversal = grit_core::Traversal::default();
        if let Some(options) = options {
            if let Some(depth) = options.depth {
                traversal = traversal.depth(depth);
            }
            if let Some(max_nodes) = options.max_nodes {
                traversal = traversal.max_nodes(max_nodes as usize);
            }
            if let Some(group) = options.group_id {
                traversal = traversal.group(group);
            }
            if let Some(as_of) = options.as_of.as_deref() {
                traversal = traversal.as_of(parse_time(Some(as_of))?.timestamp_millis());
            }
            if let Some(as_at) = options.as_at.as_deref() {
                traversal = traversal.as_at(parse_time(Some(as_at))?.timestamp_millis());
            }
        }
        let sub = self.grit.traverse(&seeds, &traversal).map_err(generic)?;
        Ok(SubgraphJs {
            nodes: sub.nodes.iter().map(node_row).collect(),
            edges: sub.edges.iter().map(edge_row).collect(),
        })
    }

    /// A node's bi-temporal audit trail: the node row (even if expired)
    /// plus every incident edge ever believed, oldest belief first.
    #[napi]
    pub fn node_history(&self, id: String) -> Result<NodeHistoryJs> {
        let id: uuid::Uuid = id
            .parse()
            .map_err(|e| Error::new(Status::InvalidArg, format!("node id: {e}")))?;
        let history = self.grit.node_history(id).map_err(generic)?;
        Ok(NodeHistoryJs {
            node: node_row(&history.node),
            edges: history.edges.iter().map(edge_row).collect(),
        })
    }

    /// Episode ids attributing a node or edge (provenance drill-down).
    #[napi]
    pub fn mentions_of(&self, id: String) -> Result<Vec<String>> {
        let id: uuid::Uuid = id
            .parse()
            .map_err(|e| Error::new(Status::InvalidArg, format!("target id: {e}")))?;
        Ok(self
            .grit
            .mentions_of(id)
            .map_err(generic)?
            .iter()
            .map(|id| id.to_string())
            .collect())
    }

    /// The previous-episode context window `addEpisode` would use at
    /// `reference` (ISO-8601; omit for now) — last `lastN` episodes,
    /// chronological. Useful for showing "what the pipeline will see".
    #[napi]
    pub fn previous_episodes(
        &self,
        group_id: String,
        reference: Option<String>,
        last_n: Option<u32>,
    ) -> Result<Vec<EpisodeRow>> {
        let reference = parse_time(reference.as_deref())?;
        let last_n = last_n.map_or(PREVIOUS_EPISODE_WINDOW, |n| n as usize);
        // The pipeline helper returns prompt-shaped inputs; the JS surface
        // wants full rows — re-derive from the same chronological scan.
        let mut episodes = self.grit.episodes_in_group(&group_id).map_err(generic)?;
        episodes.retain(|e| e.occurred_at <= reference.timestamp_millis());
        let start = episodes.len().saturating_sub(last_n);
        Ok(episodes[start..].iter().map(episode_row).collect())
    }

    // ------------------------------------------------------------------
    // Deletion. Synchronous like the read path — no model call is
    // involved, only grit — but these are WRITES: each commits one
    // `Purge` in a single transaction, so a deletion happens whole or not
    // at all. "Forgotten" means gone from the live file: rows, indexes,
    // vectors and the op-log entries that carried them, with the freed
    // pages zeroed. What stays out of reach of any library — OS backups,
    // copies taken earlier, devices that never receive the op — is
    // documented on `nacre_core::forget`, alongside the maintenance call
    // (`Grit::compact`) that finishes the byte-level job.
    // ------------------------------------------------------------------

    /// Forget one episode, and the graph that exists only because of it:
    /// nodes this episode is the sole evidence for and facts it created
    /// go; anything another episode also evidences stays. Returns both
    /// halves.
    ///
    /// Rejects with [`EPISODE_NOT_FOUND_CODE`] when the id is not stored,
    /// including a second delete of an episode already deleted — nothing
    /// is written in that case.
    ///
    /// The deleted id is **spent**: grit tombstones it permanently, so a
    /// later ingest naming it stores neither the episode row nor its
    /// provenance while the nodes and edges beside it land. Re-ingesting
    /// after a delete needs a NEW id.
    #[napi]
    pub fn delete_episode(&self, id: String) -> Result<DeleteEpisodeResultJs> {
        let id: uuid::Uuid = id
            .parse()
            .map_err(|e| Error::new(Status::InvalidArg, format!("episode id: {e}")))?;
        let outcome = delete_episode(&self.grit, id).map_err(pipeline_error)?;
        Ok(DeleteEpisodeResultJs {
            episode_id: outcome.episode_id.to_string(),
            purged_node_ids: ids(&outcome.purged_node_ids),
            purged_edge_ids: ids(&outcome.purged_edge_ids),
            kept_node_ids: ids(&outcome.kept_node_ids),
            kept_edge_ids: ids(&outcome.kept_edge_ids),
        })
    }

    /// Forget a whole namespace: every node, edge and episode row tagged
    /// with `groupId`, erased in one `Purge`. This enumerates rows, not
    /// evidence — the full bi-temporal record, expired and merged-away
    /// rows included.
    ///
    /// An unknown or empty group is **not** an error: it is a group with
    /// no rows, and the answer is a zero-count result with nothing
    /// written.
    #[napi]
    pub fn delete_group(&self, group_id: String) -> Result<DeleteGroupResultJs> {
        let outcome = delete_group(&self.grit, &group_id).map_err(pipeline_error)?;
        Ok(DeleteGroupResultJs {
            purged_node_ids: ids(&outcome.purged_node_ids),
            purged_edge_ids: ids(&outcome.purged_edge_ids),
            purged_episode_ids: ids(&outcome.purged_episode_ids),
        })
    }
}

/// One search hit, in fused rank order.
#[napi(object)]
pub struct SearchHitJs {
    /// Edge id.
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    /// Relation label.
    pub name: String,
    /// The fact sentence.
    pub fact: String,
    /// Event time the fact became true.
    pub valid_at: Option<String>,
    /// Event time the fact stopped being true.
    pub invalid_at: Option<String>,
    /// Provenance episode ids.
    pub episodes: Vec<String>,
}

#[napi]
impl Memory {
    /// Hybrid recall: the query is embedded and fused (RRF) with BM25 over
    /// the stored fact embeddings; currently-valid edges return in rank
    /// order with provenance.
    #[napi]
    pub async fn search_edges(
        &self,
        query: String,
        group_id: String,
        limit: u32,
        embedder: EmbedderConfig,
    ) -> Result<Vec<SearchHitJs>> {
        let embedder = build_embedder(embedder)?;
        let hits = search_edges(&self.grit, &embedder, &query, &group_id, limit as usize)
            .await
            .map_err(generic)?;
        Ok(hits
            .into_iter()
            .map(|hit| SearchHitJs {
                id: hit.id,
                source_id: hit.source_id,
                target_id: hit.target_id,
                name: hit.name,
                fact: hit.fact,
                valid_at: hit
                    .valid_at
                    .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
                invalid_at: hit
                    .invalid_at
                    .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
                episodes: hit.episodes,
            })
            .collect())
    }
}

/// Options for [`Memory::traverse`].
#[napi(object)]
pub struct TraverseOptions {
    /// Maximum hops from the seed set (default 3).
    pub depth: Option<u32>,
    /// Node budget — the walk halts once this many nodes are reached
    /// (default 256).
    pub max_nodes: Option<u32>,
    /// Restrict to one group.
    pub group_id: Option<String>,
    /// Event-time instant (ISO-8601).
    pub as_of: Option<String>,
    /// Belief-time instant (ISO-8601) — the time-travel knob.
    pub as_at: Option<String>,
}
