//! Swift bindings for nacre — the Layer 3 gateway for a native Apple host.
//!
//! Thin [UniFFI](https://mozilla.github.io/uniffi-rs/) glue over
//! `nacre-core` + `grit-core`; all logic lives in those crates. The host
//! app implements two protocols ([`LlmProvider`], [`EmbeddingProvider`])
//! and owns keys plus network — this crate performs no I/O of its own —
//! while every pipeline judgment (extraction, dedup, temporal
//! invalidation, summaries, retrieval fusion) stays in Rust. See
//! [`providers`] for the contract.
//!
//! Plain `cargo build -p nacre-swift` type-checks and compiles the
//! libraries with no Xcode or Swift toolchain present; the SwiftPM package
//! (staticlib slices → XCFramework → generated Swift sources) is a
//! separate build step, documented in this crate's README.
//!
//! Unlike the rest of the workspace this crate does **not** carry
//! `#![deny(unsafe_code)]`: UniFFI's proc macros generate the `unsafe`
//! `extern "C"` scaffolding that is the whole point of the crate.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use nacre_swift::{EmbedderSpec, Memory};
//! # async fn demo(llm: Arc<dyn nacre_swift::LlmProvider>,
//! #               embedder: Arc<dyn nacre_swift::EmbeddingProvider>)
//! #     -> Result<(), nacre_swift::NacreError> {
//! let memory = Memory::open("memory.db".into(), "my-device".into())?;
//! let spec = EmbedderSpec {
//!     model_id: "embedding-3".into(),
//!     dim: 1024,
//!     model_version: String::new(),
//! };
//! let hits = memory
//!     .search_edges("Where does Priya work?".into(), "chat".into(), 10, embedder, spec)
//!     .await?;
//! # let _ = (hits, llm);
//! # Ok(())
//! # }
//! ```

pub mod providers;
pub mod rows;

use std::sync::Arc;

use chrono::{DateTime, Utc};

use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::model::{Embedder, LanguageModel, ModelError};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, PipelineError, add_episode,
    retrieve_previous_episodes,
};
use nacre_core::search::search_edges;

pub use providers::{
    ChatMessage, ChatRequest, ChatRole, EmbedderSpec, EmbeddingProvider, LlmProvider, ModelTier,
};
pub use rows::{EdgeRow, EpisodeRow, NodeHistory, NodeRow, Subgraph};

use providers::{host_embedder, host_model, replay_embedder, replay_model};
use rows::{edge_row, episode_row, node_row};

uniffi::setup_scaffolding!();

/// Everything that can go wrong across the boundary. Swift sees an
/// `enum NacreError: Error` with associated values.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum NacreError {
    /// An argument did not parse (a malformed id, a timestamp without a
    /// UTC offset, an unknown episode source).
    #[error("invalid input: {message}")]
    InvalidInput {
        /// What was wrong, naming the offending field.
        message: String,
    },
    /// A host protocol failed, or kept failing until the pipeline gave up.
    #[error("provider: {message}")]
    Provider {
        /// The failure as reported by the host, with pipeline context.
        message: String,
    },
    /// A replayed model was asked something no recording answers. Offline
    /// replay refuses to guess; the canonical request is included so the
    /// diff against the recordings file is mechanical.
    #[error("no recording matches this request: {canonical_request}")]
    ReplayMiss {
        /// The unmatched request, canonically rendered.
        canonical_request: String,
    },
    /// Any other model-side failure (decode, recording I/O, format).
    #[error("model: {message}")]
    Model {
        /// The underlying model error, rendered.
        message: String,
    },
    /// A grit storage failure.
    #[error("storage: {message}")]
    Storage {
        /// The underlying grit error, rendered.
        message: String,
    },
}

/// MANDATORY for foreign traits: the generated callback scaffolding panics
/// without a conversion for errors a Swift implementation throws that are
/// not `NacreError` values (Swift can throw anything).
impl From<uniffi::UnexpectedUniFFICallbackError> for NacreError {
    fn from(e: uniffi::UnexpectedUniFFICallbackError) -> Self {
        NacreError::Provider {
            message: e.to_string(),
        }
    }
}

impl From<ModelError> for NacreError {
    fn from(e: ModelError) -> Self {
        match e {
            ModelError::ReplayMiss { canonical_request } => {
                NacreError::ReplayMiss { canonical_request }
            }
            // A host protocol that threw (or kept throwing until the
            // pipeline gave up) arrives here wrapped in
            // `ModelError::Provider` — keep it distinguishable from
            // nacre's own decode/recording failures, which is what the
            // host can actually act on.
            ModelError::Provider(message) => NacreError::Provider { message },
            other => NacreError::Model {
                message: other.to_string(),
            },
        }
    }
}

impl From<PipelineError> for NacreError {
    fn from(e: PipelineError) -> Self {
        match e {
            PipelineError::Model(e) => e.into(),
            // grit_core::Error is #[non_exhaustive]: render it.
            PipelineError::Storage(e) => NacreError::Storage {
                message: e.to_string(),
            },
        }
    }
}

impl From<grit_core::Error> for NacreError {
    fn from(e: grit_core::Error) -> Self {
        NacreError::Storage {
            message: e.to_string(),
        }
    }
}

/// One episode to ingest.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Episode {
    /// Short episode name (metadata only; not stored in the graph).
    pub name: String,
    /// Raw text of the episode.
    pub content: String,
    /// `"message"` (default) | `"text"` | `"json"` — picks the extraction
    /// prompt, and scopes the previous-episode window the pipeline reads.
    pub source: Option<String>,
    /// Where this came from, e.g. `"chat"` or `"doc:notes.md"`.
    pub source_description: String,
    /// Namespace; every query filters on it.
    pub group_id: String,
    /// RFC-3339 event time **with a UTC offset**. Also the pipeline's
    /// injected clock for this ingestion; defaults to the current system
    /// time.
    pub valid_at: Option<String>,
}

/// A `(from, into)` node merge the pipeline executed.
#[derive(Debug, Clone, uniffi::Record)]
pub struct MergePair {
    /// The node that was merged away (kept, expired, for audit).
    pub from: String,
    /// The canonical node it folded into.
    pub into: String,
}

/// What one ingestion changed — the UI animation feed.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AddEpisodeResult {
    /// Stored episode id.
    pub episode_id: String,
    /// Final node id per extracted entity (post-merge), in draft order.
    pub node_ids: Vec<String>,
    /// Edges created by this ingestion.
    pub new_edge_ids: Vec<String>,
    /// Node merges executed.
    pub merges: Vec<MergePair>,
    /// Stored edges this ingestion invalidated.
    pub invalidated_edge_ids: Vec<String>,
}

/// One search hit, in fused rank order.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SearchHit {
    /// Edge id.
    pub id: String,
    /// Source node id.
    pub source_id: String,
    /// Target node id.
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

/// Options for [`Memory::traverse`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct TraverseOptions {
    /// Maximum hops from the seed set (default 3).
    pub depth: Option<u32>,
    /// Node budget — the walk halts once this many nodes are reached
    /// (default 256).
    pub max_nodes: Option<u32>,
    /// Restrict to one group.
    pub group_id: Option<String>,
    /// Event-time instant (RFC-3339 with offset).
    pub as_of: Option<String>,
    /// Belief-time instant (RFC-3339 with offset) — the time-travel knob.
    pub as_at: Option<String>,
}

/// Parse an RFC-3339 timestamp, defaulting to now. The offset is
/// REQUIRED — a bare local time is ambiguous and the pipeline's clock
/// must not be guessed. `field` names the offending input in the error.
fn parse_time(iso: Option<&str>, field: &str) -> Result<DateTime<Utc>, NacreError> {
    match iso {
        None => Ok(Utc::now()),
        Some(t) => DateTime::parse_from_rfc3339(t)
            .map(|t| t.with_timezone(&Utc))
            .map_err(|e| NacreError::InvalidInput {
                message: format!("{field}: {e}"),
            }),
    }
}

fn parse_id(id: &str, field: &str) -> Result<uuid::Uuid, NacreError> {
    id.parse().map_err(|e| NacreError::InvalidInput {
        message: format!("{field} {id:?}: {e}"),
    })
}

fn episode_source(source: Option<&str>) -> Result<EpisodeSource, NacreError> {
    match source {
        None | Some("message") => Ok(EpisodeSource::Message),
        Some("text") => Ok(EpisodeSource::Text),
        Some("json") => Ok(EpisodeSource::Json),
        Some(other) => Err(NacreError::InvalidInput {
            message: format!("unknown episode source {other:?}"),
        }),
    }
}

/// A memory file: one grit graph plus the pipeline that writes it.
///
/// Safe to share across threads; grit serializes writes internally. Call
/// `addEpisode` **sequentially per group** all the same — the pipeline
/// reads the previous-episode window and then mutates the graph
/// (upstream Graphiti's prescribed queueing model), so concurrent
/// ingestion into one group would race on context, not on storage.
#[derive(uniffi::Object)]
pub struct Memory {
    grit: grit_core::Grit,
}

#[uniffi::export]
impl Memory {
    /// Open (or create) a memory file. `device_id` names this device in
    /// the op-log — keep it stable per installation.
    ///
    /// In-memory databases are rejected by grit: pass a real path (on
    /// iOS, something under the app container).
    #[uniffi::constructor]
    pub fn open(path: String, device_id: String) -> Result<Arc<Self>, NacreError> {
        let grit = grit_core::Grit::open(&path, grit_core::Options::new(device_id))?;
        Ok(Arc::new(Memory { grit }))
    }

    /// Ingest one episode through the full pipeline (extraction, dedup,
    /// temporal invalidation, summaries, embeddings), running every model
    /// call through the host's protocols. Previous-episode context comes
    /// from the graph itself. Returns the change set.
    pub async fn add_episode(
        &self,
        episode: Episode,
        llm: Arc<dyn LlmProvider>,
        embedder: Arc<dyn EmbeddingProvider>,
        embedder_spec: EmbedderSpec,
    ) -> Result<AddEpisodeResult, NacreError> {
        let model = host_model(llm);
        let embedder = host_embedder(embedder, embedder_spec);
        self.ingest(episode, &model, &embedder).await
    }

    /// Hybrid recall: the query is embedded through the host's protocol
    /// and fused (RRF) with keyword legs over the stored facts;
    /// currently-valid edges return in rank order with provenance.
    pub async fn search_edges(
        &self,
        query: String,
        group_id: String,
        limit: u32,
        embedder: Arc<dyn EmbeddingProvider>,
        embedder_spec: EmbedderSpec,
    ) -> Result<Vec<SearchHit>, NacreError> {
        let embedder = host_embedder(embedder, embedder_spec);
        self.recall(&query, &group_id, limit, &embedder).await
    }

    /// [`Memory::add_episode`] against recorded responses instead of live
    /// protocols — the offline test path. The recordings files are the
    /// same JSON the oracle captures (`oracle/fixtures/<trace>/`); an
    /// unrecorded request fails as [`NacreError::ReplayMiss`].
    pub async fn add_episode_replay(
        &self,
        episode: Episode,
        llm_recordings_path: String,
        embedder_recordings_path: String,
        embedder_spec: EmbedderSpec,
    ) -> Result<AddEpisodeResult, NacreError> {
        let model = replay_model(&llm_recordings_path)?;
        let embedder = replay_embedder(&embedder_recordings_path, embedder_spec)?;
        self.ingest(episode, &model, &embedder).await
    }

    /// [`Memory::search_edges`] against a recorded query embedding.
    pub async fn search_edges_replay(
        &self,
        query: String,
        group_id: String,
        limit: u32,
        recordings_path: String,
        embedder_spec: EmbedderSpec,
    ) -> Result<Vec<SearchHit>, NacreError> {
        let embedder = replay_embedder(&recordings_path, embedder_spec)?;
        self.recall(&query, &group_id, limit, &embedder).await
    }

    // ------------------------------------------------------------------
    // Read path. The calls below are the entire data contract the
    // memory-graph visualization needs: full dump (nodes_in_group /
    // edges_in_group / episodes_in_group), focus+context (traverse), and
    // drill-down (node_history + mentions_of). They are SYNCHRONOUS —
    // grit is an embedded library and these are single-digit-millisecond
    // SQLite reads — but they do touch the disk, so call them off the
    // main thread for large groups.
    // ------------------------------------------------------------------

    /// Every node in a group, id-ordered — full bi-temporal record
    /// including merged-away rows (filter `expired_at` for the live
    /// view).
    pub fn nodes_in_group(&self, group_id: String) -> Result<Vec<NodeRow>, NacreError> {
        Ok(self
            .grit
            .nodes_in_group(&group_id)?
            .iter()
            .map(node_row)
            .collect())
    }

    /// Every edge in a group, id-ordered — invalidated and expired rows
    /// included (that's the archaeology view).
    pub fn edges_in_group(&self, group_id: String) -> Result<Vec<EdgeRow>, NacreError> {
        Ok(self
            .grit
            .edges_in_group(&group_id)?
            .iter()
            .map(edge_row)
            .collect())
    }

    /// Every episode in a group, chronological.
    pub fn episodes_in_group(&self, group_id: String) -> Result<Vec<EpisodeRow>, NacreError> {
        Ok(self
            .grit
            .episodes_in_group(&group_id)?
            .iter()
            .map(episode_row)
            .collect())
    }

    /// Bounded neighborhood around seed nodes. `as_of` filters by event
    /// time, `as_at` by belief time — both RFC-3339 with offset; omit for
    /// "now". The node budget keeps the walk viewport-sized (default
    /// 256).
    pub fn traverse(
        &self,
        seeds: Vec<String>,
        options: Option<TraverseOptions>,
    ) -> Result<Subgraph, NacreError> {
        let seeds: Vec<uuid::Uuid> = seeds
            .iter()
            .map(|s| parse_id(s, "seed id"))
            .collect::<Result<_, _>>()?;
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
                traversal = traversal.as_of(parse_time(Some(as_of), "asOf")?.timestamp_millis());
            }
            if let Some(as_at) = options.as_at.as_deref() {
                traversal = traversal.as_at(parse_time(Some(as_at), "asAt")?.timestamp_millis());
            }
        }
        let sub = self.grit.traverse(&seeds, &traversal)?;
        Ok(Subgraph {
            nodes: sub.nodes.iter().map(node_row).collect(),
            edges: sub.edges.iter().map(edge_row).collect(),
        })
    }

    /// A node's bi-temporal audit trail: the node row (even if expired)
    /// plus every incident edge ever believed, oldest belief first.
    pub fn node_history(&self, id: String) -> Result<NodeHistory, NacreError> {
        let history = self.grit.node_history(parse_id(&id, "node id")?)?;
        Ok(NodeHistory {
            node: node_row(&history.node),
            edges: history.edges.iter().map(edge_row).collect(),
        })
    }

    /// Episode ids attributing a node or edge (provenance drill-down).
    pub fn mentions_of(&self, id: String) -> Result<Vec<String>, NacreError> {
        Ok(self
            .grit
            .mentions_of(parse_id(&id, "target id")?)?
            .iter()
            .map(|id| id.to_string())
            .collect())
    }

    /// The last `last_n` episodes of a group at `reference` (RFC-3339
    /// with offset; omit for now), chronological — the "what has been
    /// said so far" feed.
    ///
    /// Deliberately NOT filtered by source kind, unlike the window the
    /// pipeline itself assembles for extraction (upstream Graphiti scopes
    /// that one to the incoming episode's source). This is the display
    /// contract, not the prompt contract.
    pub fn previous_episodes(
        &self,
        group_id: String,
        reference: Option<String>,
        last_n: Option<u32>,
    ) -> Result<Vec<EpisodeRow>, NacreError> {
        let reference = parse_time(reference.as_deref(), "reference")?;
        let last_n = last_n.map_or(PREVIOUS_EPISODE_WINDOW, |n| n as usize);
        let mut episodes = self.grit.episodes_in_group(&group_id)?;
        episodes.retain(|e| e.occurred_at <= reference.timestamp_millis());
        let start = episodes.len().saturating_sub(last_n);
        Ok(episodes[start..].iter().map(episode_row).collect())
    }
}

impl Memory {
    /// The shared ingestion body: decode the episode, fetch the
    /// previous-episode window from the graph, run the pipeline, project
    /// the outcome. Mirrors `nacre-node`'s `addEpisode` exactly, so both
    /// hosts produce identical graphs from identical inputs.
    async fn ingest<M: LanguageModel, E: Embedder>(
        &self,
        episode: Episode,
        model: &M,
        embedder: &E,
    ) -> Result<AddEpisodeResult, NacreError> {
        let source = episode_source(episode.source.as_deref())?;
        let now = parse_time(episode.valid_at.as_deref(), "validAt")?;
        let input = EpisodeInput {
            name: episode.name,
            content: episode.content,
            source,
            source_description: episode.source_description,
            group_id: episode.group_id.clone(),
            // The raw input string rides through verbatim when present:
            // prompt and trace fidelity depend on its exact spelling.
            valid_at: episode
                .valid_at
                .clone()
                .or_else(|| Some(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false))),
        };
        let previous = retrieve_previous_episodes(
            &self.grit,
            &episode.group_id,
            source,
            now,
            PREVIOUS_EPISODE_WINDOW,
        )?;
        let outcome = add_episode(
            &self.grit,
            model,
            embedder,
            &input,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await?;
        Ok(AddEpisodeResult {
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
                .map(|(from, into)| MergePair {
                    from: from.to_string(),
                    into: into.to_string(),
                })
                .collect(),
            invalidated_edge_ids: outcome
                .invalidated_edge_ids
                .iter()
                .map(|id| id.to_string())
                .collect(),
        })
    }

    /// The shared recall body.
    async fn recall<E: Embedder>(
        &self,
        query: &str,
        group_id: &str,
        limit: u32,
        embedder: &E,
    ) -> Result<Vec<SearchHit>, NacreError> {
        let hits = search_edges(&self.grit, embedder, query, group_id, limit as usize).await?;
        Ok(hits
            .into_iter()
            .map(|hit| SearchHit {
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

/// The nacre-swift crate version (binding load smoke-check).
#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// grit rejects in-memory databases, so even a unit test needs a real
    /// file. Named per test; removed before use, not after, so a failure
    /// leaves the file to look at.
    fn temp_memory(name: &str) -> Arc<Memory> {
        let dir = std::env::temp_dir().join(format!("nacre-swift-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!("{name}.db"));
        let _ = std::fs::remove_file(&path);
        Memory::open(path.to_string_lossy().into_owned(), "unit-test".into()).expect("open")
    }

    #[test]
    fn parse_time_requires_an_offset() {
        let err = parse_time(Some("2026-07-14T00:00:00"), "validAt")
            .expect_err("a bare local time is ambiguous");
        match err {
            NacreError::InvalidInput { message } => assert!(
                message.starts_with("validAt: "),
                "the error names the field: {message}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        assert!(parse_time(Some("2026-07-14T00:00:00Z"), "validAt").is_ok());
        assert!(parse_time(Some("2026-07-14T02:00:00+02:00"), "validAt").is_ok());
        // No timestamp at all means "now", not an error.
        assert!(parse_time(None, "validAt").is_ok());
    }

    #[test]
    fn traverse_errors_name_the_offending_field() {
        let memory = temp_memory("traverse-errors");
        // A seed that is not a UUID is rejected before grit is touched, and
        // the message says which argument was wrong — `traverse` takes two
        // kinds of string, and "invalid input" alone would not tell a host
        // which one to fix.
        match memory
            .traverse(vec!["not-a-uuid".into()], None)
            .expect_err("garbage seed ids are rejected")
        {
            NacreError::InvalidInput { message } => {
                assert!(message.starts_with("seed id "), "{message}");
                assert!(message.contains("not-a-uuid"), "{message}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Same for the time-travel knobs: an instant without a UTC offset
        // is ambiguous, and `asOf` is not `asAt`.
        let options = TraverseOptions {
            depth: None,
            max_nodes: None,
            group_id: None,
            as_of: Some("2026-07-14T00:00:00".into()),
            as_at: None,
        };
        match memory
            .traverse(Vec::new(), Some(options))
            .expect_err("a bare local time is ambiguous")
        {
            NacreError::InvalidInput { message } => {
                assert!(message.starts_with("asOf: "), "{message}")
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn episode_source_decodes_the_three_kinds() {
        assert_eq!(episode_source(None).unwrap(), EpisodeSource::Message);
        assert_eq!(
            episode_source(Some("message")).unwrap(),
            EpisodeSource::Message
        );
        assert_eq!(episode_source(Some("text")).unwrap(), EpisodeSource::Text);
        assert_eq!(episode_source(Some("json")).unwrap(), EpisodeSource::Json);
        match episode_source(Some("audio")).expect_err("unknown source") {
            NacreError::InvalidInput { message } => assert!(message.contains("audio")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn replay_miss_survives_the_error_mapping() {
        let miss = PipelineError::Model(ModelError::ReplayMiss {
            canonical_request: "{\"messages\":[]}".into(),
        });
        match NacreError::from(miss) {
            NacreError::ReplayMiss { canonical_request } => {
                assert_eq!(canonical_request, "{\"messages\":[]}");
            }
            other => panic!("expected ReplayMiss, got {other:?}"),
        }
        // Host failures stay distinguishable from nacre's own.
        let provider = PipelineError::Model(ModelError::Provider("boom".into()));
        match NacreError::from(provider) {
            NacreError::Provider { message } => assert_eq!(message, "boom"),
            other => panic!("expected Provider, got {other:?}"),
        }
        // Everything else collapses into Model{message}.
        let decode = PipelineError::Model(ModelError::RecordingIo {
            path: "recordings.json".into(),
            source: std::io::Error::other("nope"),
        });
        match NacreError::from(decode) {
            NacreError::Model { message } => assert!(message.contains("recordings.json")),
            other => panic!("expected Model, got {other:?}"),
        }
    }

    #[test]
    fn unexpected_callback_errors_map_to_provider() {
        let err = NacreError::from(uniffi::UnexpectedUniFFICallbackError::new(
            "swift threw something else",
        ));
        match err {
            NacreError::Provider { message } => {
                assert!(message.contains("swift threw something else"), "{message}");
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }
}
