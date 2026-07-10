//! Node.js bindings for nacre — the Layer 3 gateway.
//!
//! Thin `#[napi]` glue over `nacre-core` + `grit-core`; all logic lives in
//! those crates. Built into a loadable `.node` addon by `@napi-rs/cli`
//! (`npm run build` in this directory); plain `cargo build -p nacre-node`
//! type-checks and compiles the cdylib without any Node toolchain.

mod providers;

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use providers::{EmbedderConfig, LlmConfig, build_embedder, build_model};

use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, add_episode, retrieve_previous_episodes,
};

/// The nacre-node crate version (addon load smoke-check).
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// One episode to ingest.
#[napi(object)]
pub struct EpisodeJs {
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

fn generic(e: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
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
        let input = EpisodeInput {
            name: episode.name,
            content: episode.content,
            source,
            source_description: episode.source_description,
            group_id: episode.group_id.clone(),
            valid_at: episode
                .valid_at
                .clone()
                .or_else(|| Some(now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false))),
        };
        let model = build_model(&llm)?;
        let embedder = build_embedder(&embedder)?;
        let previous =
            retrieve_previous_episodes(&self.grit, &episode.group_id, now, PREVIOUS_EPISODE_WINDOW)
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
        .map_err(generic)?;
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
}
