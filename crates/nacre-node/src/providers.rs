//! JS-configurable model/embedder providers.
//!
//! `LanguageModel`/`Embedder` use RPITIT (not dyn-compatible), so JS
//! configs dispatch through enums. `replay` mode serves recordings from a
//! path — the same JSON format as the oracle fixtures — making Node-side
//! tests offline and deterministic.

use nacre::model::claude::{ClaudeConfig, ClaudeModel, StructuredOutput};
use nacre::model::openai_embed::{OpenAiEmbedConfig, OpenAiEmbedder};
use nacre::model::{
    CompletionRequest, Embedder, EmbedderMeta, LanguageModel, ModelError, RecordingStore,
    ReplayEmbedder, ReplayModel,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// LLM configuration passed from JS.
#[napi(object)]
#[derive(Clone)]
pub struct LlmConfig {
    /// "anthropic" | "deepseek" | "replay"
    pub provider: String,
    /// API key (anthropic/deepseek).
    pub api_key: Option<String>,
    /// Recordings JSON path (replay).
    pub recordings_path: Option<String>,
    /// Override the API root (no trailing slash).
    pub base_url: Option<String>,
    /// Override the default-tier model id.
    pub medium_model: Option<String>,
    /// Override the small-tier model id.
    pub small_model: Option<String>,
}

/// Embedder configuration passed from JS.
#[napi(object)]
#[derive(Clone)]
pub struct EmbedderConfig {
    /// "zhipu" | "openai-compatible" | "replay"
    pub provider: String,
    /// API key (zhipu/openai-compatible).
    pub api_key: Option<String>,
    /// Recordings JSON path (replay).
    pub recordings_path: Option<String>,
    /// API root for openai-compatible (no trailing slash).
    pub base_url: Option<String>,
    /// Model id (also the identity vectors persist under).
    pub model: Option<String>,
    /// Vector dimension (default 1024).
    pub dim: Option<u32>,
}

pub enum AnyModel {
    Claude(ClaudeModel),
    Replay(ReplayModel),
}

impl LanguageModel for AnyModel {
    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> std::result::Result<serde_json::Value, ModelError> {
        match self {
            AnyModel::Claude(m) => m.complete(request).await,
            AnyModel::Replay(m) => m.complete(request).await,
        }
    }
}

pub enum AnyEmbedder {
    OpenAi(OpenAiEmbedder),
    Replay(ReplayEmbedder),
}

impl Embedder for AnyEmbedder {
    fn meta(&self) -> EmbedderMeta {
        match self {
            AnyEmbedder::OpenAi(e) => e.meta(),
            AnyEmbedder::Replay(e) => e.meta(),
        }
    }

    async fn embed(&self, inputs: &[String]) -> std::result::Result<Vec<Vec<f32>>, ModelError> {
        match self {
            AnyEmbedder::OpenAi(e) => e.embed(inputs).await,
            AnyEmbedder::Replay(e) => e.embed(inputs).await,
        }
    }
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

pub fn build_model(config: &LlmConfig) -> Result<AnyModel> {
    match config.provider.as_str() {
        "anthropic" | "deepseek" => {
            let key = config
                .api_key
                .clone()
                .ok_or_else(|| invalid("apiKey required for live providers"))?;
            let mut cc = if config.provider == "deepseek" {
                ClaudeConfig::deepseek(key)
            } else {
                ClaudeConfig::new(key)
            };
            if let Some(url) = &config.base_url {
                cc.base_url = url.clone();
                if config.provider == "anthropic" && *url != "https://api.anthropic.com" {
                    // A compatible endpoint may not support native
                    // structured outputs; prompt mode is the safe default.
                    cc.structured_output = StructuredOutput::SchemaInPrompt;
                }
            }
            if let Some(m) = &config.medium_model {
                cc.medium_model = m.clone();
            }
            if let Some(m) = &config.small_model {
                cc.small_model = m.clone();
            }
            Ok(AnyModel::Claude(ClaudeModel::new(cc)))
        }
        "replay" => {
            let path = config
                .recordings_path
                .clone()
                .ok_or_else(|| invalid("recordingsPath required for replay"))?;
            let store = RecordingStore::load(std::path::Path::new(&path))
                .map_err(|e| invalid(format!("loading recordings: {e}")))?;
            Ok(AnyModel::Replay(ReplayModel::new(store)))
        }
        other => Err(invalid(format!("unknown llm provider {other:?}"))),
    }
}

pub fn build_embedder(config: &EmbedderConfig) -> Result<AnyEmbedder> {
    let dim = config.dim.unwrap_or(1024);
    match config.provider.as_str() {
        "zhipu" | "openai-compatible" => {
            let key = config
                .api_key
                .clone()
                .ok_or_else(|| invalid("apiKey required for live providers"))?;
            let mut ec = OpenAiEmbedConfig::zhipu(key);
            if let Some(url) = &config.base_url {
                ec.base_url = url.clone();
            } else if config.provider == "openai-compatible" {
                return Err(invalid("baseUrl required for openai-compatible"));
            }
            if let Some(m) = &config.model {
                ec.model = m.clone();
            }
            ec.dim = dim;
            Ok(AnyEmbedder::OpenAi(OpenAiEmbedder::new(ec)))
        }
        "replay" => {
            let path = config
                .recordings_path
                .clone()
                .ok_or_else(|| invalid("recordingsPath required for replay"))?;
            let store = RecordingStore::load(std::path::Path::new(&path))
                .map_err(|e| invalid(format!("loading recordings: {e}")))?;
            Ok(AnyEmbedder::Replay(ReplayEmbedder::new(
                store,
                EmbedderMeta {
                    model_id: config.model.clone().unwrap_or("embedding-3".into()),
                    dim,
                    model_version: String::new(),
                },
            )))
        }
        other => Err(invalid(format!("unknown embedder provider {other:?}"))),
    }
}
