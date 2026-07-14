//! JS-configurable model/embedder providers.
//!
//! `LanguageModel`/`Embedder` use RPITIT (not dyn-compatible), so JS
//! configs dispatch through enums. `replay` mode serves recordings from a
//! path — the same JSON format as the oracle fixtures — making Node-side
//! tests offline and deterministic. `host` mode makes NO network requests
//! from the addon at all: the host app supplies `chat`/`embed` callbacks
//! and owns the transport (nacre keeps the schema/retry judgment logic in
//! Rust). The reqwest-backed `anthropic`/`deepseek`/`zhipu` providers only
//! exist behind the off-by-default `live-providers` feature, for the
//! opt-in live smoke scripts.

#[cfg(feature = "live-providers")]
use nacre_core::model::claude::{ClaudeConfig, ClaudeModel, StructuredOutput};
#[cfg(feature = "live-providers")]
use nacre_core::model::openai_embed::{OpenAiEmbedConfig, OpenAiEmbedder};
use nacre_core::model::prompted::{ChatTransport, PromptedModel};
use nacre_core::model::{
    CompletionRequest, Embedder, EmbedderMeta, LanguageModel, Message, ModelError, ModelSize,
    RecordingStore, ReplayEmbedder, ReplayModel, Role,
};
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;

/// One prompt message handed to the host `chat` callback.
#[napi(object)]
#[derive(Clone)]
pub struct HostChatMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    /// The message text (schema block already appended where needed).
    pub content: String,
}

/// One chat round handed to the host `chat` callback. The callback runs the
/// completion over the host's own transport and resolves with the model's
/// raw response text; nacre parses/validates/retries.
#[napi(object)]
#[derive(Clone)]
pub struct HostChatRequest {
    /// The full prompt, in order.
    pub messages: Vec<HostChatMessage>,
    /// Completion budget the pipeline wants.
    pub max_tokens: u32,
    /// "small" | "medium" — tier routing hint.
    pub model_size: String,
}

type ChatCallback =
    ThreadsafeFunction<HostChatRequest, Promise<String>, HostChatRequest, Status, false>;
type EmbedCallback =
    ThreadsafeFunction<Vec<String>, Promise<Vec<Vec<f64>>>, Vec<String>, Status, false>;

/// LLM configuration passed from JS.
#[napi(object, object_to_js = false)]
pub struct LlmConfig {
    /// "host" | "replay" | (live-providers builds only) "anthropic" | "deepseek"
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
    /// Host transport (host): runs one chat completion, resolves the raw
    /// response text.
    #[napi(ts_type = "(request: HostChatRequest) => Promise<string>")]
    pub chat: Option<ChatCallback>,
}

/// Embedder configuration passed from JS.
#[napi(object, object_to_js = false)]
pub struct EmbedderConfig {
    /// "host" | "replay" | (live-providers builds only) "zhipu" | "openai-compatible"
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
    /// Host transport (host): embeds a batch, resolves one vector per
    /// input in input order.
    #[napi(ts_type = "(inputs: Array<string>) => Promise<Array<Array<number>>>")]
    pub embed: Option<EmbedCallback>,
}

/// [`ChatTransport`] over a host-supplied JS callback.
pub struct HostChatTransport {
    callback: ChatCallback,
}

impl ChatTransport for HostChatTransport {
    async fn chat(
        &self,
        messages: &[Message],
        max_tokens: u32,
        model_size: ModelSize,
    ) -> std::result::Result<String, ModelError> {
        let request = HostChatRequest {
            messages: messages
                .iter()
                .map(|m| HostChatMessage {
                    role: match m.role {
                        Role::System => "system".to_owned(),
                        Role::User => "user".to_owned(),
                        Role::Assistant => "assistant".to_owned(),
                    },
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens,
            model_size: match model_size {
                ModelSize::Small => "small".to_owned(),
                ModelSize::Medium => "medium".to_owned(),
            },
        };
        let promise = self
            .callback
            .call_async(request)
            .await
            .map_err(|e| ModelError::Provider(format!("host chat callback: {e}")))?;
        promise
            .await
            .map_err(|e| ModelError::Provider(format!("host chat callback rejected: {e}")))
    }
}

/// [`Embedder`] over a host-supplied JS callback.
pub struct HostEmbedder {
    callback: EmbedCallback,
    meta: EmbedderMeta,
}

impl Embedder for HostEmbedder {
    fn meta(&self) -> EmbedderMeta {
        self.meta.clone()
    }

    async fn embed(&self, inputs: &[String]) -> std::result::Result<Vec<Vec<f32>>, ModelError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let promise = self
            .callback
            .call_async(inputs.to_vec())
            .await
            .map_err(|e| ModelError::Provider(format!("host embed callback: {e}")))?;
        let vectors = promise
            .await
            .map_err(|e| ModelError::Provider(format!("host embed callback rejected: {e}")))?;
        if vectors.len() != inputs.len() {
            return Err(ModelError::Provider(format!(
                "host embed callback returned {} vectors for {} inputs",
                vectors.len(),
                inputs.len()
            )));
        }
        let dim = self.meta.dim as usize;
        vectors
            .into_iter()
            .map(|vector| {
                if vector.len() < dim {
                    return Err(ModelError::Provider(format!(
                        "host embed callback returned a {}-dim vector; expected at least {dim}",
                        vector.len()
                    )));
                }
                // MRL slice to the configured dim, matching OpenAiEmbedder.
                Ok(vector.into_iter().take(dim).map(|x| x as f32).collect())
            })
            .collect()
    }
}

pub enum AnyModel {
    #[cfg(feature = "live-providers")]
    Claude(ClaudeModel),
    Host(PromptedModel<HostChatTransport>),
    Replay(ReplayModel),
}

impl LanguageModel for AnyModel {
    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> std::result::Result<serde_json::Value, ModelError> {
        match self {
            #[cfg(feature = "live-providers")]
            AnyModel::Claude(m) => m.complete(request).await,
            AnyModel::Host(m) => m.complete(request).await,
            AnyModel::Replay(m) => m.complete(request).await,
        }
    }
}

pub enum AnyEmbedder {
    #[cfg(feature = "live-providers")]
    OpenAi(OpenAiEmbedder),
    Host(HostEmbedder),
    Replay(ReplayEmbedder),
}

impl Embedder for AnyEmbedder {
    fn meta(&self) -> EmbedderMeta {
        match self {
            #[cfg(feature = "live-providers")]
            AnyEmbedder::OpenAi(e) => e.meta(),
            AnyEmbedder::Host(e) => e.meta(),
            AnyEmbedder::Replay(e) => e.meta(),
        }
    }

    async fn embed(&self, inputs: &[String]) -> std::result::Result<Vec<Vec<f32>>, ModelError> {
        match self {
            #[cfg(feature = "live-providers")]
            AnyEmbedder::OpenAi(e) => e.embed(inputs).await,
            AnyEmbedder::Host(e) => e.embed(inputs).await,
            AnyEmbedder::Replay(e) => e.embed(inputs).await,
        }
    }
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

pub fn build_model(config: LlmConfig) -> Result<AnyModel> {
    match config.provider.as_str() {
        "host" => {
            let callback = config
                .chat
                .ok_or_else(|| invalid("chat callback required for host provider"))?;
            Ok(AnyModel::Host(PromptedModel::new(HostChatTransport {
                callback,
            })))
        }
        #[cfg(feature = "live-providers")]
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
        #[cfg(not(feature = "live-providers"))]
        "anthropic" | "deepseek" => Err(invalid(
            "this build ships without networked providers; use provider \"host\" \
             (or rebuild with the live-providers feature)",
        )),
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

pub fn build_embedder(config: EmbedderConfig) -> Result<AnyEmbedder> {
    let dim = config.dim.unwrap_or(1024);
    match config.provider.as_str() {
        "host" => {
            let callback = config
                .embed
                .ok_or_else(|| invalid("embed callback required for host provider"))?;
            Ok(AnyEmbedder::Host(HostEmbedder {
                callback,
                meta: EmbedderMeta {
                    model_id: config.model.clone().unwrap_or("embedding-3".into()),
                    dim,
                    model_version: String::new(),
                },
            }))
        }
        #[cfg(feature = "live-providers")]
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
        #[cfg(not(feature = "live-providers"))]
        "zhipu" | "openai-compatible" => Err(invalid(
            "this build ships without networked providers; use provider \"host\" \
             (or rebuild with the live-providers feature)",
        )),
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
