//! Host-fulfilled model I/O: the Swift protocols and their Rust adapters.
//!
//! A native Apple host cannot ship API keys and cannot fork a sidecar, so
//! the adopting app owns the transport: it implements two protocols
//! ([`LlmProvider`], [`EmbeddingProvider`]) and nacre keeps every piece of
//! pipeline judgment — schema-in-prompt structured output, fence
//! stripping, shape salvage, validation and the correction-retry loop
//! ([`nacre_core::model::prompted::PromptedModel`]) — in Rust. This crate
//! therefore performs no network I/O of its own and compiles none of
//! nacre-core's optional HTTP clients.
//!
//! `replay` mode serves recorded responses from a path — the same JSON
//! format as the oracle fixtures — which is what makes the Rust-side and
//! Swift-side tests offline and deterministic.
//!
//! Mirrors `nacre-node/src/providers.rs`: `LanguageModel`/`Embedder` use
//! RPITIT and so are not dyn-compatible, hence the dispatch enums.

use std::sync::Arc;

use nacre_core::model::prompted::{ChatTransport, PromptedModel};
use nacre_core::model::{
    CompletionRequest, Embedder, EmbedderMeta, LanguageModel, Message, ModelError, ModelSize,
    RecordingStore, ReplayEmbedder, ReplayModel, Role,
};

use crate::NacreError;

/// Who authored one prompt message. Mirrors
/// [`nacre_core::model::Role`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ChatRole {
    /// System / instruction message.
    System,
    /// User-authored message.
    User,
    /// Model-authored message.
    Assistant,
}

/// Model tier a pipeline step asks for — a routing hint hosts may ignore.
/// Mirrors [`nacre_core::model::ModelSize`]; cheap judgment calls (dedup,
/// summaries) ask for [`ModelTier::Small`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ModelTier {
    /// The cheaper/faster tier.
    Small,
    /// The default tier.
    Medium,
}

/// One message in a chat request.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct ChatMessage {
    /// Who authored it.
    pub role: ChatRole,
    /// The message text.
    pub content: String,
}

/// One chat round for the host to run.
///
/// The prompt is **post-mutation**: the JSON schema block nacre appends is
/// already part of the last message, and a retry after a validation
/// failure arrives with the error-context message appended. Hosts relay
/// the messages as-is.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct ChatRequest {
    /// The full prompt, in order.
    pub messages: Vec<ChatMessage>,
    /// Completion budget the pipeline wants (default 16000 — clamp it if
    /// your relay enforces a lower ceiling).
    pub max_tokens: u32,
    /// Which tier the step would like.
    pub tier: ModelTier,
}

/// Identity of the host's embedding model.
///
/// grit tags every stored vector with this triple, so it MUST stay stable
/// for a given memory file: changing `model_id` or `dim` between runs
/// makes previously persisted vectors unusable (grit fails loudly on a
/// dimension change rather than mixing vector spaces).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct EmbedderSpec {
    /// Provider-scoped model identifier — also the identity vectors
    /// persist under.
    #[uniffi(default = "embedding-3")]
    pub model_id: String,
    /// Vector dimensionality; longer vectors are MRL-truncated to it.
    #[uniffi(default = 1024)]
    pub dim: u32,
    /// Provider model revision, if meaningful; empty otherwise.
    #[uniffi(default = "")]
    pub model_version: String,
}

impl From<EmbedderSpec> for EmbedderMeta {
    fn from(spec: EmbedderSpec) -> Self {
        EmbedderMeta {
            model_id: spec.model_id,
            dim: spec.dim,
            model_version: spec.model_version,
        }
    }
}

/// The host's LLM transport.
///
/// Implemented in Swift (an `actor` works — the method is `async` and
/// throwing); nacre calls it whenever a pipeline step needs a completion.
#[uniffi::export(rust, foreign)]
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Run ONE chat completion over the host's own transport and return
    /// the model's raw response text.
    ///
    /// MUST NOT parse, validate, or retry — nacre owns that judgment.
    /// Idempotency matters: a response that fails schema validation is
    /// retried (up to 3 attempts) with the validation error fed back.
    async fn chat(&self, request: ChatRequest) -> Result<String, NacreError>;
}

/// The host's embedding transport.
#[uniffi::export(rust, foreign)]
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch and return exactly one vector per input, in input
    /// order.
    ///
    /// Vectors longer than the configured [`EmbedderSpec::dim`] are
    /// MRL-truncated by nacre; shorter ones are an error. Never called
    /// with an empty batch.
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f64>>, NacreError>;
}

/// [`ChatTransport`] over a host-implemented [`LlmProvider`].
pub(crate) struct SwiftChatTransport {
    provider: Arc<dyn LlmProvider>,
}

impl ChatTransport for SwiftChatTransport {
    async fn chat(
        &self,
        messages: &[Message],
        max_tokens: u32,
        model_size: ModelSize,
    ) -> Result<String, ModelError> {
        let request = ChatRequest {
            messages: messages
                .iter()
                .map(|m| ChatMessage {
                    role: match m.role {
                        Role::System => ChatRole::System,
                        Role::User => ChatRole::User,
                        Role::Assistant => ChatRole::Assistant,
                    },
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens,
            tier: match model_size {
                ModelSize::Small => ModelTier::Small,
                ModelSize::Medium => ModelTier::Medium,
            },
        };
        self.provider
            .chat(request)
            .await
            .map_err(|e| ModelError::Provider(format!("host chat provider: {e}")))
    }
}

/// [`Embedder`] over a host-implemented [`EmbeddingProvider`].
pub(crate) struct HostEmbedder {
    provider: Arc<dyn EmbeddingProvider>,
    meta: EmbedderMeta,
}

impl Embedder for HostEmbedder {
    fn meta(&self) -> EmbedderMeta {
        self.meta.clone()
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
        // Short-circuit before crossing the FFI: the protocol promises the
        // host is never called with an empty batch.
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self
            .provider
            .embed(inputs.to_vec())
            .await
            .map_err(|e| ModelError::Provider(format!("host embedding provider: {e}")))?;
        if vectors.len() != inputs.len() {
            return Err(ModelError::Provider(format!(
                "host embedding provider returned {} vectors for {} inputs",
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
                        "host embedding provider returned a {}-dim vector; expected at least {dim}",
                        vector.len()
                    )));
                }
                // MRL slice to the configured dim, matching OpenAiEmbedder.
                Ok(vector.into_iter().take(dim).map(|x| x as f32).collect())
            })
            .collect()
    }
}

/// The language models this crate can drive: host transport, or recorded
/// responses. (RPITIT traits are not dyn-compatible — hence an enum.)
pub(crate) enum AnyModel {
    Host(PromptedModel<SwiftChatTransport>),
    Replay(ReplayModel),
}

impl LanguageModel for AnyModel {
    async fn complete(&self, request: &CompletionRequest) -> Result<serde_json::Value, ModelError> {
        match self {
            AnyModel::Host(m) => m.complete(request).await,
            AnyModel::Replay(m) => m.complete(request).await,
        }
    }
}

/// The embedders this crate can drive: host transport, or recorded
/// responses.
pub(crate) enum AnyEmbedder {
    Host(HostEmbedder),
    Replay(ReplayEmbedder),
}

impl Embedder for AnyEmbedder {
    fn meta(&self) -> EmbedderMeta {
        match self {
            AnyEmbedder::Host(e) => e.meta(),
            AnyEmbedder::Replay(e) => e.meta(),
        }
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
        match self {
            AnyEmbedder::Host(e) => e.embed(inputs).await,
            AnyEmbedder::Replay(e) => e.embed(inputs).await,
        }
    }
}

/// Wrap a host protocol in the schema/retry driver.
pub(crate) fn host_model(provider: Arc<dyn LlmProvider>) -> AnyModel {
    AnyModel::Host(PromptedModel::new(SwiftChatTransport { provider }))
}

/// Wrap a host protocol as the pipeline's embedder.
pub(crate) fn host_embedder(
    provider: Arc<dyn EmbeddingProvider>,
    spec: EmbedderSpec,
) -> AnyEmbedder {
    AnyEmbedder::Host(HostEmbedder {
        provider,
        meta: spec.into(),
    })
}

fn load_recordings(path: &str) -> Result<RecordingStore, NacreError> {
    RecordingStore::load(std::path::Path::new(path)).map_err(|e| NacreError::InvalidInput {
        message: format!("loading recordings from {path:?}: {e}"),
    })
}

/// A model serving recorded completions; an unrecorded request fails
/// loudly as [`NacreError::ReplayMiss`] instead of calling out.
pub(crate) fn replay_model(recordings_path: &str) -> Result<AnyModel, NacreError> {
    Ok(AnyModel::Replay(ReplayModel::new(load_recordings(
        recordings_path,
    )?)))
}

/// An embedder serving recorded vectors.
pub(crate) fn replay_embedder(
    recordings_path: &str,
    spec: EmbedderSpec,
) -> Result<AnyEmbedder, NacreError> {
    Ok(AnyEmbedder::Replay(ReplayEmbedder::new(
        load_recordings(recordings_path)?,
        spec.into(),
    )))
}
