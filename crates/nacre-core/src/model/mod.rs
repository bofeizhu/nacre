//! The LLM and embedding seams — the only places I/O happens in nacre.
//!
//! Everything between two model calls is a pure function (AGENTS.md
//! invariant 3), and every call through these traits is recordable and
//! replayable (invariant 4): tests replay recordings through [`ReplayModel`]
//! and [`ReplayEmbedder`] and never touch the network — a replay type that
//! is asked something no recording answers fails loudly instead of guessing
//! or calling out.

#[cfg(feature = "claude")]
pub mod claude;
#[cfg(feature = "openai-embed")]
pub mod openai_embed;
mod recording;

pub use recording::{
    Recording, RecordingModel, RecordingStore, ReplayEmbedder, ReplayModel, canonical_json,
};

use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Errors from model calls and the recording/replay plumbing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A replayed model was asked something no recording answers. In an
    /// offline test this almost always means the code under test built a
    /// different request than the one captured; the canonical request is
    /// included so the diff against the recording file is mechanical.
    #[error(
        "no recording matches this request (offline replay refuses to guess): {canonical_request}"
    )]
    ReplayMiss {
        /// The unmatched request, rendered with [`canonical_json`].
        canonical_request: String,
    },
    /// A model response did not decode into the expected schema.
    #[error("response did not match schema `{schema_name}`: {source}")]
    Decode {
        /// Name of the schema the response was expected to satisfy.
        schema_name: String,
        #[source]
        source: serde_json::Error,
    },
    /// Reading or writing a recordings file failed.
    #[error("recording I/O at {path}: {source}")]
    RecordingIo {
        /// The file involved.
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// A recordings file did not parse as a JSON array of recordings.
    #[error("recordings file {path} is not valid: {source}")]
    RecordingFormat {
        /// The file involved.
        path: String,
        #[source]
        source: serde_json::Error,
    },
    /// A real provider failed (network, auth, rate limiting). Never
    /// produced by the replay types.
    #[error("provider error: {0}")]
    Provider(String),
}

/// Chat role. Serializes lowercase, matching what the oracle harness records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System / instruction message.
    System,
    /// User-authored message.
    User,
    /// Model-authored message.
    Assistant,
}

/// One chat message in a completion request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message text.
    pub content: String,
}

/// Model tier for a pipeline step — mirrors Graphiti's `ModelSize`, which
/// routes cheap judgment calls (e.g. dedup) to a smaller model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    /// The cheaper/faster tier.
    Small,
    /// The default tier.
    #[default]
    Medium,
}

/// A structured-output completion request: messages in, JSON matching a
/// named schema out.
///
/// The serialized form of this struct is the identity of a recording —
/// see [`RecordingStore`]. Keep changes to its fields deliberate: renaming
/// a field invalidates every existing recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The prompt, as an ordered message list.
    pub messages: Vec<Message>,
    /// Name of the expected response schema. Mirrors Graphiti's
    /// `response_model` class name and is part of the recording identity.
    pub schema_name: String,
    /// Optional completion budget; omitted from the recording key when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Which model tier the step wants.
    #[serde(default)]
    pub model_size: ModelSize,
}

/// A language model that answers [`CompletionRequest`]s with structured JSON.
///
/// Implementations: [`ReplayModel`] (offline, test workhorse),
/// [`RecordingModel`] (wraps another model and logs every exchange), and —
/// feature-gated, later — a real Claude API client.
pub trait LanguageModel: Send + Sync {
    /// Complete `request` into a JSON value matching `request.schema_name`.
    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> impl Future<Output = Result<Value, ModelError>> + Send;
}

/// Identity of an embedding model, mirroring grit's `(model_id, dim,
/// model_version)` embedding metadata — grit tags stored vectors with this
/// so devices know when to re-embed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderMeta {
    /// Provider-scoped model identifier.
    pub model_id: String,
    /// Vector dimensionality.
    pub dim: u32,
    /// Provider model revision, if meaningful; empty string otherwise.
    pub model_version: String,
}

/// A batched text-embedding model.
pub trait Embedder: Send + Sync {
    /// The identity grit should tag vectors produced by this embedder with.
    fn meta(&self) -> EmbedderMeta;

    /// Embed a batch of inputs; the result has one vector per input, in
    /// input order.
    fn embed(
        &self,
        inputs: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, ModelError>> + Send;
}

/// Decode a model's JSON response into the typed schema `T`, attributing
/// failures to `schema_name`.
///
/// # Example
///
/// ```
/// use nacre_core::model::decode_response;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Extracted {
///     names: Vec<String>,
/// }
///
/// let value = serde_json::json!({ "names": ["Yoneda lemma"] });
/// let typed: Extracted = decode_response(value, "Extracted").unwrap();
/// assert_eq!(typed.names, ["Yoneda lemma"]);
/// ```
pub fn decode_response<T: serde::de::DeserializeOwned>(
    response: Value,
    schema_name: &str,
) -> Result<T, ModelError> {
    serde_json::from_value(response).map_err(|source| ModelError::Decode {
        schema_name: schema_name.to_owned(),
        source,
    })
}

#[cfg(any(feature = "claude", feature = "openai-embed"))]
/// Minimal async sleep without a tokio dependency in the library: reqwest
/// already requires a tokio runtime, but the `time` feature may be absent —
/// spawn the wait on a blocking thread.
async fn tokio_sleep(secs: u64) {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(secs));
        let _ = tx.send(());
    });
    // Poll the channel without blocking the async executor.
    loop {
        match rx.try_recv() {
            Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            Err(std::sync::mpsc::TryRecvError::Empty) => yield_now().await,
        }
    }
}

#[cfg(any(feature = "claude", feature = "openai-embed"))]
async fn yield_now() {
    struct YieldOnce(bool);
    impl std::future::Future for YieldOnce {
        type Output = ();
        fn poll(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            if self.0 {
                std::task::Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
        }
    }
    YieldOnce(false).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_response_attributes_schema_on_failure() {
        let err = decode_response::<Vec<String>>(json!({"not": "a list"}), "NameList")
            .expect_err("shape mismatch must fail");
        match err {
            ModelError::Decode { schema_name, .. } => assert_eq!(schema_name, "NameList"),
            other => panic!("expected Decode error, got {other:?}"),
        }
    }

    #[test]
    fn completion_request_key_omits_unset_max_tokens() {
        let request = CompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: "hi".into(),
            }],
            schema_name: "X".into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("max_tokens").is_none());
        assert_eq!(value["model_size"], "medium");
        assert_eq!(value["messages"][0]["role"], "user");
    }
}
