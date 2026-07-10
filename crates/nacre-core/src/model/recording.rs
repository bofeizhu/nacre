//! Recording and replay of model calls.
//!
//! Recordings are stored as human-readable JSON arrays of
//! `{request, response}` pairs and matched by **canonical request JSON**
//! (object keys sorted recursively) rather than by hash — a miss can be
//! diffed against the file by eye, and the Python oracle harness can write
//! the same format with `json.dumps(..., sort_keys=True)`.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{CompletionRequest, Embedder, EmbedderMeta, LanguageModel, ModelError};

/// Render `value` as canonical JSON: object keys sorted recursively, arrays
/// kept in order, compact separators. Two structurally equal values always
/// canonicalize to the same string, regardless of key insertion order.
///
/// # Example
///
/// ```
/// use nacre_core::model::canonical_json;
/// use serde_json::json;
///
/// let value = json!({"b": 1, "a": {"d": [ {"z": 1, "y": 2} ]}});
/// assert_eq!(canonical_json(&value), r#"{"a":{"d":[{"y":2,"z":1}]},"b":1}"#);
/// ```
pub fn canonical_json(value: &Value) -> String {
    fn canon(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut sorted = serde_json::Map::new();
                for key in keys {
                    sorted.insert(key.clone(), canon(&map[key]));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(canon).collect()),
            other => other.clone(),
        }
    }
    canon(value).to_string()
}

/// One recorded model exchange. `request` is the serialized form of the
/// request (for LLM calls, a [`CompletionRequest`]); `response` is the raw
/// JSON the model returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    /// The request, in its serialized form.
    pub request: Value,
    /// The response the model gave.
    pub response: Value,
}

/// An immutable set of recordings, indexed by canonical request JSON.
///
/// Later recordings win when two share a request — the capture layer appends,
/// so the last exchange reflects the freshest response.
pub struct RecordingStore {
    by_request: HashMap<String, Value>,
}

impl RecordingStore {
    /// Build a store from recordings already in memory.
    pub fn new(recordings: impl IntoIterator<Item = Recording>) -> Self {
        let by_request = recordings
            .into_iter()
            .map(|recording| (canonical_json(&recording.request), recording.response))
            .collect();
        Self { by_request }
    }

    /// Load a store from a JSON file containing an array of recordings.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(|source| ModelError::RecordingIo {
            path: path.display().to_string(),
            source,
        })?;
        let recordings: Vec<Recording> =
            serde_json::from_str(&text).map_err(|source| ModelError::RecordingFormat {
                path: path.display().to_string(),
                source,
            })?;
        Ok(Self::new(recordings))
    }

    /// Look up the recorded response for a request, by structural equality.
    pub fn get(&self, request: &Value) -> Option<Value> {
        self.by_request.get(&canonical_json(request)).cloned()
    }

    /// Number of distinct recorded requests.
    pub fn len(&self) -> usize {
        self.by_request.len()
    }

    /// Whether the store holds no recordings.
    pub fn is_empty(&self) -> bool {
        self.by_request.is_empty()
    }
}

/// A [`LanguageModel`] that only serves recordings — the offline test
/// workhorse. A request with no matching recording is an error
/// ([`ModelError::ReplayMiss`]), never a network call.
///
/// # Example
///
/// ```
/// use nacre_core::model::{
///     CompletionRequest, LanguageModel, Message, ModelSize, Recording, RecordingStore,
///     ReplayModel, Role,
/// };
/// use serde_json::json;
///
/// let request = CompletionRequest {
///     messages: vec![Message { role: Role::User, content: "extract".into() }],
///     schema_name: "ExtractedEntities".into(),
///     max_tokens: None,
///     model_size: ModelSize::Medium,
/// };
/// let recording = Recording {
///     request: serde_json::to_value(&request).unwrap(),
///     response: json!({"extracted_entities": []}),
/// };
/// let model = ReplayModel::new(RecordingStore::new([recording]));
///
/// let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
/// let response = runtime.block_on(model.complete(&request)).unwrap();
/// assert_eq!(response, json!({"extracted_entities": []}));
/// ```
pub struct ReplayModel {
    store: RecordingStore,
}

impl ReplayModel {
    /// Replay from an in-memory store.
    pub fn new(store: RecordingStore) -> Self {
        Self { store }
    }

    /// Replay from a recordings file (see [`RecordingStore::load`]).
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        Ok(Self::new(RecordingStore::load(path)?))
    }
}

impl LanguageModel for ReplayModel {
    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> impl Future<Output = Result<Value, ModelError>> + Send {
        // A plain struct of strings and enums always serializes.
        let request_value =
            serde_json::to_value(request).expect("CompletionRequest serialization is infallible");
        let result = match self.store.get(&request_value) {
            Some(response) => Ok(response),
            None => Err(ModelError::ReplayMiss {
                canonical_request: canonical_json(&request_value),
            }),
        };
        async move { result }
    }
}

/// An [`Embedder`] that only serves recordings. Embedding recordings use
/// `{"inputs": [...], "model_id": "..."}` as their request form and an array
/// of vectors as the response.
pub struct ReplayEmbedder {
    store: RecordingStore,
    meta: EmbedderMeta,
}

impl ReplayEmbedder {
    /// Replay from an in-memory store, reporting `meta` as the embedder
    /// identity.
    pub fn new(store: RecordingStore, meta: EmbedderMeta) -> Self {
        Self { store, meta }
    }

    /// The request form under which an embedding exchange is recorded.
    pub fn request_value(inputs: &[String], model_id: &str) -> Value {
        serde_json::json!({ "inputs": inputs, "model_id": model_id })
    }
}

impl Embedder for ReplayEmbedder {
    fn meta(&self) -> EmbedderMeta {
        self.meta.clone()
    }

    fn embed(
        &self,
        inputs: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, ModelError>> + Send {
        let request_value = Self::request_value(inputs, &self.meta.model_id);
        let result = match self.store.get(&request_value) {
            Some(response) => {
                serde_json::from_value(response).map_err(|source| ModelError::Decode {
                    schema_name: "embeddings".to_owned(),
                    source,
                })
            }
            None => Err(ModelError::ReplayMiss {
                canonical_request: canonical_json(&request_value),
            }),
        };
        async move { result }
    }
}

/// Wraps another [`LanguageModel`] and logs every exchange, so a capture run
/// against a real model produces recordings that [`ReplayModel`] can serve
/// later. This is the recording layer AGENTS.md invariant 4 requires every
/// model call to flow through.
pub struct RecordingModel<M> {
    inner: M,
    log: Mutex<Vec<Recording>>,
}

impl<M> RecordingModel<M> {
    /// Wrap `inner`, starting with an empty log.
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            log: Mutex::new(Vec::new()),
        }
    }

    /// A snapshot of everything recorded so far.
    pub fn recordings(&self) -> Vec<Recording> {
        self.log
            .lock()
            .expect("recording log lock poisoned")
            .clone()
    }

    /// Write the recordings to `path` as pretty-printed JSON.
    pub fn save(&self, path: &Path) -> Result<(), ModelError> {
        let recordings = self.recordings();
        let text = serde_json::to_string_pretty(&recordings)
            .expect("recordings are plain JSON values and always serialize");
        std::fs::write(path, text).map_err(|source| ModelError::RecordingIo {
            path: path.display().to_string(),
            source,
        })
    }
}

impl<M: LanguageModel> LanguageModel for RecordingModel<M> {
    async fn complete(&self, request: &CompletionRequest) -> Result<Value, ModelError> {
        let response = self.inner.complete(request).await?;
        let request_value =
            serde_json::to_value(request).expect("CompletionRequest serialization is infallible");
        self.log
            .lock()
            .expect("recording log lock poisoned")
            .push(Recording {
                request: request_value,
                response: response.clone(),
            });
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Message, ModelSize, Role};
    use super::*;
    use serde_json::json;

    /// A stub model that answers every request with a fixed value.
    struct FixedModel(Value);

    impl LanguageModel for FixedModel {
        fn complete(
            &self,
            _request: &CompletionRequest,
        ) -> impl Future<Output = Result<Value, ModelError>> + Send {
            let response = Ok(self.0.clone());
            async move { response }
        }
    }

    fn request(content: &str) -> CompletionRequest {
        CompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: content.into(),
            }],
            schema_name: "X".into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        }
    }

    #[test]
    fn canonical_json_is_stable_under_key_order() {
        let value = json!({"b": 1, "a": {"d": 2, "c": [{"z": 1, "y": 2}]}});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"c":[{"y":2,"z":1}],"d":2},"b":1}"#
        );
    }

    #[tokio::test]
    async fn replay_hits_regardless_of_recording_key_order() {
        // Same request as `request("hi")`, but with object keys written in a
        // deliberately different order than serde emits.
        let recording = Recording {
            request: json!({
                "schema_name": "X",
                "model_size": "medium",
                "messages": [{"content": "hi", "role": "user"}],
            }),
            response: json!({"ok": true}),
        };
        let model = ReplayModel::new(RecordingStore::new([recording]));
        let response = model.complete(&request("hi")).await.unwrap();
        assert_eq!(response, json!({"ok": true}));
    }

    #[tokio::test]
    async fn replay_miss_fails_loudly() {
        let model = ReplayModel::new(RecordingStore::new([]));
        let err = model.complete(&request("unrecorded")).await.unwrap_err();
        match err {
            ModelError::ReplayMiss { canonical_request } => {
                assert!(canonical_request.contains("unrecorded"));
            }
            other => panic!("expected ReplayMiss, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn record_then_replay_round_trips() {
        let live = RecordingModel::new(FixedModel(json!({"answer": 42})));
        let first = live.complete(&request("one")).await.unwrap();
        let second = live.complete(&request("two")).await.unwrap();
        assert_eq!(first, second);

        let replay = ReplayModel::new(RecordingStore::new(live.recordings()));
        assert_eq!(
            replay.complete(&request("one")).await.unwrap(),
            json!({"answer": 42})
        );
        assert_eq!(
            replay.complete(&request("two")).await.unwrap(),
            json!({"answer": 42})
        );
        // But nothing beyond what was recorded.
        assert!(replay.complete(&request("three")).await.is_err());
    }

    #[tokio::test]
    async fn replay_embedder_round_trips_and_reports_meta() {
        let meta = EmbedderMeta {
            model_id: "test-embed".into(),
            dim: 3,
            model_version: String::new(),
        };
        let inputs = vec!["alpha".to_owned(), "beta".to_owned()];
        let recording = Recording {
            request: ReplayEmbedder::request_value(&inputs, "test-embed"),
            response: json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        };
        let embedder = ReplayEmbedder::new(RecordingStore::new([recording]), meta.clone());
        assert_eq!(embedder.meta(), meta);

        let vectors = embedder.embed(&inputs).await.unwrap();
        assert_eq!(vectors, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);

        let miss = embedder.embed(&["gamma".to_owned()]).await.unwrap_err();
        assert!(matches!(miss, ModelError::ReplayMiss { .. }));
    }

    #[tokio::test]
    async fn store_save_and_load_round_trip() {
        let live = RecordingModel::new(FixedModel(json!({"saved": true})));
        live.complete(&request("persist me")).await.unwrap();

        let path =
            std::env::temp_dir().join(format!("nacre-recordings-test-{}.json", std::process::id()));
        live.save(&path).unwrap();

        let replay = ReplayModel::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            replay.complete(&request("persist me")).await.unwrap(),
            json!({"saved": true})
        );
    }
}
