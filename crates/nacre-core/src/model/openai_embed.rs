//! OpenAI-compatible embeddings client (feature `openai-embed`).
//!
//! Speaks `POST {base_url}/embeddings` — the de-facto standard surface
//! served by OpenAI, Zhipu (bigmodel.cn, the oracle's embedding provider),
//! and most other vendors. Vectors are truncated client-side to the
//! configured dimension, matching Graphiti's `EMBEDDING_DIM` behavior for
//! MRL models like embedding-3 (slice, no renormalization).
//!
//! Never used by `cargo test` (AGENTS.md invariant 4): tests replay
//! recordings; the pure request-builder and response-parser below carry
//! the offline coverage. Wrap in a recording layer on capture runs.

use serde_json::{Value, json};

use super::{Embedder, EmbedderMeta, ModelError};

const MAX_RETRIES: u32 = 2;

/// Configuration for [`OpenAiEmbedder`].
#[derive(Debug, Clone)]
pub struct OpenAiEmbedConfig {
    /// API root, e.g. `https://open.bigmodel.cn/api/paas/v4` or
    /// `https://api.openai.com/v1` (no trailing slash; `/embeddings` is
    /// appended).
    pub base_url: String,
    /// Bearer token.
    pub api_key: String,
    /// Model identifier, e.g. `embedding-3`.
    pub model: String,
    /// Vectors are truncated to this many components (Graphiti's
    /// `EMBEDDING_DIM`; 1024 is a natively supported embedding-3 size).
    pub dim: u32,
    /// Provider model revision for grit's embedding identity; empty when
    /// not meaningful.
    pub model_version: String,
    /// Inputs per HTTP request; longer batches are chunked. Zhipu caps
    /// batches at 64, OpenAI at 2048 — the conservative default works
    /// everywhere.
    pub max_batch: usize,
}

impl OpenAiEmbedConfig {
    /// Zhipu embedding-3 defaults (the oracle's provider): 1024 dims,
    /// 64-input batches.
    pub fn zhipu(api_key: impl Into<String>) -> Self {
        Self {
            base_url: "https://open.bigmodel.cn/api/paas/v4".to_owned(),
            api_key: api_key.into(),
            model: "embedding-3".to_owned(),
            dim: 1024,
            model_version: String::new(),
            max_batch: 64,
        }
    }
}

/// An [`Embedder`] backed by an OpenAI-compatible `/embeddings` endpoint.
pub struct OpenAiEmbedder {
    config: OpenAiEmbedConfig,
    client: reqwest::Client,
}

impl OpenAiEmbedder {
    /// Build a client with the given configuration.
    pub fn new(config: OpenAiEmbedConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    async fn embed_chunk(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
        let body = build_request_body(&self.config.model, inputs);
        let url = format!("{}/embeddings", self.config.base_url);
        let mut attempt = 0;
        loop {
            let response = self
                .client
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| ModelError::Provider(e.to_string()))?;
            let status = response.status();
            if status.as_u16() == 429 || status.is_server_error() {
                if attempt < MAX_RETRIES {
                    attempt += 1;
                    super::tokio_sleep(1u64 << attempt).await;
                    continue;
                }
                return Err(ModelError::Provider(format!(
                    "embeddings endpoint returned {status} after {MAX_RETRIES} retries"
                )));
            }
            if !status.is_success() {
                // Error bodies are provider diagnostics; keys never appear.
                let text = response.text().await.unwrap_or_default();
                return Err(ModelError::Provider(format!(
                    "embeddings endpoint returned {status}: {text}"
                )));
            }
            let value: Value = response
                .json()
                .await
                .map_err(|e| ModelError::Provider(format!("embeddings response body: {e}")))?;
            return parse_response(&value, inputs.len(), self.config.dim as usize);
        }
    }
}

impl Embedder for OpenAiEmbedder {
    fn meta(&self) -> EmbedderMeta {
        EmbedderMeta {
            model_id: self.config.model.clone(),
            dim: self.config.dim,
            model_version: self.config.model_version.clone(),
        }
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(self.config.max_batch.max(1)) {
            out.extend(self.embed_chunk(chunk).await?);
        }
        Ok(out)
    }
}

/// The request body (pure — unit-tested offline).
fn build_request_body(model: &str, inputs: &[String]) -> Value {
    json!({ "model": model, "input": inputs })
}

/// Decode `{"data": [{"index": n, "embedding": [...]}, ...]}`: order by
/// `index` (the spec does not guarantee response order), truncate each
/// vector to `dim`, reject count or length mismatches loudly.
fn parse_response(value: &Value, expected: usize, dim: usize) -> Result<Vec<Vec<f32>>, ModelError> {
    let data = value["data"]
        .as_array()
        .ok_or_else(|| ModelError::Provider("embeddings response has no data array".into()))?;
    if data.len() != expected {
        return Err(ModelError::Provider(format!(
            "embeddings response has {} vectors for {} inputs",
            data.len(),
            expected
        )));
    }
    let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
    for item in data {
        let index = item["index"]
            .as_u64()
            .ok_or_else(|| ModelError::Provider("embedding item missing index".into()))?
            as usize;
        let raw = item["embedding"]
            .as_array()
            .ok_or_else(|| ModelError::Provider("embedding item missing vector".into()))?;
        if raw.len() < dim {
            return Err(ModelError::Provider(format!(
                "provider returned a {}-dim vector, need at least {dim}",
                raw.len()
            )));
        }
        let vector: Vec<f32> = raw
            .iter()
            .take(dim)
            .map(|v| v.as_f64().unwrap_or_default() as f32)
            .collect();
        indexed.push((index, vector));
    }
    indexed.sort_by_key(|(i, _)| *i);
    Ok(indexed.into_iter().map(|(_, v)| v).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_shape() {
        let body = build_request_body("embedding-3", &["a".into(), "b".into()]);
        assert_eq!(body["model"], "embedding-3");
        assert_eq!(body["input"], json!(["a", "b"]));
    }

    #[test]
    fn response_parses_sorted_and_truncated() {
        // Out-of-order response, 4-dim vectors truncated to 2.
        let value = json!({"data": [
            {"index": 1, "embedding": [10.0, 11.0, 12.0, 13.0]},
            {"index": 0, "embedding": [0.5, 0.25, 0.125, 0.0625]},
        ]});
        let vectors = parse_response(&value, 2, 2).unwrap();
        assert_eq!(vectors, vec![vec![0.5, 0.25], vec![10.0, 11.0]]);
    }

    #[test]
    fn response_mismatches_fail_loudly() {
        let short = json!({"data": [{"index": 0, "embedding": [1.0]}]});
        assert!(parse_response(&short, 1, 4).is_err(), "under-dimensioned");
        assert!(parse_response(&short, 2, 1).is_err(), "count mismatch");
        assert!(
            parse_response(&json!({"nope": []}), 0, 1).is_err(),
            "no data array"
        );
    }
}
