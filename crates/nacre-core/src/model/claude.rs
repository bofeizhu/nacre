//! The real Claude API [`LanguageModel`] client (feature `claude`).
//!
//! Raw HTTP against `POST /v1/messages` (Rust has no official Anthropic
//! SDK). Structured JSON is enforced natively via the API's structured
//! outputs (`output_config.format` with a `json_schema`) — the recording
//! contract keys exchanges on the PRE-mutation prompt messages, so this
//! client does not replicate upstream Python's schema-append prompt
//! mutation; the mechanism producing schema-conforming JSON is a
//! client-internal detail.
//!
//! Never used by `cargo test` (AGENTS.md invariant 4): tests replay
//! recordings. Wrap this client in [`super::RecordingModel`] on capture
//! runs so its exchanges become replayable.

use serde_json::{Value, json};

use super::{CompletionRequest, LanguageModel, Message, ModelError, ModelSize, Role};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Upstream `LLMClient` default when the caller passes no max_tokens.
const DEFAULT_MAX_TOKENS: u32 = 16000;
const MAX_RETRIES: u32 = 2;

/// Configuration for [`ClaudeModel`].
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    /// API key (`sk-ant-...`).
    pub api_key: String,
    /// Model used for [`ModelSize::Medium`] requests.
    pub medium_model: String,
    /// Model used for [`ModelSize::Small`] requests.
    pub small_model: String,
}

impl ClaudeConfig {
    /// Defaults: Opus 4.8 for the default tier, Haiku 4.5 for the small
    /// tier (Graphiti routes cheap judgment calls to a smaller model).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            medium_model: "claude-opus-4-8".to_owned(),
            small_model: "claude-haiku-4-5".to_owned(),
        }
    }
}

/// A [`LanguageModel`] backed by the Claude API.
pub struct ClaudeModel {
    config: ClaudeConfig,
    client: reqwest::Client,
}

impl ClaudeModel {
    /// Build a client with the given configuration.
    pub fn new(config: ClaudeConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    fn model_for(&self, size: ModelSize) -> &str {
        match size {
            ModelSize::Small => &self.config.small_model,
            ModelSize::Medium => &self.config.medium_model,
        }
    }
}

/// Build the Messages API request body for a [`CompletionRequest`].
///
/// System messages map to the top-level `system` parameter; user/assistant
/// messages stay in `messages`. When the schema name is registered, the
/// response format is enforced with structured outputs.
pub fn build_request_body(request: &CompletionRequest, model: &str) -> Value {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for Message { role, content } in &request.messages {
        match role {
            Role::System => system_parts.push(content),
            Role::User => messages.push(json!({"role": "user", "content": content})),
            Role::Assistant => messages.push(json!({"role": "assistant", "content": content})),
        }
    }

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
    });
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
    if let Some(schema) = schema_for(&request.schema_name) {
        body["output_config"] = json!({
            "format": {"type": "json_schema", "schema": schema}
        });
    }
    body
}

/// JSON schema per response-schema name, mirroring `src/schemas.rs` (which
/// mirrors the pinned pydantic models). Only the top-level schemas the
/// pipeline actually requests are registered; an unregistered name gets no
/// output constraint (the model is prompted for JSON but not enforced).
pub fn schema_for(name: &str) -> Option<Value> {
    let no_extra = |properties: Value, required: Value| -> Value {
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    };
    let string_or_null = json!({"type": ["string", "null"]});
    let int_array = json!({"type": "array", "items": {"type": "integer"}});

    match name {
        "ExtractedEntities" => Some(no_extra(
            json!({
                "extracted_entities": {"type": "array", "items": no_extra(
                    json!({
                        "name": {"type": "string"},
                        "entity_type_id": {"type": "integer"},
                        "episode_indices": int_array,
                    }),
                    json!(["name", "entity_type_id", "episode_indices"]),
                )},
            }),
            json!(["extracted_entities"]),
        )),
        "ExtractedEdges" => Some(no_extra(
            json!({
                "edges": {"type": "array", "items": no_extra(
                    json!({
                        "source_entity_name": {"type": "string"},
                        "target_entity_name": {"type": "string"},
                        "relation_type": {"type": "string"},
                        "fact": {"type": "string"},
                        "valid_at": string_or_null,
                        "invalid_at": string_or_null,
                        "episode_indices": int_array,
                    }),
                    json!([
                        "source_entity_name", "target_entity_name", "relation_type",
                        "fact", "valid_at", "invalid_at", "episode_indices"
                    ]),
                )},
            }),
            json!(["edges"]),
        )),
        "EdgeTimestamps" => Some(no_extra(
            json!({"valid_at": string_or_null, "invalid_at": string_or_null}),
            json!(["valid_at", "invalid_at"]),
        )),
        "BatchEdgeTimestamps" => Some(no_extra(
            json!({
                "timestamps": {"type": "array", "items": no_extra(
                    json!({"valid_at": string_or_null, "invalid_at": string_or_null}),
                    json!(["valid_at", "invalid_at"]),
                )},
            }),
            json!(["timestamps"]),
        )),
        "NodeResolutions" => Some(no_extra(
            json!({
                "entity_resolutions": {"type": "array", "items": no_extra(
                    json!({
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "duplicate_candidate_id": {"type": "integer"},
                    }),
                    json!(["id", "name", "duplicate_candidate_id"]),
                )},
            }),
            json!(["entity_resolutions"]),
        )),
        "EdgeDuplicate" => Some(no_extra(
            json!({"duplicate_facts": int_array, "contradicted_facts": int_array}),
            json!(["duplicate_facts", "contradicted_facts"]),
        )),
        "SummarizedEntities" => Some(no_extra(
            json!({
                "summaries": {"type": "array", "items": no_extra(
                    json!({"name": {"type": "string"}, "summary": {"type": "string"}}),
                    json!(["name", "summary"]),
                )},
            }),
            json!(["summaries"]),
        )),
        "EntitySummary" | "Summary" => Some(no_extra(
            json!({"summary": {"type": "string"}}),
            json!(["summary"]),
        )),
        "SummaryDescription" => Some(no_extra(
            json!({"description": {"type": "string"}}),
            json!(["description"]),
        )),
        _ => None,
    }
}

/// Pull the structured JSON out of a Messages API response: the first text
/// block, parsed. Refusals and truncation surface as provider errors.
pub fn parse_response_body(body: &Value) -> Result<Value, ModelError> {
    if let Some(error) = body.get("error") {
        return Err(ModelError::Provider(format!("API error: {error}")));
    }
    match body["stop_reason"].as_str() {
        Some("refusal") => {
            return Err(ModelError::Provider(format!(
                "model refused: {:?}",
                body.get("stop_details")
            )));
        }
        Some("max_tokens") => {
            return Err(ModelError::Provider(
                "output truncated at max_tokens; raise CompletionRequest::max_tokens".to_owned(),
            ));
        }
        _ => {}
    }
    let text = body["content"]
        .as_array()
        .and_then(|blocks| {
            blocks
                .iter()
                .find(|b| b["type"] == "text")
                .and_then(|b| b["text"].as_str())
        })
        .ok_or_else(|| ModelError::Provider("response has no text block".to_owned()))?;
    serde_json::from_str(text)
        .map_err(|e| ModelError::Provider(format!("response text is not valid JSON: {e}")))
}

impl LanguageModel for ClaudeModel {
    async fn complete(&self, request: &CompletionRequest) -> Result<Value, ModelError> {
        let body = build_request_body(request, self.model_for(request.model_size));
        let mut last_error = String::new();
        for attempt in 0..=MAX_RETRIES {
            let response = self
                .client
                .post(API_URL)
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body)
                .send()
                .await;
            match response {
                Err(e) => last_error = format!("request failed: {e}"),
                Ok(response) => {
                    let status = response.status().as_u16();
                    let retryable = status == 429 || status == 408 || status >= 500;
                    let value: Value = response
                        .json()
                        .await
                        .map_err(|e| ModelError::Provider(format!("invalid response: {e}")))?;
                    if !retryable {
                        return parse_response_body(&value);
                    }
                    last_error = format!("HTTP {status}: {value}");
                }
            }
            if attempt < MAX_RETRIES {
                // Exponential backoff: 1s, 2s.
                tokio_sleep(1u64 << attempt).await;
            }
        }
        Err(ModelError::Provider(format!(
            "giving up after {} attempts: {last_error}",
            MAX_RETRIES + 1
        )))
    }
}

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

    fn request(schema: &str) -> CompletionRequest {
        CompletionRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: "You are an entity extraction specialist.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Extract entities.".into(),
                },
            ],
            schema_name: schema.into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        }
    }

    #[test]
    fn request_body_maps_roles_and_enforces_schema() {
        let body = build_request_body(&request("ExtractedEntities"), "claude-opus-4-8");
        assert_eq!(body["model"], "claude-opus-4-8");
        assert_eq!(body["max_tokens"], 16000);
        assert_eq!(body["system"], "You are an entity extraction specialist.");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(
            body["output_config"]["format"]["type"], "json_schema",
            "registered schemas are enforced via structured outputs"
        );
        assert_eq!(
            body["output_config"]["format"]["schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn unregistered_schema_gets_no_output_constraint() {
        let body = build_request_body(&request("SomethingCustom"), "claude-opus-4-8");
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn every_pipeline_schema_is_registered() {
        for name in [
            "ExtractedEntities",
            "ExtractedEdges",
            "EdgeTimestamps",
            "BatchEdgeTimestamps",
            "NodeResolutions",
            "EdgeDuplicate",
            "SummarizedEntities",
            "EntitySummary",
            "Summary",
            "SummaryDescription",
        ] {
            let schema = schema_for(name).unwrap_or_else(|| panic!("{name} unregistered"));
            assert_eq!(schema["additionalProperties"], false, "{name}");
        }
    }

    #[test]
    fn response_parsing_handles_success_refusal_and_truncation() {
        let ok = json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "{\"extracted_entities\": []}"}],
        });
        assert_eq!(
            parse_response_body(&ok).unwrap(),
            json!({"extracted_entities": []})
        );

        let refusal = json!({"stop_reason": "refusal", "stop_details": {"category": "cyber"}});
        assert!(matches!(
            parse_response_body(&refusal),
            Err(ModelError::Provider(_))
        ));

        let truncated = json!({"stop_reason": "max_tokens", "content": []});
        assert!(parse_response_body(&truncated).is_err());

        let api_error = json!({"type": "error", "error": {"type": "invalid_request_error"}});
        assert!(parse_response_body(&api_error).is_err());
    }
}
