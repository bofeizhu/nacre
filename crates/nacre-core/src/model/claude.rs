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

use super::prompted::{SCHEMA_PROMPT_HEADER, required_keys, salvage_shape, strip_code_fences};
use super::{CompletionRequest, LanguageModel, Message, ModelError, ModelSize, Role};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Upstream `LLMClient` default when the caller passes no max_tokens.
const DEFAULT_MAX_TOKENS: u32 = 16000;
const MAX_RETRIES: u32 = 2;

/// How schema conformance is obtained from the endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutput {
    /// Anthropic-native `output_config.format` json_schema enforcement.
    JsonSchema,
    /// Append the schema to the last prompt message and trust the model —
    /// upstream's `OpenAIGenericClient` json_object strategy, for
    /// Anthropic-COMPATIBLE endpoints (e.g. DeepSeek's) that speak the
    /// Messages API but not structured outputs. A client-internal mutation:
    /// the recording contract keys on pre-mutation messages.
    SchemaInPrompt,
}

/// Configuration for [`ClaudeModel`].
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    /// API root (no trailing slash; `/v1/messages` is appended). Defaults
    /// to `https://api.anthropic.com`.
    pub base_url: String,
    /// API key (`sk-ant-...` for Anthropic; provider-specific otherwise).
    pub api_key: String,
    /// Model used for [`ModelSize::Medium`] requests.
    pub medium_model: String,
    /// Model used for [`ModelSize::Small`] requests.
    pub small_model: String,
    /// Schema conformance mechanism.
    pub structured_output: StructuredOutput,
}

impl ClaudeConfig {
    /// Anthropic defaults: Opus 4.8 for the default tier, Haiku 4.5 for the
    /// small tier (Graphiti routes cheap judgment calls to a smaller
    /// model), native structured outputs.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            api_key: api_key.into(),
            medium_model: "claude-opus-4-8".to_owned(),
            small_model: "claude-haiku-4-5".to_owned(),
            structured_output: StructuredOutput::JsonSchema,
        }
    }

    /// DeepSeek's Anthropic-style endpoint: same wire format, no native
    /// structured outputs — the schema rides in the prompt instead.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self {
            base_url: "https://api.deepseek.com/anthropic".to_owned(),
            api_key: api_key.into(),
            medium_model: "deepseek-chat".to_owned(),
            small_model: "deepseek-chat".to_owned(),
            structured_output: StructuredOutput::SchemaInPrompt,
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
/// response format is enforced with structured outputs
/// ([`StructuredOutput::JsonSchema`]) or requested via a schema block
/// appended to the last message ([`StructuredOutput::SchemaInPrompt`]).
pub fn build_request_body(
    request: &CompletionRequest,
    model: &str,
    mode: StructuredOutput,
) -> Value {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for Message { role, content } in &request.messages {
        match role {
            Role::System => system_parts.push(content),
            Role::User => messages.push(json!({"role": "user", "content": content})),
            Role::Assistant => messages.push(json!({"role": "assistant", "content": content})),
        }
    }

    let schema = schema_for(&request.schema_name);
    if mode == StructuredOutput::SchemaInPrompt
        && let Some(schema) = &schema
        && let Some(last) = messages.last_mut()
    {
        // ports: openai_generic_client.py json_object mode — the schema is
        // appended to the last message; wording kept identical.
        let appended = format!(
            "{}\n\n{SCHEMA_PROMPT_HEADER}\n\n{}",
            last["content"].as_str().unwrap_or_default(),
            schema
        );
        last["content"] = json!(appended);
    }

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
    });
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
    if mode == StructuredOutput::JsonSchema
        && let Some(schema) = schema
    {
        body["output_config"] = json!({
            "format": {"type": "json_schema", "schema": schema}
        });
    }
    body
}

pub use super::prompted::schema_for;

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
    // Compatible endpoints in schema-in-prompt mode often wrap JSON in a
    // markdown fence; native structured outputs never do. Stripping is
    // harmless either way.
    serde_json::from_str(strip_code_fences(text))
        .map_err(|e| ModelError::Provider(format!("response text is not valid JSON: {e}")))
}

impl LanguageModel for ClaudeModel {
    async fn complete(&self, request: &CompletionRequest) -> Result<Value, ModelError> {
        // `body` is mutated across attempts in SchemaInPrompt mode: a failed
        // schema validation appends an error-context user message so the
        // retry can correct itself (blind retries at deterministic
        // temperature just reproduce the same bad shape).
        let mut body = build_request_body(
            request,
            self.model_for(request.model_size),
            self.config.structured_output,
        );
        let url = format!("{}/v1/messages", self.config.base_url);
        let mut last_error = String::new();
        for attempt in 0..=MAX_RETRIES {
            let response = self
                .client
                .post(&url)
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
                        let parsed = parse_response_body(&value)?;
                        // Prompt-only schema conformance is best-effort:
                        // salvage known bad shapes, then validate the FULL
                        // response model (key presence alone misses
                        // wrong-typed values). On a miss, feed the error
                        // back as a user message and burn a retry — the
                        // upstream clients' correction loop.
                        // ports: openai_base_client.py::_generate_response_with_retry
                        if self.config.structured_output == StructuredOutput::SchemaInPrompt {
                            let required = required_keys(&request.schema_name);
                            let parsed = salvage_shape(parsed, &required);
                            match crate::schemas::validate_response(&request.schema_name, &parsed) {
                                Ok(()) => return Ok(parsed),
                                Err(e) => {
                                    last_error = format!(
                                        "response did not validate against schema {}: {e}",
                                        request.schema_name
                                    );
                                    let error_context = format!(
                                        "The previous response attempt was invalid. \
                                         Error type: ValidationError. \
                                         Error details: {e}. \
                                         Please try again with a valid response, ensuring the \
                                         output matches the expected format and constraints."
                                    );
                                    if let Some(messages) = body["messages"].as_array_mut() {
                                        messages.push(
                                            json!({"role": "user", "content": error_context}),
                                        );
                                    }
                                }
                            }
                        } else {
                            return Ok(parsed);
                        }
                    } else {
                        last_error = format!("HTTP {status}: {value}");
                    }
                }
            }
            if attempt < MAX_RETRIES {
                // Exponential backoff: 1s, 2s.
                super::tokio_sleep(1u64 << attempt).await;
            }
        }
        Err(ModelError::Provider(format!(
            "giving up after {} attempts: {last_error}",
            MAX_RETRIES + 1
        )))
    }
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
        let body = build_request_body(
            &request("ExtractedEntities"),
            "claude-opus-4-8",
            StructuredOutput::JsonSchema,
        );
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
        let body = build_request_body(
            &request("SomethingCustom"),
            "claude-opus-4-8",
            StructuredOutput::JsonSchema,
        );
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
    #[test]
    fn schema_in_prompt_mode_appends_to_last_message() {
        let body = build_request_body(
            &request("ExtractedEntities"),
            "deepseek-chat",
            StructuredOutput::SchemaInPrompt,
        );
        assert!(body.get("output_config").is_none(), "no native enforcement");
        let last = body["messages"].as_array().unwrap().last().unwrap();
        let content = last["content"].as_str().unwrap();
        assert!(content.contains("Respond with a JSON object in the following format:"));
        assert!(content.contains("extracted_entities"));
    }

    #[test]
    fn code_fences_are_stripped() {
        let fenced = json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "```json\n{\"a\": 1}\n```"}],
        });
        assert_eq!(parse_response_body(&fenced).unwrap(), json!({"a": 1}));
    }
}
