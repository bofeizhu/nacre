//! Transport-agnostic structured-output driver.
//!
//! [`PromptedModel`] owns everything about coaxing schema-shaped JSON out of
//! a plain chat endpoint — appending the schema block to the prompt,
//! stripping markdown fences, salvaging known bad shapes, validating, and
//! feeding validation errors back on retry — while delegating the actual
//! chat round to a [`ChatTransport`]. Hosts that relay LLM traffic through
//! their own network stack implement `ChatTransport` and keep all of the
//! Graphiti-ported judgment logic here in the library.

use std::future::Future;

use serde_json::Value;

use super::{CompletionRequest, LanguageModel, Message, ModelError, ModelSize, Role};

/// Wording of the appended schema request. Kept identical to the upstream
/// clients (openai_generic_client.py json_object mode); recordings depend on
/// the exact bytes.
pub(crate) const SCHEMA_PROMPT_HEADER: &str = "Respond with a JSON object in the following format:";

/// Completion budget when the request does not set one.
pub const DEFAULT_MAX_TOKENS: u32 = 16000;

const MAX_RETRIES: u32 = 2;

/// One plain chat round: prompt messages in, response text out. The only
/// thing a host has to provide — no schema handling, no retries.
pub trait ChatTransport: Send + Sync {
    /// Run one chat completion and return the model's text.
    fn chat(
        &self,
        messages: &[Message],
        max_tokens: u32,
        model_size: ModelSize,
    ) -> impl Future<Output = Result<String, ModelError>> + Send;
}

/// A [`LanguageModel`] over any [`ChatTransport`], using schema-in-prompt
/// structured output with the upstream correction loop.
pub struct PromptedModel<T: ChatTransport> {
    transport: T,
}

impl<T: ChatTransport> PromptedModel<T> {
    /// Wrap a transport.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}

impl<T: ChatTransport> LanguageModel for PromptedModel<T> {
    async fn complete(&self, request: &CompletionRequest) -> Result<Value, ModelError> {
        // `messages` is mutated across attempts: a failed schema validation
        // appends an error-context user message so the retry can correct
        // itself (blind retries at deterministic temperature just reproduce
        // the same bad shape). Mirrors ClaudeModel's SchemaInPrompt path.
        let mut messages = request.messages.clone();
        if let Some(schema) = schema_for(&request.schema_name)
            && let Some(last) = messages.last_mut()
        {
            last.content = format!("{}\n\n{SCHEMA_PROMPT_HEADER}\n\n{schema}", last.content);
        }
        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let mut last_error = String::new();
        for attempt in 0..=MAX_RETRIES {
            match self
                .transport
                .chat(&messages, max_tokens, request.model_size)
                .await
            {
                Err(e) => last_error = format!("transport failed: {e}"),
                Ok(text) => {
                    let parsed: Value =
                        serde_json::from_str(strip_code_fences(&text)).map_err(|e| {
                            ModelError::Provider(format!("response text is not valid JSON: {e}"))
                        })?;
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
                            messages.push(Message {
                                role: Role::User,
                                content: error_context,
                            });
                        }
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

/// JSON schema per response-schema name, mirroring `src/schemas.rs` (which
/// mirrors the pinned pydantic models). Only the top-level schemas the
/// pipeline actually requests are registered; an unregistered name gets no
/// output constraint (the model is prompted for JSON but not enforced).
pub fn schema_for(name: &str) -> Option<Value> {
    use serde_json::json;
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

/// Trim a ```json ... ``` (or bare ```) fence if the whole payload is
/// wrapped in one.
// ports: openai_generic_client.py::_strip_code_fences
pub(crate) fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// The top-level `required` keys of a registered schema, for cheap
/// response-shape validation in schema-in-prompt mode (native mode is
/// enforced by the API and needs none).
pub(crate) fn required_keys(schema_name: &str) -> Vec<String> {
    schema_for(schema_name)
        .and_then(|s| {
            s["required"].as_array().map(|keys| {
                keys.iter()
                    .filter_map(|k| k.as_str().map(str::to_owned))
                    .collect()
            })
        })
        .unwrap_or_default()
}

/// Salvage known non-conforming shapes prompt-only models produce: an
/// object nested under `"properties"` (the model echoed the schema
/// scaffolding — observed from deepseek-chat during oracle capture).
pub(crate) fn salvage_shape(value: Value, required: &[String]) -> Value {
    let conforms = |v: &Value| -> bool { required.iter().all(|k| v.get(k.as_str()).is_some()) };
    if conforms(&value) {
        return value;
    }
    if let Some(inner) = value.get("properties")
        && conforms(inner)
    {
        return inner.clone();
    }
    value
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    fn request(schema: &str) -> CompletionRequest {
        CompletionRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: "You are an entity summarizer.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Summarize.".into(),
                },
            ],
            schema_name: schema.into(),
            max_tokens: None,
            model_size: ModelSize::Medium,
        }
    }

    /// Scripted transport: pops responses front-first and records the
    /// message lists it was called with.
    struct ScriptedTransport {
        responses: Mutex<Vec<Result<String, ModelError>>>,
        calls: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedTransport {
        fn new(responses: Vec<Result<String, ModelError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl ChatTransport for ScriptedTransport {
        async fn chat(
            &self,
            messages: &[Message],
            _max_tokens: u32,
            _model_size: ModelSize,
        ) -> Result<String, ModelError> {
            self.calls.lock().unwrap().push(messages.to_vec());
            self.responses.lock().unwrap().remove(0)
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        // The retry backoff polls a channel with yield_now, so a trivial
        // busy-poll executor is enough here.
        use std::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut future = Box::pin(future);
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return output;
            }
        }
    }

    #[test]
    fn appends_schema_to_last_message() {
        let transport = ScriptedTransport::new(vec![Ok(r#"{"summary": "ok"}"#.into())]);
        let model = PromptedModel::new(transport);
        let value = block_on(model.complete(&request("EntitySummary"))).unwrap();
        assert_eq!(value, json!({"summary": "ok"}));
        let calls = model.transport.calls.lock().unwrap();
        let last = &calls[0].last().unwrap().content;
        assert!(last.starts_with("Summarize."));
        assert!(last.contains(SCHEMA_PROMPT_HEADER));
    }

    #[test]
    fn feeds_validation_error_back_and_retries() {
        let transport = ScriptedTransport::new(vec![
            Ok(r#"{"wrong": 1}"#.into()),
            Ok(r#"{"summary": "fixed"}"#.into()),
        ]);
        let model = PromptedModel::new(transport);
        let value = block_on(model.complete(&request("EntitySummary"))).unwrap();
        assert_eq!(value, json!({"summary": "fixed"}));
        let calls = model.transport.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        let correction = &calls[1].last().unwrap();
        assert_eq!(correction.role, Role::User);
        assert!(correction.content.contains("ValidationError"));
    }

    #[test]
    fn salvages_schema_scaffolding_echo() {
        let transport =
            ScriptedTransport::new(vec![Ok(r#"{"properties": {"summary": "nested"}}"#.into())]);
        let model = PromptedModel::new(transport);
        let value = block_on(model.complete(&request("EntitySummary"))).unwrap();
        assert_eq!(value, json!({"summary": "nested"}));
    }

    #[test]
    fn strips_markdown_fences() {
        let transport =
            ScriptedTransport::new(vec![Ok("```json\n{\"summary\": \"ok\"}\n```".into())]);
        let model = PromptedModel::new(transport);
        let value = block_on(model.complete(&request("EntitySummary"))).unwrap();
        assert_eq!(value, json!({"summary": "ok"}));
    }

    #[test]
    fn invalid_json_is_a_hard_error() {
        let transport = ScriptedTransport::new(vec![Ok("not json".into())]);
        let model = PromptedModel::new(transport);
        let error = block_on(model.complete(&request("EntitySummary"))).unwrap_err();
        assert!(error.to_string().contains("not valid JSON"));
    }

    #[test]
    fn transport_errors_exhaust_retries() {
        let transport = ScriptedTransport::new(vec![
            Err(ModelError::Provider("boom 1".into())),
            Err(ModelError::Provider("boom 2".into())),
            Err(ModelError::Provider("boom 3".into())),
        ]);
        let model = PromptedModel::new(transport);
        let error = block_on(model.complete(&request("EntitySummary"))).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("giving up after 3 attempts"), "{message}");
        assert!(message.contains("boom 3"), "{message}");
    }
}
