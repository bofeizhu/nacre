//! Ports `graphiti_core/prompts/summarize_nodes.py` (pinned v0.29.2).
//!
//! Prompt text is byte-verbatim; fidelity is enforced against fixtures
//! rendered from the actual upstream Python — see `tests/prompt_fidelity.rs`.

use serde_json::Value;

use super::helpers::to_prompt_json;
use super::py::py_interp;
use super::snippets::summary_instructions;
use super::{MAX_SUMMARY_CHARS, msg};
use crate::model::{Message, Role};

// ports: graphiti_core/prompts/summarize_nodes.py::summarize_pair
pub fn summarize_pair(context: &Value) -> Vec<Message> {
    let sys_prompt =
        "You are a helpful assistant that combines summaries into a single dense factual summary.";

    let user_prompt = format!(
        r#"
        Synthesize the information from the following two summaries into a single information-dense summary.

        IMPORTANT:
        - Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
        - Prefer compact factual sentences over vague thematic phrasing.
        - When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        - Use communication verbs only when the act of speaking, asking, sharing, presenting, or announcing is itself the important fact.
        - Avoid filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", and "indicated" unless the communication act itself matters.
        - SUMMARIES MUST BE LESS THAN {MAX_SUMMARY_CHARS} CHARACTERS.

        Summaries:
        {node_summaries}
        "#,
        node_summaries = to_prompt_json(&context["node_summaries"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}

// ports: graphiti_core/prompts/summarize_nodes.py::summarize_context
pub fn summarize_context(context: &Value) -> Vec<Message> {
    let sys_prompt = "You are a helpful assistant that generates detailed, information-dense summaries and attributes from provided text.";

    let user_prompt = format!(
        r#"
        Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use
        information from the provided MESSAGES. Your summary should also only contain information relevant to the
        provided ENTITY.

        In addition, extract any values for the provided entity properties based on their descriptions.
        If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None.

        {summary_instructions}

        <MESSAGES>
        {previous_episodes}
        {episode_content}
        </MESSAGES>

        <ENTITY>
        {node_name}
        </ENTITY>

        <ENTITY CONTEXT>
        {node_summary}
        </ENTITY CONTEXT>

        <ATTRIBUTES>
        {attributes}
        </ATTRIBUTES>
        "#,
        summary_instructions = summary_instructions(),
        previous_episodes = to_prompt_json(&context["previous_episodes"]),
        episode_content = to_prompt_json(&context["episode_content"]),
        node_name = py_interp(&context["node_name"]),
        node_summary = py_interp(&context["node_summary"]),
        attributes = to_prompt_json(&context["attributes"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}

// ports: graphiti_core/prompts/summarize_nodes.py::summary_description
pub fn summary_description(context: &Value) -> Vec<Message> {
    let sys_prompt =
        "You are a helpful assistant that describes provided contents in a single sentence.";

    let user_prompt = format!(
        r#"
        Create a short one sentence description of the summary that explains what kind of information is summarized.
        Summaries must be under {MAX_SUMMARY_CHARS} characters.

        Summary:
        {summary}
        "#,
        summary = to_prompt_json(&context["summary"]),
    );
    vec![msg(Role::System, sys_prompt), msg(Role::User, user_prompt)]
}
