//! Byte-fidelity tests for the prompt ports.
//!
//! Fixtures under `tests/fixtures/prompts/` are rendered from the *actual*
//! upstream Python (pinned v0.29.2) by `oracle/promptgen/
//! gen_prompt_fixtures.py`. The Rust port replays the same context and must
//! reproduce every message byte-for-byte. Offline: fixtures are committed.

use nacre_core::model::{Message, Role};
use nacre_core::prompts::{dedupe_edges, dedupe_nodes, extract_edges, extract_nodes};
use serde_json::Value;

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn assert_case_matches(function: &str, case_name: &str, got: &[Message], expected: &Value) {
    let expected = expected.as_array().expect("messages array");
    assert_eq!(
        got.len(),
        expected.len(),
        "{function}/{case_name}: message count"
    );
    for (i, (message, want)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            role_str(message.role),
            want["role"].as_str().unwrap(),
            "{function}/{case_name}: message {i} role"
        );
        let want_content = want["content"].as_str().unwrap();
        if message.content != want_content {
            // Locate the first divergence to make transcription slips
            // mechanical to fix.
            let byte = message
                .content
                .bytes()
                .zip(want_content.bytes())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| message.content.len().min(want_content.len()));
            let lo = byte.saturating_sub(80);
            panic!(
                "{function}/{case_name}: message {i} content diverges at byte {byte}\n\
                 got:  {:?}\n\
                 want: {:?}",
                &message.content[lo..(byte + 80).min(message.content.len())],
                &want_content[lo..(byte + 80).min(want_content.len())],
            );
        }
    }
}

#[test]
fn extract_nodes_prompts_match_pinned_python() {
    let cases: Value = serde_json::from_str(include_str!("fixtures/prompts/extract_nodes.json"))
        .expect("fixture parses");

    let mut seen = 0;
    for case in cases.as_array().expect("fixture is an array") {
        let function = case["function"].as_str().unwrap();
        let case_name = case["case"].as_str().unwrap();
        let context = &case["context"];
        let got = match function {
            "extract_message" => extract_nodes::extract_message(context),
            "extract_json" => extract_nodes::extract_json(context),
            "extract_text" => extract_nodes::extract_text(context),
            "classify_nodes" => extract_nodes::classify_nodes(context),
            "extract_attributes" => extract_nodes::extract_attributes(context),
            "extract_summary" => extract_nodes::extract_summary(context),
            "extract_summaries_batch" => extract_nodes::extract_summaries_batch(context),
            "extract_entity_summaries_from_episodes" => {
                extract_nodes::extract_entity_summaries_from_episodes(context)
            }
            other => panic!("fixture references unported function {other}"),
        };
        assert_case_matches(function, case_name, &got, &case["messages"]);
        seen += 1;
    }
    // Every function in the family, including the optional-section variants.
    assert_eq!(seen, 11, "fixture case count");
}

#[test]
fn extract_edges_prompts_match_pinned_python() {
    let cases: Value = serde_json::from_str(include_str!("fixtures/prompts/extract_edges.json"))
        .expect("fixture parses");

    let mut seen = 0;
    for case in cases.as_array().expect("fixture is an array") {
        let function = case["function"].as_str().unwrap();
        let case_name = case["case"].as_str().unwrap();
        let context = &case["context"];
        let got = match function {
            "edge" => extract_edges::edge(context),
            "extract_attributes" => extract_edges::extract_attributes(context),
            "extract_timestamps" => extract_edges::extract_timestamps(context),
            "extract_timestamps_batch" => extract_edges::extract_timestamps_batch(context),
            other => panic!("fixture references unported function {other}"),
        };
        assert_case_matches(function, case_name, &got, &case["messages"]);
        seen += 1;
    }
    assert_eq!(seen, 5, "fixture case count");
}

#[test]
fn dedupe_prompts_match_pinned_python() {
    let node_cases: Value =
        serde_json::from_str(include_str!("fixtures/prompts/dedupe_nodes.json"))
            .expect("fixture parses");
    let edge_cases: Value =
        serde_json::from_str(include_str!("fixtures/prompts/dedupe_edges.json"))
            .expect("fixture parses");

    let mut seen = 0;
    for case in node_cases
        .as_array()
        .unwrap()
        .iter()
        .chain(edge_cases.as_array().unwrap())
    {
        let function = case["function"].as_str().unwrap();
        let case_name = case["case"].as_str().unwrap();
        let context = &case["context"];
        let got = match function {
            "node" => dedupe_nodes::node(context),
            "nodes" => dedupe_nodes::nodes(context),
            "node_list" => dedupe_nodes::node_list(context),
            "resolve_edge" => dedupe_edges::resolve_edge(context),
            other => panic!("fixture references unported function {other}"),
        };
        assert_case_matches(function, case_name, &got, &case["messages"]);
        seen += 1;
    }
    assert_eq!(seen, 4, "fixture case count");
}
