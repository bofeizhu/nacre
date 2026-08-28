//! The host-fulfilled path end-to-end, with Rust stubs standing in for the
//! Swift protocols an app implements. Offline by construction: the stubs
//! ARE the transport, so nothing here can reach the network.
//!
//! Ports the spirit of `nacre-node/test/host.test.mjs`: the chat stub
//! recognizes which schema a round asked for by the schema block nacre
//! appends to the prompt, and the embedding stub deliberately returns
//! over-long vectors so the MRL slice is exercised.

use std::sync::{Arc, Mutex};

use nacre_swift::{
    ChatRequest, EmbedderSpec, EmbeddingProvider, Episode, LlmProvider, Memory, NacreError,
};

/// Deliberately tiny: grit tags vectors with (model_id, dim, version), and
/// a stub embedder has no reason to pretend to be 1024-dimensional.
const DIM: u32 = 8;

fn spec() -> EmbedderSpec {
    EmbedderSpec {
        model_id: "stub-embedding".into(),
        dim: DIM,
        model_version: String::new(),
    }
}

fn temp_db(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nacre-swift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    // grit needs a real file (two connections must see one database).
    let path = dir.join(format!("{name}.db"));
    let _ = std::fs::remove_file(&path);
    path
}

/// Answers whichever schema the round asked for with a minimal valid
/// response, and records every request it saw.
struct StubLlm {
    requests: Mutex<Vec<ChatRequest>>,
}

impl StubLlm {
    fn new() -> Arc<Self> {
        Arc::new(StubLlm {
            requests: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl LlmProvider for StubLlm {
    async fn chat(&self, request: ChatRequest) -> Result<String, NacreError> {
        let prompt: String = request
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.requests.lock().unwrap().push(request);
        // The appended schema block names the required top-level keys.
        let response = if prompt.contains("\"extracted_entities\"") {
            r#"{"extracted_entities": [{"name": "Ada", "entity_type_id": 0, "episode_indices": [0]}]}"#
        } else if prompt.contains("\"edges\"") {
            r#"{"edges": []}"#
        } else if prompt.contains("\"entity_resolutions\"") {
            r#"{"entity_resolutions": []}"#
        } else if prompt.contains("\"duplicate_facts\"") {
            r#"{"duplicate_facts": [], "contradicted_facts": []}"#
        } else if prompt.contains("\"summaries\"") {
            r#"{"summaries": [{"name": "Ada", "summary": "Ada is a person."}]}"#
        } else if prompt.contains("\"timestamps\"") {
            r#"{"timestamps": []}"#
        } else if prompt.contains("\"description\"") {
            r#"{"description": "d"}"#
        } else if prompt.contains("\"summary\"") {
            r#"{"summary": "s"}"#
        } else {
            return Err(NacreError::Provider {
                message: format!(
                    "stub got an unrecognized prompt: {}",
                    &prompt[..200.min(prompt.len())]
                ),
            });
        };
        Ok(response.to_owned())
    }
}

/// Deterministic vectors, `extra` components LONGER than the configured
/// dim so the MRL truncation in `HostEmbedder` is exercised. `extra` may
/// be negative to produce vectors that are too SHORT (an error).
struct StubEmbedder {
    extra: i32,
    batches: Mutex<Vec<Vec<String>>>,
}

impl StubEmbedder {
    fn new(extra: i32) -> Arc<Self> {
        Arc::new(StubEmbedder {
            extra,
            batches: Mutex::new(Vec::new()),
        })
    }

    fn saw(&self, input: &str) -> bool {
        self.batches
            .lock()
            .unwrap()
            .iter()
            .any(|batch| batch.iter().any(|s| s == input))
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for StubEmbedder {
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f64>>, NacreError> {
        assert!(!inputs.is_empty(), "nacre never asks for an empty batch");
        let len = (DIM as i32 + self.extra).max(0) as usize;
        let vectors = inputs
            .iter()
            .enumerate()
            .map(|(i, _)| {
                (0..len)
                    .map(|j| (i as f64 + 1.0) / (j as f64 + 1.0))
                    .collect()
            })
            .collect();
        self.batches.lock().unwrap().push(inputs);
        Ok(vectors)
    }
}

/// Always throws — a host whose relay is down.
struct BrokenLlm;

#[async_trait::async_trait]
impl LlmProvider for BrokenLlm {
    async fn chat(&self, _request: ChatRequest) -> Result<String, NacreError> {
        Err(NacreError::Provider {
            message: "host transport down".into(),
        })
    }
}

fn episode(name: &str, content: &str, valid_at: &str) -> Episode {
    Episode {
        id: None,
        name: name.into(),
        content: content.into(),
        source: None,
        source_description: "chat".into(),
        group_id: "host-group".into(),
        valid_at: Some(valid_at.into()),
        metadata: None,
    }
}

#[tokio::test]
async fn host_protocols_drive_the_whole_pipeline() {
    let memory = Memory::open(
        temp_db("host").to_string_lossy().into_owned(),
        "host-test".into(),
    )
    .expect("open");
    let llm = StubLlm::new();
    let embedder = StubEmbedder::new(4);

    let outcome = memory
        .add_episode(
            episode(
                "ep-0",
                "Ada wrote the first program.",
                "2026-07-14T00:00:00Z",
            ),
            llm.clone(),
            embedder.clone(),
            spec(),
        )
        .await
        .expect("ingestion runs entirely through the stubs");

    assert_eq!(outcome.node_ids.len(), 1, "one entity extracted");
    assert!(outcome.merges.is_empty());
    assert!(outcome.invalidated_edge_ids.is_empty());
    let requests = llm.requests.lock().unwrap().clone();
    assert!(
        requests.len() >= 2,
        "the chat protocol drove the pipeline ({} rounds)",
        requests.len()
    );
    assert!(
        requests.iter().all(|r| r.max_tokens > 0),
        "every round carries a completion budget"
    );
    assert!(
        requests
            .iter()
            .any(|r| r.messages.iter().any(|m| m.content.contains("Ada wrote"))),
        "the episode text reached the host verbatim"
    );

    // Read path sees what the pipeline wrote.
    let nodes = memory.nodes_in_group("host-group".into()).expect("nodes");
    let ada = nodes
        .iter()
        .find(|n| n.name == "Ada")
        .expect("Ada is a live node");
    assert!(ada.labels.contains(&"Entity".to_owned()));
    assert!(ada.created_at.ends_with('Z'));
    assert_eq!(
        memory
            .episodes_in_group("host-group".into())
            .expect("episodes")
            .len(),
        1
    );
    assert!(
        !memory
            .mentions_of(ada.id.clone())
            .expect("mentions")
            .is_empty(),
        "the node is attributed to the episode that produced it"
    );

    // Search embeds the query through the same protocol.
    let hits = memory
        .search_edges(
            "who wrote programs?".into(),
            "host-group".into(),
            3,
            embedder.clone(),
            spec(),
        )
        .await
        .expect("search");
    assert!(
        hits.is_empty(),
        "the stub extracted no facts, so there is nothing to recall"
    );
    assert!(
        embedder.saw("who wrote programs?"),
        "the query embedding went through the host protocol"
    );
    // The stub returned DIM+4 components per vector; grit rejects a
    // vector whose length disagrees with the registered dim, so a green
    // ingestion above IS the proof that HostEmbedder MRL-truncated them.
}

#[tokio::test]
async fn a_throwing_llm_protocol_surfaces_as_a_provider_error() {
    let memory = Memory::open(
        temp_db("host-throws").to_string_lossy().into_owned(),
        "host-test".into(),
    )
    .expect("open");
    let error = memory
        .add_episode(
            episode("ep-err", "x", "2026-07-14T00:01:00Z"),
            Arc::new(BrokenLlm),
            StubEmbedder::new(4),
            spec(),
        )
        .await
        .expect_err("a dead relay must not be swallowed");
    match error {
        NacreError::Provider { message } => {
            assert!(message.contains("host transport down"), "{message}");
            assert!(message.contains("giving up after 3 attempts"), "{message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}

#[tokio::test]
async fn a_short_vector_is_an_error_not_a_pad() {
    let memory = Memory::open(
        temp_db("host-short-vec").to_string_lossy().into_owned(),
        "host-test".into(),
    )
    .expect("open");
    let error = memory
        .add_episode(
            episode("ep-short", "Ada wrote it.", "2026-07-14T00:02:00Z"),
            StubLlm::new(),
            StubEmbedder::new(-1),
            spec(),
        )
        .await
        .expect_err("vectors shorter than the configured dim are a contract breach");
    match error {
        NacreError::Provider { message } => {
            assert!(message.contains("expected at least 8"), "{message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}

#[tokio::test]
async fn timestamps_and_sources_are_validated_before_any_model_call() {
    let memory = Memory::open(
        temp_db("host-args").to_string_lossy().into_owned(),
        "host-test".into(),
    )
    .expect("open");
    // The stubs are kept, not discarded: "before any model call" is the
    // actual claim, and the only witness to it is that these two recorded
    // nothing. A validation that ran after the first round would still
    // return InvalidInput while having spent a host request.
    let llm = StubLlm::new();
    let embedder = StubEmbedder::new(4);
    let untouched = |what: &str| {
        assert!(
            llm.requests.lock().unwrap().is_empty(),
            "{what} reached the chat protocol"
        );
        assert!(
            embedder.batches.lock().unwrap().is_empty(),
            "{what} reached the embedding protocol"
        );
    };

    let mut bad_time = episode("ep-0", "x", "2026-07-14T00:00:00");
    bad_time.valid_at = Some("2026-07-14T00:00:00".into()); // no offset
    match memory
        .add_episode(bad_time, llm.clone(), embedder.clone(), spec())
        .await
        .expect_err("an offset-less timestamp is ambiguous")
    {
        NacreError::InvalidInput { message } => assert!(message.starts_with("validAt: ")),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    untouched("an offset-less timestamp");

    let mut bad_source = episode("ep-0", "x", "2026-07-14T00:00:00Z");
    bad_source.source = Some("audio".into());
    match memory
        .add_episode(bad_source, llm.clone(), embedder.clone(), spec())
        .await
        .expect_err("unknown sources are rejected")
    {
        NacreError::InvalidInput { message } => assert!(message.contains("audio")),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    untouched("an unknown episode source");

    match memory
        .traverse(vec!["not-a-uuid".into()], None)
        .expect_err("garbage seed ids are rejected")
    {
        NacreError::InvalidInput { message } => assert!(message.contains("seed id")),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}
