//! The replay path against the committed golden trace — the same
//! recordings the conformance suite and the Node bindings replay, so a
//! divergence here is a bindings bug and nothing else.
//!
//! Offline and deterministic: `ReplayModel`/`ReplayEmbedder` fail loudly
//! on any request the trace did not record, which is what keeps
//! `cargo test` off the network.

use std::path::{Path, PathBuf};

use nacre_swift::{EmbedderSpec, Episode, Memory, NacreError};
use serde_json::Value;

/// Trace1's embedder identity — vectors persist under it.
fn spec() -> EmbedderSpec {
    EmbedderSpec {
        model_id: "embedding-3".into(),
        dim: 1024,
        model_version: String::new(),
    }
}

fn trace_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures/trace1")
}

fn recordings(name: &str) -> String {
    trace_dir().join(name).to_string_lossy().into_owned()
}

fn temp_db(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("nacre-swift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{name}.db"));
    let _ = std::fs::remove_file(&path);
    path.to_string_lossy().into_owned()
}

/// The trace spec, or `None` when the capture has not been run (a bare
/// checkout without `oracle/fixtures/` — skip loudly, like the
/// conformance suite).
fn trace_spec() -> Option<Value> {
    let path = trace_dir().join("episodes.json");
    if !path.exists() {
        eprintln!(
            "SKIP: golden trace 1 not captured yet ({}). Run oracle/capture.py first.",
            path.display()
        );
        return None;
    }
    Some(serde_json::from_str(&std::fs::read_to_string(&path).expect("read spec")).expect("spec"))
}

fn episode(spec: &Value, index: usize) -> Episode {
    let e = &spec["episodes"][index];
    Episode {
        id: None,
        name: e["name"].as_str().unwrap().to_owned(),
        content: e["content"].as_str().unwrap().to_owned(),
        source: e["source"].as_str().map(str::to_owned),
        source_description: e["source_description"].as_str().unwrap().to_owned(),
        group_id: spec["group_id"].as_str().unwrap().to_owned(),
        valid_at: Some(e["reference_time"].as_str().unwrap().to_owned()),
        metadata: None,
    }
}

#[tokio::test]
async fn replays_the_golden_trace_through_the_bindings() {
    let Some(trace) = trace_spec() else { return };
    let group = trace["group_id"].as_str().unwrap().to_owned();
    let memory = Memory::open(temp_db("replay"), "swift-test".into()).expect("open");

    let ingest = async |index: usize| {
        memory
            .add_episode_replay(
                episode(&trace, index),
                recordings("llm_recordings.json"),
                recordings("embedder_recordings.json"),
                spec(),
            )
            .await
    };

    // Deltas are exact under replay — the same numbers the Node bindings
    // assert for ep-0.
    let first = ingest(0).await.expect("ep-0 replays");
    assert_eq!(first.node_ids.len(), 4, "ep-0 extracts 4 entities");
    assert_eq!(first.new_edge_ids.len(), 2, "ep-0 creates 2 facts");
    assert!(first.merges.is_empty());
    assert!(first.invalidated_edge_ids.is_empty());
    assert!(!first.episode_id.is_empty());

    let second = ingest(1).await.expect("ep-1 replays");
    assert!(
        !second.merges.is_empty(),
        "ep-1 dedups returning entities into ep-0's nodes"
    );
    assert!(
        second
            .merges
            .iter()
            .all(|m| !m.from.is_empty() && !m.into.is_empty()),
        "merge pairs carry both endpoints"
    );

    // --- read path: the visualization data contract ---
    let nodes = memory.nodes_in_group(group.clone()).expect("nodes");
    let live: Vec<_> = nodes.iter().filter(|n| n.expired_at.is_none()).collect();
    assert!(
        nodes.len() > live.len(),
        "merged-away drafts are retained for audit"
    );
    assert!(
        nodes
            .iter()
            .any(|n| n.merged_into.is_some() && n.expired_at.is_some()),
        "a merged-away row points at its canonical node"
    );
    let priya = live
        .iter()
        .find(|n| n.name == "Priya")
        .expect("Priya is live");
    assert!(!priya.summary.is_empty(), "summaries persisted");
    assert!(priya.labels.contains(&"Entity".to_owned()));
    assert_eq!(
        priya.attrs_json, "{}",
        "attrs ride as a JSON object string, labels unpacked out of it"
    );

    let edges = memory.edges_in_group(group.clone()).expect("edges");
    assert!(edges.len() >= 4, "edges: {}", edges.len());
    let episodes = memory.episodes_in_group(group.clone()).expect("episodes");
    assert_eq!(episodes.len(), 2);
    assert_eq!(episodes[0].kind, "message");
    assert!(episodes[0].occurred_at.ends_with('Z'));

    let sub = memory
        .traverse(vec![priya.id.clone()], None)
        .expect("traverse");
    assert!(sub.nodes.iter().any(|n| n.id == priya.id));
    assert!(!sub.edges.is_empty(), "Priya has incident facts");

    let history = memory.node_history(priya.id.clone()).expect("history");
    assert_eq!(history.node.id, priya.id);
    assert!(!history.edges.is_empty());
    assert!(
        !memory
            .mentions_of(history.edges[0].id.clone())
            .expect("mentions")
            .is_empty(),
        "every edge is attributed to the episode that produced it"
    );

    // previous_episodes is the DISPLAY window: unfiltered by source kind,
    // bounded by the reference instant.
    let window = memory
        .previous_episodes(group.clone(), None, Some(1))
        .expect("window");
    assert_eq!(window.len(), 1, "last_n bounds the window");
    let before = memory
        .previous_episodes(
            group.clone(),
            Some("2026-03-03T00:00:00+00:00".into()),
            None,
        )
        .expect("window");
    assert_eq!(before.len(), 1, "episodes after the reference are excluded");

    // --- search: recorded query embedding, fused with the keyword legs ---
    let hits = memory
        .search_edges_replay(
            "Where does Priya work?".into(),
            group.clone(),
            5,
            recordings("embedder_recordings.json"),
            spec(),
        )
        .await
        .expect("search replays");
    assert!(!hits.is_empty(), "the trace query returns hits");
    assert!(
        hits.iter().any(|h| h.fact.contains("Northwind")),
        "the employment fact is recalled: {:?}",
        hits.iter().map(|h| &h.fact).collect::<Vec<_>>()
    );
    assert!(
        hits.iter().all(|h| !h.episodes.is_empty()),
        "hits carry provenance"
    );
}

/// The M10 surface through the bindings: an episode's own metadata on
/// every read path, and the two ways to forget. The laws themselves are
/// pinned in `nacre-core`'s `tests/deletion.rs`; what this asserts is that
/// the boundary carries them — the field, the outcome records, the
/// getter's `None`, and `EpisodeNotFound` as its own error case.
#[tokio::test]
async fn deletion_and_metadata_round_trip_through_the_bindings() {
    let Some(trace) = trace_spec() else { return };
    let group = trace["group_id"].as_str().unwrap().to_owned();
    let sibling = format!("{group}-sibling");
    let memory = Memory::open(temp_db("forget"), "swift-test".into()).expect("open");

    let ingest = async |episode: Episode| {
        memory
            .add_episode_replay(
                episode,
                recordings("llm_recordings.json"),
                recordings("embedder_recordings.json"),
                spec(),
            )
            .await
    };

    // ep-0 carries a host payload. It must not reach a prompt: a leak
    // would change the replay key and this ingest would fail as a miss,
    // so the assertion is the ingest itself.
    const PROVENANCE: &str = r#"{"outboxRow":42}"#;
    let mut keyed = episode(&trace, 0);
    keyed.metadata = Some(PROVENANCE.to_owned());
    let first = ingest(keyed).await.expect("ep-0 replays");
    let second = ingest(episode(&trace, 1)).await.expect("ep-1 replays");
    // The same trace under a second group id: byte-identical requests, an
    // id-disjoint graph — the control for what a deletion must not touch.
    let mut elsewhere = episode(&trace, 0);
    elsewhere.group_id = sibling.clone();
    ingest(elsewhere).await.expect("the sibling replays");

    // --- metadata: stored verbatim, handed back everywhere ---
    let stored = memory
        .episode(first.episode_id.clone())
        .expect("lookup")
        .expect("the episode is stored");
    assert_eq!(stored.metadata.as_deref(), Some(PROVENANCE));
    assert_eq!(stored.id, first.episode_id);
    let scanned = memory.episodes_in_group(group.clone()).expect("scan");
    assert_eq!(
        scanned[0].metadata.as_deref(),
        Some(PROVENANCE),
        "the group scan carries it too"
    );
    assert_eq!(
        memory
            .previous_episodes(group.clone(), None, Some(2))
            .expect("window")[0]
            .metadata
            .as_deref(),
        Some(PROVENANCE),
        "so does the previous-episode window"
    );
    assert_eq!(
        scanned[1].metadata, None,
        "an episode that stored nothing has none — not an empty string"
    );

    // The getter answers a missing row with None; only a malformed id is
    // an argument error, and it names the field.
    assert!(
        memory
            .episode("018f0000-0000-7000-8000-000000000000".into())
            .expect("an unknown id is not a failure")
            .is_none()
    );
    match memory.episode("not-a-uuid".into()).expect_err("garbage id") {
        NacreError::InvalidInput { message } => {
            assert!(message.starts_with("episode id "), "{message}")
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // --- delete_episode: sole evidence goes, corroborated stays ---
    let sibling_before = memory
        .nodes_in_group(sibling.clone())
        .expect("sibling")
        .len();
    let gone = memory
        .delete_episode(second.episode_id.clone())
        .expect("ep-1 is deletable");
    assert_eq!(gone.episode_id, second.episode_id);
    assert!(
        !gone.purged_node_ids.is_empty() || !gone.purged_edge_ids.is_empty(),
        "ep-1 left residue of its own"
    );
    assert!(
        !gone.kept_node_ids.is_empty(),
        "and entities ep-0 also evidences, which survive it"
    );
    assert!(
        memory
            .episode(second.episode_id.clone())
            .expect("lookup")
            .is_none(),
        "the episode row is unreachable"
    );
    let live: Vec<String> = memory
        .nodes_in_group(group.clone())
        .expect("nodes")
        .into_iter()
        .map(|n| n.id)
        .collect();
    for id in &gone.purged_node_ids {
        assert!(!live.contains(id), "{id} was purged");
    }
    for id in &gone.kept_node_ids {
        assert!(live.contains(id), "{id} was kept");
    }

    // A second delete is the mirror of the idempotency rejection: its own
    // error case, naming the id, with nothing written.
    match memory
        .delete_episode(second.episode_id.clone())
        .expect_err("an episode already deleted is not deletable")
    {
        NacreError::EpisodeNotFound { id } => assert_eq!(id, second.episode_id),
        other => panic!("expected EpisodeNotFound, got {other:?}"),
    }

    // --- delete_group: one namespace, whole; the sibling untouched ---
    let wiped = memory.delete_group(group.clone()).expect("group deletion");
    assert!(wiped.purged_episode_ids.contains(&first.episode_id));
    assert!(
        memory
            .nodes_in_group(group.clone())
            .expect("nodes")
            .is_empty()
    );
    assert!(
        memory
            .edges_in_group(group.clone())
            .expect("edges")
            .is_empty()
    );
    assert!(
        memory
            .episodes_in_group(group.clone())
            .expect("episodes")
            .is_empty()
    );
    assert_eq!(
        memory.nodes_in_group(sibling).expect("sibling").len(),
        sibling_before,
        "neither deletion touched the sibling group"
    );

    // An unknown group is a group with no rows, not an error.
    let nothing = memory
        .delete_group("never-existed".into())
        .expect("an unknown group is not an error");
    assert!(
        nothing.purged_node_ids.is_empty()
            && nothing.purged_edge_ids.is_empty()
            && nothing.purged_episode_ids.is_empty()
    );
}

#[tokio::test]
async fn an_unrecorded_request_fails_loudly_as_a_replay_miss() {
    let Some(trace) = trace_spec() else { return };
    let memory = Memory::open(temp_db("replay-miss"), "swift-test".into()).expect("open");

    let mut unrecorded = episode(&trace, 0);
    unrecorded.content = "Nothing in this sentence was ever captured.".into();
    let error = memory
        .add_episode_replay(
            unrecorded,
            recordings("llm_recordings.json"),
            recordings("embedder_recordings.json"),
            spec(),
        )
        .await
        .expect_err("offline replay must refuse to guess");
    match error {
        NacreError::ReplayMiss { canonical_request } => assert!(
            canonical_request.contains("Nothing in this sentence"),
            "the unmatched request is included for a mechanical diff"
        ),
        other => panic!("expected ReplayMiss, got {other:?}"),
    }
}

#[tokio::test]
async fn a_missing_recordings_file_is_an_invalid_input() {
    let memory = Memory::open(temp_db("replay-nofile"), "swift-test".into()).expect("open");
    let error = memory
        .search_edges_replay(
            "anything".into(),
            "trace1".into(),
            1,
            "/nonexistent/recordings.json".into(),
            spec(),
        )
        .await
        .expect_err("a bad path is caught before any pipeline work");
    match error {
        NacreError::InvalidInput { message } => {
            assert!(message.contains("loading recordings"), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn in_memory_databases_are_rejected() {
    match Memory::open(":memory:".into(), "swift-test".into())
        .err()
        .expect("grit needs a real file")
    {
        NacreError::Storage { message } => assert!(!message.is_empty()),
        other => panic!("expected Storage, got {other:?}"),
    }
}

#[test]
fn version_reports_the_crate_version() {
    assert_eq!(nacre_swift::version(), env!("CARGO_PKG_VERSION"));
}
