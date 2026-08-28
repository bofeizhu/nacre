//! `add_episode` commits an episode whole or not at all, and a caller
//! supplied episode id makes a retry recognizable as a retry.
//!
//! The failure this pins used to be real: the pipeline applied its ops one
//! at a time, with model calls in between, and wrote the episode row LAST.
//! Any error after the first op left committed graph mutations that no
//! episode evidenced — and a retrying caller had no way to say "this is the
//! same episode", so it got them twice.
//!
//! The tests drive the golden traces with deliberately truncated recording
//! sets, withholding one step's responses so the first episode that needs
//! them fails part-way through. Two truncation points, because they fail at
//! different depths:
//!
//! - **summarize withheld** — the LAST model call an episode makes. The
//!   pipeline fails having decided every node, merge, edge, invalidation
//!   and embedding of that episode: the worst case for partial persistence.
//! - **edge resolution withheld** — the first episode with a non-empty
//!   graph behind it fails mid-planning, after its drafts and its merges
//!   INTO existing nodes were decided. This one runs against a populated
//!   graph, so it also pins that pre-existing rows come out untouched.
//!
//! Either way the graph must come out byte-identical to what it was before
//! the call, and the retry must complete the ingest exactly once.

use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use grit_core::Grit;
use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::model::{
    EmbedderMeta, ModelError, Recording, RecordingStore, ReplayEmbedder, ReplayModel,
};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, PipelineError, add_episode,
    retrieve_previous_episodes,
};
use serde_json::Value;
use uuid::Uuid;

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle")
}

fn recordings(path: &Path) -> Vec<Recording> {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).expect("recordings parse")
}

fn trace_episode(spec: &Value, group_id: &str, id: Uuid) -> EpisodeInput {
    EpisodeInput {
        // The caller's idempotency key: the same value survives a failed
        // attempt and identifies the retry.
        id: Some(id),
        name: spec["name"].as_str().unwrap().into(),
        content: spec["content"].as_str().unwrap().into(),
        source: match spec["source"].as_str().unwrap_or("message") {
            "text" => EpisodeSource::Text,
            "json" => EpisodeSource::Json,
            _ => EpisodeSource::Message,
        },
        source_description: spec["source_description"].as_str().unwrap().into(),
        group_id: group_id.to_owned(),
        valid_at: Some(spec["reference_time"].as_str().unwrap().into()),
        metadata: None,
    }
}

/// Everything about a group that an ingest could possibly change: the rows
/// themselves (every field, expired ones included), their provenance links,
/// which ids hold a vector, and the oplog length.
#[derive(Debug, PartialEq)]
struct Snapshot {
    oplog: usize,
    nodes: Vec<Value>,
    edges: Vec<Value>,
    episodes: Vec<Value>,
    mentions: Vec<(Uuid, Vec<Uuid>)>,
    embedded_nodes: Vec<Uuid>,
    embedded_edges: Vec<Uuid>,
}

fn snapshot(grit: &Grit, group_id: &str) -> Snapshot {
    let nodes = grit.nodes_in_group(group_id).unwrap();
    let edges = grit.edges_in_group(group_id).unwrap();
    let episodes = grit.episodes_in_group(group_id).unwrap();
    let mut mentions: Vec<(Uuid, Vec<Uuid>)> = nodes
        .iter()
        .map(|n| n.id)
        .chain(edges.iter().map(|e| e.id))
        .map(|id| (id, grit.mentions_of(id).unwrap()))
        .collect();
    mentions.sort();
    Snapshot {
        oplog: grit.oplog_since(0).unwrap().len(),
        mentions,
        embedded_nodes: nodes
            .iter()
            .filter(|n| grit.get_node_embedding(n.id).unwrap().is_some())
            .map(|n| n.id)
            .collect(),
        embedded_edges: edges
            .iter()
            .filter(|e| grit.get_edge_embedding(e.id).unwrap().is_some())
            .map(|e| e.id)
            .collect(),
        nodes: nodes
            .iter()
            .map(|n| serde_json::to_value(n).unwrap())
            .collect(),
        edges: edges
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect(),
        episodes: episodes
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect(),
    }
}

fn reference_time(episode: &EpisodeInput) -> DateTime<Utc> {
    episode
        .valid_at
        .as_deref()
        .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
        .map(|t| t.with_timezone(&Utc))
        .unwrap_or_else(|| Utc.with_ymd_and_hms(2026, 7, 10, 0, 0, 0).unwrap())
}

/// Replay a golden trace with every `withhold` response missing. Episodes
/// that never need one ingest normally; the first that does fails, and that
/// failure must be invisible in the graph. The retry — same episode id,
/// complete recordings — must then ingest it exactly once, and a further
/// call must be refused.
async fn run_trace(trace: &str, withhold: &str, expect_prior_state: bool) {
    let trace_dir = oracle_dir().join("fixtures").join(trace);
    let episodes_path = trace_dir.join("episodes.json");
    if !episodes_path.exists() {
        eprintln!(
            "SKIP: golden trace {trace:?} not captured yet ({}). Run oracle/capture.py first.",
            episodes_path.display()
        );
        return;
    }

    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&episodes_path).unwrap())
        .expect("episodes.json parses");
    let group_id = spec["group_id"].as_str().unwrap().to_owned();
    let all = recordings(&trace_dir.join("llm_recordings.json"));
    let truncated: Vec<Recording> = all
        .iter()
        .filter(|r| r.request["schema_name"] != withhold)
        .cloned()
        .collect();
    assert!(
        truncated.len() < all.len(),
        "trace {trace:?} has no {withhold} recording to withhold"
    );
    let full_model = ReplayModel::new(RecordingStore::new(all));
    let truncated_model = ReplayModel::new(RecordingStore::new(truncated));
    let embedder = ReplayEmbedder::new(
        RecordingStore::load(&trace_dir.join("embedder_recordings.json")).unwrap(),
        EmbedderMeta {
            model_id: "embedding-3".into(),
            dim: 1024,
            model_version: String::new(),
        },
    );

    let dir = std::env::temp_dir().join(format!("nacre-atomicity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // One db per scenario: the tests of this file run in parallel.
    let path = dir.join(format!("{trace}-{withhold}.db"));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
    let grit = Grit::open(&path, grit_core::Options::new("nacre-atomicity")).unwrap();

    let mut failed_on: Option<String> = None;
    for episode_spec in spec["episodes"].as_array().unwrap() {
        let episode_id = grit.new_id();
        let episode = trace_episode(episode_spec, &group_id, episode_id);
        let now = reference_time(&episode);
        let previous = retrieve_previous_episodes(
            &grit,
            &group_id,
            episode.source,
            now,
            PREVIOUS_EPISODE_WINDOW,
        )
        .unwrap();
        let before = snapshot(&grit, &group_id);
        let attempt = add_episode(
            &grit,
            &truncated_model,
            &embedder,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await;

        let Err(error) = attempt else {
            // This episode never needed the withheld response; it ingested
            // normally and becomes context for the ones after it.
            continue;
        };

        // The failure landed where the truncation put it, not somewhere
        // incidental.
        match &error {
            PipelineError::Model(ModelError::ReplayMiss { canonical_request }) => assert!(
                canonical_request.contains(withhold),
                "expected the withheld {withhold} call to fail, got {canonical_request}"
            ),
            other => panic!("expected a replay miss, got {other:?}"),
        }
        if expect_prior_state {
            assert!(
                !before.nodes.is_empty() && !before.edges.is_empty(),
                "this scenario is only interesting once earlier episodes \
                 have populated the graph"
            );
        }

        // Nothing at all was written: no nodes, no edges, no episode row,
        // no mentions, no oplog entries. Vectors are checked by the same
        // comparison — grit refuses one whose base row is missing, so the
        // only way a failed ingest could leave a vector behind is by
        // committing the row too, which shows up here as a node/edge diff.
        let after = snapshot(&grit, &group_id);
        assert_eq!(
            after, before,
            "a failed ingest of {:?} must leave the graph untouched",
            episode.name
        );
        assert_eq!(after.oplog, before.oplog, "no oplog entries were minted");
        assert!(
            grit.episode(episode_id).unwrap().is_none(),
            "the failed episode must not be stored"
        );

        // The retry: same id, complete recordings. Exactly one ingest.
        let outcome = add_episode(
            &grit,
            &full_model,
            &embedder,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .unwrap_or_else(|e| panic!("retry of {} failed: {e}", episode.name));
        assert_eq!(
            outcome.episode_id, episode_id,
            "the stored episode carries the caller's id"
        );
        let stored = grit.episode(episode_id).unwrap().expect("episode stored");
        assert_eq!(stored.content, episode.content);
        let committed = snapshot(&grit, &group_id);
        assert_ne!(committed, before, "the retry did ingest the episode");
        // Vectors commit with the rows they describe — every row this
        // ingest created has one, and (the failed attempt having created
        // no rows) nothing holds a vector for an id that does not exist.
        for id in outcome.node_ids.iter().chain(&outcome.new_edge_ids) {
            assert!(
                grit.get_node_embedding(*id).unwrap().is_some()
                    || grit.get_edge_embedding(*id).unwrap().is_some(),
                "{id} was committed without its vector"
            );
        }

        // A third call with the same id is refused before doing any work,
        // and changes nothing.
        let again = add_episode(
            &grit,
            &full_model,
            &embedder,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await;
        match again {
            Err(PipelineError::EpisodeExists(id)) => assert_eq!(id, episode_id),
            Err(other) => panic!("expected EpisodeExists, got {other:?}"),
            Ok(_) => panic!("a second successful ingest of the same id must be an error"),
        }
        assert_eq!(
            snapshot(&grit, &group_id),
            committed,
            "the refused call must not touch the graph"
        );

        failed_on = Some(episode.name.clone());
        break;
    }

    assert!(
        failed_on.is_some(),
        "no episode of {trace:?} needed a withheld {withhold} response — \
         the truncation no longer tests anything"
    );

    drop(grit);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
}

/// Failure at the last model call of an episode: everything was decided,
/// nothing is stored.
#[tokio::test]
async fn trace1_failure_at_the_last_model_call_leaves_no_trace() {
    run_trace("trace1", "SummarizedEntities", false).await;
}

#[tokio::test]
async fn trace2_failure_at_the_last_model_call_leaves_no_trace() {
    run_trace("trace2", "SummarizedEntities", false).await;
}

/// Failure against a populated graph, after this episode's merges into
/// existing nodes were decided: those nodes must come out untouched.
#[tokio::test]
async fn trace1_failure_after_merges_leaves_stored_rows_untouched() {
    run_trace("trace1", "EdgeDuplicate", true).await;
}

#[tokio::test]
async fn trace2_failure_after_merges_leaves_stored_rows_untouched() {
    run_trace("trace2", "EdgeDuplicate", true).await;
}
