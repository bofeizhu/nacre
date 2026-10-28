//! Pre-refactor capture: the exact ordered sequence of grit writes that
//! `add_episode` performs for every episode of every golden trace.
//!
//! Why this exists: `add_episode` is about to become two-phase (plan → one
//! atomic `apply_batch`). The refactor must not change WHICH ops are applied
//! or in WHAT order — only when they commit. This test freezes the current
//! sequence so the new code can be asserted byte-equal against it.
//!
//! What is captured, per episode: every new `oplog_since` entry's `op` value
//! (the whole op — kind, names, summaries, attrs, facts, group_id,
//! occurred_at, invalid_at, mentions …), with UUIDs replaced by stable
//! placeholders. Plus the set of ids that hold an embedding vector after the
//! episode — embeddings are local state, not oplog entries, so their
//! interleaving is not observable; only their per-episode effect is.
//!
//! Normalization rules (see `normalize_op`):
//!   * every UUID is replaced by `$nK`, K = order of first appearance across
//!     the trace, scanning ops in oplog order and, within an op, in a fixed
//!     per-variant field order (`register_uuids`);
//!   * `OplogEntry.id` / `.hlc` / `.device_id` and the local `seq` are
//!     dropped — they are freshly minted per run and carry no pipeline
//!     decision;
//!   * nothing else is touched: the op's own `op` tag and every payload
//!     field survive verbatim.
//!
//! After the last episode, the trace's last episode is **deleted** and the
//! deletion's batch is pinned too (`deletion`): the `Purge`, the
//! `PurgeNodeFields` resetting the kept nodes' summaries, and the rebuilt
//! summaries — so a change to what forgetting writes is reviewed the same
//! way a change to what ingesting writes is. One extra rule there: the ids
//! inside a purge op, and the order of the rebuilt-summary updates, follow
//! uuid order, and uuids minted in the same millisecond order randomly
//! between runs, so both are sorted by placeholder before pinning.
//!
//! Modes: `NACRE_UPDATE_OP_SEQUENCE=1 cargo test -p nacre-core --test
//! op_sequence` rewrites `oracle/fixtures/<trace>/op_sequence.json`; without
//! the variable the test compares and fails on any diff.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use grit_core::{GraphOp, Grit};
use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::forget::delete_episode;
use nacre_core::model::{EmbedderMeta, RecordingStore, ReplayEmbedder, ReplayModel};
use nacre_core::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, add_episode, retrieve_previous_episodes,
};
use serde_json::{Value, json};
use uuid::Uuid;

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle")
}

/// Stable placeholders for freshly minted ids: `$n0`, `$n1`, … keyed by
/// first appearance within the trace.
#[derive(Default)]
struct Placeholders {
    names: HashMap<Uuid, String>,
    order: Vec<Uuid>,
}

impl Placeholders {
    fn intern(&mut self, id: Uuid) -> String {
        if let Some(name) = self.names.get(&id) {
            return name.clone();
        }
        let name = format!("$n{}", self.order.len());
        self.names.insert(id, name.clone());
        self.order.push(id);
        name
    }
}

/// Register an op's uuids in a fixed, human-meaningful order so placeholder
/// numbering does not depend on JSON key order.
fn register_uuids(op: &GraphOp, ph: &mut Placeholders) {
    match op {
        GraphOp::AddNode { id, .. } | GraphOp::UpdateNode { id, .. } => {
            ph.intern(*id);
        }
        GraphOp::AddEdge { id, src, dst, .. } => {
            ph.intern(*id);
            ph.intern(*src);
            ph.intern(*dst);
        }
        GraphOp::AddEpisode { id, mentions, .. } => {
            ph.intern(*id);
            for m in mentions {
                ph.intern(*m);
            }
        }
        GraphOp::InvalidateEdge { edge_id, .. } => {
            ph.intern(*edge_id);
        }
        GraphOp::MergeNodes { from, into } => {
            ph.intern(*from);
            ph.intern(*into);
        }
        GraphOp::Purge { ids } | GraphOp::PurgeNodeFields { ids, .. } => {
            for id in ids {
                ph.intern(*id);
            }
        }
    }
}

/// Replace every uuid-shaped string anywhere in the value with its
/// placeholder (interning any the registration pass missed, so a new op
/// variant can never silently leak a raw uuid into the fixture).
fn substitute(value: &mut Value, ph: &mut Placeholders) {
    match value {
        Value::String(s) => {
            if let Ok(id) = Uuid::parse_str(s) {
                *s = ph.intern(id);
            }
        }
        Value::Array(items) => {
            for item in items {
                substitute(item, ph);
            }
        }
        Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                substitute(v, ph);
            }
        }
        _ => {}
    }
}

fn normalize_op(op: &GraphOp, ph: &mut Placeholders) -> Value {
    register_uuids(op, ph);
    let mut value = serde_json::to_value(op).expect("op serializes");
    substitute(&mut value, ph);
    value
}

fn trace_episode(spec: &Value) -> EpisodeInput {
    EpisodeInput {
        id: None,
        name: spec["name"].as_str().unwrap().into(),
        content: spec["content"].as_str().unwrap().into(),
        source: match spec["source"].as_str().unwrap_or("message") {
            "text" => EpisodeSource::Text,
            "json" => EpisodeSource::Json,
            _ => EpisodeSource::Message,
        },
        source_description: spec["source_description"].as_str().unwrap().into(),
        group_id: String::new(), // filled by caller from the trace spec
        valid_at: Some(spec["reference_time"].as_str().unwrap().into()),
        // The traces carry no application payload. `None` serializes
        // away in the op (grit skips it), so the pins stay byte-identical.
        metadata: None,
    }
}

/// Replay one golden trace and capture the per-episode write sequence.
async fn capture(trace: &str) -> Option<Value> {
    let trace_dir = oracle_dir().join("fixtures").join(trace);
    let episodes_path = trace_dir.join("episodes.json");
    if !episodes_path.exists() {
        eprintln!(
            "SKIP: golden trace {trace:?} not captured yet ({}). Run oracle/capture.py first.",
            episodes_path.display()
        );
        return None;
    }

    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&episodes_path).unwrap())
        .expect("episodes.json parses");
    let model =
        ReplayModel::new(RecordingStore::load(&trace_dir.join("llm_recordings.json")).unwrap());
    let embedder = ReplayEmbedder::new(
        RecordingStore::load(&trace_dir.join("embedder_recordings.json")).unwrap(),
        EmbedderMeta {
            model_id: "embedding-3".into(),
            dim: 1024,
            model_version: String::new(),
        },
    );

    let group_id = spec["group_id"].as_str().unwrap().to_owned();
    let dir = std::env::temp_dir().join(format!("nacre-opseq-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{trace}.db"));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
    let grit = Grit::open(&path, grit_core::Options::new("nacre-op-sequence")).unwrap();

    let mut ph = Placeholders::default();
    let mut last_seq = 0i64;
    let mut episodes_out: Vec<Value> = Vec::new();

    for episode_spec in spec["episodes"].as_array().unwrap() {
        let mut episode = trace_episode(episode_spec);
        episode.group_id = group_id.clone();
        let now = episode
            .valid_at
            .as_deref()
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|t| t.with_timezone(&Utc))
            .unwrap_or_else(|| Utc.with_ymd_and_hms(2026, 7, 10, 0, 0, 0).unwrap());
        let previous = retrieve_previous_episodes(
            &grit,
            &group_id,
            episode.source,
            now,
            PREVIOUS_EPISODE_WINDOW,
        )
        .unwrap();
        add_episode(
            &grit,
            &model,
            &embedder,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .unwrap_or_else(|e| panic!("add_episode({}) failed: {e}", episode.name));

        let fresh = grit.oplog_since(last_seq).expect("oplog read");
        let mut ops: Vec<Value> = Vec::with_capacity(fresh.len());
        for (seq, entry) in &fresh {
            last_seq = (*seq).max(last_seq);
            ops.push(normalize_op(&entry.op, &mut ph));
        }

        // Embeddings are not oplog entries (recomputable local state), so
        // only their cumulative effect is observable: which of the ids seen
        // so far hold a vector once this episode is done. Placeholder order
        // keeps the listing deterministic.
        let mut embedded_nodes: Vec<String> = Vec::new();
        let mut embedded_edges: Vec<String> = Vec::new();
        let known: Vec<Uuid> = ph.order.clone();
        for id in known {
            let name = ph.names[&id].clone();
            if grit.get_node_embedding(id).unwrap().is_some() {
                embedded_nodes.push(name.clone());
            }
            if grit.get_edge_embedding(id).unwrap().is_some() {
                embedded_edges.push(name);
            }
        }

        episodes_out.push(json!({
            "episode": episode.name,
            "op_count": ops.len(),
            "ops": ops,
            "embedded_nodes": embedded_nodes,
            "embedded_edges": embedded_edges,
        }));
    }

    // Forget the last episode and pin what that writes.
    let last = grit.episodes_in_group(&group_id).unwrap().pop().unwrap().id;
    delete_episode(&grit, last).expect("delete_episode");
    let rank = |ph: &Placeholders, id: &Value| -> usize {
        let name = id.as_str().unwrap();
        ph.order.iter().position(|u| ph.names[u] == name).unwrap()
    };
    let mut deletion: Vec<Value> = grit
        .oplog_since(last_seq)
        .expect("oplog read")
        .iter()
        .map(|(_, entry)| normalize_op(&entry.op, &mut ph))
        .collect();
    for op in &mut deletion {
        if let Some(ids) = op.get_mut("ids").and_then(Value::as_array_mut) {
            ids.sort_by_key(|id| rank(&ph, id));
        }
    }
    let updates_from = deletion
        .iter()
        .position(|op| op["op"] == "update_node")
        .unwrap_or(deletion.len());
    deletion[updates_from..].sort_by_key(|op| rank(&ph, &op["id"]));

    drop(grit);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }

    Some(json!({
        "trace": trace,
        "group_id": group_id,
        "placeholders": "UUIDs replaced by $nK, K = order of first appearance across the trace",
        "episodes": episodes_out,
        "deletion": {
            "deleted_episode": ph.names[&last],
            "op_count": deletion.len(),
            "ops": deletion,
        },
    }))
}

/// Compare pretty-printed JSON and report the first differing lines.
fn assert_matches(expected: &Value, actual: &Value, fixture: &Path) {
    if expected == actual {
        return;
    }
    let expected_text = serde_json::to_string_pretty(expected).unwrap();
    let actual_text = serde_json::to_string_pretty(actual).unwrap();
    let mut diffs: Vec<String> = Vec::new();
    let expected_lines: Vec<&str> = expected_text.lines().collect();
    let actual_lines: Vec<&str> = actual_text.lines().collect();
    for i in 0..expected_lines.len().max(actual_lines.len()) {
        let e = expected_lines.get(i).copied().unwrap_or("<missing>");
        let a = actual_lines.get(i).copied().unwrap_or("<missing>");
        if e != a {
            diffs.push(format!(
                "  line {}:\n    expected: {e}\n    actual:   {a}",
                i + 1
            ));
        }
        if diffs.len() == 20 {
            diffs.push("  … (truncated)".to_owned());
            break;
        }
    }
    panic!(
        "grit write sequence diverges from {} ({} expected lines vs {} actual).\n\
         The two-phase refactor must apply the same ops in the same order.\n\
         If the change is intended, re-capture with \
         NACRE_UPDATE_OP_SEQUENCE=1 and review the diff.\n{}",
        fixture.display(),
        expected_lines.len(),
        actual_lines.len(),
        diffs.join("\n")
    );
}

async fn run_trace(trace: &str) {
    let Some(actual) = capture(trace).await else {
        return;
    };
    let fixture = oracle_dir()
        .join("fixtures")
        .join(trace)
        .join("op_sequence.json");
    if std::env::var("NACRE_UPDATE_OP_SEQUENCE").as_deref() == Ok("1") {
        let mut text = serde_json::to_string_pretty(&actual).unwrap();
        text.push('\n');
        std::fs::write(&fixture, text).expect("fixture write");
        eprintln!("WROTE {}", fixture.display());
        return;
    }
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap_or_else(|e| {
            panic!(
                "missing op-sequence fixture {}: {e}. Capture it with \
                 NACRE_UPDATE_OP_SEQUENCE=1 cargo test -p nacre-core --test op_sequence",
                fixture.display()
            )
        }))
        .expect("op_sequence.json parses");
    assert_matches(&expected, &actual, &fixture);
}

#[tokio::test]
async fn trace1_op_sequence() {
    run_trace("trace1").await;
}

#[tokio::test]
async fn trace2_op_sequence() {
    run_trace("trace2").await;
}
