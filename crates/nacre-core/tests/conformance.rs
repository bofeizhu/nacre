//! Golden-trace conformance: the Rust stack (nacre + grit) must reproduce
//! the frozen outputs of pinned Python Graphiti (oracle/fixtures/<trace>/).
//!
//! This file carries the loader, the grit-side state dumper (mirroring
//! oracle/capture.py's aliased dump), and the field-for-field differ. The
//! end-to-end assertion runs only when golden trace #1 exists — capture is
//! user-gated (Docker + API key); until then the trace test skips loudly
//! and the differ is exercised by self-tests.
//!
//! Diff policy: `created_at` fields are wall-clock on the Python side and
//! injected-clock here, so they are excluded from comparison. Everything
//! else — names, labels (sorted), summaries, facts, endpoints, episode
//! attribution, valid_at/invalid_at/expired_at (to the second), retrieval
//! rank order — must match exactly or be recorded in DEVIATIONS.md.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use grit_core::Grit;
use nacre_core::extract::{EpisodeInput, EpisodeSource};
use nacre_core::model::{RecordingStore, ReplayModel};
use nacre_core::pipeline::{AddEpisodeOptions, add_episode};
use nacre_core::search::search_edges;
use serde_json::{Value, json};
use uuid::Uuid;

/// Upstream fetches this many previous episodes as extraction context.
// ports: graphiti_core/utils/maintenance/graph_data_operations.py::EPISODE_WINDOW_LEN
const EPISODE_WINDOW_LEN: usize = 3;

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle")
}

/// Normalize a timestamp to the trace's isoformat spelling, second
/// precision (`2026-06-05T00:00:00+00:00`).
fn iso(t: Option<DateTime<Utc>>) -> Value {
    match t {
        Some(t) => json!(t.to_rfc3339_opts(chrono::SecondsFormat::Secs, false)),
        None => Value::Null,
    }
}

fn iso_ms(ms: Option<i64>) -> Value {
    iso(ms.and_then(DateTime::<Utc>::from_timestamp_millis))
}

/// Dump a grit group in oracle/capture.py's aliased shape. Mirrors
/// `build_aliases` + `dump_graph_state` exactly (content-derived aliases,
/// sorted collections, no embeddings).
fn dump_graph_state(grit: &Grit, group_id: &str, live_only: bool) -> Value {
    // grit exposes getters by id only; enumerate the group from the JSONL
    // export stream (flat records tagged `"t"`; the extra hlc field is
    // ignored by serde).
    let mut buffer: Vec<u8> = Vec::new();
    grit.export_jsonl(&mut buffer).expect("export succeeds");
    let export = String::from_utf8(buffer).expect("export is UTF-8");
    let mut nodes: Vec<grit_core::Node> = Vec::new();
    let mut edges: Vec<grit_core::Edge> = Vec::new();
    let mut episodes: Vec<grit_core::Episode> = Vec::new();
    for line in export.lines() {
        let value: Value = serde_json::from_str(line).expect("export line parses");
        match value["t"].as_str() {
            Some("node") => {
                let node: grit_core::Node =
                    serde_json::from_value(value.clone()).expect("node record");
                if node.group_id == group_id && (!live_only || node.expired_at.is_none()) {
                    nodes.push(node);
                }
            }
            Some("edge") => {
                let edge: grit_core::Edge =
                    serde_json::from_value(value.clone()).expect("edge record");
                if edge.group_id == group_id {
                    edges.push(edge);
                }
            }
            Some("episode") => {
                let episode: grit_core::Episode =
                    serde_json::from_value(value.clone()).expect("episode record");
                if episode.group_id == group_id {
                    episodes.push(episode);
                }
            }
            _ => {}
        }
    }

    // Aliases: content-derived, matching capture.py.
    let mut aliases: HashMap<Uuid, String> = HashMap::new();
    let mut sorted_nodes: Vec<&grit_core::Node> = nodes.iter().collect();
    sorted_nodes.sort_by(|a, b| (&a.name, &a.summary).cmp(&(&b.name, &b.summary)));
    for (i, node) in sorted_nodes.iter().enumerate() {
        aliases.insert(node.id, format!("n{i}"));
    }
    let ep_key = |e: &grit_core::Episode| {
        (
            iso_ms(Some(e.occurred_at))
                .as_str()
                .unwrap_or("")
                .to_owned(),
            e.content.clone(),
        )
    };
    let mut sorted_episodes: Vec<&grit_core::Episode> = episodes.iter().collect();
    sorted_episodes.sort_by_key(|e| ep_key(e));
    for (i, episode) in sorted_episodes.iter().enumerate() {
        aliases.insert(episode.id, format!("ep{i}"));
    }
    let alias = |aliases: &HashMap<Uuid, String>, id: Uuid| -> String {
        aliases
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("UNALIASED:{id}"))
    };
    let mut sorted_edges: Vec<&grit_core::Edge> = edges.iter().collect();
    sorted_edges.sort_by_key(|e| {
        (
            e.fact.clone(),
            alias(&aliases, e.src),
            alias(&aliases, e.dst),
        )
    });
    for (i, edge) in sorted_edges.iter().enumerate() {
        aliases.insert(edge.id, format!("e{i}"));
    }

    let node_dump: Vec<Value> = {
        let mut out: Vec<Value> = nodes
            .iter()
            .map(|n| {
                let mut labels: Vec<String> = n.attrs["labels"]
                    .as_array()
                    .map(|l| {
                        l.iter()
                            .filter_map(|v| v.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![n.kind.clone()]);
                labels.sort();
                json!({
                    "uuid": alias(&aliases, n.id),
                    "name": n.name,
                    "labels": labels,
                    "summary": n.summary,
                    "attributes": {},
                    "created_at": iso_ms(Some(n.created_at)),
                })
            })
            .collect();
        out.sort_by_key(|v| v["uuid"].as_str().unwrap_or("").to_owned());
        out
    };
    let edge_dump: Vec<Value> = {
        let mut out: Vec<Value> = edges
            .iter()
            .map(|e| {
                let mut mentions: Vec<String> = grit
                    .mentions_of(e.id)
                    .expect("mentions query")
                    .into_iter()
                    .map(|id| alias(&aliases, id))
                    .collect();
                mentions.sort();
                json!({
                    "uuid": alias(&aliases, e.id),
                    "source": alias(&aliases, e.src),
                    "target": alias(&aliases, e.dst),
                    "name": e.rel,
                    "fact": e.fact,
                    "episodes": mentions,
                    "attributes": {},
                    "created_at": iso_ms(Some(e.created_at)),
                    "valid_at": iso_ms(e.valid_at),
                    "invalid_at": iso_ms(e.invalid_at),
                    "expired_at": iso_ms(e.expired_at),
                })
            })
            .collect();
        out.sort_by_key(|v| v["uuid"].as_str().unwrap_or("").to_owned());
        out
    };
    let episode_dump: Vec<Value> = {
        let mut out: Vec<Value> = episodes
            .iter()
            .map(|e| {
                json!({
                    "uuid": alias(&aliases, e.id),
                    "content": e.content,
                    "source_description": e.source,
                    "valid_at": iso_ms(Some(e.occurred_at)),
                    "created_at": iso_ms(Some(e.created_at)),
                })
            })
            .collect();
        out.sort_by_key(|v| v["uuid"].as_str().unwrap_or("").to_owned());
        out
    };
    json!({"nodes": node_dump, "edges": edge_dump, "episodes": episode_dump})
}

/// Field-for-field diff, `created_at` excluded (wall-clock vs injected)
/// plus any explicitly ignored keys. Returns human-readable differences.
fn diff_states(expected: &Value, actual: &Value, ignore: &[&str]) -> Vec<String> {
    let mut diffs = Vec::new();
    diff_value("", expected, actual, ignore, &mut diffs);
    diffs
}

fn diff_value(
    path: &str,
    expected: &Value,
    actual: &Value,
    ignore: &[&str],
    out: &mut Vec<String>,
) {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            for (key, ev) in e {
                if ignore.contains(&key.as_str()) {
                    continue;
                }
                let sub = format!("{path}/{key}");
                match a.get(key) {
                    Some(av) => diff_value(&sub, ev, av, ignore, out),
                    None => out.push(format!("{sub}: missing (expected {ev})")),
                }
            }
            for key in a.keys() {
                if !ignore.contains(&key.as_str()) && !e.contains_key(key) {
                    out.push(format!("{path}/{key}: unexpected"));
                }
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            if e.len() != a.len() {
                out.push(format!("{path}: length {} vs {}", e.len(), a.len()));
            }
            for (i, (ev, av)) in e.iter().zip(a).enumerate() {
                diff_value(&format!("{path}[{i}]"), ev, av, ignore, out);
            }
        }
        _ => {
            if expected != actual {
                out.push(format!("{path}: {expected} vs {actual}"));
            }
        }
    }
}

fn trace_episode(spec: &Value) -> EpisodeInput {
    EpisodeInput {
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
    }
}

/// End-to-end: replay golden trace #1 through nacre + grit and diff.
/// Skips (loudly) until the capture run has produced the fixture.
#[tokio::test]
async fn golden_trace1_conformance() {
    let trace_dir = oracle_dir().join("fixtures/trace1");
    let graph_state_path = trace_dir.join("graph_state.json");
    if !graph_state_path.exists() {
        eprintln!(
            "SKIP: golden trace #1 not captured yet ({}). Run oracle/capture.py first.",
            graph_state_path.display()
        );
        return;
    }

    let spec: Value =
        serde_json::from_str(&std::fs::read_to_string(trace_dir.join("episodes.json")).unwrap())
            .unwrap();
    let expected_state: Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_state_path).unwrap()).unwrap();
    let expected_retrieval: Value =
        serde_json::from_str(&std::fs::read_to_string(trace_dir.join("retrieval.json")).unwrap())
            .unwrap();
    let model =
        ReplayModel::new(RecordingStore::load(&trace_dir.join("llm_recordings.json")).unwrap());

    let group_id = spec["group_id"].as_str().unwrap().to_owned();
    let dir = std::env::temp_dir().join(format!("nacre-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace1.db");
    let _ = std::fs::remove_file(&path);
    let grit = Grit::open(&path, grit_core::Options::new("nacre-conformance")).unwrap();

    let mut episodes: Vec<EpisodeInput> = Vec::new();
    for episode_spec in spec["episodes"].as_array().unwrap() {
        let mut episode = trace_episode(episode_spec);
        episode.group_id = group_id.clone();
        let window_start = episodes.len().saturating_sub(EPISODE_WINDOW_LEN);
        let previous = episodes[window_start..].to_vec();
        let now = episode
            .valid_at
            .as_deref()
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|t| t.with_timezone(&Utc))
            .unwrap_or_else(|| Utc.with_ymd_and_hms(2026, 7, 10, 0, 0, 0).unwrap());
        add_episode(
            &grit,
            &model,
            &episode,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .unwrap_or_else(|e| panic!("add_episode({}) failed: {e}", episode.name));
        episodes.push(episode);
    }

    let actual_state = dump_graph_state(&grit, &group_id, true);
    let state_diffs = diff_states(&expected_state, &actual_state, &["created_at"]);
    assert!(
        state_diffs.is_empty(),
        "graph state diverges from golden trace #1:\n{}",
        state_diffs.join("\n")
    );

    for query_spec in expected_retrieval.as_array().unwrap() {
        let query = query_spec["query"].as_str().unwrap();
        let hits = search_edges(&grit, query, &group_id, 10).unwrap();
        let actual: Vec<&str> = hits.iter().map(|h| h.fact.as_str()).collect();
        let expected: Vec<&str> = query_spec["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["fact"].as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "retrieval rank order for {query:?}");
    }
}

/// The differ must be able to certify identity and catch real differences —
/// exercised without a golden trace.
#[test]
fn differ_self_test() {
    let dir = std::env::temp_dir().join(format!("nacre-conf-self-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("self.db");
    let _ = std::fs::remove_file(&path);
    let grit = Grit::open(&path, grit_core::Options::new("nacre-test")).unwrap();

    let a = grit.new_id();
    let b = grit.new_id();
    for (id, name) in [(a, "Priya"), (b, "Northwind Labs")] {
        grit.apply(grit_core::GraphOp::AddNode {
            id,
            kind: "Entity".into(),
            name: name.into(),
            summary: String::new(),
            attrs: json!({"labels": ["Entity"]}),
            group_id: "g".into(),
        })
        .unwrap();
    }
    grit.apply(grit_core::GraphOp::AddEdge {
        id: grit.new_id(),
        src: a,
        dst: b,
        rel: "WORKS_AT".into(),
        fact: "Priya works at Northwind Labs.".into(),
        attrs: json!({}),
        group_id: "g".into(),
        valid_at: Some(1_770_000_000_000),
    })
    .unwrap();

    let state = dump_graph_state(&grit, "g", true);
    assert!(diff_states(&state, &state, &["created_at"]).is_empty());

    // A mutated copy is caught, and created_at differences are ignored.
    let mut mutated = state.clone();
    mutated["edges"][0]["fact"] = json!("Priya works elsewhere.");
    mutated["edges"][0]["created_at"] = json!("1999-01-01T00:00:00+00:00");
    let diffs = diff_states(&state, &mutated, &["created_at"]);
    assert_eq!(diffs.len(), 1, "only the fact difference: {diffs:?}");
    assert!(diffs[0].contains("/edges[0]/fact"));

    drop(grit);
    let _ = std::fs::remove_file(&path);
}
