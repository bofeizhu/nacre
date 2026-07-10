//! Live end-to-end smoke run: real LLM + real embeddings + a fresh grit
//! file, no recordings anywhere. NOT part of `cargo test` (requires the
//! `claude` + `openai-embed` features and live API keys).
//!
//! ```sh
//! set -a && . oracle/.env && set +a
//! cargo run --example live_smoke --features claude,openai-embed
//! ```
//!
//! Env (never printed): `CAPTURE_LLM_API_KEY` — LLM key, used against
//! DeepSeek's Anthropic-style endpoint by default, or set
//! `NACRE_SMOKE_ANTHROPIC_KEY` to run against Anthropic instead;
//! `CAPTURE_EMBEDDER_API_KEY` — Zhipu embedding-3 key.

use nacre::extract::{EpisodeInput, EpisodeSource};
use nacre::model::claude::{ClaudeConfig, ClaudeModel};
use nacre::model::openai_embed::{OpenAiEmbedConfig, OpenAiEmbedder};
use nacre::pipeline::{
    AddEpisodeOptions, PREVIOUS_EPISODE_WINDOW, add_episode, retrieve_previous_episodes,
};
use nacre::search::search_edges;

const GROUP: &str = "smoke";

fn episode(name: &str, content: &str, at: &str) -> EpisodeInput {
    EpisodeInput {
        name: name.into(),
        content: content.into(),
        source: EpisodeSource::Message,
        source_description: "message".into(),
        group_id: GROUP.into(),
        valid_at: Some(at.into()),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let llm = match std::env::var("NACRE_SMOKE_ANTHROPIC_KEY") {
        Ok(key) => {
            println!("LLM: Anthropic (native structured outputs)");
            ClaudeModel::new(ClaudeConfig::new(key))
        }
        Err(_) => {
            let key = std::env::var("CAPTURE_LLM_API_KEY")
                .expect("CAPTURE_LLM_API_KEY not set — source oracle/.env first");
            println!("LLM: DeepSeek via Anthropic-style endpoint (schema in prompt)");
            ClaudeModel::new(ClaudeConfig::deepseek(key))
        }
    };
    let embed_key = std::env::var("CAPTURE_EMBEDDER_API_KEY")
        .expect("CAPTURE_EMBEDDER_API_KEY not set — source oracle/.env first");
    let embedder = OpenAiEmbedder::new(OpenAiEmbedConfig::zhipu(embed_key));

    let dir = std::env::temp_dir().join(format!("nacre-live-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("smoke.db");
    let _ = std::fs::remove_file(&path);
    let grit = grit_core::Grit::open(&path, grit_core::Options::new("live-smoke")).unwrap();
    println!("grit file: {}", path.display());

    let episodes = [
        episode(
            "turn-0",
            "Dana: Guess what — I finally adopted a greyhound! Her name is Comet.\n\
             Felix: That's great! Where are you keeping her, your flat in Rotterdam?",
            "2026-05-04T10:00:00+00:00",
        ),
        episode(
            "turn-1",
            "Dana: Yes, though Comet needs more space, so I'm moving to Utrecht next month.\n\
             Felix: Makes sense. I'm still at Brightwater Analytics, by the way — got promoted to data lead.",
            "2026-05-11T09:30:00+00:00",
        ),
        episode(
            "turn-2",
            "Dana: Congrats! Utrecht move is confirmed for June 1st, found a place near Wilhelminapark.\n\
             Felix: Perfect for Comet. Bring her to the office sometime, Brightwater is dog-friendly now.",
            "2026-05-18T16:45:00+00:00",
        ),
    ];

    for ep in &episodes {
        let now = chrono::DateTime::parse_from_rfc3339(ep.valid_at.as_deref().unwrap())
            .unwrap()
            .with_timezone(&chrono::Utc);
        let previous =
            retrieve_previous_episodes(&grit, GROUP, ep.source, now, PREVIOUS_EPISODE_WINDOW)
                .unwrap();
        let outcome = add_episode(
            &grit,
            &llm,
            &embedder,
            ep,
            &previous,
            &AddEpisodeOptions::default(),
            now,
        )
        .await
        .unwrap_or_else(|e| panic!("add_episode({}) failed: {e}", ep.name));
        println!(
            "ingested {}: {} nodes, {} new edges, {} merges, {} invalidations",
            ep.name,
            outcome.node_ids.len(),
            outcome.new_edge_ids.len(),
            outcome.merges.len(),
            outcome.invalidated_edge_ids.len(),
        );
    }

    let nodes = grit.nodes_in_group(GROUP).unwrap();
    let live: Vec<&str> = nodes
        .iter()
        .filter(|n| n.expired_at.is_none())
        .map(|n| n.name.as_str())
        .collect();
    println!(
        "\ngraph: {} live nodes: {live:?}\n       {} edges",
        live.len(),
        grit.edges_in_group(GROUP).unwrap().len()
    );

    for query in [
        "Where does Felix work?",
        "What kind of dog is Comet?",
        "Where is Dana moving?",
    ] {
        let hits = search_edges(&grit, &embedder, query, GROUP, 5)
            .await
            .unwrap();
        println!("\nquery: {query}");
        if hits.is_empty() {
            println!("  (no hits)");
        }
        for hit in hits {
            println!("  - {} [episodes: {:?}]", hit.fact, hit.episodes);
        }
    }
}
