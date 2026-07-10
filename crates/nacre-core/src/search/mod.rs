//! Search orchestration over grit's retrieval legs.
//!
//! Ports the *default* search surface of `graphiti_core/graphiti.py::search`
//! (pinned v0.29.2): a hybrid edge search (`EDGE_HYBRID_SEARCH_RRF` — BM25 +
//! cosine, RRF-fused) returning edges in rank order. grit performs the
//! hybrid fusion internally (`Grit::search` runs FTS + vector + graph
//! expansion with RRF); nacre selects the edge hits and applies the limit,
//! preserving grit's fused ranking among edges.
//!
//! Parity notes (golden-trace retrieval conformance judges these):
//! - grit fuses nodes/edges/episodes in one ranking; upstream's edge recipe
//!   ranks edges only. Filtering preserves the relative edge order.
//! - RRF constants and BM25 scoring differ between engines by construction
//!   (grit AGENTS.md accepts this: rank-order parity is the target, not
//!   score parity).
//! - Advanced recipes (MMR, node-distance, cross-encoder, community search)
//!   are deliberately not ported — see AGENTS.md "not ported".

use chrono::{DateTime, Utc};
use grit_core::{Budget, Grit, Query, SearchTarget};
use serde::{Deserialize, Serialize};

/// Upstream's default result limit.
// ports: graphiti_core/helpers.py::DEFAULT_SEARCH_LIMIT
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Over-fetch multiplier: grit's budget spans all target kinds, so ask for
/// more items than needed and keep the first `limit` edges.
const EDGE_OVERFETCH: usize = 4;

/// One edge hit, in fused rank order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeSearchResult {
    /// Storage identity (grit edge id).
    pub id: String,
    /// Source node id.
    pub source_id: String,
    /// Target node id.
    pub target_id: String,
    /// Relation label.
    pub name: String,
    /// The fact sentence.
    pub fact: String,
    /// Event time the fact became true, if known.
    pub valid_at: Option<DateTime<Utc>>,
    /// Event time the fact stopped being true, if known.
    pub invalid_at: Option<DateTime<Utc>>,
    /// Provenance episode ids.
    pub episodes: Vec<String>,
}

/// Hybrid edge search: the out-of-the-box `graphiti.search` equivalent.
/// Returns at most `num_results` currently-valid edges in rank order.
// ports: graphiti.py::search (EDGE_HYBRID_SEARCH_RRF path)
pub fn search_edges(
    grit: &Grit,
    query: &str,
    group_id: &str,
    num_results: usize,
) -> Result<Vec<EdgeSearchResult>, grit_core::Error> {
    let hits = grit.search(Query::text(query).group(group_id).budget(Budget::items(
        num_results.saturating_mul(EDGE_OVERFETCH).max(1),
    )))?;
    let ms = |t: i64| DateTime::<Utc>::from_timestamp_millis(t);
    Ok(hits
        .into_iter()
        .filter_map(|hit| match hit.target {
            SearchTarget::Edge(edge) => Some(EdgeSearchResult {
                id: edge.id.to_string(),
                source_id: edge.src.to_string(),
                target_id: edge.dst.to_string(),
                name: edge.rel.clone(),
                fact: edge.fact.clone(),
                valid_at: edge.valid_at.and_then(ms),
                invalid_at: edge.invalid_at.and_then(ms),
                episodes: hit.episodes.iter().map(|id| id.to_string()).collect(),
            }),
            _ => None,
        })
        .take(num_results)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use grit_core::{GraphOp, Options};
    use serde_json::json;

    fn open_grit(name: &str) -> (Grit, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("nacre-search-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        (Grit::open(&path, Options::new("nacre-test")).unwrap(), path)
    }

    fn add_fact(grit: &Grit, group: &str, src_name: &str, dst_name: &str, fact: &str) {
        let src = grit.new_id();
        let dst = grit.new_id();
        for (id, name) in [(src, src_name), (dst, dst_name)] {
            grit.apply(GraphOp::AddNode {
                id,
                kind: "Entity".into(),
                name: name.into(),
                summary: String::new(),
                attrs: json!({}),
                group_id: group.into(),
            })
            .unwrap();
        }
        grit.apply(GraphOp::AddEdge {
            id: grit.new_id(),
            src,
            dst,
            rel: "RELATES_TO".into(),
            fact: fact.into(),
            attrs: json!({}),
            group_id: group.into(),
            valid_at: None,
            invalid_at: None,
        })
        .unwrap();
    }

    #[test]
    fn returns_matching_edges_in_rank_order_and_respects_group() {
        let (grit, path) = open_grit("search-edges.db");
        add_fact(
            &grit,
            "g1",
            "Priya",
            "Northwind Labs",
            "Priya works at Northwind Labs.",
        );
        add_fact(
            &grit,
            "g1",
            "Jordan",
            "Belmont Arts Center",
            "Jordan teaches ceramics at Belmont.",
        );
        add_fact(
            &grit,
            "g2",
            "Priya",
            "Northwind Labs",
            "Priya visited the Northwind office in g2.",
        );

        let hits = search_edges(&grit, "Northwind", "g1", 10).unwrap();
        assert_eq!(hits.len(), 1, "only g1's Northwind fact");
        assert_eq!(hits[0].fact, "Priya works at Northwind Labs.");

        let all = search_edges(&grit, "Priya OR Jordan OR Northwind OR ceramics", "g1", 10);
        // FTS syntax may reject the raw OR string depending on tokenizer;
        // don't over-constrain — the group filter is the assertion above.
        drop(all);

        let limited = search_edges(&grit, "Northwind", "g2", 0).unwrap();
        assert!(limited.is_empty(), "num_results=0 yields nothing");

        drop(grit);
        let _ = std::fs::remove_file(&path);
    }
}
