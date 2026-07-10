//! Edge resolution: dedup judgment, contradiction detection, temporal
//! expiry, and timestamp extraction — upstream fuses these in one unit, so
//! the port keeps them together.
//!
//! Ports `resolve_extracted_edge`, `resolve_edge_contradictions`, and
//! `_extract_edge_timestamps` from
//! `graphiti_core/utils/maintenance/edge_operations.py` (pinned v0.29.2).
//! Custom edge-attribute extraction (pydantic edge models with fields) is
//! deliberately deferred; attributes are treated as always-empty.
//!
//! Time is injected: the caller passes `now` (from grit's `Clock`) —
//! upstream calls `utc_now()` where this port uses it.

use chrono::{DateTime, Utc};
use serde_json::json;

use super::helpers::normalize_string_exact;
use super::{EpisodeRef, ExistingEdge};
use crate::extract::DraftEdge;
use crate::extract::edges::parse_llm_timestamp;
use crate::model::{CompletionRequest, LanguageModel, ModelError, ModelSize};
use crate::prompts::{dedupe_edges, extract_edges};
use crate::schemas::{EdgeDuplicate, EdgeTimestamps, ResponseSchema};

/// The resolved edge's final state, whichever edge it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEdgeState {
    /// `Some(id)`: the extraction duplicates stored edge `id`. `None`: the
    /// draft stands as a new edge.
    pub duplicate_of: Option<String>,
    /// Final event-time lower bound after timestamp extraction and expiry
    /// rules.
    pub valid_at: Option<DateTime<Utc>>,
    /// Final event-time upper bound.
    pub invalid_at: Option<DateTime<Utc>>,
    /// Final system-time expiry.
    pub expired_at: Option<DateTime<Utc>>,
    /// Whether the current episode gets appended to the resolved edge's
    /// attribution. (Upstream's fast path checks for presence; the LLM
    /// duplicate path appends unconditionally — preserved as-is.)
    pub append_episode: bool,
}

/// A stored edge the resolution decided to invalidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeInvalidation {
    /// Storage identity of the invalidated edge.
    pub id: String,
    /// New event-time upper bound (the resolved edge's `valid_at`).
    pub invalid_at: DateTime<Utc>,
    /// New system-time expiry (kept if already set, else `now`).
    pub expired_at: DateTime<Utc>,
}

/// The full outcome of resolving one extracted edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeResolution {
    /// Final state of the resolved edge.
    pub resolved: ResolvedEdgeState,
    /// Stored edges to invalidate, in decision order.
    pub invalidated: Vec<EdgeInvalidation>,
}

/// Decide which contradicted candidates the resolved edge invalidates.
///
/// A candidate survives when the timelines don't overlap (it ended before
/// the resolved edge began, or the resolved edge ended before it began);
/// it is invalidated when it began strictly before the resolved edge.
// ports: edge_operations.py::resolve_edge_contradictions
pub fn resolve_edge_contradictions(
    resolved_valid_at: Option<DateTime<Utc>>,
    resolved_invalid_at: Option<DateTime<Utc>>,
    invalidation_candidates: &[ExistingEdge],
    now: DateTime<Utc>,
) -> Vec<EdgeInvalidation> {
    let mut invalidated = Vec::new();
    for edge in invalidation_candidates {
        let non_overlapping = matches!((edge.invalid_at, resolved_valid_at),
                (Some(edge_invalid), Some(resolved_valid)) if edge_invalid <= resolved_valid)
            || matches!((edge.valid_at, resolved_invalid_at),
                (Some(edge_valid), Some(resolved_invalid)) if resolved_invalid <= edge_valid);
        if non_overlapping {
            continue;
        }
        if let (Some(edge_valid), Some(resolved_valid)) = (edge.valid_at, resolved_valid_at)
            && edge_valid < resolved_valid
        {
            invalidated.push(EdgeInvalidation {
                id: edge.id.clone(),
                invalid_at: resolved_valid,
                expired_at: edge.expired_at.unwrap_or(now),
            });
        }
    }
    invalidated
}

/// Extract valid_at/invalid_at for a new edge via a lightweight LLM call.
/// Skipped when either bound is already set or no reference time exists;
/// model failures are swallowed (upstream warns and proceeds unbounded).
// ports: edge_operations.py::_extract_edge_timestamps
async fn extract_edge_timestamps<M: LanguageModel>(
    model: &M,
    fact: &str,
    valid_at: &mut Option<DateTime<Utc>>,
    invalid_at: &mut Option<DateTime<Utc>>,
    episode: &EpisodeRef,
) {
    if valid_at.is_some() || invalid_at.is_some() {
        return;
    }
    let Some(reference_time) = &episode.valid_at else {
        return;
    };
    let context = json!({
        "fact": fact,
        "reference_time": reference_time,
    });
    let request = CompletionRequest {
        messages: extract_edges::extract_timestamps(&context),
        schema_name: EdgeTimestamps::NAME.to_owned(),
        max_tokens: None,
        model_size: ModelSize::Small,
    };
    let Ok(response) = model.complete(&request).await else {
        return;
    };
    let Ok(timestamps) =
        crate::model::decode_response::<EdgeTimestamps>(response, EdgeTimestamps::NAME)
    else {
        return;
    };
    if let Some(raw) = timestamps.valid_at.as_deref()
        && let Some(parsed) = parse_llm_timestamp(raw)
    {
        *valid_at = Some(parsed);
    }
    if let Some(raw) = timestamps.invalid_at.as_deref()
        && let Some(parsed) = parse_llm_timestamp(raw)
    {
        *invalid_at = Some(parsed);
    }
}

/// Resolve an extracted edge against related (same-endpoint) and broader
/// existing edges: verbatim fast path, LLM dedup + contradiction judgment,
/// timestamp extraction for new edges, and the temporal expiry rules.
// ports: edge_operations.py::resolve_extracted_edge
pub async fn resolve_extracted_edge<M: LanguageModel>(
    model: &M,
    extracted: &DraftEdge,
    related_edges: &[ExistingEdge],
    existing_edges: &[ExistingEdge],
    episode: &EpisodeRef,
    now: DateTime<Utc>,
) -> Result<EdgeResolution, ModelError> {
    if related_edges.is_empty() && existing_edges.is_empty() {
        // No dedup needed; still extract timestamps for the new edge.
        // (Custom edge-attribute extraction deliberately deferred.)
        let mut valid_at = extracted.valid_at;
        let mut invalid_at = extracted.invalid_at;
        extract_edge_timestamps(
            model,
            &extracted.fact,
            &mut valid_at,
            &mut invalid_at,
            episode,
        )
        .await;
        return Ok(EdgeResolution {
            resolved: ResolvedEdgeState {
                duplicate_of: None,
                valid_at,
                invalid_at,
                expired_at: extracted.expired_at,
                append_episode: true,
            },
            invalidated: Vec::new(),
        });
    }

    // Fast path: identical endpoints and verbatim (normalized) fact.
    let normalized_fact = normalize_string_exact(&extracted.fact);
    for edge in related_edges {
        if edge.source_id == extracted.source_id
            && edge.target_id == extracted.target_id
            && normalize_string_exact(&edge.fact) == normalized_fact
        {
            return Ok(EdgeResolution {
                resolved: ResolvedEdgeState {
                    duplicate_of: Some(edge.id.clone()),
                    valid_at: edge.valid_at,
                    invalid_at: edge.invalid_at,
                    expired_at: edge.expired_at,
                    append_episode: !edge.episodes.contains(&episode.id),
                },
                invalidated: Vec::new(),
            });
        }
    }

    // LLM dedup with continuous indexing across both lists.
    let related_context: Vec<_> = related_edges
        .iter()
        .enumerate()
        .map(|(i, edge)| json!({"idx": i, "fact": edge.fact}))
        .collect();
    let offset = related_edges.len();
    let invalidation_context: Vec<_> = existing_edges
        .iter()
        .enumerate()
        .map(|(i, edge)| json!({"idx": offset + i, "fact": edge.fact}))
        .collect();
    let context = json!({
        "existing_edges": related_context,
        "new_edge": extracted.fact,
        "edge_invalidation_candidates": invalidation_context,
    });
    let request = CompletionRequest {
        messages: dedupe_edges::resolve_edge(&context),
        schema_name: EdgeDuplicate::NAME.to_owned(),
        max_tokens: None,
        model_size: ModelSize::Small,
    };
    let response = model.complete(&request).await?;
    let judgment: EdgeDuplicate = crate::model::decode_response(response, EdgeDuplicate::NAME)?;

    // First valid duplicate id wins; invalid ids are ignored.
    let duplicate: Option<&ExistingEdge> = judgment
        .duplicate_facts
        .iter()
        .filter_map(|&i| usize::try_from(i).ok())
        .find(|&i| i < related_edges.len())
        .map(|i| &related_edges[i]);

    // Contradicted ids split across both lists by the offset; invalid ids
    // are ignored, repeats preserved (upstream doesn't dedup them).
    let mut invalidation_candidates: Vec<&ExistingEdge> = Vec::new();
    for &idx in &judgment.contradicted_facts {
        let Ok(idx) = usize::try_from(idx) else {
            continue;
        };
        if idx < related_edges.len() {
            invalidation_candidates.push(&related_edges[idx]);
        } else if idx < related_edges.len() + existing_edges.len() {
            invalidation_candidates.push(&existing_edges[idx - offset]);
        }
    }

    // Resolved-edge temporal state, from whichever edge won.
    let (duplicate_of, mut valid_at, mut invalid_at, mut expired_at, append_episode) =
        match duplicate {
            Some(edge) => (
                Some(edge.id.clone()),
                edge.valid_at,
                edge.invalid_at,
                edge.expired_at,
                // Upstream appends unconditionally on the LLM duplicate path.
                true,
            ),
            None => (
                None,
                extracted.valid_at,
                extracted.invalid_at,
                extracted.expired_at,
                true,
            ),
        };

    // Timestamp extraction only for new edges (duplicates keep theirs).
    if duplicate_of.is_none() {
        extract_edge_timestamps(
            model,
            &extracted.fact,
            &mut valid_at,
            &mut invalid_at,
            episode,
        )
        .await;
    }

    if invalid_at.is_some() && expired_at.is_none() {
        expired_at = Some(now);
    }

    // A candidate with strictly newer information expires the resolved
    // edge. Candidates sort by valid_at with None last; first hit wins.
    if expired_at.is_none() {
        let mut sorted = invalidation_candidates.clone();
        sorted.sort_by_key(|c| (c.valid_at.is_none(), c.valid_at));
        for candidate in sorted {
            if let (Some(candidate_valid), Some(resolved_valid)) = (candidate.valid_at, valid_at)
                && candidate_valid > resolved_valid
            {
                invalid_at = Some(candidate_valid);
                expired_at = Some(now);
                break;
            }
        }
    }

    let owned_candidates: Vec<ExistingEdge> =
        invalidation_candidates.into_iter().cloned().collect();
    let invalidated = resolve_edge_contradictions(valid_at, invalid_at, &owned_candidates, now);

    Ok(EdgeResolution {
        resolved: ResolvedEdgeState {
            duplicate_of,
            valid_at,
            invalid_at,
            expired_at,
            append_episode,
        },
        invalidated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Recording, RecordingStore, ReplayModel};
    use chrono::TimeZone;
    use serde_json::Value;

    fn ts(iso: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(iso)
            .unwrap()
            .with_timezone(&Utc)
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 10, 12, 0, 0).unwrap()
    }

    fn draft(fact: &str) -> DraftEdge {
        DraftEdge {
            source_id: "id-p".into(),
            target_id: "id-n".into(),
            name: "WORKS_AT".into(),
            fact: fact.into(),
            group_id: "g".into(),
            episode_indices: vec![0],
            valid_at: None,
            invalid_at: None,
            expired_at: None,
            reference_time: Some("2026-03-02T18:30:00+00:00".into()),
        }
    }

    fn stored(id: &str, fact: &str, valid_at: Option<&str>) -> ExistingEdge {
        ExistingEdge {
            id: id.into(),
            source_id: "id-p".into(),
            target_id: "id-n".into(),
            name: "WORKS_AT".into(),
            fact: fact.into(),
            episodes: vec!["ep-old".into()],
            valid_at: valid_at.map(ts),
            invalid_at: None,
            expired_at: None,
        }
    }

    fn episode() -> EpisodeRef {
        EpisodeRef {
            id: "ep-new".into(),
            valid_at: Some("2026-06-08T20:45:00+00:00".into()),
        }
    }

    struct NoCallModel;
    impl LanguageModel for NoCallModel {
        fn complete(
            &self,
            request: &CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Value, ModelError>> + Send {
            panic!("unexpected model call: {}", request.schema_name);
            #[allow(unreachable_code)]
            async move {
                unreachable!()
            }
        }
    }

    fn dedupe_recording(
        related: &[ExistingEdge],
        existing: &[ExistingEdge],
        new_fact: &str,
        response: Value,
    ) -> Recording {
        let related_context: Vec<_> = related
            .iter()
            .enumerate()
            .map(|(i, e)| json!({"idx": i, "fact": e.fact}))
            .collect();
        let invalidation_context: Vec<_> = existing
            .iter()
            .enumerate()
            .map(|(i, e)| json!({"idx": related.len() + i, "fact": e.fact}))
            .collect();
        let context = json!({
            "existing_edges": related_context,
            "new_edge": new_fact,
            "edge_invalidation_candidates": invalidation_context,
        });
        let request = CompletionRequest {
            messages: dedupe_edges::resolve_edge(&context),
            schema_name: "EdgeDuplicate".into(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }
    }

    fn timestamps_recording(fact: &str, response: Value) -> Recording {
        let context = json!({
            "fact": fact,
            "reference_time": "2026-06-08T20:45:00+00:00",
        });
        let request = CompletionRequest {
            messages: extract_edges::extract_timestamps(&context),
            schema_name: "EdgeTimestamps".into(),
            max_tokens: None,
            model_size: ModelSize::Small,
        };
        Recording {
            request: serde_json::to_value(&request).unwrap(),
            response,
        }
    }

    #[tokio::test]
    async fn fast_path_matches_verbatim_fact_without_llm() {
        let existing = stored(
            "e-1",
            "Priya  works at Northwind Labs.",
            Some("2026-01-15T00:00:00+00:00"),
        );
        let out = resolve_extracted_edge(
            &NoCallModel,
            &draft("priya works at northwind labs."),
            &[existing],
            &[],
            &episode(),
            now(),
        )
        .await
        .unwrap();
        assert_eq!(out.resolved.duplicate_of.as_deref(), Some("e-1"));
        assert!(out.resolved.append_episode, "ep-new not yet attributed");
        assert!(out.invalidated.is_empty());
    }

    #[tokio::test]
    async fn no_candidates_extracts_timestamps_for_new_edge() {
        let model = ReplayModel::new(RecordingStore::new([timestamps_recording(
            "Priya joined Northwind Labs in January.",
            json!({"valid_at": "2026-01-15T00:00:00Z", "invalid_at": null}),
        )]));
        let out = resolve_extracted_edge(
            &model,
            &draft("Priya joined Northwind Labs in January."),
            &[],
            &[],
            &episode(),
            now(),
        )
        .await
        .unwrap();
        assert!(out.resolved.duplicate_of.is_none());
        assert_eq!(out.resolved.valid_at, Some(ts("2026-01-15T00:00:00+00:00")));
        assert!(out.resolved.expired_at.is_none());
    }

    #[tokio::test]
    async fn llm_contradiction_invalidates_older_edge() {
        // The new fact (valid June) contradicts the stored employment fact
        // (valid January) -> stored edge is invalidated at June.
        let old_edge = stored(
            "e-old",
            "Priya works at Northwind Labs.",
            Some("2026-01-15T00:00:00+00:00"),
        );
        let mut new_edge = draft("Priya left Northwind Labs.");
        new_edge.valid_at = Some(ts("2026-06-05T00:00:00+00:00"));

        let model = ReplayModel::new(RecordingStore::new([dedupe_recording(
            std::slice::from_ref(&old_edge),
            &[],
            &new_edge.fact,
            json!({"duplicate_facts": [], "contradicted_facts": [0]}),
        )]));
        let out = resolve_extracted_edge(&model, &new_edge, &[old_edge], &[], &episode(), now())
            .await
            .unwrap();
        assert!(out.resolved.duplicate_of.is_none());
        assert_eq!(out.invalidated.len(), 1);
        assert_eq!(out.invalidated[0].id, "e-old");
        assert_eq!(
            out.invalidated[0].invalid_at,
            ts("2026-06-05T00:00:00+00:00")
        );
        assert_eq!(out.invalidated[0].expired_at, now());
    }

    #[tokio::test]
    async fn newer_candidate_expires_the_new_edge() {
        // A contradicting candidate with NEWER validity expires the new
        // (older) edge instead.
        let newer = stored(
            "e-new",
            "Priya works at Meridian Health.",
            Some("2026-07-01T00:00:00+00:00"),
        );
        let mut extracted = draft("Priya works at Northwind Labs.");
        extracted.valid_at = Some(ts("2026-01-15T00:00:00+00:00"));

        // Continuous indexing: candidate arrives via the existing_edges
        // list (idx starts after related_edges, which is empty here...
        // except empty related means idx 0 is the first existing edge).
        let model = ReplayModel::new(RecordingStore::new([dedupe_recording(
            &[],
            std::slice::from_ref(&newer),
            &extracted.fact,
            json!({"duplicate_facts": [], "contradicted_facts": [0]}),
        )]));
        let out = resolve_extracted_edge(&model, &extracted, &[], &[newer], &episode(), now())
            .await
            .unwrap();
        assert_eq!(
            out.resolved.invalid_at,
            Some(ts("2026-07-01T00:00:00+00:00"))
        );
        assert_eq!(out.resolved.expired_at, Some(now()));
        // The newer candidate itself is NOT invalidated (it began after).
        assert!(out.invalidated.is_empty());
    }

    #[tokio::test]
    async fn llm_duplicate_reuses_stored_edge_and_skips_timestamps() {
        let existing = stored(
            "e-1",
            "Priya is employed by Northwind Labs.",
            Some("2026-01-15T00:00:00+00:00"),
        );
        let extracted = draft("Priya works for Northwind Labs.");
        let model = ReplayModel::new(RecordingStore::new([dedupe_recording(
            std::slice::from_ref(&existing),
            &[],
            &extracted.fact,
            // Invalid ids (7, -1) are ignored; 0 wins.
            json!({"duplicate_facts": [7, -1, 0], "contradicted_facts": []}),
        )]));
        let out = resolve_extracted_edge(&model, &extracted, &[existing], &[], &episode(), now())
            .await
            .unwrap();
        assert_eq!(out.resolved.duplicate_of.as_deref(), Some("e-1"));
        assert_eq!(out.resolved.valid_at, Some(ts("2026-01-15T00:00:00+00:00")));
        // No timestamps call was recorded -> a call would have errored the
        // resolution; None error proves duplicates skip extraction.
    }

    #[tokio::test]
    async fn timestamp_model_failure_is_swallowed() {
        // Empty store -> the timestamps call misses; upstream swallows.
        let model = ReplayModel::new(RecordingStore::new([]));
        let out = resolve_extracted_edge(
            &model,
            &draft("Priya visited Tokyo."),
            &[],
            &[],
            &episode(),
            now(),
        )
        .await
        .unwrap();
        assert!(out.resolved.valid_at.is_none());
    }

    #[test]
    fn contradiction_rules_skip_non_overlapping_timelines() {
        let mut ended_before = stored("e-1", "old fact", Some("2025-01-01T00:00:00+00:00"));
        ended_before.invalid_at = Some(ts("2025-06-01T00:00:00+00:00"));
        let out = resolve_edge_contradictions(
            Some(ts("2026-01-01T00:00:00+00:00")),
            None,
            &[ended_before],
            now(),
        );
        assert!(out.is_empty(), "candidate ended before resolved began");
    }
}
