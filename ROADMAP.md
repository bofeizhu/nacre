# ROADMAP.md — the build queue

Ordered task queue for building nacre. Worked top-to-bottom, one task per
increment. Conventions (binding for any agent working this file):

- Take the **first unchecked task that is not `BLOCKED(...)`**. Finish it
  completely (code + tests + docs) before touching the next.
- A task that turns out to need user input gets `BLOCKED(reason)` prepended,
  not silently skipped. Unblocking is the user's move.
- Discovered work is **inserted** where it belongs in the order (never
  appended blindly, never done "while I'm here" without a line item).
- Tasks are checked off only with the full gate green:
  `cargo fmt --all --check`, `clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo doc` (warnings deny). AGENTS.md wins over
  this file if they ever conflict.

## Milestone 2 groundwork + Rust replay infrastructure (offline, unblocked)

- [x] `model/` module in nacre-core: `LanguageModel` + `Embedder` async traits
      (tokio), request/response types mirroring what Graphiti's pipeline needs
      (structured JSON output against a schema; batched embeddings). Include a
      `RecordingStore` (JSON files, keyed by **canonical request JSON** — not
      hash, so misses are eye-diffable and Python's
      `json.dumps(sort_keys=True)` writes the same format), a `ReplayModel` /
      `ReplayEmbedder` that serve recordings and **fail loudly on a miss**,
      and a `RecordingModel` capture wrapper. Unit tests with hand-written
      recordings.
- [x] Port the prompt *output* models: serde structs for every response schema
      in `refs/graphiti/graphiti_core/prompts/models.py` and the per-prompt
      response models — field names byte-identical to the Python (that's what
      lands in recordings/traces). Round-trip serde tests. → `src/schemas.rs`:
      all 17 response models across the six prompt modules, pydantic-matching
      defaults (`episode_indices=[0]`, optionals as explicit nulls), and a
      `ResponseSchema::NAME` trait carrying the Python class name into
      `CompletionRequest::schema_name`.
- [x] Verbatim prompt port: `extract_nodes` family → `src/prompts/extract_nodes.rs`
      (all 8 functions) with upstream-path comments + PROMPTS.md ledger rows.
      Established the fidelity mechanism used by every prompt port from here on:
      `src/prompts/py.rs` (Python str()/repr interpolation emulation),
      `src/prompts/helpers.rs` (json.dumps-compatible `to_prompt_json`;
      serde_json now needs `preserve_order`), and pinned fixtures rendered from
      the actual upstream Python by `oracle/promptgen/gen_prompt_fixtures.py`
      (manual, offline) asserted byte-identical in `tests/prompt_fidelity.rs`.
      Prompt modules live under `src/prompts/` 1:1 with upstream (not inside
      the step modules) so diffs against the pin stay mechanical.
- [x] Verbatim prompt port: `extract_edges` family →
      `src/prompts/extract_edges.rs` (edge, extract_attributes,
      extract_timestamps, extract_timestamps_batch), 5 fixture cases including
      the optional FACT_TYPES section.
- [x] Verbatim prompt port: `dedupe_nodes` + `dedupe_edges` families →
      `src/prompts/dedupe_nodes.rs` (node, nodes incl. len() arithmetic,
      node_list) + `src/prompts/dedupe_edges.rs` (resolve_edge), 4 fixture
      cases.
- [x] Verbatim prompt port: `summarize_nodes` family →
      `src/prompts/summarize_nodes.rs` (summarize_pair, summarize_context,
      summary_description), 3 fixture cases. All five prompt families are now
      ported (23 fixture cases total); PROMPTS.md records what was deliberately
      NOT ported (combined extraction, sagas, eval, lib registry).
- [x] oracle/ harness **code** (no networked run yet): uv project pinning
      `graphiti-core[falkordb]==0.29.2`, docker-compose for FalkorDB,
      recording + replay LLM/embedder wrappers (`recording_clients.py` —
      its docstring is THE RECORDING CONTRACT: pre-mutation messages are the
      request identity, matching nacre-core's CompletionRequest), and
      `capture.py` (clears DB, ingests episodes, dumps aliased graph state +
      RRF retrieval + recordings; `--replay` mode for offline determinism
      verification; cross-encoder is a fail-loud stub). Syntax-checked only —
      runtime verification happens at the first capture run.
- [x] Curated episode fixture set #1 → `oracle/episodes/trace1.json`:
      5 episodes / 5 queries; employment fact contradicted (ep-0 → ep-3
      invalidation), NYC vs New York City + Priya vs Priya Raman dedup,
      possessive qualification (Priya's dog Biscuit), unicode (ep-4),
      strictly increasing reference times.
- [ ] BLOCKED(user: needs Docker running + an LLM API key for the one-time
      capture) First capture run → commit golden trace #1 + recordings.

## Milestone 3 — the pipeline port (each step: logic + replay tests green)

- [x] `extract/nodes.rs`: episode → extracted entity nodes (ports
      `node_operations.py` extraction path: context construction incl.
      multi-episode attribution, prompt routing by source, empty-name filter,
      type mapping + exclusion, index clamping, exact-duplicate collapse with
      specificity rules; plus `concatenate_episodes` and
      `_normalize_string_exact`). `extract/mod.rs` defines the pipeline input
      types (EpisodeInput with ISO-string timestamps for prompt fidelity;
      DraftNode with positional identity — grit assigns durable ids at
      apply time). Replay tests w/ synthetic recordings.
- [x] `dedupe/nodes.rs`: candidate resolution → per-draft outcomes (ports
      `dedup_helpers.py` + `node_operations.py` dedup path): deterministic
      pass (exact normalized match with ambiguity escalation, entropy gate,
      MinHash/LSH fuzzy with blake2b hashing bit-identical to Python —
      pinned test vector), then one batched LLM escalation with upstream's
      guardrails (out-of-range/duplicate ids ignored, invalid candidate ids
      and omissions → new node), label promotion. Candidate gathering via
      grit's `find_merge_candidates` + MergeNodes op construction moved to
      the `pipeline.rs` task (grit requires persisted nodes).
- [x] `extract/edges.rs`: entity pairs → fat edges with fact sentences (ports
      `edge_operations.py` extraction path): edge-types context with
      signature map + default signature, latest-episode reference time
      rendered as Python `str(datetime)` (space separator — fidelity trap),
      max_tokens=16384, endpoint validation (unknown names + self-edges
      dropped), empty-fact filter, lenient `fromisoformat`/`ensure_utc`
      timestamp parsing (chrono), raw-first-index reference_time semantics.
      DraftEdge/NodeRef/EdgeTypeSpec types in extract/mod.rs.
- [x] `dedupe/edges.rs`: edge resolution — dedup judgment AND temporal
      invalidation, ported together because upstream fuses them in
      `resolve_extracted_edge`: verbatim fast path (endpoints + normalized
      fact), LLM resolve_edge with continuous idx across related/existing
      lists and invalid-id guardrails, timestamp extraction for new edges
      (small model; failures swallowed like upstream), invalid_at→expired_at
      rule, newer-candidate-expires-new-edge rule, and
      `resolve_edge_contradictions`. Time injected via a `now` parameter.
      This also covers the separate `invalidate/` roadmap item.
- [x] `invalidate/`: folded into `dedupe/edges.rs` above (upstream keeps
      dedup + invalidation in one function; splitting would hurt fidelity).
- [x] `summarize/`: node summary refresh (ports
      `_extract_entity_summaries_batch` + `_process_summary_flight` +
      `truncate_at_sentence` + `_truncate_type_description`): fact-append
      shortcut under 2×MAX_SUMMARY_CHARS, MAX_NODES=30 flights (sequential —
      recordings identical to upstream's concurrent flights), small-model
      SummarizedEntities calls, case-insensitive name application with
      sentence-aware truncation, skip_fact_appending episode-prompt path,
      per-node filter hook.
- [x] `pipeline.rs`: `add_episode` seam stringing the steps onto grit's op
      vocabulary: drafts → AddNode (kind = specific label, full labels in
      attrs), find_merge_candidates pools (min 0.6, top 15, batch-mates
      excluded) → resolve → MergeNodes, extract_edges over resolved NodeRefs,
      per-edge resolution with related/invalidation pools from 1-hop
      traversal, AddEdge (+InvalidateEdge for pre-bounded facts),
      InvalidateEdge for contradictions, AddEpisode last with mentions.
      End-to-end offline test against a real grit DB (two episodes: merge +
      invalidation verified in storage). `now` injected.
- [ ] BLOCKED(user decision: grit op-vocabulary extension) Persist summary
      refresh + label promotion: grit v0.1 has no node-update op (AddNode /
      AddEdge / AddEpisode / InvalidateEdge / MergeNodes / Purge only), so
      the summarize step's output and dedup label promotion currently can't
      be written back. Needs a grit-side decision (e.g. a `SetNodeSummary` /
      `UpdateNode` op — touches the sync vocabulary, convergence property
      tests, and a grit release + version bump per the umbrella co-dev flow).
- [ ] Edge invalidation-candidate gathering parity: pipeline uses 1-hop
      traversal around the endpoints; upstream gathers candidates by hybrid
      search over edge facts. Revisit once embeddings are wired (grit search
      needs vectors) and judge against golden trace #1 — record in
      DEVIATIONS.md if the difference survives.
- [x] `search/`: the default `graphiti.search` surface (EDGE_HYBRID_SEARCH_RRF
      path) over grit's fused hybrid retrieval — edge hits filtered from
      grit's ranking with over-fetch, limit applied; rank-order parity is the
      conformance target (score parity impossible by construction). Advanced
      recipes (MMR, node-distance, cross-encoder, communities) deliberately
      not ported per AGENTS.md.
- [ ] Conformance harness: `tests/conformance.rs` loading `oracle/` fixtures,
      diffing graph state field-for-field and retrieval by rank.
      BLOCKED(until golden trace #1 exists) for the assertion half; the
      loader + differ can land first.
- [ ] Real Claude API `LanguageModel` client behind a `claude` feature flag
      (off by default; manual integration test only — never in `cargo test`).

## Later (do not start without a user decision)

- [ ] Custom edge-attribute extraction (pydantic edge models with fields →
      `extract_edges.extract_attributes` + `apply_capped_attributes`) —
      deferred from `dedupe/edges.rs`; not exercised by trace1. Port when a
      trace or Layer 3 needs custom edge ontologies.
- Golden traces #2+ (bulk-ish episode volume, group_id isolation, purge).
- Search cross-encoder reranking evaluation.
- Communities port. Saga summarization. crates.io publish of nacre-core.
