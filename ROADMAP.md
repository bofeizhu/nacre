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
- [x] First capture run → golden trace #1 + recordings committed
      (2026-07-10, DeepSeek V4 LLM + Zhipu embedding-3, FalkorDB digest
      pinned in docker-compose.yml). Getting the trace *deterministic*
      surfaced four harness/port fixes, all landed: (1) `prompt_library`'s
      `VersionWrapper` appends `DO_NOT_ESCAPE_UNICODE` to every system
      message at render time — was missing from the Rust port AND the
      fixture generator (both fixed, fixtures regenerated); (2) `clear_data`
      only clears the default graph — the group's shard needed clearing too;
      (3) LLM responses are now validated against the response model before
      recording (DeepSeek occasionally echoes the schema in json_object
      mode); (4) FalkorDB's search is nondeterministic across processes
      (collect() ignores ORDER BY, HNSW random levels) — candidate pools for
      node dedup and edge dedup/invalidation are now engine-free on both
      sides of the oracle (see DEVIATIONS.md, three entries).
      `capture.py --replay` verifies graph-state determinism and fails
      loudly; retrieval fixtures are advisory (DEVIATIONS.md).
- [x] Conformance GREEN for trace1 (2026-07-10): `cargo test --test
      conformance` replays all 5 episodes byte-exactly and the graph-state
      diff is clean. Unblocked by grit 0.2's `UpdateNode` (per-field LWW —
      summaries, name/label promotion persist) and `AddEdge.invalid_at`
      (extraction-time bounds without belief retraction). The shakeout
      surfaced and fixed, in nacre: merged-away drafts leaking into
      candidate pools, edge extraction seeing resolved instead of extracted
      names, upstream's draft-edge dedup prologue, directed
      get_between_nodes semantics, the 10-episode previous-window
      (RELEVANT_SCHEMA_LIMIT, not EPISODE_WINDOW_LEN), upstream's
      shared-object summary accumulation, and FalkorDB's first-write-wins
      bulk-save semantics for duplicate uuids (three new DEVIATIONS.md
      entries pin the last one and the dump normalizations).

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
- [x] Persist summary refresh + label promotion — resolved 2026-07-10 by
      grit 0.2.0's `UpdateNode` (user chose it over SetNodeSummary; per-field
      LWW, schema v2, released + published). Wired in pipeline.rs; asserted
      by the green conformance test.
- [x] `search/`: the default `graphiti.search` surface (EDGE_HYBRID_SEARCH_RRF
      path) over grit's fused hybrid retrieval — edge hits filtered from
      grit's ranking with over-fetch, limit applied; rank-order parity is the
      conformance target (score parity impossible by construction). Advanced
      recipes (MMR, node-distance, cross-encoder, communities) deliberately
      not ported per AGENTS.md.
- [x] Conformance harness: `tests/conformance.rs` — grit-side aliased state
      dumper mirroring capture.py (content-derived aliases; capture.py's
      edge-sort tiebreak fixed to use aliases so ordering is portable),
      recursive field-for-field differ with created_at excluded, retrieval
      rank comparison, and the full trace-replay test body (episode window 3,
      per-episode reference-time clock). Skips loudly until golden trace #1
      exists; the differ is exercised by an offline self-test. Expect
      first-contact shakeout when the trace lands.
- [x] Real Claude API `LanguageModel` client behind a `claude` feature flag
      (off by default; never in `cargo test`): raw HTTP against /v1/messages
      (no official Rust SDK), structured outputs via `output_config.format`
      json_schema with a registry mirroring src/schemas.rs (the recording
      contract keys on pre-mutation messages, so the client is free to use
      the native mechanism instead of upstream's schema-append mutation),
      Opus 4.8 / Haiku 4.5 tier mapping, refusal/truncation handling,
      429/5xx retry with backoff. Offline unit tests cover the request
      builder, schema registry, and response parsing. Wrap in
      RecordingModel for capture runs.
- [x] Edge invalidation-candidate gathering parity — resolved 2026-07-10:
      the 1-hop traversal was replaced by engine-free full-group pools
      (directed same-pair split, fact-sorted) applied identically by the
      capture harness; membership and order are pinned by every EdgeDuplicate
      replay lookup in the green conformance test (DEVIATIONS.md "Edge
      dedup/invalidation candidate pools").

## Milestone 4 — Phase 1.5: retrieval that works (embeddings persisted + query leg)

Rationale: nacre computes embeddings for dedup but never persists them, so
grit's hybrid search runs FTS-only (AND-semantics; question-form queries
return nothing) and dedup re-embeds every existing name each episode.
Division of labor stays sharp: nacre embeds, grit stores and serves back.
Cross-repo increments use the umbrella co-dev flow — UNCOMMITTED
`[patch.crates-io]` override while working, grit gets its own full gate
(`cargo fmt/clippy/test/doc` in ../grit) and its own commit per increment.
Conformance (`cargo test --test conformance`) is the regression net for
every step: it must stay green, and fewer/identical embedder requests are
fine (replay only fails on unrecorded requests nacre makes).

- [x] grit 0.2.1 (additive, no schema change): embedding getters
      `get_node_embedding(id)` / `get_edge_embedding(id)` (read the vector a
      caller stored — the write half already exists) and a group-scan API
      (`nodes_in_group` / `edges_in_group` / `episodes_in_group` or
      equivalent, live/all filtering like the export view) so callers stop
      parsing `export_jsonl`. Doc-commented with examples, unit tests incl.
      "getter returns exactly what the setter stored" and scan/export
      consistency. Full gate in ../grit, commit there (version 0.2.1), then
      add the uncommitted `[patch.crates-io]` override in nacre.
      → done 2026-07-10 (grit fa75fad): scans return full rows (callers
      filter on expired_at — no boolean params); episodes chronological;
      getters None when no model/vector; note `cargo update -p grit-core`
      is needed after adding the patch override (the lockfile pins the
      registry 0.2.0 otherwise and the patch is silently unused).
- [x] nacre: replace the `export_jsonl`-parse snapshots in pipeline.rs with
      grit's group-scan API (pure refactor; conformance green proves it).
      → done 2026-07-10: group_edges_snapshot deleted outright
      (edges_in_group call inlined); group_nodes_snapshot kept as a thin
      live-only filter over nodes_in_group (the merged-away-drafts exclusion
      rationale lives there). Scan order matches the old export order
      (id-ordered), so candidate pools are byte-stable — conformance green.
- [ ] nacre: persist embeddings at write time, mirroring upstream's
      `create_entity_node_embeddings` / `create_entity_edge_embeddings`:
      consult the pinned Python for the exact input strings (e.g.
      `name.replace('\n', ' ')`) and batch composition, and verify each
      batch request key exists in trace1's `embedder_recordings.json` BEFORE
      wiring (the conformance ReplayEmbedder fails loudly on any unrecorded
      request). `register_embedding_model("embedding-3", 1024, "")` at
      pipeline setup; `set_node_embedding` after AddNode/UpdateNode(name),
      `set_edge_embedding` after AddEdge. add_episode signature may grow a
      setup step; update all callers.
      → done 2026-07-10 (eefdf21): verified batch formats first — node batch
      is RAW names (no newline replace; that's the singleton path only),
      per-draft RESOLVED names with duplicates preserved; edge batch is raw
      draft facts post-dedup. register_embedding_model runs from
      Embedder::meta() each add_episode (idempotent). Signature unchanged.
      Conformance now also asserts every live node + every edge carries a
      1024-dim vector.
- [x] nacre: dedup reads stored vectors — for existing nodes, take
      `get_node_embedding` instead of re-embedding every name each episode;
      embed only names with no stored vector. Values are identical (same
      input string, same recorded vector, same f32 truncation), so candidate
      pools and prompts do not move; conformance green is the proof.
      → done 2026-07-10: conformance passes with ZERO fallback embeds (all
      existing names served from vectors persisted by earlier episodes);
      the missing-names fallback keeps the harness's sorted-unique batch
      shape for cold graphs.
- [x] nacre: query-embedding leg in `search/` — `search_edges` accepts an
      embedder (or pre-computed query vector), embeds the query the way
      upstream does for `graphiti.search` (verify the exact input string
      against the recorded `{"inputs": [query]}` keys in trace1), and passes
      the vector into grit's `Query` so RRF actually fuses vector + FTS.
      Update the conformance retrieval sanity block to pass the
      ReplayEmbedder; a question-form query returning hits is the smoke
      signal.
      → done 2026-07-10: verified input = `query.replace('\n', ' ')`
      singleton; search_edges is now async with an Embedder param and
      returns PipelineError. Conformance asserts non-empty hits for ALL
      five trace queries — "Where does Priya work?" (previously zero hits,
      FTS-AND only) now resolves through the fused vector leg.
- [x] nacre: previous-episodes helper — fetch the last-10 window
      (`RELEVANT_SCHEMA_LIMIT`, occurred_at <= reference, ascending) from
      grit via the group-scan API, mirroring upstream `retrieve_episodes`;
      switch the conformance test's hand-threaded window to it (staying
      green proves the helper reproduces the recorded prompt windows).
      → done 2026-07-10: `pipeline::retrieve_previous_episodes` +
      `PREVIOUS_EPISODE_WINDOW` (=10); inclusive <= reference, chronological;
      timestamps re-render at second precision matching Python isoformat.
      Conformance now sources its windows from grit itself — green.
- [x] nacre: real embedder client behind an `openai-embed` feature flag
      (reqwest, OpenAI-compatible `/embeddings`, configurable base URL +
      model + dim truncation — Zhipu embedding-3 is the first target; never
      compiled into `cargo test` default). Offline unit tests for request
      building and response parsing only.
      → done 2026-07-10: `model::openai_embed::{OpenAiEmbedder,
      OpenAiEmbedConfig}` with a `zhipu()` preset (1024 dims, 64-input
      batch chunking — Zhipu's cap); responses re-ordered by index and
      truncated client-side (MRL slice, no renormalization); 429/5xx retry
      shares claude.rs's tokio-free sleep (hoisted to model/mod.rs).
- [x] nacre: `claude.rs` configurable base URL + API key env (defaults
      unchanged: api.anthropic.com). Purpose: DeepSeek's Anthropic-style
      endpoint becomes usable through the same client. Verify whether it
      supports `output_config` json_schema; if not, add the
      schema-append-to-prompt fallback (the pattern upstream's
      OpenAIGenericClient json_object mode uses — nacre ported its prompt
      side already).
      → done 2026-07-10: ClaudeConfig gains base_url +
      StructuredOutput::{JsonSchema, SchemaInPrompt}; `deepseek()` preset
      (api.deepseek.com/anthropic, deepseek-chat, schema-in-prompt) since
      output_config support there is unverified offline — the smoke test
      decides empirically. Fence stripping added to response parsing
      (harmless for native mode). Keys stay caller-provided; env reading
      belongs to the example, not the library.
- [x] Live smoke example (`examples/`, feature-gated, requires env keys,
      NEVER in cargo test): ingest a handful of real conversation turns with
      a live LLM (Claude or DeepSeek) + live Zhipu embeddings into a fresh
      grit file, then run a few searches and print results with provenance.
      First end-to-end run outside replay; expect to shake out retry/rate
      limit/error-surface gaps — fix them as part of this task.
      → done 2026-07-10 in 2 live runs: examples/live_smoke.rs (DeepSeek
      Anthropic-endpoint default, NACRE_SMOKE_ANTHROPIC_KEY switches to
      Anthropic). Run 1 shook out exactly the expected gap — deepseek-chat
      echoes schema scaffolding in prompt-only mode (same failure class the
      Python capture hit) — fixed with required-keys validation +
      salvage_shape unwrap + retry in claude.rs SchemaInPrompt mode. Run 2:
      3 turns ingested (extraction, 5 merges, 1 invalidation
      Rotterdam→Utrecht, summaries, embeddings persisted); all 3 queries
      answered correctly with provenance. The full stack works live.
- [x] Release grit 0.2.1 to crates.io, drop nacre's patch override,
      regenerate Cargo.lock against the registry, full gates both repos,
      commit + push both. Nacre's `grit-core = "0.2"` requirement already
      accepts 0.2.1.
      → done 2026-07-10 (user-approved publish): grit-core + grit-compat
      0.2.1 on crates.io, grit tagged v0.2.1 and pushed (fa75fad); nacre
      locked to the registry checksum and pushed (c6a4762). Milestone 4
      complete.

## Milestone 5 — Layer 3 gateway: napi-rs bindings, visualization-first

Rationale: the Electron app ("Electron + RTK + napi-rs") is the product,
and memory-graph visualization is a must-have feature. Visualization is a
pure read-path concern and every primitive already exists (group scans,
budgeted traversal, bi-temporal as_of/as_at, node_history, mentions,
AddEpisodeOutcome deltas) — this milestone exposes them across the FFI
with TypeScript types, plus the write path and search. NOT in scope: the
Electron app itself (own repo later), npm publishing (user-gated, like
crates.io), and any new pipeline features — bindings wrap what exists.
Conformance stays the regression net for any core change a binding
motivates; core changes must not alter recorded requests.

- [x] `nacre-node` workspace crate (napi-rs, cdylib): scaffolding that
      `cargo build -p nacre-node` compiles and `npm run build` (via
      @napi-rs/cli, dev-dependency inside the crate dir) packages into a
      loadable `.node` addon with generated `.d.ts`. Workspace `cargo test`
      must stay green without any Node toolchain installed (the crate's
      own tests are Rust-side only). Document the build in the crate
      README.
      → done 2026-07-10: napi 3 / napi-build 2, Node v24 verified
      (`version()` loads). Note: feature unification means the default
      workspace test build now compiles nacre-core WITH claude+openai-embed
      (their offline unit tests run too — 69); tests remain zero-network.
- [x] Handle + write path: `Memory.open(path, deviceId)` wrapping Grit;
      `addEpisode(episode, options)` running the full pipeline with
      provider config passed from JS — `{provider: "anthropic" | "deepseek"
      | "replay", ...}` mapping to ClaudeConfig presets or a
      RecordingStore path (replay mode makes Node-side tests offline and
      deterministic, same recordings format as the oracle). Returns the
      AddEpisodeOutcome deltas (new/merged/invalidated ids) — the UI
      animation feed.
      → done 2026-07-10: providers dispatch through AnyModel/AnyEmbedder
      enums (RPITIT traits aren't dyn-compatible); previous-episode
      windows fetched internally; verified end-to-end offline — replay
      addEpisode of trace1 ep-0 through the built addon returns the
      expected deltas. Core change: SummarizeOptions' filter hook is now
      Send + Sync so pipeline futures cross the FFI (type-level only;
      conformance untouched).
- [x] Read path for the graph view: `nodesInGroup` / `edgesInGroup` /
      `episodesInGroup` (full rows, JSON), `traverse(seeds, {depth,
      budget, asOf, asAt})` returning the Subgraph, `nodeHistory(id)`,
      `mentionsOf(id)`, `retrievePreviousEpisodes`. Timestamps as ISO
      strings; ids as strings. These five calls are the entire data
      contract the graph visualization needs — document that explicitly
      in the .d.ts docs.
      → done 2026-07-10: rows.rs converters (labels unpacked from
      attrs["labels"], RFC-3339 ms timestamps, remaining attrs as plain
      JSON); previousEpisodes returns full rows (the pipeline helper's
      prompt-shaped output stays internal). Exercised end-to-end from
      Node: replay-ingest 2 episodes, then dump/traverse/history/
      mentions/window all verified.
- [x] Search: `searchEdges(query, group, limit)` with the embedder config
      from JS ({provider: "zhipu" | "openai-compatible" | "replay", ...});
      hits carry facts, validity, provenance episode ids.
      → done 2026-07-10: shook out a real search bug — small limits could
      return ZERO edges because grit's budget was spent on a mixed
      node-heavy fused ranking (EDGE_OVERFETCH=4 insufficient). Fixed
      properly in grit 0.2.2 (db1df19): Query::targets(&[SearchKind])
      filters the fused list BEFORE the budget cut; nacre's search_edges
      now uses edge-only targets with the exact limit (overfetch deleted).
      All five trace queries return hits through the FFI at limit 3.
- [x] Node-side offline test: a small JS test (node --test) driving
      open → addEpisode(replay recordings from a committed mini-fixture)
      → reads → searchEdges(replay), asserting the outcome deltas and a
      traversal shape. Runs only when the addon has been built; skipped
      (loudly) otherwise so cargo-only CI stays green.
      → done 2026-07-10: test/memory.test.mjs replays the committed trace1
      recordings (no new fixture needed — they ARE the committed
      deterministic fixture); covers deltas, all five viz reads, search
      with provenance, and the fail-loudly replay-miss surface. Skip path
      verified by hiding index.js.
- [ ] Live Node smoke (preapproved keys, same guardrails as the Rust
      smoke): a script mirroring examples/live_smoke.rs through the
      bindings — ingest 3 turns via DeepSeek+Zhipu, print deltas, run
      queries. Proves the FFI end-to-end.
- [ ] Viz starter artifact: `examples/viz/` — a script that dumps a smoke
      graph as JSON (nodes/edges with labels, validity, provenance) plus a
      minimal self-contained viewer.html (offline, no CDN) rendering it as
      a force-directed graph with an as-of time slider stub. Not a
      product — a data-contract proof the Electron app can copy from.
- [ ] Golden trace #2: text/json episode sources — the doc-ingestion bet
      for Layer 3. Curate `oracle/episodes/trace2.json` (a text document
      chunk + a json event episode alongside a message turn; 3-4 episodes,
      2-3 queries; exercises prompt routing by source and cross-source
      dedup). Capture against the digest-pinned FalkorDB (docker compose up
      in oracle/, keys from oracle/.env), verify `--replay` determinism,
      commit fixtures. Generalize `tests/conformance.rs` to run one test
      body per fixture directory under oracle/fixtures/ (trace1 must stay
      green untouched). Expect a shakeout on the text/json extraction
      prompts — fix in nacre or pin in DEVIATIONS.md, same discipline as
      trace1.
- [ ] BLOCKED(user: npm publish decision) Package/publish story for
      `nacre-node` (name, platforms, prebuilds). Everything before this
      works from a local build.

## Later (do not start without a user decision)

Coverage deepening, in the order agreed 2026-07-10 (activate when the
Electron app demonstrates the need, or on user say-so):

- [ ] Custom entity/edge ontologies: node + edge attribute extraction
      (`extract_attributes` + `apply_capped_attributes` paths) with a
      golden trace exercising typed entities — the highest-value gap for
      a real memory product.
- [ ] Golden traces #3+: bulk-ish episode volume (candidate-pool
      saturation), group_id isolation, purge/right-to-forget.
      (Trace #2 — text/json sources — was promoted into Milestone 5.)
- Search cross-encoder reranking evaluation.
- Communities-equivalent topic rollups, designed natively (not a port).
- crates.io publish of nacre-core.
