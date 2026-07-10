# DEVIATIONS.md — accepted divergences from the golden traces

Every place the Rust stack (nacre + grit) deliberately does NOT reproduce a
golden-trace fixture is recorded here: what diverges, why it is correct or
acceptable, and which fixture/test asserts the divergent behavior. A red
conformance test may only merge together with its entry here.

## Edge dedup/invalidation candidate pools are engine-free and fact-sorted

**What diverges:** Upstream builds the `EXISTING FACTS` and `FACT
INVALIDATION CANDIDATES` prompt lists from a FalkorDB hybrid search
(RRF over RediSearch BM25 + HNSW cosine, truncated to
`RELEVANT_SCHEMA_LIMIT` = 10) and renders them in engine result order.
Neither the membership at the truncation margin nor the order is
reproducible: FalkorDB's `collect()` does not honor the preceding
`ORDER BY`, index scan order varies per process (verified empirically —
capture and replay shuffled identical candidate sets; the order matches
neither cosine-score-descending nor creation order), HNSW is built with
random level draws, and BM25/HNSW rankings are engine-specific, so no
other storage engine can reproduce the top-10 cut. Both sides of the
oracle replace the search with an engine-free equivalent:

- pool = ALL edges saved in the group before the current episode
  (upstream bulk-saves an episode's edges after resolving the whole
  batch, so same-episode edges are never candidates);
- same-endpoint-pair edges → `EXISTING FACTS` (upstream's
  `get_between_nodes` semantics, preserved through the `edge_uuids`
  search filter); the rest → `FACT INVALIDATION CANDIDATES`;
- both lists sorted by `(fact, uuid)` — a total, portable order (UTF-8
  byte order == code-point order, so Rust and Python agree); the uuid
  tie-break only reorders byte-identical facts, which render identically.

Harness side: `oracle/capture.py` monkeypatches the edge-resolution
`search` binding. nacre side: `pipeline.rs::group_edges_snapshot` + the
sort before `resolve_extracted_edge`.

**Why acceptable:** Candidate pools feed LLM prompts, so unlike retrieval
rank order they change graph state and must be pinned deterministically.
At oracle-trace scale (well under ~10 relevant edges per query in spirit,
17 edges total in trace1) the engine-free pool is a superset of upstream's
top-10 relevance cut and carries the same information to the LLM; the
related/invalidation split and the same-episode exclusion — the semantics
that matter — are preserved exactly.

**Asserted by:** `tests/conformance.rs::golden_trace1_conformance` (every
`EdgeDuplicate` recording lookup embeds the pools — wrong membership or
order is a replay miss).

## Node dedup candidate search is engine-free pinned-arithmetic cosine

**What diverges:** Upstream's `_semantic_candidate_search` ranks node dedup
candidates with an in-engine vector search (`node_similarity_search`,
limit 15, strict `score > 0.6`) — the same nondeterminism class as the
edge hybrid search (engine scan order, HNSW randomness, engine-internal
float paths). Both sides of the oracle replace it with an equivalent
either can compute bit-for-bit:

- embed the extracted names (upstream's own query batch) and every
  distinct existing group-node name, sorted, so capture and nacre issue
  byte-identical — and therefore recordable/replayable — embedder requests;
- rank existing nodes by cosine over f32-truncated components with
  sequential f64 accumulation (`pipeline.rs::cosine_f64` ==
  `capture.py::_cosine_f64`, IEEE-identical), strict `> 0.6`, limit 15,
  ties broken by uuid.

Harness side: `oracle/capture.py` monkeypatches
`node_operations._semantic_candidate_search`. nacre side: the candidate
block in `pipeline.rs::add_episode` (this replaced grit's
`find_merge_candidates` in the pipeline; grit keeps that API for its own
callers).

**Why acceptable:** Candidate pools feed the dedupe prompt, so they must be
pinned deterministically (see the edge-pool entry). The selection rule —
cosine over the same recorded embeddings, same threshold, same limit — is
upstream's own semantics with the engine variability removed. The uuid
tie-break is per-run, but score ties require byte-identical names, which
the exact-match fast path resolves before any prompt is built. Cost: the
harness re-embeds existing names once per episode (recorded, so
replay/conformance never touch the network); nacre reads the vectors it
persisted at write time and embeds only names with no stored vector —
identical values, fewer requests.

**Asserted by:** `tests/conformance.rs::golden_trace1_conformance` (every
`NodeResolutions` recording lookup embeds the pools and their order).

## `expired_at` compared by presence only, derived from `invalid_at`

**What diverges:** Upstream stamps `expired_at` with `utc_now()` whenever a
resolved edge carries `invalid_at` at save time and on contradiction
invalidations — pure wall-clock provenance, different in every run. In this
pipeline the invariant is `expired_at present ⟺ invalid_at present`. grit
deliberately does not set edge `expired_at` for invalidations (in grit's
read model an expired row is retracted from belief entirely, which would
break time-travel queries; the belief history of an invalidation lives in
`edge_invalidations.recorded_at`). The conformance dumper derives the
Graphiti-facing flag from `invalid_at`; the differ compares presence only.

**Why acceptable:** Unlike `valid_at`/`invalid_at` (semantic times asserted
to the second), `expired_at` carries no information here beyond the
invariant above.

**Asserted by:** `diff_states` in `tests/conformance.rs` (a wrongly
expired / wrongly live edge still fails; only the timestamp value is
ignored). Same normalization in `oracle/capture.py::_replay_verdict`.

## Edge invalidations: first decision wins within an episode

**What diverges:** Upstream mutates hydrated edge objects per resolution
and saves every mutated object in ONE FalkorDB bulk `UNWIND … MERGE … SET`.
For duplicate uuids in that list, FalkorDB keeps the FIRST occurrence
(verified empirically on trace1: the same edge was assigned invalid_at
2026-07-01 then 2026-06-08 across two resolutions; the database holds
2026-07-01). nacre reproduces this by collecting one invalidation per edge
per episode — first decision wins — and applying a single `InvalidateEdge`
after the resolution loop. A duplicate resolution of a stored edge occupies
that edge's first-wins slot (upstream's resolved entries precede the
invalidated ones in the bulk list), blocking later contradictions of it.

**Why acceptable:** Some deterministic rule must be chosen and the oracle
defines it. Note grit's own `InvalidateEdge` folds concurrent invalidations
to the MINIMUM event time across ops (a sync-convergence law); nacre's
one-op-per-episode collection keeps the two rules from meeting. A future
episode that "raises" an earlier invalid_at cannot be expressed through
grit's monotone fold — if a trace ever exercises that, it needs a grit
decision, not a silent workaround.

**Asserted by:** `tests/conformance.rs::golden_trace1_conformance`
(edges e8/e14 in trace1 pin the first-wins outcome).

## Episode `entity_edges` compared as the attributed set

**What diverges:** Upstream's episode `entity_edges` lists every resolved
edge once per draft (with duplicates) PLUS contradiction-invalidated edges,
without attributing the episode on those invalidated edges' own `episodes`
lists. grit has one mentions table (a set) driving both directions, so
nacre's episodes mention exactly the attributed edges. The differ restricts
both sides' `entity_edges` to edges whose `episodes` list contains the
episode, sorted and deduplicated.

**Why acceptable:** Multiplicity carries no information (upstream itself
set-deduplicates on read paths), and the invalidation linkage dropped by
the normalization is recoverable in grit from the oplog (the InvalidateEdge
op). If invalidation provenance is ever needed as first-class data, the op
could carry an episode id — a grit decision for later.

**Asserted by:** `filter_unattributed_entity_edges` +
`normalize_expired_at` in `tests/conformance.rs`.

## Retrieval rank order is not asserted against the fixture

**What diverges:** `retrieval.json` records Graphiti's RRF search results at
capture time, but FalkorDB's HNSW vector index is built with random level
draws, so approximate-KNN results vary per process. Verified empirically:
`capture.py --replay` (identical recorded embeddings, identical graph) gets
different rank order on 3 of 5 trace1 queries and different top-10
membership on 2 of them. Pinned Python Graphiti cannot reproduce its own
retrieval, so it cannot be a rank oracle for grit.

**Why acceptable:** Rank-order conformance is only meaningful against a
deterministic reference. grit's RRF search is deterministic and covered by
grit's own tests. The conformance test still asserts retrieval sanity:
every fixture result exists in the ingested corpus and every trace query
returns hits from grit. `capture.py --replay` prints a per-query
rank/set comparison as an advisory signal.

**Asserted by:** the sanity block in
`tests/conformance.rs::golden_trace1_conformance`.
