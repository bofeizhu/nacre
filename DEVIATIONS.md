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

## Atomic episode commit

**What diverges:** Upstream finishes every model step first — the
resolution phase issues no writes at all — and then persists the episode in
a single call to `add_nodes_and_edges_bulk`, reached once from
`add_episode` after extraction, resolution and attribute hydration. That
call is not a transaction on the FalkorDB driver:
`FalkorDriverSession.execute_write` simply awaits the function it is handed
with the session itself standing in for the transaction (no BEGIN, no
commit), so `add_nodes_and_edges_bulk_tx` generates whatever name/fact
embeddings are still missing and then issues four independent queries —
episode nodes, entity nodes, episodic edges, entity edges. The embedder
calls all precede the first write, so an embedding failure writes nothing;
the exposure is the gaps between the four queries. A failure or crash
between them leaves the earlier writes standing — an episode plus its
entities and mention edges, without the entity edges that carry the facts,
which are saved last. That seam carries no idempotency key either, so a
retry is a fresh episode over a graph already holding half of the first
attempt. nacre commits an episode's entire write set (nodes, merges, label
promotions, edges, invalidations, summaries, embeddings and the episode
row) in ONE grit transaction, at the end of a planning phase that touches
no storage. Nothing is written until everything is decided; an error writes
nothing at all. `EpisodeInput.id` makes the retry expressible: a caller
that re-submits a failed ingest under the same id either completes it
exactly once or gets `PipelineError::EpisodeExists`.

Consequence for the pipeline's internals: reads that used to see
already-applied ops now consult a pending-write overlay
(`pipeline.rs::Pending`) that answers with what the batch will store. Op
kinds and their order are unchanged — the overlay exists to keep them
unchanged.

**Why acceptable:** Strictly stronger than upstream, and invisible in
end-state. Both traces produce the same graph, op-for-op, and any run that
fails is a run upstream would have left half-written — a state the fixtures
never describe. The one behavior a caller can observe is the new error on a
duplicate id, which is nacre-only API surface (upstream has no idempotency
key at this seam).

**Asserted by:** `tests/op_sequence.rs` (the ops and their order are
byte-identical to the pre-batch pipeline, both traces),
`tests/conformance.rs` (end state unchanged), `tests/atomicity.rs` (a
truncated recording set fails an ingest mid-flight; the graph, its
provenance links, its vectors and the oplog come out identical, and the
retry ingests exactly once), and
`pipeline.rs::tests::overlay_serves_pending_writes_within_one_episode`
(the label-promotion path, which neither trace exercises).

## Episode deletion: earliest evidence stands in for `episodes[0]`

**What diverges:** Upstream's `remove_episode` deletes an edge when
`edge.episodes[0] == episode.uuid`. Its episode attribution is an
append-ordered array, so slot 0 names the episode that first stated the
fact. grit's `mentions` is a SET (primary key `(episode_id, target_id)`)
with no insertion order and no column that could carry one, so that index
does not exist here. `forget::delete_episode` derives the same fact
instead: an edge's creator is the earliest-recorded episode mentioning it —
`Episode.created_at`, ties broken by id. The node rule needs no derivation:
upstream counts the episodes mentioning a node and deletes at exactly one,
which is `mentions_of(id) == [episode_id]`.

Three consequences of grit's data model, all deliberate:

- **`mentions_of` resolves merge aliases and unions across them**, while
  `episode_targets` returns the raw ids the episode wrote. The asymmetry is
  what makes the node count right — a canonical node inherits the mentions
  of everything folded into it, so "would anything still evidence this
  entity?" is answered over the surviving entity, not over one alias. Its
  inverse is the one place the asymmetry bites: a target merged away AFTER
  the mention was recorded is weighed and erased through its stub, leaving
  the node that absorbed it standing with no evidence. Only a hand-written
  or synced graph gets there — the pipeline attributes canonical ids and
  merges only fresh drafts into stored nodes.
- **Merged-away draft rows are not reachable from any episode.** The
  pipeline attributes the CANONICAL id in `mentions`, and grit keeps the
  absorbed draft as an expired audit row; nothing links that row to the
  episode that drafted it, so no episode deletion erases it. Upstream never
  persists such a row at all, so there is no upstream behavior to match.
  `forget::delete_group` does sweep them — it enumerates rows, not
  evidence.
- **No incident-edge cascade.** Upstream's node deletion is `DETACH
  DELETE`, which takes every relationship touching the node; grit's purge
  deliberately does not cascade (a cascade would make the op
  order-dependent against concurrent writes). The two agree on anything
  this pipeline wrote, because an episode mentions the endpoints of every
  edge it mentions: a node whose only evidence is this episode therefore
  has no incident edge with other evidence, and both rules erase the same
  set. The divergence is reachable only in a hand-written graph that breaks
  that invariant, where an edge can be left with a purged endpoint —
  filtered out of traversal and search, still present in the group scan.

**Why acceptable:** `created_at` is stamped from the op's HLC wall time —
the issuing device's clock — so for a graph built by sequential ingestion
it IS the order upstream's array was appended in, and the two rules name
the same episode. They can differ only where upstream's own array order is
not creation order: an episode ingested out of order, or one arriving by
sync from a device whose clock is skewed. Neither is expressible upstream
(no idempotency key, no sync), so there is no upstream behavior being
broken — and the id tie-break keeps the rule total and deterministic on
every device.

**Asserted by:**
`tests/deletion.rs::delete_episode_erases_sole_evidence_and_keeps_the_corroborated`
(all four cases — sole-mention node erased, corroborated node kept, created
edge erased, re-stated edge kept — in one graph) and
`::trace1_deletion_leaves_the_sibling_group_byte_identical` (the same rule
over pipeline-written state, against a sibling group as control). The alias
corner is pinned by
`::a_target_merged_away_after_the_mention_is_erased_through_its_stub`.

One thing the batch does not extend to: the *decision* is a sequence of
reads outside it, and grit's reader never blocks behind its writer, so a
concurrent ingest on the same handle can corroborate a node after
`delete_episode` has weighed it and still lose it to the purge. grit exposes
no transaction to Layer 2 by design, so the fix is the caller's — serialize
deletes against ingests on the same group. Upstream holds no transaction at
any point, so this is not a fidelity regression.

## Episode deletion resets kept nodes' summaries

**What diverges:** Upstream's `remove_episode` keeps every node another
episode also mentions and leaves its summary exactly as it is.
`forget::delete_episode` resets the summary of every node it keeps and writes
a fresh list of the node's surviving facts in its place. Same transaction as
the purge: the batch is the `Purge`, one `PurgeNodeFields` on the kept nodes'
`summary` (grit ≥ 0.5), then one `UpdateNode` per kept node with a surviving
fact. The rebuild is model-free: the node's incident edges in creation
order, in the node's own group, with both endpoints live, newest whole facts
within `2 × MAX_SUMMARY_CHARS`.

**Why:** the summarize pass rewrote that summary while the deleted episode
was being ingested. In the fact-append shortcut it literally appended the
facts of the edges that episode created — the very facts the purge erases —
and every later summary of the node descended from that one. Kept as is,
the deleted episode's words survive in the summary and go back to the model
on the next ingest that touches the node: the dedupe prompt's EXISTING
ENTITIES shows the first 120 characters, the summarize prompt the whole
summary. Measured on both golden traces before the fix: every episode
deletion left the facts it created, verbatim, in the summaries of nodes it
kept. "Take one memory back" was not true for any host that routes many
episodes through one shared entity — which is the shape of most agent
memory (one user, one subject).

**Why acceptable:** deletion makes no model call, so no oracle applies
(README); the question is only what state is honest. A reset alone would be
honest but costly: every other episode's contribution to the summary would
go with it, and in the fact-append regime the next passes append only their
own facts, so it would never come back — a shared entity would lose its
whole summary on every deletion. The rebuild restores what the graph still
holds, in the shape upstream itself stores (an unsummarized fact list is
what the fact-append shortcut writes, up to the same bound). Over the
golden traces it is exact: deleting each trace's last episode, six of the
seven rebuilt summaries equal the old summary with the deleted episode's
facts removed, and the seventh differs only where upstream appended one
fact twice (a node two drafts of one episode resolved to), which the
rebuild lists once. Where an LLM pass wrote part of a summary, that part is
lost and its facts return as list lines — stated, not hidden.

What the reset reaches is grit's tiered erasure for fields
(`PurgeNodeFields`): every earlier value leaves the live row, the per-field
record, the FTS mirrors (delete-segment residue until `compact()`) and the
op-log, and a floor strips a stale value that arrives later. What it does
not reach is listed in `forget.rs` ("What forgotten means"): facts a later
episode extracted with the deleted one in its context window, invalidations
the deleted episode caused, label promotions and names, and a concurrent
writer whose summary refresh carries a later HLC.

**Asserted by:**
`tests/deletion.rs::delete_episode_resets_and_rebuilds_the_summaries_it_keeps`
(rebuild rule incl. the cross-group and ghost-endpoint exclusions; the
forgotten text absent from summaries, op-log, export and search; the batch
shape), `::a_rebuilt_summary_keeps_the_newest_facts_within_the_bound`,
`::a_later_ingest_never_sees_what_the_deleted_episode_taught` (every request
of a later ingest swept, with positive controls that the rebuilt summaries
reached both the dedupe prompt and an LLM summary pass), and
`::trace1_deletion_leaves_the_sibling_group_byte_identical` (the fidelity
claim above, over pipeline-written state; the sibling stays byte-identical
and the op-log moves by exactly the documented entries). The deletion batch
of both traces is pinned in `oracle/fixtures/*/op_sequence.json`
(`deletion`).

## Episode metadata is a nacre extension

**What diverges:** `EpisodeInput.metadata` has no upstream counterpart —
Graphiti's `EpisodicNode` carries no application payload. Nacre forwards it
verbatim into `GraphOp::AddEpisode` (grit schema v6) and hands it back on
every episode read path, including the previous-episode window
`retrieve_previous_episodes` rebuilds.

It exists because Layer 3 needs to tie a memory back to whatever produced
it — the outbox row, the document version, a consent tag — and the honest
place for that is the episode itself. The alternative is a side table
keyed by episode id, which drifts out of sync exactly when it matters
(after a partial write, or after a deletion).

**Why acceptable:** Trace-neutral by construction, in three senses.
`None` is skipped by grit's op serialization, so an op that does not use it
is byte-identical to a pre-v6 one — the golden traces set no metadata and
their op-sequence fixtures regenerate unchanged. No prompt context builder
reads it (all six cherry-pick `content`, `valid_at` and
`source_description`), so it cannot reach a model, change a request, or
invalidate a recording. And it plays no part in identity: the ingest
idempotency check keys on the id alone.

One consequence a caller must know: episode metadata is **write-once**.
grit has no update op for episodes and `add_episode` refuses a second
ingest under a stored id, so changing it means `forget::delete_episode`
plus a fresh ingest **under a new id**. The delete tombstones the old id
permanently, and grit gates its inserts on the tombstone table: re-ingesting
under it passes the idempotency check (the row really is absent), runs every
model call and commits, but stores neither the episode row nor its mentions
while the nodes and edges beside them land — `Ok`, over a graph whose new
rows no episode evidences. grit exposes no per-id tombstone read, so nacre
cannot refuse the reuse; retiring the id is the caller's job.

**Asserted by:**
`pipeline.rs::tests::metadata_is_stored_read_back_and_never_reaches_a_prompt`
(a sentinel on both the ingested episode and the previous-episode window,
swept across every request the run issues, plus the round-trip through
`Grit::episode` and the group scan) and `tests/op_sequence.rs` (both
traces' pins regenerate byte-identical with the field present). The
delete-and-re-ingest trap above is pinned by
`tests/deletion.rs::re_adding_a_purged_id_is_a_silent_no_op`.
