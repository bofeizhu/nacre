# nacre sidecar

A tiny ndjson-over-stdio server over the built `nacre-node` addon, so
non-Node hosts (the Hermes memory provider in
[`integrations/hermes/`](../../../integrations/hermes/)) can drive nacre
as a supervised child process. stdio, not a port: the sidecar dies with
its parent, allocates nothing, and every request/response is a plain JSON
line you can replay by hand.

```sh
npm run build   # once — the sidecar loads ../index.js
node sidecar/sidecar.mjs
```

## Protocol

One request per stdin line, one response per stdout line, **strictly in
order** (episode order is meaningful; the sidecar never interleaves).

```
→ {"id": 1, "method": "init", "params": {...}}
← {"id": 1, "result": {...}}          | {"id": 1, "error": "message"}
```

| Method | Params | Result |
|---|---|---|
| `init` | `dbPath`, `deviceId`, `groupId` | `{version, dbPath, groupId}` — opens the grit file, validates env credentials up front |
| `addEpisode` | `content`, `sourceDescription`, `name?`, `source?` ("message"\|"text"\|"json"), `groupId?`, `validAt?` (ISO-8601), `id?` (UUID), `metadata?` | the change set: `{episodeId, nodeIds, newEdgeIds, merges, invalidatedEdgeIds}` |
| `searchEdges` | `query`, `groupId?`, `limit?` (default 10) | `{hits: [{id, sourceId, targetId, name, fact, validAt, invalidAt, episodes}]}` |
| `episode` | `id` | `{episode: {id, source, kind, content, occurredAt, groupId, createdAt, metadata} \| null}` — the point lookup behind a provenance id |
| `deleteEpisode` | `id` | `{episodeId, purgedNodeIds, purgedEdgeIds, keptNodeIds, keptEdgeIds}` — forgets one episode and what it alone evidenced |
| `deleteGroup` | `groupId?` (defaults to init's) | `{purgedNodeIds, purgedEdgeIds, purgedEpisodeIds}` — forgets a whole namespace; an unknown group is zero counts, not an error |
| `status` | — | `{version, initialized, groupId, episodes, liveNodes, edges}` |
| `shutdown` | — | `{ok: true}`, then the process exits; closing stdin also exits cleanly |

Errors are responses (`{"id", "error"}`), never crashes — the caller owns
retry/skip/circuit-breaker policy. `addEpisode` runs nacre's full pipeline
(≈5–10 LLM calls at smoke scale) — callers should queue it off the hot
path.

Two error messages carry a **stable prefix** to branch on, followed by
`": "` (the addon exports both as constants; napi cannot put a custom code
on an async rejection, so it rides in the message):

- `EPISODE_EXISTS` — `addEpisode` under an `id` the graph already holds.
  That ingest landed in full; mark it done rather than retrying.
- `EPISODE_NOT_FOUND` — `deleteEpisode` on an id the graph does not hold,
  including a second delete. Nothing was written.

`id` is the caller's ingest idempotency key and `metadata` its own opaque
payload — stored verbatim, never shown to a model, and handed back by
`episode`. Both are write-once: a deleted episode id is tombstoned
permanently, so re-ingesting a forgotten memory needs a new one.

The read path (`nodesInGroup`, `traverse`, `nodeHistory`, …) is
deliberately **not** exposed here — `status` aggregates what a supervising
parent needs, and `episode` exists only because `searchEdges` hands back
provenance ids that would otherwise be unreadable through this protocol.

## Credentials — environment only

Never on argv, never in a request. The parent passes these when spawning:

| Variable | Meaning |
|---|---|
| `NACRE_LLM_PROVIDER` | `deepseek` (default), `anthropic`, or `replay` |
| `NACRE_LLM_API_KEY` | required unless replay |
| `NACRE_LLM_BASE_URL`, `NACRE_LLM_MODEL`, `NACRE_LLM_SMALL_MODEL` | optional overrides |
| `NACRE_LLM_RECORDINGS` | replay mode: recordings path |
| `NACRE_EMBEDDER_PROVIDER` | `zhipu` (default), `openai-compatible`, or `replay` |
| `NACRE_EMBEDDER_API_KEY` | required unless replay |
| `NACRE_EMBEDDER_BASE_URL`, `NACRE_EMBEDDER_MODEL`, `NACRE_EMBEDDER_DIM` | optional overrides |
| `NACRE_EMBEDDER_RECORDINGS` | replay mode: recordings path |

Replay mode (`provider=replay` + recordings paths) makes the sidecar
fully offline and deterministic — that's how `npm test` exercises it
(`test/sidecar.test.mjs` replays the committed golden-trace recordings
through the real child process).
