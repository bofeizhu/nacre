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
| `addEpisode` | `content`, `sourceDescription`, `name?`, `source?` ("message"\|"text"\|"json"), `groupId?`, `validAt?` (ISO-8601) | the change set: `{episodeId, nodeIds, newEdgeIds, merges, invalidatedEdgeIds}` |
| `searchEdges` | `query`, `groupId?`, `limit?` (default 10) | `{hits: [{id, sourceId, targetId, name, fact, validAt, invalidAt, episodes}]}` |
| `status` | — | `{version, initialized, groupId, episodes, liveNodes, edges}` |
| `shutdown` | — | `{ok: true}`, then the process exits; closing stdin also exits cleanly |

Errors are responses (`{"id", "error"}`), never crashes — the caller owns
retry/skip/circuit-breaker policy. `addEpisode` runs nacre's full pipeline
(≈5–10 LLM calls at smoke scale) — callers should queue it off the hot
path.

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
