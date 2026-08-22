# nacre-node

Node.js bindings for [nacre](../../README.md) — the Layer 3 gateway.
Thin `#[napi]` glue over `nacre-core` + `grit-core`; all logic lives in
those crates.

## Build

```sh
# Type-check / compile without any Node toolchain:
cargo build -p nacre-node

# Build the loadable addon (+ generated index.js / index.d.ts):
cd crates/nacre-node
npm install
npm run build            # napi build --platform   (debug)
npm run build:release    # optimized

node -e "console.log(require('./index.js').version())"
```

Generated artifacts (`*.node`, `index.js`, `index.d.ts`) and
`node_modules/` are gitignored; `package.json` + `package-lock.json` are
committed. The plain `cargo test --workspace` gate passes with no Node
toolchain installed — the addon is packaged only by `@napi-rs/cli`.

## Try it

- `npm test` — offline Node-side tests (replays the committed golden
  trace; skips loudly if the addon isn't built).
- [`examples/viz/`](examples/viz/README.md) — a self-contained
  memory-graph viewer proving the read-path data contract (dump script +
  `viewer.html` with an as-of time slider).
- `scripts/live-smoke.mjs` — live end-to-end smoke (requires API keys and a
  `npm run build:live` build; never runs in `npm test`).

## Ingestion idempotency

`Memory.addEpisode` accepts an optional `id` (a UUID string) on the episode
object — the idempotency key. Omit it and the pipeline mints one, as
before. A caller that retries failed ingests passes the **same** id on
every attempt for the same episode:

```js
const id = randomUUID();          // persisted by the caller, alongside the row it ingests
try {
  await memory.addEpisode({ id, content, sourceDescription, groupId }, llm, embedder);
} catch (e) {
  if (e.message.startsWith(`${EPISODE_EXISTS_CODE}: `)) {
    // The earlier attempt landed after all — this episode is already in
    // the graph, in full. Mark it done; do NOT ingest it again.
  } else {
    // Nothing was written (one episode, one transaction) — safe to retry
    // with the same id.
  }
}
```

The whole ingest — nodes, merges, edges, invalidations, summaries,
embeddings and the episode row — commits in one grit transaction, so a
stored episode id means that ingest completed. A second attempt with a
stored id is therefore rejected before any model call, rather than
silently doing nothing: only the caller knows whether that is a retry
landing late or a bug in its own bookkeeping.

`EPISODE_EXISTS_CODE` is exported by the addon (`"EPISODE_EXISTS"`) and
leads the rejection's message, followed by `": "`. It is not in
`error.code`: napi derives that from its own closed `Status` enum, so an
async binding cannot put a domain-specific code there. Match the prefix,
never the prose after it. A malformed id rejects with `code`
`"InvalidArg"` alongside the other argument errors.

## Providers

`Memory.addEpisode` / `Memory.searchEdges` take `LlmConfig` / `EmbedderConfig`
objects whose `provider` field selects the backend:

| provider | network | notes |
|---|---|---|
| `"host"` | **none in the addon** — the host app supplies callbacks | the shipped default; contract below |
| `"replay"` | none | serves committed recordings; the offline test workhorse |
| `"anthropic"` / `"deepseek"` / `"zhipu"` / `"openai-compatible"` | reqwest, in-process | **compiled out by default**; `npm run build:live` (`live-providers` feature) re-enables them for the smoke script |

### The `"host"` provider contract

The default build performs no network I/O. Hosts own the transport by passing
two callbacks (napi threadsafe functions):

```ts
const llm = {
  provider: 'host',
  // One plain chat round: run the completion, resolve the raw response text.
  chat: (request: HostChatRequest) => Promise<string>
}
const embedder = {
  provider: 'host',
  model: 'embedding-3',   // identity vectors persist under
  dim: 1024,              // vectors are MRL-sliced to this
  // One batch: resolve exactly one vector per input, in input order.
  embed: (inputs: string[]) => Promise<number[][]>
}
```

Division of labor:

- **Rust keeps the judgment logic.** Schema-in-prompt structured output —
  appending the schema block, stripping markdown fences, salvaging known bad
  shapes, validating, and feeding validation errors back on retry — lives in
  `nacre_core::model::prompted::PromptedModel`. The `chat` callback must NOT
  parse or validate; run the round, return the model's text.
- `HostChatRequest` carries `messages` (system/user/assistant), `maxTokens`
  (the pipeline's completion budget, default 16000 — clamp it if your relay
  enforces a lower ceiling), and `modelSize` (`"small" | "medium"`, a
  tier-routing hint hosts may ignore).
- `embed` must return one vector per input in input order; vectors may be
  longer than `dim` (truncated, never padded — shorter is an error).
- Callback rejections surface as provider errors; the pipeline retries
  (3 attempts, exponential backoff), so keep callbacks idempotent.
- Call `addEpisode` **sequentially per group** — the pipeline reads the
  previous-episode window and mutates the graph (upstream Graphiti's
  prescribed queueing model). Hosts should serialize ingestion.

Reference host integration: a native desktop host wires these callbacks to
its own model/embedding backend behind the `nacre-node` API and serializes
ingestion per group; the full behavior contract is the callback + threading
rules documented above.
