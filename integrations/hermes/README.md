# nacre × Hermes — capture-only memory provider (Stage 1)

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory
provider plugin that ingests every conversation turn into a **local
bi-temporal memory graph** — nacre's LLM extraction/dedup/invalidation
pipeline over a grit (SQLite) file. Built to dogfood nacre on real
conversational workload with **zero influence on your sessions**.

Coded against Hermes Agent **v0.18.2, upstream commit `8e734810`**
(`agent/memory_provider.py` ABC). The ABC is a moving target — if a newer
Hermes breaks the shim, that pin is what we diffed against.

## What Stage 1 does — and deliberately does not

| Does | Does NOT |
|---|---|
| `sync_turn` → nacre `addEpisode` on a background thread (never blocks a turn) | no `prefetch` — nothing is ever injected into context |
| builds entities, facts, validity intervals, provenance in `$HERMES_HOME/nacre/memory.db` | no tools exposed to the model |
| fail-open circuit breaker: 3 strikes → capture stops, chat unaffected | no `system_prompt_block` — the model can't tell it's there |
| primary context only (subagents/cron never write) | never touches Hermes's built-in MEMORY.md/USER.md |

Recall (Stage 2) is gated on an offline review of a few weeks of real
captured graph: inspect it with the [viz
viewer](../../crates/nacre-node/examples/viz/README.md), probe it with
`searchEdges`, and only then decide to let anything flow back.

## Architecture

```
Hermes (Python) ── MemoryProvider ABC
   └─ this shim (stdlib-only, no pip deps)
        └─ node sidecar/sidecar.mjs        (ndjson over stdio, dies with parent)
             └─ nacre-node addon → nacre pipeline → grit/SQLite
                  └─ its OWN LLM/embedder keys (DeepSeek + Zhipu default)
```

The pipeline makes ~5–10 small-model calls per captured turn on **your
nacre keys** — a budget fully separate from (and much smaller than)
Hermes's chat model. See [the sidecar
README](../../crates/nacre-node/sidecar/README.md) for the protocol.

## Install (does not activate)

```sh
cd crates/nacre-node && npm install && npm run build   # once
./integrations/hermes/install.sh                       # symlink + discovery check
```

Nothing changes in Hermes until **you** run `hermes memory setup` and
select `nacre`. Setup collects (secrets go to Hermes's `.env`):

| Field | Env var | Where to get it |
|---|---|---|
| DeepSeek API key (extraction pipeline) | `NACRE_LLM_API_KEY` | platform.deepseek.com |
| Zhipu API key (embedding-3) | `NACRE_EMBEDDER_API_KEY` | open.bigmodel.cn |
| group id / node binary / crate dir | — | defaults are fine |

The sidecar needs Node ≥ 20 on Hermes's PATH (set `node_bin` in setup if
your default `node` is older).

## Kill switches

- `hermes memory off` — detach the provider (built-in memory unaffected)
- `rm $HERMES_HOME/nacre/memory.db` — erase the captured graph entirely
- the breaker does this automatically per-session on repeated failures

## Reviewing what it captured

```sh
node crates/nacre-node/examples/viz/dump-graph.mjs   # adapt dbPath/group, or:
cd integrations/hermes && uv run pytest -q           # offline tests (replay mode)
```

Tests run entirely offline: the real sidecar in replay mode against the
committed golden-trace recordings, with a structural stub of the Hermes
ABC — no Hermes install, no network, no keys.
