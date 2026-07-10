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
- `scripts/live-smoke.mjs` — live end-to-end smoke (requires API keys;
  never runs in `npm test`).
