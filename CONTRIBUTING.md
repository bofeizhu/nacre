# Contributing to nacre

nacre is a port with a falsifiable correctness claim: pinned Python
Graphiti's behavior, reproduced byte-for-byte. Contributions live or die
by that oracle, which makes the definition of done unusually crisp.

## The one rule

[AGENTS.md](AGENTS.md) is the design contract, and it is binding. The
parts you'll hit first:

- **The oracle decides.** Golden-trace conformance (recorded LLM/embedder
  responses replayed through the full Rust stack, diffed against frozen
  Python Graphiti output) must stay GREEN. A change that shifts a prompt
  byte or a graph row either fixes a real divergence or documents an
  accepted one in [DEVIATIONS.md](DEVIATIONS.md) — with a test asserting it.
- **Prompt bytes are load-bearing.** Rust prompt rendering is asserted
  byte-identical to fixtures from the pinned upstream
  ([PROMPTS.md](PROMPTS.md) is the ledger). Cosmetic rewording is a
  conformance break, not a cleanup.
- **`cargo test` never touches the network.** The default build has no
  HTTP client compiled in; live clients live behind feature flags
  (`claude`, `openai-embed`). Replay recordings drive every judgment path.
- **Scope is a decision.** The README's coverage tiers are deliberate;
  out-of-scope features (bulk ingestion, communities, sagas, advanced
  search recipes) need a proven need and an issue, not a PR.

When in doubt, open an issue before writing code — especially for
anything that would touch recorded requests or the conformance surface.

## Dev loop

```bash
cargo test --workspace     # offline: replay + property + prompt-fidelity + conformance
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

Live smoke (optional, needs API keys):

```bash
cargo run -p nacre-core --example live_smoke --features claude,openai-embed
```

The oracle capture harness (Docker FalkorDB, digest-pinned) lives in
[`oracle/`](oracle/) — you only need it when *capturing new traces*, never
for running the test suite.

## Where help is wanted

The README's **"Ported, awaiting trace coverage"** tier is the standing
contribution menu: each item is a well-scoped piece of work whose
acceptance test is mechanical — extend or add a golden trace that
exercises it, and conformance must pass. Issues labeled
[`good first issue`](https://github.com/bofeizhu/nacre/labels/good%20first%20issue)
and [`help wanted`](https://github.com/bofeizhu/nacre/labels/help%20wanted)
point at specific ones.

Also welcome: replay-test coverage for edge cases in dedup/invalidation
judgment paths, docs fixes, and provider presets for
Anthropic-compatible / OpenAI-compatible endpoints (behind the existing
feature flags, with recorded fixtures).

Commit messages follow the existing log style: lowercase, imperative,
scope-prefixed where it helps (`oracle: …`, `nacre-node: …`).

## License

Apache-2.0. Contains material ported from
[Graphiti](https://github.com/getzep/graphiti) (Apache-2.0, Zep Software,
Inc.) — attribution lives in [NOTICE](NOTICE); keep it accurate when
porting more. Unless you state otherwise, contributions you intentionally
submit are Apache-2.0, without additional terms.
