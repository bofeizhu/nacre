# nacre-swift

Swift bindings for [nacre](../../README.md) — the Layer 3 gateway for a
native Apple host (iOS / iPadOS / macOS). Thin
[UniFFI](https://mozilla.github.io/uniffi-rs/) glue over `nacre-core` +
`grit-core`; all logic lives in those crates.

The shape is the same as [`nacre-node`](../nacre-node/README.md) minus the
sidecar: one `Memory` object over a grit file, the full ingest / read /
search surface, and **host-fulfilled model I/O** — the app implements two
Swift protocols and owns keys plus network, while every pipeline judgment
(extraction, dedup, temporal invalidation, summaries, retrieval fusion)
stays in Rust.

## Build

The two build stories are independent on purpose.

```sh
# Rust only — no Xcode, no Swift toolchain, nothing to install:
cargo build -p nacre-swift          # lib + staticlib + cdylib
cargo test  -p nacre-swift          # offline; stubs and recordings only

# The Apple artifacts, from the repo root:
bash scripts/build-swift.sh         # slices → bindgen → XCFramework → generated Swift
swift build && swift test
```

`scripts/build-swift.sh` builds release static libraries for
`aarch64-apple-ios`, `aarch64-apple-ios-sim` and `aarch64-apple-darwin`
(arm64 only — add Intel slices there if they are ever needed, `lipo`-fused
one per platform), runs the generator, and assembles
`target/swift/nacre.xcframework` plus `swift/Sources/NacreFFI/`. Both are
gitignored and both are prerequisites of `swift build`: the root
[`Package.swift`](../../Package.swift) points straight at them. The script
exports the deployment targets the package declares (iOS 17 / macOS 14) so
the Rust and C objects match its floor, and it is idempotent — rerun it
after every change to the Rust side.

The generator itself is `crates/uniffi-bindgen`, an in-workspace
`uniffi-bindgen-swift` binary — UniFFI's codegen has to run at the exact
version the bindings crate compiled against, so it is a workspace member
rather than an installed CLI:

```sh
cargo run -p uniffi-bindgen -- \
    target/aarch64-apple-darwin/release/libnacre_swift.a target/swift/staging \
    --swift-sources --headers --modulemap \
    --module-name nacre_swiftFFI --modulemap-filename module.modulemap
```

The module name is not cosmetic: `--module-name` names the **C** module in
the generated modulemap, and the generated Swift imports
`nacre_swiftFFI` (crate namespace + `FFI`) behind a `#if canImport`. Any
other name makes that import silently vanish and the FFI symbols
disappear with it. `NacreFFI` is the *Swift* target holding the generated
sources — a different module, in the SwiftPM sense.

## The Swift package

| SwiftPM target | What |
|---|---|
| `nacreFFI` (binary) | `target/swift/nacre.xcframework` — the Rust static library, one arm64 slice per platform |
| `NacreFFI` | the generated bindings, pinned to Swift language mode **5** — uniffi 0.32.0's async foreign-trait codegen does not compile under Swift 6 mode ([#2929](https://github.com/mozilla/uniffi-rs/issues/2929); the fix merged after the 0.32.0 tag and is unreleased). Hosts are unaffected — everything above this target builds in Swift 6 mode |
| `NacreKit` | the hand-written surface: `@_exported import NacreFFI` plus the host contract in doc form. Deliberately thin |
| `NacreKitTests` | the XCTest suite (below) |

An app consumes it path-locally while the two are developed together —
`.package(path: "../nacre")`, product `NacreKit` — and `import NacreKit`
brings the whole surface in. Publishing (a tagged release with a
checksummed XCFramework zip) is deliberately deferred until the app ships.

## The host protocols

Swift implements these; nacre calls them. An `actor` is a fine conformer —
both methods are `async` and throwing.

```swift
public protocol LlmProvider: AnyObject, Sendable {
    /// Run ONE chat completion over your own transport; return the raw text.
    func chat(request: ChatRequest) async throws -> String
}

public protocol EmbeddingProvider: AnyObject, Sendable {
    /// Embed a batch; return exactly one vector per input, in input order.
    func embed(inputs: [String]) async throws -> [[Double]]
}
```

Division of labor — the same contract the Node bindings document, restated
for Swift:

- **Rust keeps the judgment logic.** Schema-in-prompt structured output —
  appending the schema block, stripping markdown fences, salvaging known
  bad shapes, validating, and feeding validation errors back on retry —
  lives in `nacre_core::model::prompted::PromptedModel`. `chat` must NOT
  parse, validate, or retry; run the round, return the model's text.
- `ChatRequest` arrives **post-mutation**: `messages` already carries the
  appended schema block (and, on a retry, the validation-error message).
  It also carries `maxTokens` (the pipeline's completion budget, default
  16000 — clamp it if your relay enforces a lower ceiling) and `tier`
  (`.small` / `.medium`, a routing hint you may ignore).
- `embed` must return one vector per input in input order. Vectors longer
  than `EmbedderSpec.dim` are MRL-truncated, never padded; shorter is an
  error. It is never called with an empty batch.
- One ingestion issues its model calls **sequentially** (the port keeps
  upstream's recorded ordering), so a conformer never sees overlapping
  calls from a single `addEpisode` — but nothing stops the app from
  running two `Memory` calls at once, so keep it thread-safe.
- Throws surface as `NacreError.Provider`; the pipeline retries (3
  attempts, exponential backoff), so keep implementations idempotent. A
  Swift error that is not a `NacreError` is converted rather than
  trapping.
- `EmbedderSpec` (`modelId`, `dim`, `modelVersion`) is the identity grit
  tags stored vectors with. It MUST stay stable for a given memory file —
  changing `dim` between runs fails loudly rather than mixing vector
  spaces. A `modelId`/`modelVersion` swap at the *same* dim is accepted
  silently and mixes two vector spaces in one index; nothing catches that
  but the host, and it surfaces as bad recall rather than an error.
- Call `addEpisode` **sequentially per group**: the pipeline reads the
  previous-episode window and then mutates the graph (upstream Graphiti's
  prescribed queueing model). grit serializes storage writes, but ordering
  is the host's job.
- `addEpisode` is **not cancellable**. Swift `Task` cancellation does not
  propagate into Rust — uniffi 0.32 generates no cancellation path (the
  bindings never call `rust_future_cancel`), so cancelling the awaiting
  Task abandons the await only: the pipeline runs to completion, keeps
  invoking the host protocols, and commits its writes. Gate UI departure
  on the returned outcome, not on `Task.cancel()`.
- **Retry with the same `Episode.id`.** The optional id is the ingest
  idempotency key: pass one (and persist it next to whatever the host is
  ingesting) and every attempt at that episode carries it. The whole
  ingest commits in one transaction, so a failure leaves the graph exactly
  as it was — retry freely — and an id that is already stored comes back
  as `NacreError.EpisodeExists(id:)`, meaning the earlier attempt landed
  in full: mark it done rather than ingesting it again. Omit the id and
  the pipeline mints one, as before, and a retry duplicates the episode.
  An id that is not a UUID is `NacreError.InvalidInput`, raised with the
  other argument checks before any protocol call.

## Surface

| Swift | what |
|---|---|
| `Memory.open(path:deviceId:)` | open/create a memory file (a real path — in-memory databases are rejected) |
| `addEpisode(episode:llm:embedder:embedderSpec:)` | full ingest, one transaction; returns the change set (new / merged / invalidated ids). `Episode.id` is the optional idempotency key |
| `searchEdges(query:groupId:limit:embedder:embedderSpec:)` | hybrid recall with provenance |
| `addEpisodeReplay(…)` / `searchEdgesReplay(…)` | the same paths against recorded responses — offline tests, no protocols needed |
| `nodesInGroup` / `edgesInGroup` / `episodesInGroup` / `traverse` / `nodeHistory` / `mentionsOf` / `previousEpisodes` | the read path — the entire data contract a memory-graph view needs |
| `episode(id:)` | one episode by id, or `nil` — the point lookup behind a provenance id, and how a host reads its own `Episode.metadata` back |
| `deleteEpisode(id:)` | forget one episode and what it alone evidenced; returns both halves (`purged*` / `kept*`). `NacreError.EpisodeNotFound` when the id is not stored |
| `deleteGroup(groupId:)` | forget a whole namespace; an unknown group is a zero-count result, not an error |
| `version()` | crate version (load smoke-check) |

The read path is **synchronous** (grit is an embedded library and these
are millisecond SQLite reads) — call it off the main thread for large
groups. Node/edge `attrsJson` is a JSON object *string*: UniFFI has no
dynamic-value type, so hosts decode it themselves.

Deletion is synchronous for the same reason — no model call is involved —
but both calls are **writes**, each committing a single grit `Purge` in
one transaction, so a deletion happens whole or not at all. Two rules the
type system cannot state:

- **"Forgotten" means unreachable through every read API and every index**
  (rows, update and invalidation records, provenance both ways, FTS
  mirrors, stored vectors) — *not* gone from the file. grit's op-log keeps
  the original ops by design: it is the sync unit. An app that promises a
  person erasure must make that promise on its own terms.
- **A deleted id is spent.** grit tombstones it permanently, so a later
  ingest naming it succeeds while storing neither the episode row nor its
  provenance, leaving that ingest's nodes and edges evidenced by nothing.
  Re-ingesting a forgotten memory — the one way to change write-once
  `Episode.metadata` — needs a NEW id.

## Tests

Both suites are offline by construction — the stubs *are* the transport,
and replay refuses to guess.

`cargo test -p nacre-swift` (no Swift toolchain needed):

- `tests/host_transport.rs` — Rust stubs stand in for the Swift protocols
  and drive a real ingestion end-to-end against a temp grit file
  (including the MRL truncation, a throwing provider, and a too-short
  vector).
- `tests/replay.rs` — the committed golden trace (`oracle/fixtures/trace1`)
  replayed through the bindings: exact ingestion deltas, the whole read
  path, search with provenance, and the fail-loudly replay-miss surface.
  Plus the M10 surface: `metadata` on every episode read path (a leak into
  a prompt would change the replay key and fail the ingest as a miss, so
  the ingest is the assertion), both deletions with a second group of the
  same database as the control, the getter's `nil`, and
  `NacreError.EpisodeNotFound` on a second delete.

`swift test` (repo root, after `scripts/build-swift.sh`) runs the same two
shapes through the real bindings, which is what proves the *packaging*:

- `swift/Tests/NacreKitTests/HostProviderTests.swift` — a Swift `actor`
  conforming to both protocols drives addEpisode → reads → searchEdges;
  over-long vectors prove the MRL slice, a throwing conformer comes back
  as `NacreError.Provider`, and the whole test target builds in Swift 6
  mode.
- `swift/Tests/NacreKitTests/ReplayTests.swift` — trace 1 replayed from
  Swift, asserting the same ep-0 deltas the Rust and Node suites assert,
  plus the idempotency key (a second ingest under a stored id is refused
  as `NacreError.EpisodeExists` and writes nothing) and the forgetting
  surface (metadata round-trip, `deleteEpisode`'s two halves,
  `NacreError.EpisodeNotFound` on the second delete, `deleteGroup` leaving
  a sibling group untouched).

The Swift tests resolve the fixtures from `#filePath`, so they work from
any checkout, and they `XCTSkip` **loudly** when the build artifacts or
the oracle fixtures are absent rather than failing.

## Releasing (maintainers)

`main`'s `Package.swift` is the DEV manifest: a `path:` binary target and
gitignored generated bindings, rebuilt by `scripts/build-swift.sh`. That
manifest cannot resolve for remote SwiftPM consumers, so every release tag
must carry a RELEASE MANIFEST instead (first done for v0.1.2, commit
d411f34):

1. From the release commit on `main`: run `scripts/build-swift.sh`, then
   zip the artifact and compute its checksum —
   `ditto -c -k --sequesterRsrc --keepParent target/swift/nacre.xcframework nacre-xcframework-vX.Y.Z.zip`
   and `swift package compute-checksum <zip>`.
2. Create the GitHub release for the tag name and attach the zip. The zip
   is the canonical artifact: the xcframework's `Info.plist` is not
   byte-deterministic across rebuilds, so never recompute the checksum
   from a rebuild.
3. On a `release/vX.Y.Z` branch: swap the binary target to the asset's
   `url:` + `checksum:`, commit the generated
   `swift/Sources/NacreFFI/*.swift` (`git add -f`), commit.
4. Point the version tag at that commit and push it. Verify from a bare
   scratch package: `.package(url: "https://github.com/bofeizhu/nacre.git",
   from: "X.Y.Z")` must fetch, download, build, and run.
