/// # NacreKit
///
/// Agent memory for a native Apple host: episodes in, a bi-temporal knowledge
/// graph out. The pipeline (entity extraction, dedup judgment, temporal
/// invalidation, summaries, hybrid retrieval) is nacre's Rust code; the graph
/// lives in a single grit/SQLite file the app owns.
///
/// Everything public comes from the UniFFI-generated bindings, re-exported
/// below: ``Memory`` and its records, the ``LlmProvider`` / ``EmbeddingProvider``
/// protocols, and ``NacreError``. This target stays deliberately thin — it is
/// the seam where Swift-only sugar (an async read path, ingestion
/// serialization helpers) would land if the app asks for it, and every line
/// added here is a line the Rust tests do not cover.
///
/// ## The host contract
///
/// The app owns the transport. nacre performs no network I/O of its own: it
/// hands you a prompt and expects the model's raw text back.
///
/// ```swift
/// actor Relay: LlmProvider, EmbeddingProvider {
///     func chat(request: ChatRequest) async throws -> String {
///         // Relay `request.messages` verbatim over your own client and
///         // return the response text. Do NOT parse, validate, or retry:
///         // schema handling and the correction-retry loop are nacre's.
///     }
///     func embed(inputs: [String]) async throws -> [[Double]] {
///         // Exactly one vector per input, in input order. Vectors longer
///         // than `EmbedderSpec.dim` are MRL-truncated by nacre; shorter is
///         // an error. Never called with an empty batch.
///     }
/// }
///
/// let memory = try Memory.open(path: url.path, deviceId: deviceId)
/// let change = try await memory.addEpisode(
///     episode: Episode(
///         name: "ep-0",
///         content: "Priya joined Northwind last March.",
///         source: nil,
///         sourceDescription: "chat",
///         groupId: "chat",
///         validAt: "2026-03-04T09:00:00+00:00"
///     ),
///     llm: relay,
///     embedder: relay,
///     embedderSpec: EmbedderSpec()
/// )
/// ```
///
/// Four rules the compiler cannot enforce:
///
/// - **Ingest sequentially per group.** ``Memory`` is safe to share across
///   threads and grit serializes storage writes, but `addEpisode` reads the
///   previous-episode window and *then* mutates the graph (upstream
///   Graphiti's prescribed queueing model). Two concurrent ingestions into
///   one group race on context, not on storage. Serialize them — an actor
///   per group, or one ingestion queue.
/// - **Treat `addEpisode` as non-cancellable.** Swift `Task` cancellation
///   does not reach Rust — uniffi 0.32 generates no cancellation path, so
///   cancelling the awaiting `Task` abandons only the *await*: the pipeline
///   runs to completion, still calling your providers and still committing
///   its writes. Gate UI departure on the returned outcome, never on
///   `Task.cancel()`, and expect provider calls to keep arriving after the
///   screen that started them is gone.
/// - **Keep `EmbedderSpec` stable per memory file.** It is the identity grit
///   tags stored vectors with. Only a `dim` change fails loudly (grit
///   rejects a vector whose length disagrees with the registered dim); a
///   `modelId` or `modelVersion` swap at the same dim is accepted silently
///   and quietly mixes two vector spaces in one index, which shows up as
///   bad recall, not as an error. Pin the spec per memory file.
/// - **Call the read path off the main thread.** `nodesInGroup`,
///   `edgesInGroup`, `episodesInGroup`, `traverse`, `nodeHistory`,
///   `mentionsOf` and `previousEpisodes` are synchronous — they are
///   millisecond SQLite reads, but they are disk reads.
///
/// Provider failures come back as ``NacreError/Provider(message:)``, after
/// nacre's own retries; a thrown Swift error that is not a `NacreError` is
/// converted rather than trapping. Node and edge attributes ride as an
/// `attrsJson` object *string* — UniFFI has no dynamic-value type, so decode
/// it yourself.
@_exported import NacreFFI
