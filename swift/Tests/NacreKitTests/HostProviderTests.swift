// The host-fulfilled path end-to-end, driven by a Swift stub that stands in
// for an app's transport — the proof that the protocols this milestone ships
// are implementable from Swift 6 code and that nothing in the pipeline needs
// the network.
//
// Ports `crates/nacre-node/test/host.test.mjs` (and its Rust twin,
// `crates/nacre-swift/tests/host_transport.rs`): the chat stub recognizes
// which schema a round asked for by the schema block nacre appends to the
// prompt, and the embedding stub deliberately returns over-long vectors so
// the MRL truncation is exercised.

import Foundation
import NacreKit
import XCTest

/// Deliberately tiny: grit tags vectors with (modelId, dim, version), and a
/// stub embedder has no reason to pretend to be 1024-dimensional.
private let stubDimensions: UInt32 = 8

private func stubSpec() -> EmbedderSpec {
    EmbedderSpec(modelId: "stub-embedding", dim: stubDimensions, modelVersion: "")
}

private func episode(_ name: String, _ content: String, at validAt: String) -> Episode {
    Episode(
        name: name,
        content: content,
        source: nil,
        sourceDescription: "chat",
        groupId: "host-group",
        validAt: validAt
    )
}

/// An `actor` conformer, which is the shape an app will reach for: both
/// protocol requirements are `async throws`, so actor isolation satisfies
/// them without a single `nonisolated`.
private actor StubHost: LlmProvider, EmbeddingProvider {
    private(set) var chatRequests: [ChatRequest] = []
    private(set) var embedBatches: [[String]] = []

    /// Components ADDED to the configured dim. Positive proves the MRL
    /// slice; negative produces vectors that are too short (an error).
    private let extraComponents: Int

    init(extraComponents: Int = 4) {
        self.extraComponents = extraComponents
    }

    func chat(request: ChatRequest) async throws -> String {
        chatRequests.append(request)
        let prompt = request.messages.map(\.content).joined(separator: "\n")
        // The appended schema block names the required top-level keys; answer
        // whichever schema this round asked for. One entity is extracted, so
        // the run also exercises summaries and embeddings.
        if prompt.contains("\"extracted_entities\"") {
            return """
                {"extracted_entities": [{"name": "Ada", "entity_type_id": 0, "episode_indices": [0]}]}
                """
        }
        if prompt.contains("\"edges\"") { return #"{"edges": []}"# }
        if prompt.contains("\"entity_resolutions\"") { return #"{"entity_resolutions": []}"# }
        if prompt.contains("\"duplicate_facts\"") {
            return #"{"duplicate_facts": [], "contradicted_facts": []}"#
        }
        if prompt.contains("\"summaries\"") {
            return #"{"summaries": [{"name": "Ada", "summary": "Ada is a person."}]}"#
        }
        if prompt.contains("\"timestamps\"") { return #"{"timestamps": []}"# }
        if prompt.contains("\"description\"") { return #"{"description": "d"}"# }
        if prompt.contains("\"summary\"") { return #"{"summary": "s"}"# }
        throw NacreError.Provider(
            message: "stub got an unrecognized prompt: \(prompt.prefix(200))")
    }

    func embed(inputs: [String]) async throws -> [[Double]] {
        XCTAssertFalse(inputs.isEmpty, "nacre never asks for an empty batch")
        embedBatches.append(inputs)
        let length = max(0, Int(stubDimensions) + extraComponents)
        return inputs.indices.map { index in
            (0..<length).map { component in Double(index + 1) / Double(component + 1) }
        }
    }

    func sawEmbedding(of input: String) -> Bool {
        embedBatches.contains { $0.contains(input) }
    }
}

/// A host whose relay is down.
private final class BrokenLlm: LlmProvider {
    func chat(request: ChatRequest) async throws -> String {
        throw NacreError.Provider(message: "host transport down")
    }
}

/// A host that throws something that is NOT a `NacreError` — a URLSession
/// error, a `CancellationError`, anything. Swift protocols cannot constrain
/// what a conformer throws, so this path has to convert rather than trap.
private final class OddThrowingLlm: LlmProvider {
    struct Oops: Error {}

    func chat(request: ChatRequest) async throws -> String {
        throw Oops()
    }
}

final class HostProviderTests: XCTestCase {
    func testHostProtocolsDriveTheWholePipeline() async throws {
        try Artifacts.requireBuildArtifacts()
        let memory = try Memory.open(
            path: temporaryMemoryPath("host", cleanUp: self), deviceId: "swift-host-test")
        let host = StubHost()

        let change = try await memory.addEpisode(
            episode: episode("ep-0", "Ada wrote the first program.", at: "2026-07-14T00:00:00Z"),
            llm: host,
            embedder: host,
            embedderSpec: stubSpec()
        )

        XCTAssertEqual(change.nodeIds.count, 1, "one entity extracted through the chat protocol")
        XCTAssertTrue(change.merges.isEmpty)
        XCTAssertTrue(change.invalidatedEdgeIds.isEmpty)
        XCTAssertFalse(change.episodeId.isEmpty)

        let requests = await host.chatRequests
        XCTAssertGreaterThanOrEqual(
            requests.count, 2, "the chat protocol drove the pipeline (\(requests.count) rounds)")
        XCTAssertTrue(
            requests.allSatisfy { $0.maxTokens > 0 }, "every round carries a completion budget")
        XCTAssertTrue(
            requests.contains { $0.messages.contains { $0.content.contains("Ada wrote") } },
            "the episode text reached the host verbatim")

        // The read path sees what the pipeline wrote.
        let nodes = try memory.nodesInGroup(groupId: "host-group")
        let ada = try XCTUnwrap(nodes.first { $0.name == "Ada" }, "Ada is a live node")
        XCTAssertTrue(ada.labels.contains("Entity"))
        XCTAssertTrue(ada.createdAt.hasSuffix("Z"), "timestamps are RFC-3339 millis, Z form")
        XCTAssertEqual(try memory.episodesInGroup(groupId: "host-group").count, 1)
        XCTAssertFalse(
            try memory.mentionsOf(id: ada.id).isEmpty,
            "the node is attributed to the episode that produced it")

        // Search embeds the query through the same protocol.
        let hits = try await memory.searchEdges(
            query: "who wrote programs?",
            groupId: "host-group",
            limit: 3,
            embedder: host,
            embedderSpec: stubSpec()
        )
        XCTAssertTrue(hits.isEmpty, "the stub extracted no facts, so there is nothing to recall")
        let embedded = await host.sawEmbedding(of: "who wrote programs?")
        XCTAssertTrue(embedded, "the query embedding went through the host protocol")

        // The stub returned dim+4 components per vector and grit rejects a
        // vector whose length disagrees with the registered dim — a green
        // ingestion IS the proof that nacre MRL-truncated them.
    }

    func testAThrowingConformerSurfacesAsAProviderError() async throws {
        try Artifacts.requireBuildArtifacts()
        let memory = try Memory.open(
            path: temporaryMemoryPath("host-throws", cleanUp: self), deviceId: "swift-host-test")
        do {
            _ = try await memory.addEpisode(
                episode: episode("ep-err", "x", at: "2026-07-14T00:01:00Z"),
                llm: BrokenLlm(),
                embedder: StubHost(),
                embedderSpec: stubSpec()
            )
            XCTFail("a dead relay must not be swallowed")
        } catch let error as NacreError {
            guard case .Provider(let message) = error else {
                return XCTFail("expected .Provider, got \(error)")
            }
            XCTAssertTrue(message.contains("host transport down"), message)
            XCTAssertTrue(
                message.contains("giving up after 3 attempts"),
                "nacre owns the retry loop: \(message)")
        }
    }

    func testAShortVectorIsAnErrorNotAPad() async throws {
        try Artifacts.requireBuildArtifacts()
        let memory = try Memory.open(
            path: temporaryMemoryPath("host-short-vector", cleanUp: self),
            deviceId: "swift-host-test")
        do {
            _ = try await memory.addEpisode(
                episode: episode("ep-short", "Ada wrote it.", at: "2026-07-14T00:02:00Z"),
                llm: StubHost(),
                embedder: StubHost(extraComponents: -1),
                embedderSpec: stubSpec()
            )
            XCTFail("vectors shorter than the configured dim are a contract breach")
        } catch let error as NacreError {
            guard case .Provider(let message) = error else {
                return XCTFail("expected .Provider, got \(error)")
            }
            XCTAssertTrue(message.contains("expected at least 8"), message)
        }
    }

    func testAConformerThrowingANonNacreErrorStillSurfacesAsProvider() async throws {
        try Artifacts.requireBuildArtifacts()
        let memory = try Memory.open(
            path: temporaryMemoryPath("host-odd-throw", cleanUp: self),
            deviceId: "swift-host-test")
        do {
            _ = try await memory.addEpisode(
                episode: episode("ep-odd", "x", at: "2026-07-14T00:03:00Z"),
                llm: OddThrowingLlm(),
                embedder: StubHost(),
                embedderSpec: stubSpec()
            )
            XCTFail("a conformer that throws anything at all must not be swallowed")
        } catch let error as NacreError {
            // Swift can throw ANY Error, and the FFI hands those to Rust as
            // an unexpected-callback error; without nacre's
            // `From<UnexpectedUniFFICallbackError>` conversion the generated
            // scaffolding panics instead of returning. Only a real round
            // trip proves the conversion is wired up.
            guard case .Provider(let message) = error else {
                return XCTFail("expected .Provider, got \(error)")
            }
            XCTAssertTrue(message.contains("Oops"), "the thrown error is described: \(message)")
        }
    }

    func testArgumentsAreValidatedBeforeAnyModelCall() async throws {
        try Artifacts.requireBuildArtifacts()
        let memory = try Memory.open(
            path: temporaryMemoryPath("host-args", cleanUp: self), deviceId: "swift-host-test")
        // The stub is kept, not discarded: "before any model call" is the
        // claim, and the only witness to it is that the host recorded
        // nothing. Validation that ran after the first round would still
        // throw .InvalidInput, having spent a request to do it.
        let host = StubHost()
        func assertHostUntouched(_ what: String) async {
            let requests = await host.chatRequests
            let batches = await host.embedBatches
            XCTAssertTrue(requests.isEmpty, "\(what) reached the chat protocol")
            XCTAssertTrue(batches.isEmpty, "\(what) reached the embedding protocol")
        }

        var offsetless = episode("ep-0", "x", at: "2026-07-14T00:00:00")  // no UTC offset
        offsetless.source = nil
        do {
            _ = try await memory.addEpisode(
                episode: offsetless, llm: host, embedder: host,
                embedderSpec: stubSpec())
            XCTFail("an offset-less timestamp is ambiguous")
        } catch let error as NacreError {
            guard case .InvalidInput(let message) = error else {
                return XCTFail("expected .InvalidInput, got \(error)")
            }
            XCTAssertTrue(message.hasPrefix("validAt: "), message)
        }
        await assertHostUntouched("an offset-less timestamp")

        var unknownSource = episode("ep-0", "x", at: "2026-07-14T00:00:00Z")
        unknownSource.source = "audio"
        do {
            _ = try await memory.addEpisode(
                episode: unknownSource, llm: host, embedder: host,
                embedderSpec: stubSpec())
            XCTFail("unknown episode sources are rejected")
        } catch let error as NacreError {
            guard case .InvalidInput(let message) = error else {
                return XCTFail("expected .InvalidInput, got \(error)")
            }
            XCTAssertTrue(message.contains("audio"), message)
        }
        await assertHostUntouched("an unknown episode source")
    }
}
