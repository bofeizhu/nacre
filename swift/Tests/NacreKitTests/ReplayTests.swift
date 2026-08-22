// The replay path against the committed golden trace, driven from Swift.
//
// Same recordings the conformance suite, the Node bindings and the Rust-side
// binding tests replay (`oracle/fixtures/trace1`), so a divergence here is a
// packaging bug and nothing else. Offline by construction: a request the
// trace never recorded fails as `NacreError.ReplayMiss` instead of reaching
// for the network.

import Foundation
import NacreKit
import XCTest

/// Trace 1's embedder identity — the one vectors persist under.
private func traceSpec() -> EmbedderSpec {
    EmbedderSpec(modelId: "embedding-3", dim: 1024, modelVersion: "")
}

/// Build episode `index` of the capture spec, exactly as the Rust and Node
/// replay tests do: `reference_time` doubles as the pipeline's injected now.
private func episode(from trace: [String: Any], index: Int) throws -> Episode {
    let episodes = try XCTUnwrap(trace["episodes"] as? [[String: Any]])
    let episode = episodes[index]
    return Episode(
        name: try XCTUnwrap(episode["name"] as? String),
        content: try XCTUnwrap(episode["content"] as? String),
        source: episode["source"] as? String,
        sourceDescription: try XCTUnwrap(episode["source_description"] as? String),
        groupId: try XCTUnwrap(trace["group_id"] as? String),
        validAt: try XCTUnwrap(episode["reference_time"] as? String)
    )
}

final class ReplayTests: XCTestCase {
    func testTheGoldenTraceReplaysThroughThePackage() async throws {
        let trace = try Artifacts.requireTrace1()
        let group = try XCTUnwrap(trace["group_id"] as? String)
        let memory = try Memory.open(
            path: temporaryMemoryPath("replay", cleanUp: self), deviceId: "swift-replay-test")

        // Deltas are exact under replay — the same numbers the Rust and Node
        // bindings assert for ep-0.
        let first = try await memory.addEpisodeReplay(
            episode: try episode(from: trace, index: 0),
            llmRecordingsPath: Artifacts.recordings("llm_recordings.json"),
            embedderRecordingsPath: Artifacts.recordings("embedder_recordings.json"),
            embedderSpec: traceSpec()
        )
        XCTAssertEqual(first.nodeIds.count, 4, "ep-0 extracts 4 entities")
        XCTAssertEqual(first.newEdgeIds.count, 2, "ep-0 creates 2 facts")
        XCTAssertTrue(first.merges.isEmpty)
        XCTAssertTrue(first.invalidatedEdgeIds.isEmpty)
        XCTAssertFalse(first.episodeId.isEmpty)

        let second = try await memory.addEpisodeReplay(
            episode: try episode(from: trace, index: 1),
            llmRecordingsPath: Artifacts.recordings("llm_recordings.json"),
            embedderRecordingsPath: Artifacts.recordings("embedder_recordings.json"),
            embedderSpec: traceSpec()
        )
        XCTAssertFalse(
            second.merges.isEmpty, "ep-1 dedups returning entities into ep-0's nodes")
        XCTAssertTrue(
            second.merges.allSatisfy { !$0.from.isEmpty && !$0.into.isEmpty },
            "merge pairs carry both endpoints")

        // --- the read path: the data contract a graph view consumes ---
        let nodes = try memory.nodesInGroup(groupId: group)
        let live = nodes.filter { $0.expiredAt == nil }
        XCTAssertGreaterThan(nodes.count, live.count, "merged-away drafts are retained for audit")
        XCTAssertTrue(
            nodes.contains { $0.mergedInto != nil && $0.expiredAt != nil },
            "a merged-away row points at its canonical node")

        let priya = try XCTUnwrap(live.first { $0.name == "Priya" }, "Priya is live")
        XCTAssertFalse(priya.summary.isEmpty, "summaries persisted")
        XCTAssertTrue(priya.labels.contains("Entity"))
        XCTAssertEqual(
            priya.attrsJson, "{}",
            "attrs ride as a JSON object string, labels unpacked out of it")

        XCTAssertGreaterThanOrEqual(try memory.edgesInGroup(groupId: group).count, 4)
        let episodes = try memory.episodesInGroup(groupId: group)
        XCTAssertEqual(episodes.count, 2)
        XCTAssertEqual(episodes[0].kind, "message")
        XCTAssertTrue(episodes[0].occurredAt.hasSuffix("Z"))

        let subgraph = try memory.traverse(seeds: [priya.id], options: nil)
        XCTAssertTrue(subgraph.nodes.contains { $0.id == priya.id })
        XCTAssertFalse(subgraph.edges.isEmpty, "Priya has incident facts")

        let history = try memory.nodeHistory(id: priya.id)
        XCTAssertEqual(history.node.id, priya.id)
        let incident = try XCTUnwrap(history.edges.first)
        XCTAssertFalse(
            try memory.mentionsOf(id: incident.id).isEmpty,
            "every edge is attributed to the episode that produced it")

        // previousEpisodes is the DISPLAY window: not filtered by source
        // kind, bounded by the reference instant.
        XCTAssertEqual(
            try memory.previousEpisodes(groupId: group, reference: nil, lastN: 1).count, 1,
            "lastN bounds the window")
        XCTAssertEqual(
            try memory.previousEpisodes(
                groupId: group, reference: "2026-03-03T00:00:00+00:00", lastN: nil
            ).count,
            1, "episodes after the reference are excluded")

        // --- search: the recorded query embedding, fused with keyword legs ---
        let hits = try await memory.searchEdgesReplay(
            query: "Where does Priya work?",
            groupId: group,
            limit: 5,
            recordingsPath: Artifacts.recordings("embedder_recordings.json"),
            embedderSpec: traceSpec()
        )
        XCTAssertFalse(hits.isEmpty, "the trace query returns hits")
        XCTAssertTrue(
            hits.contains { $0.fact.contains("Northwind") },
            "the employment fact is recalled: \(hits.map(\.fact))")
        XCTAssertTrue(hits.allSatisfy { !$0.episodes.isEmpty }, "hits carry provenance")
    }

    func testAnUnrecordedRequestFailsLoudlyAsAReplayMiss() async throws {
        let trace = try Artifacts.requireTrace1()
        let memory = try Memory.open(
            path: temporaryMemoryPath("replay-miss", cleanUp: self), deviceId: "swift-replay-test")

        var unrecorded = try episode(from: trace, index: 0)
        unrecorded.content = "Nothing in this sentence was ever captured."
        do {
            _ = try await memory.addEpisodeReplay(
                episode: unrecorded,
                llmRecordingsPath: Artifacts.recordings("llm_recordings.json"),
                embedderRecordingsPath: Artifacts.recordings("embedder_recordings.json"),
                embedderSpec: traceSpec()
            )
            XCTFail("offline replay must refuse to guess")
        } catch let error as NacreError {
            guard case .ReplayMiss(let canonicalRequest) = error else {
                return XCTFail("expected .ReplayMiss, got \(error)")
            }
            XCTAssertTrue(
                canonicalRequest.contains("Nothing in this sentence"),
                "the unmatched request is included for a mechanical diff")
        }
    }

    func testARetryUnderAStoredEpisodeIdIsRefused() async throws {
        let trace = try Artifacts.requireTrace1()
        let group = try XCTUnwrap(trace["group_id"] as? String)
        let memory = try Memory.open(
            path: temporaryMemoryPath("idempotent", cleanUp: self), deviceId: "swift-replay-test")
        func replay(_ episode: Episode) async throws -> AddEpisodeResult {
            try await memory.addEpisodeReplay(
                episode: episode,
                llmRecordingsPath: Artifacts.recordings("llm_recordings.json"),
                embedderRecordingsPath: Artifacts.recordings("embedder_recordings.json"),
                embedderSpec: traceSpec()
            )
        }

        // A host that retries failed ingests owns the id: the same one on
        // every attempt for the same episode.
        let id = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        var keyed = try episode(from: trace, index: 0)
        keyed.id = id
        let landed = try await replay(keyed)
        XCTAssertEqual(landed.episodeId, id, "the supplied id IS the stored episode id")
        let nodes = try memory.nodesInGroup(groupId: group).count

        // The retry that arrives after its first attempt actually landed is
        // refused, and refused before any model call: the episode row
        // commits with the graph mutations it evidences, so a stored id
        // means the whole ingest is already there.
        do {
            _ = try await replay(keyed)
            XCTFail("a second ingest under a stored id must not write again")
        } catch let error as NacreError {
            guard case .EpisodeExists(let existing) = error else {
                return XCTFail("expected .EpisodeExists, got \(error)")
            }
            XCTAssertEqual(existing, id, "the rejection names the id that was passed in")
        }
        XCTAssertEqual(try memory.episodesInGroup(groupId: group).count, 1, "no second episode row")
        XCTAssertEqual(
            try memory.nodesInGroup(groupId: group).count, nodes, "no second set of entities")

        // A malformed key is a host bug, caught with the other arguments.
        var malformed = try episode(from: trace, index: 1)
        malformed.id = "not-a-uuid"
        do {
            _ = try await replay(malformed)
            XCTFail("an id that is not a UUID is not an id")
        } catch let error as NacreError {
            guard case .InvalidInput(let message) = error else {
                return XCTFail("expected .InvalidInput, got \(error)")
            }
            XCTAssertTrue(message.hasPrefix("id \"not-a-uuid\""), message)
        }
    }

    func testTheBindingsLoadAndRejectInMemoryDatabases() throws {
        try Artifacts.requireBuildArtifacts()
        // Module-qualified: XCTestCase has a `version` of its own.
        XCTAssertFalse(NacreFFI.version().isEmpty, "the FFI library loaded")
        do {
            _ = try Memory.open(path: ":memory:", deviceId: "swift-replay-test")
            XCTFail("grit needs a real file")
        } catch let error as NacreError {
            guard case .Storage(let message) = error else {
                return XCTFail("expected .Storage, got \(error)")
            }
            XCTAssertFalse(message.isEmpty)
        }
    }
}
