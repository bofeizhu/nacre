// Shared scaffolding for the Swift-side suite: where the build artifacts and
// the golden-trace fixtures live, and a temp memory file per test.
//
// Both are checked at RUN time and skipped LOUDLY when absent, so a checkout
// that only does Rust work never sees a red Swift test — it sees a skip that
// says which command to run. (`swift build` itself still needs the artifacts:
// the package's binary target points straight at the XCFramework.)

import Foundation
import XCTest

enum Artifacts {
    /// The repo root, resolved from this file rather than the working
    /// directory, so the suite works from any checkout and any launcher.
    static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // NacreKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // swift
        .deletingLastPathComponent()  // <repo>

    static let xcframework = repoRoot.appendingPathComponent("target/swift/nacre.xcframework")
    static let generatedSources = repoRoot.appendingPathComponent("swift/Sources/NacreFFI")
    static let trace1 = repoRoot.appendingPathComponent("oracle/fixtures/trace1")

    private static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    /// Skip unless `scripts/build-swift.sh` has produced the XCFramework and
    /// the generated sources.
    static func requireBuildArtifacts() throws {
        guard exists(xcframework) else {
            throw XCTSkip(
                "SKIP: \(xcframework.path) is missing — run `bash scripts/build-swift.sh` first")
        }
        let swiftSources = (try? FileManager.default.contentsOfDirectory(
            atPath: generatedSources.path)) ?? []
        guard swiftSources.contains(where: { $0.hasSuffix(".swift") }) else {
            throw XCTSkip(
                "SKIP: no generated Swift in \(generatedSources.path) — run `bash scripts/build-swift.sh` first"
            )
        }
    }

    /// The trace-1 capture spec, or a loud skip when the oracle fixtures are
    /// not in this checkout (same posture as the Rust replay tests).
    static func requireTrace1() throws -> [String: Any] {
        try requireBuildArtifacts()
        let spec = trace1.appendingPathComponent("episodes.json")
        guard let data = FileManager.default.contents(atPath: spec.path) else {
            throw XCTSkip(
                "SKIP: golden trace 1 not captured (\(spec.path)) — see oracle/README.md")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("SKIP: \(spec.path) is not a JSON object")
        }
        return json
    }

    /// Absolute path of one of trace 1's recordings files.
    static func recordings(_ name: String) -> String {
        trace1.appendingPathComponent(name).path
    }
}

/// A fresh memory file under the system temp dir, removed when the test ends.
/// grit rejects in-memory databases, so every test needs a real path.
func temporaryMemoryPath(_ name: String, cleanUp test: XCTestCase) -> String {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("nacre-swift-tests-\(ProcessInfo.processInfo.processIdentifier)")
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let path = directory.appendingPathComponent("\(name).db")
    try? FileManager.default.removeItem(at: path)
    test.addTeardownBlock {
        try? FileManager.default.removeItem(at: path)
    }
    return path.path
}
