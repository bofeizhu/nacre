// swift-tools-version: 6.0
//
// The repo IS the Swift package: NacreKit is nacre's Layer 3 gateway for a
// native Apple host (iOS / iPadOS / macOS).
//
// PREREQUISITE: `scripts/build-swift.sh` builds the two artifacts this
// manifest points at — `target/swift/nacre.xcframework` (the Rust static
// library, one arm64 slice per platform) and `swift/Sources/NacreFFI/` (the
// UniFFI-generated Swift). Both are gitignored, so a bare checkout has to run
// that script once before `swift build` resolves. Nothing in the cargo gate
// needs either: `cargo test --workspace` is green with no Xcode installed.

import PackageDescription

let package = Package(
    name: "NacreKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "NacreKit", targets: ["NacreKit"])
    ],
    targets: [
        // The Rust side: nacre-core's pipeline + grit's storage, compiled to
        // a static library per platform. THIS IS THE RELEASE MANIFEST: the
        // binary target resolves to the versioned artifact attached to the
        // GitHub release this tag carries, and the generated bindings are
        // committed alongside it, so `.package(url:, from:)` consumption
        // works from a bare clone. Day-to-day development happens on `main`,
        // whose manifest uses a path binary target + scripts/build-swift.sh.
        .binaryTarget(
            name: "nacreFFI",
            url: "https://github.com/bofeizhu/nacre/releases/download/v0.1.2/nacre-xcframework-v0.1.2.zip",
            checksum: "4315a0dc1903913370e1e06e7892b7c7fbe00e3862528390a00c3a3e987e6197"
        ),

        // The generated bindings, in their own target for two reasons: it
        // keeps the FFI plumbing out of any host build settings (Xcode 26
        // defaults to MainActor isolation, which the generated code cannot
        // survive — uniffi#2818), and it is the only place that needs the
        // Swift 5 language mode.
        //
        // .v5 is REQUIRED, not a preference: uniffi 0.32.0's async
        // foreign-trait codegen does not compile under Swift 6 mode
        // (`#SendingClosureRisksDataRace` — uniffi#2929; the @Sendable fix
        // merged two weeks after the 0.32.0 tag and is still unreleased).
        // Drop this setting when a uniffi release carries the fix. Hosts are
        // unaffected: NacreKit and its clients build in Swift 6 mode.
        .target(
            name: "NacreFFI",
            dependencies: [.target(name: "nacreFFI")],
            path: "swift/Sources/NacreFFI",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),

        // The hand-written surface. Deliberately thin — see the doc header.
        .target(
            name: "NacreKit",
            dependencies: ["NacreFFI"],
            path: "swift/Sources/NacreKit"
        ),

        .testTarget(
            name: "NacreKitTests",
            dependencies: ["NacreKit"],
            path: "swift/Tests/NacreKitTests"
        ),
    ]
)
