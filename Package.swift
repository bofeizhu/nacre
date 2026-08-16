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
        // a static library per platform. Path-local while the Layer 3 app is
        // developed alongside nacre; a released package would point this at a
        // versioned `url:`/`checksum:` zip instead.
        .binaryTarget(name: "nacreFFI", path: "target/swift/nacre.xcframework"),

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
