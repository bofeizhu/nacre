//! The Swift binding generator for [`nacre-swift`](../nacre-swift).
//!
//! UniFFI's codegen has to run against the *same* uniffi version the
//! bindings crate was compiled with, so upstream's prescribed shape is a
//! tiny in-workspace binary that re-exports the bindgen entry point rather
//! than a separately installed CLI. This is that binary — it is
//! `uniffi-bindgen-swift`, which takes a built static library plus an
//! output directory and emits Swift sources, C headers, and modulemaps:
//!
//! ```sh
//! cargo run -p uniffi-bindgen -- \
//!     target/aarch64-apple-darwin/release/libnacre_swift.a \
//!     target/swift/generated --swift-sources
//! ```
//!
//! Packaging — staticlib slices → this generator → XCFramework → SwiftPM —
//! is driven by `scripts/build-swift.sh`, which calls the invocation above
//! with the headers and modulemap flags the package needs. Nothing here is
//! on the cargo gate.

fn main() {
    uniffi::uniffi_bindgen_swift()
}
