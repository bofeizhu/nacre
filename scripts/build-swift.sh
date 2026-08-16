#!/usr/bin/env bash
#
# Build the SwiftPM artifacts for `nacre-swift`: release static libraries for
# every Apple slice, the UniFFI-generated Swift/C sources, and the XCFramework
# the package's binary target points at.
#
#   scripts/build-swift.sh
#   swift build && swift test        # from the repo root
#
# Nothing in the cargo gate needs any of this — `cargo test --workspace` stays
# green on a machine with no Xcode. Run this only when working on the Swift
# package.
#
# Outputs (all gitignored, all regenerated from scratch on every run):
#   target/swift/staging/           headers + module.modulemap for the XCFramework
#   target/swift/nacre.xcframework  the binary target
#   swift/Sources/NacreFFI/         the generated Swift bindings
#
# arm64 only (aarch64-apple-ios, aarch64-apple-ios-sim, aarch64-apple-darwin):
# fresh 2026 project, Apple Silicon host, and the Layer 3 app ships nowhere an
# x86_64 slice would run. Add `x86_64-apple-ios` / `x86_64-apple-darwin` here
# and `lipo`-fuse them per platform (xcodebuild takes ONE -library per
# platform) if an Intel Mac or an Intel simulator ever has to be supported.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

crate="nacre-swift"
lib="libnacre_swift.a"
# The C module name the generated Swift imports: uniffi derives it from the
# crate namespace (`nacre_swift` + "FFI"), so the modulemap MUST declare that
# exact name or the `#if canImport(...)` in the generated sources silently
# skips the import. It is deliberately NOT "NacreFFI": that is the name of the
# *Swift* target holding the generated sources, and two modules cannot share a
# name in one build.
module="nacre_swiftFFI"
targets=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin)

staging="target/swift/staging"
xcframework="target/swift/nacre.xcframework"
generated="swift/Sources/NacreFFI"

# Keep these in step with Package.swift's `platforms:`. Without them rustc and
# the `cc` builds of SQLite/sqlite-vec inherit the host SDK's version, and the
# linker warns (correctly) that objects built for a newer OS are being linked
# into a package that claims to support an older one.
export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"

die() {
    echo "build-swift: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

# --- preflight ------------------------------------------------------------

command -v cargo >/dev/null || die "cargo not found — install Rust (https://rustup.rs)"
command -v rustup >/dev/null || die "rustup not found — the Apple targets are installed through it"
command -v xcodebuild >/dev/null ||
    die "xcodebuild not found — install Xcode, then: sudo xcode-select -s /Applications/Xcode.app"
xcodebuild -version >/dev/null 2>&1 ||
    die "xcodebuild is present but not usable (Command Line Tools only?) — sudo xcode-select -s /Applications/Xcode.app"

installed="$(rustup target list --installed)"
missing=()
for target in "${targets[@]}"; do
    grep -qx "$target" <<<"$installed" || missing+=("$target")
done
if [ ${#missing[@]} -ne 0 ]; then
    die "missing rustup targets: ${missing[*]} — rustup target add ${missing[*]}"
fi

# --- static libraries -----------------------------------------------------

for target in "${targets[@]}"; do
    step "cargo build --release --target $target"
    cargo build -p "$crate" --lib --release --target "$target"
    [ -f "target/$target/release/$lib" ] || die "expected target/$target/release/$lib"
done

# --- bindings -------------------------------------------------------------

# The generator reads UniFFI metadata out of the archive, so any slice will
# do; the host one keeps the invocation obvious. `--modulemap-filename
# module.modulemap` is mandatory for the `-library/-headers` XCFramework
# layout (Clang looks for that exact name); `--xcframework` is NOT — that flag
# is for the other layout, where headers live inside a real .framework.
step "uniffi-bindgen-swift → $staging"
rm -rf "$staging"
mkdir -p "$staging"
cargo run --quiet -p uniffi-bindgen -- \
    "target/aarch64-apple-darwin/release/$lib" "$staging" \
    --swift-sources --headers --modulemap \
    --module-name "$module" --modulemap-filename module.modulemap
[ -f "$staging/module.modulemap" ] || die "the generator produced no module.modulemap"

# The generated Swift belongs to the plain SwiftPM target; only headers and
# the modulemap may stay in the directory xcodebuild copies into every slice.
step "generated Swift sources → $generated"
mkdir -p "$generated"
rm -f "$generated"/*.swift
mv "$staging"/*.swift "$generated"/
ls "$generated"/*.swift >/dev/null 2>&1 || die "no Swift sources were generated"

# --- xcframework ----------------------------------------------------------

step "xcodebuild -create-xcframework → $xcframework"
rm -rf "$xcframework"
slices=()
for target in "${targets[@]}"; do
    slices+=(-library "target/$target/release/$lib" -headers "$staging")
done
xcodebuild -create-xcframework "${slices[@]}" -output "$xcframework"

echo
echo "Done. The SwiftPM package is ready:"
echo "    swift build"
echo "    swift test"
