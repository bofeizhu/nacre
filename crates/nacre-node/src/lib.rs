//! Node.js bindings for nacre — the Layer 3 gateway.
//!
//! Thin `#[napi]` glue over `nacre-core` + `grit-core`; all logic lives in
//! those crates. Built into a loadable `.node` addon by `@napi-rs/cli`
//! (`npm run build` in this directory); plain `cargo build -p nacre-node`
//! type-checks and compiles the cdylib without any Node toolchain.

use napi_derive::napi;

/// The nacre-node crate version (addon load smoke-check).
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}
