//! # nacre-core
//!
//! The LLM extraction pipeline for agent memory: Graphiti's pipeline
//! (entity extraction, dedup judgment, temporal invalidation, summarization)
//! ported to Rust, speaking [`grit_core`]'s typed API.
//!
//! Nacre decides; grit executes. Every graph mutation leaves this crate as a
//! `grit_core` op — nacre owns the LLM judgment calls and embedding
//! computation, never the storage.
//!
//! This crate is under construction; see `AGENTS.md` for the binding design
//! contract and port order. The first real modules land behind the
//! golden-trace capture harness (`oracle/`).

#![deny(unsafe_code)]

pub mod model;

pub use grit_core;
