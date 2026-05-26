pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod metal;
pub mod model;
pub mod pack;
pub mod parsers;
pub mod runtime;
pub mod serve;
pub mod store;

/// Semver plus short git SHA from `build.rs`, shown by `modelc --version`.
pub const CLI_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (git ",
    env!("MODELC_GIT_SHA"),
    ")"
);
