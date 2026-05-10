pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod model;
pub mod parsers;
pub mod runtime;

/// Semver plus short git SHA from `build.rs`, shown by `modelc --version`.
pub const CLI_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (git ",
    env!("MODELC_GIT_SHA"),
    ")"
);
