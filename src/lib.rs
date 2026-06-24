pub mod arch;
pub mod chat_template;
pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod config;
pub mod constraint;
pub mod containerize;
pub mod draft;
pub mod generate;
pub mod json_schema;
pub mod lora;
pub mod metal;
pub mod model;
pub mod onnx_exec;
pub mod pack;
pub mod prefix_cache;
pub mod parsers;
pub mod runtime;
pub mod serve;
pub mod store;
pub mod tokenizer;

/// Semver plus short git SHA from `build.rs`, shown by `modelc --version`.
pub const CLI_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (git ",
    env!("MODELC_GIT_SHA"),
    ")"
);

/// Short git SHA from `build.rs`.
pub const GIT_SHA: &str = env!("MODELC_GIT_SHA");
