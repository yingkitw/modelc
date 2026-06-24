mod common;

use modelc::generate::{GenerationConfig, generate_token_ids, generate_token_ids_with_cache};
use modelc::prefix_cache::PrefixCache;
use modelc::runtime::serve::Runtime;
use modelc::tokenizer::BpeTokenizer;

fn config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        ..GenerationConfig::default()
    }
}

/// The prefix cache must be transparent: identical output whether or not a cache
/// is supplied, and regardless of hit vs. miss. This is the central correctness
/// invariant for the feature.
#[test]
fn cache_is_transparent_for_identical_prompt() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let prompt = "system: you are helpful. user: hello";
    let cfg = config(8);

    // Baseline: no cache.
    let baseline = generate_token_ids(&runtime, "gpt2", hidden, &tokenizer, prompt, &cfg, None);

    // With a fresh cache (cold miss → inserts).
    let mut cache = PrefixCache::new(8);
    let cold = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt,
        &cfg,
        Some(&mut cache),
        None,
    );
    assert_eq!(
        cold, baseline,
        "cold (miss) output must match the no-cache baseline"
    );
    assert_eq!(cache.len(), 1, "cold call should populate the cache");

    // With a warm cache (exact hit → reuses KV + stored logits).
    let warm = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt,
        &cfg,
        Some(&mut cache),
        None,
    );
    assert_eq!(
        warm, baseline,
        "warm (hit) output must match the no-cache baseline"
    );
    assert_eq!(cache.len(), 1, "exact repeat should not grow the cache");
}

/// A prompt that shares a prefix with a cached prompt must reuse the cached prefix
/// KV and still produce output identical to a cold generation of that same prompt.
#[test]
fn cache_reuses_shared_prefix() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let cfg = config(6);

    let prompt_a = "system: you are helpful.";
    let prompt_b = "system: you are helpful. user: what is 2+2?";

    // Baseline for B without any shared-prefix caching.
    let baseline_b = generate_token_ids(&runtime, "gpt2", hidden, &tokenizer, prompt_b, &cfg, None);

    // Prime the cache with A, then generate B (B starts with A → prefix reuse).
    let mut cache = PrefixCache::new(8);
    let _ = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt_a,
        &cfg,
        Some(&mut cache),
        None,
    );
    assert_eq!(cache.len(), 1);

    let b_cached = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt_b,
        &cfg,
        Some(&mut cache),
        None,
    );
    assert_eq!(
        b_cached, baseline_b,
        "prefix-reuse output must match the cold baseline for the same prompt"
    );
    // A was a prefix of B, and B is now cached too.
    assert_eq!(cache.len(), 2);
}

/// A non-prefix cached entry must never be reused for an unrelated prompt.
#[test]
fn cache_ignores_non_prefix_entries() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let cfg = config(4);

    let unrelated = "completely different prompt xyz";
    let baseline = generate_token_ids(&runtime, "gpt2", hidden, &tokenizer, unrelated, &cfg, None);

    let mut cache = PrefixCache::new(8);
    // Insert something that is NOT a prefix of `unrelated`.
    let _ = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        "system prompt one",
        &cfg,
        Some(&mut cache),
        None,
    );
    let got = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        unrelated,
        &cfg,
        Some(&mut cache),
        None,
    );
    assert_eq!(
        got, baseline,
        "unrelated prompt must produce its cold baseline"
    );
    assert_eq!(cache.len(), 2);
}

/// Speculative generation with a neural draft model produces the same output
/// as the non-speculative path (gamma = 0) when the draft model is deterministic.
#[test]
fn neural_draft_model_speculative_matches_non_speculative() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();

    // Non-speculative baseline.
    let baseline_cfg = config(6);
    let baseline = generate_token_ids(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        "hello",
        &baseline_cfg,
        None,
    );

    // Build a tiny draft model that always proposes token 0 (the most common
    // fallback).  Because the draft will almost always be rejected, the
    // speculative path falls back to the target model on the first mismatch,
    // producing output identical to the non-speculative baseline.
    let draft = modelc::draft::MlpDraftModel::from_runtime(
        &runtime,
        tokenizer.vocab_size(),
        hidden,
        2,
        0.0,
        0.0,
    );

    // If no draft weights are present in the test model, `from_runtime` returns
    // None and we fall back to the n-gram draft model — still a valid test.
    let spec_cfg = GenerationConfig {
        gamma: 3,
        ..baseline_cfg
    };
    let spec = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        "hello",
        &spec_cfg,
        None,
        draft.as_ref().map(|d| d as &dyn modelc::draft::DraftModel),
    );

    assert_eq!(
        spec, baseline,
        "speculative with draft must match non-speculative baseline"
    );
}

/// A pre-tripped cancellation flag must stop generation before the first token,
/// proving the loop honors the flag each iteration.
#[test]
fn cancellation_flag_stops_generation() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let prompt = "system: you are helpful. user: hello";

    // Baseline: uncancelled generation produces tokens up to max_tokens.
    let cfg = config(8);
    let baseline = generate_token_ids(&runtime, "gpt2", hidden, &tokenizer, prompt, &cfg, None);
    assert!(
        !baseline.is_empty(),
        "uncancelled generation should produce tokens"
    );

    // With the cancel flag pre-set, the loop bails immediately → no tokens.
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let cancelled_cfg = GenerationConfig {
        cancel: Some(cancel),
        ..config(8)
    };
    let ids = generate_token_ids(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt,
        &cancelled_cfg,
        None,
    );
    assert!(
        ids.is_empty(),
        "cancelled generation should produce no tokens, got {}",
        ids.len()
    );
}
