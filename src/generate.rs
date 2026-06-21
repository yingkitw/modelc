//! Autoregressive text generation for GPT-2 / LLaMA transformer models.
//!
//! Given a prompt, tokenizes it, runs the transformer forward pass in a loop with a KV cache,
//! samples the next token, and decodes the generated text.
//!
//! Also supports speculative decoding via an n-gram draft model that proposes candidate
//! tokens from prompt context, which the target model verifies in a single forward pass loop.

use std::collections::HashMap;

use crate::prefix_cache::{CachedPrefix, PrefixCache};
use crate::runtime::serve::Runtime;
use crate::runtime::transformer::{KvCache, forward_gpt2_cached, forward_llama_cached};
use crate::tokenizer::BpeTokenizer;

/// Create a KvCache based on the generation config flags.
fn make_kv_cache(n_layers: usize, hidden: usize, config: &GenerationConfig) -> KvCache {
    if config.use_mixed_kv {
        KvCache::new_mixed(n_layers, hidden, 64)
    } else {
        KvCache::new_quantized(n_layers, hidden, config.use_int8_kv)
    }
}

/// Apply context shifting when token count exceeds max_context.
/// If `anchor_tokens > 0`, preserves the first N tokens (StreamingLLM-style)
/// and evicts from the middle; otherwise evicts oldest tokens.
fn context_shift(
    token_ids: &mut Vec<u32>,
    prompt_len: &mut usize,
    kv_cache_opt: &mut Option<KvCache>,
    max_ctx: usize,
    anchor_tokens: usize,
) {
    let len = token_ids.len();
    let overflow = len - max_ctx + max_ctx / 4; // keep ~75%

    if anchor_tokens > 0 && overflow > 0 && len > anchor_tokens + overflow {
        // Anchored eviction: preserve first anchor_tokens, remove from middle.
        if let Some(kv) = kv_cache_opt {
            kv.shift_anchored(overflow, anchor_tokens);
        }
        let suffix_start = anchor_tokens + overflow;
        let suffix = token_ids[suffix_start..].to_vec();
        token_ids.truncate(anchor_tokens);
        token_ids.extend_from_slice(&suffix);
        if *prompt_len > anchor_tokens {
            let non_anchor = *prompt_len - anchor_tokens;
            let removed = overflow.min(non_anchor);
            *prompt_len = anchor_tokens + non_anchor - removed;
        }
    } else {
        // Standard eviction: remove oldest tokens.
        if let Some(kv) = kv_cache_opt {
            kv.shift(overflow);
        }
        *token_ids = token_ids.split_off(overflow);
        *prompt_len = prompt_len.saturating_sub(overflow);
    }
}

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    /// Nucleus sampling threshold (0.0 = disabled). Keeps the smallest set of tokens whose
    /// cumulative probability exceeds `top_p`, then renormalizes and samples.
    pub top_p: f32,
    /// Number of draft tokens to propose per speculative step (0 = disabled).
    pub gamma: usize,
    /// Use INT8 quantization for the KV cache (4x memory reduction).
    pub use_int8_kv: bool,
    /// Use mixed-precision KV cache: recent tokens in FP32, older in INT8.
    pub use_mixed_kv: bool,
    /// Optional grammar constraint (e.g., regex) applied during sampling.
    pub constraint: Option<std::sync::Arc<dyn crate::constraint::Constraint>>,
    /// Maximum context length before KV cache shifting. When total tokens exceed this,
    /// the oldest tokens are discarded and remaining cache is shifted left. `None` = no limit.
    pub max_context: Option<usize>,
    /// Number of initial "anchor" tokens to preserve during KV cache eviction.
    /// Inspired by StreamingLLM: the first few tokens act as "attention sinks" and
    /// retaining them stabilizes attention quality when context is truncated.
    /// Only effective when `max_context` is set. Default 0 (no anchor preservation).
    pub anchor_tokens: usize,
    /// Optional stop sequences. Generation halts when any sequence appears in the output.
    pub stop: Vec<String>,
}

impl Clone for GenerationConfig {
    fn clone(&self) -> Self {
        Self {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            gamma: self.gamma,
            use_int8_kv: self.use_int8_kv,
            use_mixed_kv: self.use_mixed_kv,
            constraint: self.constraint.clone(),
            max_context: self.max_context,
            anchor_tokens: self.anchor_tokens,
            stop: self.stop.clone(),
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0, // 0.0 = greedy
            top_p: 0.0,
            gamma: 0,
            use_int8_kv: false,
            use_mixed_kv: false,
            constraint: None,
            max_context: None,
            anchor_tokens: 0,
            stop: Vec::new(),
        }
    }
}

/// Generate text autoregressively from a prompt.
///
/// When `config.gamma > 0`, uses speculative decoding with an n-gram draft model.
/// Returns the newly generated text (excluding the prompt).
pub fn generate(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    draft_model: Option<&dyn crate::draft::DraftModel>,
) -> String {
    let generated = generate_token_ids(runtime, architecture, hidden, tokenizer, prompt, config, draft_model);
    tokenizer.decode(&generated)
}

/// Generate token IDs autoregressively from a prompt.
///
/// Returns the newly generated token IDs (excluding the prompt tokens). When
/// `config.gamma > 0`, uses the speculative path; otherwise the prompt is fully
/// primed into the KV cache before generation (so the model attends to the whole
/// prompt, not just its last token).
pub fn generate_token_ids(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    draft_model: Option<&dyn crate::draft::DraftModel>,
) -> Vec<u32> {
    generate_token_ids_with_cache(runtime, architecture, hidden, tokenizer, prompt, config, None, draft_model)
}

/// Like [`generate_token_ids`], but consults an optional [`PrefixCache`] to skip
/// recomputation of K/V for prompt prefixes shared with a prior request.
///
/// The output is identical to the non-cached path; the cache only affects speed.
/// Both the standard and speculative paths use the cache.
#[allow(clippy::too_many_arguments)]
pub fn generate_token_ids_with_cache(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    mut cache: Option<&mut PrefixCache>,
    draft_model: Option<&dyn crate::draft::DraftModel>,
) -> Vec<u32> {
    let prompt_ids = tokenizer.encode(prompt);
    let lookup = cache.as_ref().and_then(|c| c.lookup(&prompt_ids));
    if config.gamma > 0 {
        let (text, maybe_kv) = speculative_generate(
            runtime, architecture, hidden, tokenizer, prompt, config, lookup, draft_model,
        );
        let all_ids = tokenizer.encode(&(prompt.to_string() + &text));
        let ids = if all_ids.len() > prompt_ids.len() {
            all_ids[prompt_ids.len()..].to_vec()
        } else {
            Vec::new()
        };
        if let Some(ref mut c) = cache
            && let Some(kv) = maybe_kv
        {
            c.insert(
                all_ids,
                CachedPrefix {
                    kv,
                    last_logits: Vec::new(),
                },
            );
        }
        return ids;
    }

    let (ids, _logprobs, maybe_kv) = generate_core(
        runtime,
        architecture,
        hidden,
        tokenizer,
        prompt,
        config,
        lookup,
        false,
        0,
    );
    if let Some(ref mut c) = cache
        && let Some(kv) = maybe_kv
    {
        let mut full_ids = prompt_ids.clone();
        full_ids.extend_from_slice(&ids);
        c.insert(
            full_ids,
            CachedPrefix {
                kv,
                last_logits: Vec::new(),
            },
        );
    }
    ids
}

/// Per-token logprob information produced by `generate_with_logprobs`.
///
/// `logprob` and the entries in `top_logprobs` are natural logs of probabilities
/// computed from the model's **raw** (pre-temperature) softmax distribution,
/// so they are deterministic regardless of the sampling temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The token ID that was sampled for this position.
    pub token: u32,
    /// `ln(P(token))` under the raw softmax of the logits.
    pub logprob: f32,
    /// Up to `top_n` most-probable tokens as `(token_id, logprob)`, sorted by
    /// descending probability. May or may not include `token`.
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Generate token IDs autoregressively, also returning per-token logprobs.
///
/// Records the logprob of each sampled token plus the top-`top_logprobs`
/// alternatives. Logprobs are derived from the raw (pre-temperature) softmax of
/// the logits, so they are deterministic regardless of sampling temperature.
///
/// Returns the newly generated token IDs (excluding the prompt) and one
/// `TokenLogprob` per generated token. For unknown architectures both vectors
/// are empty.
pub fn generate_with_logprobs(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    top_logprobs: usize,
) -> (Vec<u32>, Vec<TokenLogprob>) {
    generate_with_logprobs_with_cache(
        runtime,
        architecture,
        hidden,
        tokenizer,
        prompt,
        config,
        top_logprobs,
        None,
    )
}

/// Like [`generate_with_logprobs`], but consults an optional [`PrefixCache`].
/// Output is identical to the non-cached path; the cache only affects speed.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_logprobs_with_cache(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    top_logprobs: usize,
    mut cache: Option<&mut PrefixCache>,
) -> (Vec<u32>, Vec<TokenLogprob>) {
    let prompt_ids = tokenizer.encode(prompt);
    let lookup = cache.as_ref().and_then(|c| c.lookup(&prompt_ids));
    let (ids, logprobs, maybe_kv) = generate_core(
        runtime,
        architecture,
        hidden,
        tokenizer,
        prompt,
        config,
        lookup,
        true,
        top_logprobs,
    );
    if let Some(ref mut c) = cache
        && let Some(kv) = maybe_kv
    {
        let mut full_ids = prompt_ids.clone();
        full_ids.extend_from_slice(&ids);
        c.insert(
            full_ids,
            CachedPrefix {
                kv,
                last_logits: Vec::new(),
            },
        );
    }
    (ids, logprobs)
}

/// Core autoregressive loop shared by the token-id and logprobs entry points.
///
/// Pipeline:
/// 1. Tokenize the prompt.
/// 2. If a cache is supplied, restore the longest cached prefix's K/V state.
/// 3. Prime: process every prompt token beyond the matched prefix through the
///    cached forward, so the KV cache reflects the full prompt context. The last
///    step's logits seed the first generated token.
/// 4. Store the primed KV (keyed by the full prompt) back into the cache.
/// 5. Sample + generate up to `max_tokens`, optionally recording per-token logprobs.
///
/// Step 3 also fixes a prior limitation where only the last prompt token was
/// attended to; the model now sees the entire prompt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_core(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    cache_lookup: Option<crate::prefix_cache::CacheLookup>,
    collect_logprobs: bool,
    top_logprobs: usize,
) -> (Vec<u32>, Vec<TokenLogprob>, Option<KvCache>) {
    let mut token_ids = tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return (Vec::new(), Vec::new(), None);
    }
    let mut prompt_len = token_ids.len();
    let n_layers = count_layers(runtime, architecture);

    // --- Restore the longest cached prefix, if any. ---
    let mut matched_len = 0usize;
    let mut kv_cache_opt: Option<KvCache>;
    let mut logits: Option<Vec<f32>> = None;

    if let Some(look) = cache_lookup {
        kv_cache_opt = Some(look.kv);
        matched_len = look.matched_len;
        logits = look.last_logits; // `Some` only on an exact full-match.
    } else {
        kv_cache_opt = Some(make_kv_cache(n_layers, hidden, config));
    }

    // --- Prime: process prompt tokens beyond the matched prefix. ---
    // Each forward appends one position to the KV cache; the last step's logits
    // predict the first generated token. Skipped entirely on an exact cache hit.
    if logits.is_none() {
        for &id in &token_ids[matched_len..] {
            let embedding = token_embedding(runtime, architecture, id, hidden);
            logits = match architecture {
                "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
                "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
                _ => break,
            };
        }
    }

    // --- Generation loop. ---
    // If priming produced no logits (unknown architecture / no output head), there is
    // nothing to sample from — bail out with an empty result.
    if logits.is_none() {
        return (Vec::new(), Vec::new(), kv_cache_opt.clone());
    }
    let mut generated = 0usize;
    let mut logprobs: Vec<TokenLogprob> = Vec::new();
    loop {
        if generated >= config.max_tokens {
            break;
        }
        let next_token = sample_constrained(
            logits.as_deref(),
            config,
            tokenizer,
            &token_ids[prompt_len..],
        );
        if collect_logprobs {
            logprobs.push(match logits.as_deref() {
                Some(l) => logprob_for_step(l, next_token, top_logprobs),
                None => TokenLogprob {
                    token: next_token,
                    logprob: f32::NEG_INFINITY,
                    top_logprobs: Vec::new(),
                },
            });
        }
        token_ids.push(next_token);
        generated += 1;

        // --- Stop sequence check ---
        if !config.stop.is_empty() {
            let text = tokenizer.decode(&token_ids[prompt_len..]);
            if let Some(pos) = config.stop.iter().filter_map(|s| text.find(s)).min() {
                // Truncate tokens to just before the stop sequence starts.
                let prefix = &text[..pos];
                let prefix_ids = tokenizer.encode(prefix);
                let target_len = prompt_len + prefix_ids.len();
                token_ids.truncate(target_len);
                break;
            }
        }

        if let Some(max_ctx) = config.max_context
            && token_ids.len() > max_ctx
        {
            context_shift(&mut token_ids, &mut prompt_len, &mut kv_cache_opt, max_ctx, config.anchor_tokens);
        }

        if tokenizer.vocab_size() > 0 && next_token as usize >= tokenizer.vocab_size() {
            break;
        }

        // Compute logits for the just-appended token (drives the next sample).
        let embedding = token_embedding(runtime, architecture, next_token, hidden);
        let next_logits = match architecture {
            "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
            "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
            _ => break,
        };
        match next_logits {
            Some(l) => logits = Some(l),
            None => break,
        }
    }

    let cut = prompt_len.min(token_ids.len());
    let final_kv = kv_cache_opt.clone();
    (token_ids[cut..].to_vec(), logprobs, final_kv)
}

/// Check whether any stop sequence appears in the generated text.
/// If found, truncates `token_ids` to just before the stop sequence and returns
/// the truncated vec. Otherwise returns `None`.
fn check_stop_sequences(
    tokenizer: &crate::tokenizer::BpeTokenizer,
    stop: &[String],
    token_ids: &mut Vec<u32>,
    prompt_len: usize,
) -> Option<Vec<u32>> {
    if stop.is_empty() {
        return None;
    }
    let text = tokenizer.decode(&token_ids[prompt_len..]);
    let pos = stop.iter().filter_map(|s| text.find(s)).min()?;
    let prefix = &text[..pos];
    let prefix_ids = tokenizer.encode(prefix);
    token_ids.truncate(prompt_len + prefix_ids.len());
    Some(token_ids.clone())
}

/// Compute per-token logprob info from raw logits.
///
/// `logprob` is `ln(P(chosen))` from the softmax of `logits`. `top_logprobs`
/// contains up to `top_n` `(token_id, logprob)` pairs sorted by descending
/// probability.
fn logprob_for_step(logits: &[f32], chosen: u32, top_n: usize) -> TokenLogprob {
    if logits.is_empty() {
        return TokenLogprob {
            token: chosen,
            logprob: f32::NEG_INFINITY,
            top_logprobs: Vec::new(),
        };
    }

    let probs = softmax(logits);
    let logprob = probs
        .get(chosen as usize)
        .map(|p| p.ln())
        .unwrap_or(f32::NEG_INFINITY);

    let top_logprobs = if top_n == 0 {
        Vec::new()
    } else {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(top_n)
            .map(|(i, p)| (i as u32, p.ln()))
            .collect()
    };

    TokenLogprob {
        token: chosen,
        logprob,
        top_logprobs,
    }
}

/// Speculative generation using an n-gram draft model.
///
/// 1. Build an n-gram index from the prompt tokens.
/// 2. For each step, draft `gamma` candidate tokens by looking up the most recent n-gram
///    in the prefix and appending its historical continuation.
/// 3. Verify each draft token with the target model (one forward pass per token, reusing
///    the KV cache). If the model's sampled token matches the draft, accept it.
/// 4. On the first mismatch, use the model's sampled token and restart drafting.
///
/// This reduces per-token sampling overhead when the draft model (n-gram lookup) is
/// accurate, and establishes the infrastructure for future faster draft models.
#[allow(clippy::too_many_arguments)]
pub(crate) fn speculative_generate(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
    cache_lookup: Option<crate::prefix_cache::CacheLookup>,
    draft_model: Option<&dyn crate::draft::DraftModel>,
) -> (String, Option<KvCache>) {
    let mut token_ids = tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return (String::new(), None);
    }

    let n_layers = count_layers(runtime, architecture);
    let mut prompt_len = token_ids.len();
    let n = 3usize; // trigram
    let mut ngram_index = build_ngram_index(&token_ids, n);

    // --- Restore the longest cached prefix, if any. ---
    let mut matched_len = 0usize;
    let mut kv_cache_opt: Option<KvCache>;
    if let Some(look) = cache_lookup {
        kv_cache_opt = Some(look.kv);
        matched_len = look.matched_len;
    } else {
        kv_cache_opt = Some(make_kv_cache(n_layers, hidden, config));
    }

    // --- Prime: process prompt tokens beyond the matched prefix. ---
    for &id in &token_ids[matched_len..] {
        let embedding = token_embedding(runtime, architecture, id, hidden);
        let _ = match architecture {
            "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
            "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
            _ => break,
        };
    }


    while token_ids.len() - prompt_len < config.max_tokens {
        let draft = if let Some(dm) = draft_model {
            dm.draft(&token_ids, config.gamma)
        } else {
            draft_tokens(&ngram_index, &token_ids, n, config.gamma)
        };
        if draft.is_empty() {
            // No draft available — fall back to one standard generation step.
            let last_id = token_ids.last().copied().unwrap_or(0);
            let embedding = token_embedding(runtime, architecture, last_id, hidden);
            let logits = match architecture {
                "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
                "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
                _ => break,
            };
            let next_token = sample_constrained(logits.as_deref(), config, tokenizer, &token_ids[prompt_len..]);
            token_ids.push(next_token);
            ngram_index = update_ngram_index(&ngram_index, &token_ids, n);

            if let Some(max_ctx) = config.max_context
                && token_ids.len() > max_ctx
            {
                context_shift(&mut token_ids, &mut prompt_len, &mut kv_cache_opt, max_ctx, config.anchor_tokens);
            }

            if tokenizer.vocab_size() > 0 && next_token as usize >= tokenizer.vocab_size() {
                break;
            }
            if let Some(truncated) = check_stop_sequences(tokenizer, &config.stop, &mut token_ids, prompt_len) {
                token_ids = truncated;
                break;
            }
            continue;
        }

        let mut accepted = 0usize;
        for &draft_token in &draft {
            let last_id = token_ids.last().copied().unwrap_or(0);
            let embedding = token_embedding(runtime, architecture, last_id, hidden);
            let logits = match architecture {
                "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
                "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
                _ => break,
            };
            let next_token = sample_constrained(logits.as_deref(), config, tokenizer, &token_ids[prompt_len..]);
            if next_token == draft_token {
                token_ids.push(draft_token);
                accepted += 1;
                if tokenizer.vocab_size() > 0 && draft_token as usize >= tokenizer.vocab_size() {
                    break;
                }
            } else {
                token_ids.push(next_token);
                break;
            }
        }

        // Update the n-gram index with newly accepted tokens.
        ngram_index = update_ngram_index(&ngram_index, &token_ids, n);

        if let Some(truncated) = check_stop_sequences(tokenizer, &config.stop, &mut token_ids, prompt_len) {
            token_ids = truncated;
            break;
        }

        if let Some(max_ctx) = config.max_context
            && token_ids.len() > max_ctx
        {
            context_shift(&mut token_ids, &mut prompt_len, &mut kv_cache_opt, max_ctx, config.anchor_tokens);
        }

        // Stop if we've hit max tokens or an out-of-vocab token was emitted.
        if token_ids.len() - prompt_len >= config.max_tokens {
            break;
        }
        if accepted < draft.len()
            && tokenizer.vocab_size() > 0
            && token_ids.last().copied().unwrap_or(0) as usize >= tokenizer.vocab_size()
        {
            break;
        }
    }

    let final_kv = kv_cache_opt.clone();
    (tokenizer.decode(&token_ids[prompt_len.min(token_ids.len())..]), final_kv)
}

/// Build an n-gram index from a token sequence.
/// Maps each n-gram (as a Vec<u32>) to a list of tokens that historically followed it.
fn build_ngram_index(tokens: &[u32], n: usize) -> HashMap<Vec<u32>, Vec<u32>> {
    let mut index = HashMap::new();
    if tokens.len() < n + 1 {
        return index;
    }
    for window in tokens.windows(n + 1) {
        let key = window[..n].to_vec();
        let next = window[n];
        index.entry(key).or_default().push(next);
    }
    index
}

/// Incrementally update the n-gram index with the latest tokens.
fn update_ngram_index(
    index: &HashMap<Vec<u32>, Vec<u32>>,
    tokens: &[u32],
    n: usize,
) -> HashMap<Vec<u32>, Vec<u32>> {
    let mut new_index = index.clone();
    if tokens.len() < n + 1 {
        return new_index;
    }
    // Only add the newest n-gram ending at the last token.
    let start = tokens.len().saturating_sub(n + 1);
    let window = &tokens[start..];
    let key = window[..n].to_vec();
    let next = window[n];
    new_index.entry(key).or_default().push(next);
    new_index
}

/// Draft up to `gamma` tokens by repeatedly looking up the most recent n-gram in `index`
/// and choosing the most frequent historical continuation.
fn draft_tokens(
    index: &HashMap<Vec<u32>, Vec<u32>>,
    tokens: &[u32],
    n: usize,
    gamma: usize,
) -> Vec<u32> {
    let mut draft = Vec::new();
    let mut context = tokens.to_vec();
    for _ in 0..gamma {
        let key = if context.len() >= n {
            context[context.len() - n..].to_vec()
        } else {
            context.clone()
        };
        if let Some(candidates) = index.get(&key) {
            let next = most_frequent(candidates);
            draft.push(next);
            context.push(next);
        } else {
            break;
        }
    }
    draft
}

/// Return the most frequent element in `xs`.
fn most_frequent(xs: &[u32]) -> u32 {
    let mut counts = HashMap::new();
    for &x in xs {
        *counts.entry(x).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn token_embedding(runtime: &Runtime, arch: &str, token_id: u32, hidden: usize) -> Vec<f32> {
    let weight_name = match arch {
        "gpt2" => "transformer.wte.weight",
        "llama" => "model.embed_tokens.weight",
        _ => return vec![0.0; hidden],
    };

    if let Some(w) = runtime.get(weight_name) {
        let idx = (token_id as usize) * hidden;
        if idx + hidden <= w.data.len() {
            return w.data[idx..idx + hidden].to_vec();
        }
    }

    vec![0.0; hidden]
}

fn count_layers(runtime: &Runtime, arch: &str) -> usize {
    let prefix = match arch {
        "gpt2" => "transformer.h.",
        "llama" => "model.layers.",
        _ => return 0,
    };

    let mut max = 0usize;
    for name in runtime.tensor_names() {
        if let Some(rest) = name.strip_prefix(prefix)
            && let Some(n) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
        {
            max = max.max(n);
        }
    }
    max + 1
}

fn sample(logits: Option<&[f32]>, config: &GenerationConfig) -> u32 {
    let logits = logits.unwrap_or(&[]);
    if logits.is_empty() {
        return 0;
    }

    if config.temperature <= 0.0 {
        return argmax(logits) as u32;
    }

    let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
    let mut probs = softmax(&scaled);

    if config.top_p > 0.0 && config.top_p < 1.0 {
        apply_top_p(&mut probs, config.top_p);
    }

    multinomial(&probs)
}

/// Sample with an optional grammar constraint applied to the logits.
///
/// If `config.constraint` is set, only tokens whose decoded text keeps the partial
/// output compatible with the regex are allowed. Invalid tokens are masked to
/// `-inf` before sampling.
fn sample_constrained(
    logits: Option<&[f32]>,
    config: &GenerationConfig,
    tokenizer: &BpeTokenizer,
    generated_tokens: &[u32],
) -> u32 {
    let logits = logits.unwrap_or(&[]);
    if logits.is_empty() {
        return 0;
    }

    if let Some(constraint) = &config.constraint {
        let prefix = tokenizer.decode(generated_tokens);
        let vocab_size = tokenizer.vocab_size();
        let token_bytes: Vec<Option<Vec<u8>>> = (0..vocab_size as u32)
            .map(|id| tokenizer.token_bytes(id).map(|b| b.to_vec()))
            .collect();
        let mask = constraint.valid_mask(&prefix, vocab_size, &token_bytes);
        let mut masked = logits.to_vec();
        for (i, valid) in mask.iter().enumerate() {
            if !valid && i < masked.len() {
                masked[i] = f32::NEG_INFINITY;
            }
        }
        return sample(Some(&masked), config);
    }

    sample(Some(logits), config)
}

/// In-place nucleus (top-p) filtering.
/// Sorts probabilities descending, keeps the smallest prefix whose cumulative sum >= `top_p`,
/// zeros everything else, then renormalizes.
pub(crate) fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cum = 0.0f32;
    let mut keep = 0usize;
    for (_, p) in &indexed {
        cum += *p;
        keep += 1;
        if cum >= top_p {
            break;
        }
    }

    // Zero out dropped tokens.
    let keep_set: std::collections::HashSet<usize> =
        indexed[..keep].iter().map(|(i, _)| *i).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep_set.contains(&i) {
            *p = 0.0;
        }
    }

    // Renormalize.
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

pub(crate) fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_val {
            best = i;
            best_val = x;
        }
    }
    best
}

pub(crate) fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

pub(crate) fn multinomial(probs: &[f32]) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    probs.len().saturating_sub(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn greedy_sample_picks_argmax() {
        let config = GenerationConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_p: 0.0,
            gamma: 0,
            use_int8_kv: false,
            use_mixed_kv: false,
            constraint: None,
            max_context: None,
            anchor_tokens: 0,
            stop: vec![],
        };
        assert_eq!(sample(Some(&[1.0, 5.0, 2.0]), &config), 1);
    }

    #[test]
    fn apply_top_p_keeps_nucleus() {
        let mut probs = vec![0.5f32, 0.3, 0.15, 0.05];
        apply_top_p(&mut probs, 0.8);
        // Should keep tokens 0 and 1 (0.5 + 0.3 = 0.8 >= 0.8)
        assert!((probs[2]).abs() < 1e-6, "token 2 should be zeroed");
        assert!((probs[3]).abs() < 1e-6, "token 3 should be zeroed");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "should renormalize to 1.0");
    }

    #[test]
    fn apply_top_p_with_very_high_p_keeps_all() {
        let mut probs = vec![0.4f32, 0.3, 0.2, 0.1];
        apply_top_p(&mut probs, 0.99);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "should still sum to 1.0");
        assert!(probs.iter().all(|&p| p > 0.0), "all tokens should be kept");
    }

    #[test]
    fn build_ngram_index_maps_trigrams() {
        let tokens = vec![1, 2, 3, 1, 2, 4];
        let index = build_ngram_index(&tokens, 3);
        assert_eq!(index.get([1, 2, 3].as_slice()).unwrap(), [1].as_slice());
        assert_eq!(index.get([2, 3, 1].as_slice()).unwrap(), [2].as_slice());
        assert_eq!(index.get([3, 1, 2].as_slice()).unwrap(), [4].as_slice());
    }

    #[test]
    fn draft_tokens_continues_trigram() {
        let tokens = vec![1, 2, 3, 1, 2, 4, 1, 2, 3, 5];
        let index = build_ngram_index(&tokens, 3);
        // Context [1,2,3] was followed by 1 and 5; most frequent is first seen (1)
        let draft = draft_tokens(&index, &[1, 2, 3], 3, 2);
        assert!(!draft.is_empty(), "should produce at least one draft token");
    }

    #[test]
    fn draft_tokens_stops_when_no_match() {
        let index = build_ngram_index(&[1, 2, 3, 4], 3);
        let draft = draft_tokens(&index, &[9, 9, 9], 3, 3);
        assert!(draft.is_empty(), "unknown n-gram should yield empty draft");
    }

    #[test]
    fn most_frequent_picks_mode() {
        assert_eq!(most_frequent(&[1, 2, 2, 3, 2]), 2);
        assert_eq!(most_frequent(&[5]), 5);
    }

    #[test]
    fn update_ngram_index_adds_latest() {
        let mut index = build_ngram_index(&[1, 2, 3, 4], 3);
        index = update_ngram_index(&index, &[1, 2, 3, 4, 5], 3);
        assert_eq!(index.get([2, 3, 4].as_slice()).unwrap(), [5].as_slice());
    }

    #[test]
    fn generate_token_ids_returns_empty_for_unknown_arch() {
        let runtime = Runtime::from_raw(&std::collections::HashMap::new());
        let tokenizer = BpeTokenizer::byte_fallback();
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 0.0,
            top_p: 0.0,
            gamma: 0,
            use_int8_kv: false,
            use_mixed_kv: false,
            constraint: None,
            max_context: None,
            anchor_tokens: 0,
            stop: vec![],
        };
        let ids = generate_token_ids(&runtime, "unknown", 4, &tokenizer, "hello", &config, None);
        assert!(
            ids.is_empty(),
            "unknown architecture should yield no tokens"
        );
    }

    #[test]
    fn logprob_for_step_argmax_has_highest_logprob() {
        // Token 1 is the argmax (5.0).
        let lp = logprob_for_step(&[1.0, 5.0, 2.0], 1, 0);
        assert_eq!(lp.token, 1);
        assert!(lp.logprob.is_finite());
        // ln(P(1)) must exceed ln(P(0)) and ln(P(2)).
        let lp0 = logprob_for_step(&[1.0, 5.0, 2.0], 0, 0);
        let lp2 = logprob_for_step(&[1.0, 5.0, 2.0], 2, 0);
        assert!(lp.logprob > lp0.logprob);
        assert!(lp.logprob > lp2.logprob);
    }

    #[test]
    fn logprob_for_step_exp_sums_to_one() {
        // exp(logprob) over the full distribution must sum to ~1.
        let logits = [1.0, 2.0, 3.0, 4.0];
        let sum: f32 = logits
            .iter()
            .enumerate()
            .map(|(i, _)| logprob_for_step(&logits, i as u32, 0).logprob.exp())
            .sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp(logprob) sum = {sum}");
    }

    #[test]
    fn logprob_for_step_top_logprobs_sorted_desc_and_limited() {
        let lp = logprob_for_step(&[0.1, 0.9, 5.0, 0.2], 2, 2);
        assert_eq!(lp.top_logprobs.len(), 2);
        // Descending: token 2 (5.0) then token 1 (0.9).
        assert_eq!(lp.top_logprobs[0].0, 2);
        assert_eq!(lp.top_logprobs[1].0, 1);
        assert!(lp.top_logprobs[0].1 > lp.top_logprobs[1].1);
    }

    #[test]
    fn logprob_for_step_empty_logits_yields_neg_infinity() {
        let lp = logprob_for_step(&[], 0, 3);
        assert_eq!(lp.token, 0);
        assert!(lp.logprob.is_infinite() && lp.logprob.is_sign_negative());
        assert!(lp.top_logprobs.is_empty());
    }
}
