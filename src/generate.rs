//! Autoregressive text generation for GPT-2 / LLaMA transformer models.
//!
//! Given a prompt, tokenizes it, runs the transformer forward pass in a loop with a KV cache,
//! samples the next token, and decodes the generated text.
//!
//! Also supports speculative decoding via an n-gram draft model that proposes candidate
//! tokens from prompt context, which the target model verifies in a single forward pass loop.

use std::collections::HashMap;

use crate::runtime::serve::Runtime;
use crate::runtime::transformer::{forward_gpt2_cached, forward_llama_cached, KvCache};
use crate::tokenizer::BpeTokenizer;

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    /// Number of draft tokens to propose per speculative step (0 = disabled).
    pub gamma: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0, // 0.0 = greedy
            gamma: 0,
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
) -> String {
    if config.gamma > 0 {
        return speculative_generate(runtime, architecture, hidden, tokenizer, prompt, config);
    }

    let mut token_ids = tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return String::new();
    }

    let n_layers = count_layers(runtime, architecture);
    let kv_cache = KvCache::new(n_layers);
    let mut kv_cache_opt = Some(kv_cache);

    for _ in 0..config.max_tokens {
        let last_id = token_ids.last().copied().unwrap_or(0);
        let embedding = token_embedding(runtime, architecture, last_id, hidden);

        let logits = match architecture {
            "gpt2" => {
                forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt)
            }
            "llama" => {
                forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt)
            }
            _ => break,
        };

        let next_token = sample(logits.as_deref(), config);
        token_ids.push(next_token);

        if tokenizer.vocab_size() > 0 && next_token as usize >= tokenizer.vocab_size() {
            break;
        }
    }

    // Decode only the newly generated tokens.
    let prompt_len = tokenizer.encode(prompt).len();
    tokenizer.decode(&token_ids[prompt_len.min(token_ids.len())..])
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
fn speculative_generate(
    runtime: &Runtime,
    architecture: &str,
    hidden: usize,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> String {
    let mut token_ids = tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return String::new();
    }

    let n_layers = count_layers(runtime, architecture);
    let kv_cache = KvCache::new(n_layers);
    let mut kv_cache_opt = Some(kv_cache);

    let prompt_len = token_ids.len();
    let n = 3usize; // trigram
    let mut ngram_index = build_ngram_index(&token_ids, n);

    while token_ids.len() - prompt_len < config.max_tokens {
        let draft = draft_tokens(&ngram_index, &token_ids, n, config.gamma);
        if draft.is_empty() {
            // No draft available — fall back to one standard generation step.
            let last_id = token_ids.last().copied().unwrap_or(0);
            let embedding = token_embedding(runtime, architecture, last_id, hidden);
            let logits = match architecture {
                "gpt2" => forward_gpt2_cached(runtime, &embedding, false, &mut kv_cache_opt),
                "llama" => forward_llama_cached(runtime, &embedding, false, &mut kv_cache_opt),
                _ => break,
            };
            let next_token = sample(logits.as_deref(), config);
            token_ids.push(next_token);
            ngram_index = update_ngram_index(&ngram_index, &token_ids, n);
            if tokenizer.vocab_size() > 0 && next_token as usize >= tokenizer.vocab_size() {
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
            let next_token = sample(logits.as_deref(), config);
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

    tokenizer.decode(&token_ids[prompt_len.min(token_ids.len())..])
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
        argmax(logits) as u32
    } else {
        let scaled: Vec<f32> = logits.iter().map(|l| l / config.temperature).collect();
        let probs = softmax(&scaled);
        multinomial(&probs)
    }
}

fn argmax(xs: &[f32]) -> usize {
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

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn multinomial(probs: &[f32]) -> u32 {
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
            gamma: 0,
        };
        assert_eq!(sample(Some(&[1.0, 5.0, 2.0]), &config), 1);
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
}
