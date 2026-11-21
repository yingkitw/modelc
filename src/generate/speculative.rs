//! Speculative decoding: n-gram draft proposal + target-model verification.
//! `speculative_generate` is called by `generate.rs`'s public entry points.

use std::collections::HashMap;

use super::{
    GenerationConfig, check_stop_sequences, context_shift, count_layers,
    make_kv_cache, token_embedding,
};
use super::sampling::sample_constrained;
use crate::runtime::serve::Runtime;
use crate::runtime::transformer::{KvCache, forward_gpt2_cached, forward_llama_cached};
use crate::tokenizer::BpeTokenizer;

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
    let mut generated = 0usize;
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

    while generated < config.max_tokens {
        if config.cancelled() {
            break;
        }
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
                "gpt2" => {
                    forward_gpt2_cached(runtime, embedding.as_ref(), false, &mut kv_cache_opt)
                }
                "llama" => {
                    forward_llama_cached(runtime, embedding.as_ref(), false, &mut kv_cache_opt)
                }
                _ => break,
            };
            let next_token = sample_constrained(
                logits.as_deref(),
                config,
                tokenizer,
                &token_ids[prompt_len..],
                generated,
            );
            token_ids.push(next_token);
            generated += 1;
            ngram_index = update_ngram_index(&ngram_index, &token_ids, n);

            if let Some(max_ctx) = config.max_context {
                if token_ids.len() > max_ctx {
                    context_shift(
                        &mut token_ids,
                        &mut prompt_len,
                        &mut kv_cache_opt,
                        max_ctx,
                        config.anchor_tokens,
                    );
                }
            }

            if tokenizer.vocab_size() > 0 && next_token as usize >= tokenizer.vocab_size() {
                break;
            }
            if let Some(truncated) =
                check_stop_sequences(tokenizer, &config.stop, &mut token_ids, prompt_len)
            {
                token_ids = truncated;
                break;
            }
            continue;
        }

        let mut accepted = 0usize;
        for &draft_token in &draft {
            if generated >= config.max_tokens {
                break;
            }
            let last_id = token_ids.last().copied().unwrap_or(0);
            let embedding = token_embedding(runtime, architecture, last_id, hidden);
            let logits = match architecture {
                "gpt2" => {
                    forward_gpt2_cached(runtime, embedding.as_ref(), false, &mut kv_cache_opt)
                }
                "llama" => {
                    forward_llama_cached(runtime, embedding.as_ref(), false, &mut kv_cache_opt)
                }
                _ => break,
            };
            let next_token = sample_constrained(
                logits.as_deref(),
                config,
                tokenizer,
                &token_ids[prompt_len..],
                generated,
            );
            if next_token == draft_token {
                token_ids.push(draft_token);
                generated += 1;
                accepted += 1;
                if tokenizer.vocab_size() > 0 && draft_token as usize >= tokenizer.vocab_size() {
                    break;
                }
            } else {
                token_ids.push(next_token);
                generated += 1;
                break;
            }
        }

        // Update the n-gram index with newly accepted tokens.
        ngram_index = update_ngram_index(&ngram_index, &token_ids, n);

        if let Some(truncated) =
            check_stop_sequences(tokenizer, &config.stop, &mut token_ids, prompt_len)
        {
            token_ids = truncated;
            break;
        }

        if let Some(max_ctx) = config.max_context {
            if token_ids.len() > max_ctx {
                context_shift(
                    &mut token_ids,
                    &mut prompt_len,
                    &mut kv_cache_opt,
                    max_ctx,
                    config.anchor_tokens,
                );
            }
        }

        // Stop if we've hit max tokens or an out-of-vocab token was emitted.
        if generated >= config.max_tokens {
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
    (
        tokenizer.decode(&token_ids[prompt_len.min(token_ids.len())..]),
        final_kv,
    )
}

/// Build an n-gram index from a token sequence.
/// Maps each n-gram (as a Vec<u32>) to a list of tokens that historically followed it.
pub(super) fn build_ngram_index(tokens: &[u32], n: usize) -> HashMap<Vec<u32>, Vec<u32>> {
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
pub(super) fn update_ngram_index(
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
pub(super) fn draft_tokens(
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
pub(super) fn most_frequent(xs: &[u32]) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ngram_index_maps_trigrams_to_continuations() {
        // Tokens 1,2,3,1,2,3 → trigram (1,2,3) is followed by 1; (2,3,1) by 2; (3,1,2) by 3.
        let index = build_ngram_index(&[1, 2, 3, 1, 2, 3], 3);
        assert_eq!(index.get(&vec![1, 2, 3]), Some(&vec![1]));
        assert_eq!(index.get(&vec![2, 3, 1]), Some(&vec![2]));
        assert_eq!(index.get(&vec![3, 1, 2]), Some(&vec![3]));
    }

    #[test]
    fn build_ngram_index_empty_when_too_short() {
        // Fewer than n+1 tokens can't form a complete window.
        assert!(build_ngram_index(&[1, 2, 3], 3).is_empty());
        assert!(build_ngram_index(&[], 3).is_empty());
    }

    #[test]
    fn update_ngram_index_adds_only_newest_window() {
        let index = build_ngram_index(&[1, 2, 3, 1], 3);
        // Appending 2 creates the new tail window (2,3,1)→2.
        let updated = update_ngram_index(&index, &[1, 2, 3, 1, 2], 3);
        assert_eq!(updated.get(&vec![2, 3, 1]), Some(&vec![2]));
        // Older entries untouched.
        assert_eq!(updated.get(&vec![1, 2, 3]), Some(&vec![1]));
    }

    #[test]
    fn update_ngram_index_noop_on_short_tokens() {
        let index = build_ngram_index(&[1, 2, 3, 1], 3);
        let updated = update_ngram_index(&index, &[9, 9], 3);
        assert_eq!(updated, index);
    }

    #[test]
    fn draft_tokens_follows_continuation_chain() {
        // Repeated "a b c" pattern: index maps trigrams so drafting chains.
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let index = build_ngram_index(&tokens, 3);
        let draft = draft_tokens(&index, &tokens, 3, 4);
        // Last trigram is (1,2,3) → most frequent continuation is 1, then (2,3,1)→2, …
        assert_eq!(draft, vec![1, 2, 3, 1]);
    }

    #[test]
    fn draft_tokens_empty_when_no_continuation() {
        let index = build_ngram_index(&[1, 2, 3, 1], 3);
        // Tail trigram (4,5,6) never seen → no draft.
        assert!(draft_tokens(&index, &[4, 5, 6], 3, 3).is_empty());
    }

    #[test]
    fn most_frequent_picks_majority() {
        assert_eq!(most_frequent(&[7, 7, 3, 7, 3]), 7);
        assert_eq!(most_frequent(&[]), 0);
    }
}
