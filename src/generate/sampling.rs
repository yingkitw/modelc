//! Sampling helpers for autoregressive generation: penalties, nucleus/min-p
//! truncation, `argmax`/`softmax`/`multinomial`, and the deterministic
//! `MiniRng` used for seeded sampling.

use super::GenerationConfig;
use crate::tokenizer::BpeTokenizer;

/// Apply repetition penalty to logits based on tokens already generated.
/// Tokens present in `generated_tokens` have their logits adjusted:
/// - positive logits are divided by `penalty`
/// - negative logits are multiplied by `penalty`
///
/// This pushes repeated-token probabilities down when penalty > 1.0.
pub(super) fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
    if penalty <= 1.0 || generated_tokens.is_empty() {
        return;
    }
    // Track which token IDs have appeared so we only penalize each once.
    let mut seen = [0u64; 1024]; // bitmap for IDs 0..65535 (covers most vocabs)
    for &id in generated_tokens {
        let idx = id as usize / 64;
        let bit = id as u64 % 64;
        if idx < seen.len() {
            seen[idx] |= 1u64 << bit;
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        let idx = i / 64;
        let bit = i as u64 % 64;
        if idx < seen.len() && (seen[idx] & (1u64 << bit)) != 0 {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Apply OpenAI-style presence and frequency penalties to logits.
///
/// * `presence_penalty` is added once to any token already seen in `generated_tokens`.
/// * `frequency_penalty` is subtracted `count * penalty` for each seen token.
///
/// Both are additive adjustments on logits (not multiplicative like repetition_penalty).
pub(super) fn apply_presence_frequency_penalty(
    logits: &mut [f32],
    generated_tokens: &[u32],
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    if (presence_penalty == 0.0 && frequency_penalty == 0.0) || generated_tokens.is_empty() {
        return;
    }
    // Count occurrences of each token ID. Most vocabs are < 65536 entries.
    let mut counts = [0u16; 65536];
    for &id in generated_tokens {
        let idx = id as usize;
        if idx < counts.len() {
            // Cap at u16::MAX to avoid overflow on very long sequences.
            counts[idx] = counts[idx].saturating_add(1);
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        let count = counts.get(i).copied().unwrap_or(0) as f32;
        if count > 0.0 {
            *logit -= presence_penalty;
            *logit -= count * frequency_penalty;
        }
    }
}

pub(super) fn sample(logits: Option<&[f32]>, config: &GenerationConfig, step: usize) -> u32 {
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
    if config.min_p > 0.0 && config.min_p < 1.0 {
        apply_min_p(&mut probs, config.min_p);
    }

    multinomial(&probs, step, config.seed)
}

/// Sample with optional repetition / presence / frequency penalties and grammar
/// constraint applied to the logits.
///
/// If `config.constraint` is set, only tokens whose decoded text keeps the partial
/// output compatible with the regex are allowed. Invalid tokens are masked to
/// `-inf` before sampling. Penalties (`repetition_penalty`, `presence_penalty`,
/// `frequency_penalty`) adjust the logits of tokens already present in
/// `generated_tokens` before the constraint mask is applied.
///
/// When neither penalties nor a constraint are active, this samples directly
/// without allocating a working copy (fast path).
pub(super) fn sample_constrained(
    logits: Option<&[f32]>,
    config: &GenerationConfig,
    tokenizer: &BpeTokenizer,
    generated_tokens: &[u32],
    step: usize,
) -> u32 {
    let logits = logits.unwrap_or(&[]);
    if logits.is_empty() {
        return 0;
    }

    let has_penalty = config.repetition_penalty > 1.0
        || config.presence_penalty != 0.0
        || config.frequency_penalty != 0.0;
    let has_bias = !config.logit_bias.is_empty();
    let constraint = config.constraint.as_deref();

    // Fast path: nothing to adjust — sample directly over the original logits.
    if !has_penalty && !has_bias && constraint.is_none() {
        return sample(Some(logits), config, step);
    }

    let mut working = logits.to_vec();
    if has_penalty {
        apply_repetition_penalty(&mut working, generated_tokens, config.repetition_penalty);
        apply_presence_frequency_penalty(
            &mut working,
            generated_tokens,
            config.presence_penalty,
            config.frequency_penalty,
        );
    }

    if has_bias {
        for (&token_id, &bias) in &config.logit_bias {
            let idx = token_id as usize;
            if idx < working.len() {
                working[idx] += bias;
            }
        }
    }

    if let Some(constraint) = constraint {
        let prefix = tokenizer.decode(generated_tokens);
        let vocab_size = tokenizer.vocab_size();
        let mask = constraint.valid_mask(&prefix, vocab_size, tokenizer.cached_token_bytes());
        for (i, valid) in mask.iter().enumerate() {
            if !valid && i < working.len() {
                working[i] = f32::NEG_INFINITY;
            }
        }
    }

    sample(Some(&working), config, step)
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

/// In-place min-p filtering.
///
/// Keeps tokens whose probability is at least `min_p` times the maximum
/// probability, zeros the rest, then renormalizes. Simpler and often more
/// effective than top-p (nucleus) sampling: the threshold scales with the
/// model's confidence, so it adapts per step.
pub(crate) fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = max * min_p;
    for p in probs.iter_mut() {
        if *p < threshold {
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

pub(crate) fn multinomial(probs: &[f32], step: usize, seed: Option<u64>) -> u32 {
    let r: f32 = if let Some(s) = seed {
        let mut rng = MiniRng::from_seed(s.wrapping_add(step as u64));
        rng.gen_f32()
    } else {
        let mut rng = MiniRng::new();
        rng.gen_f32()
    };
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    probs.len().saturating_sub(1) as u32
}

/// Minimal splitmix64 PRNG — replaces the `rand` dependency for sampling.
///
/// Not cryptographic. Plenty of statistical quality for token sampling.
pub(super) struct MiniRng(u64);

impl MiniRng {
    pub(super) fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdead_beef_cafe_babe);
        Self(Self::mix(seed))
    }

    pub(super) fn from_seed(seed: u64) -> Self {
        Self(Self::mix(seed))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        Self::mix(self.0)
    }

    /// Returns a value in `[0.0, 1.0)`.
    pub(super) fn gen_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    pub(super) fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod mini_rng_tests {
    use super::MiniRng;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = MiniRng::from_seed(42);
        let mut b = MiniRng::from_seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = MiniRng::from_seed(1);
        let mut b = MiniRng::from_seed(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f32_in_unit_range() {
        let mut rng = MiniRng::from_seed(123);
        for _ in 0..10_000 {
            let v = rng.gen_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
