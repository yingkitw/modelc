//! Prompt-prefix KV cache for autoregressive generation.
//!
//! Many requests share a common prefix (system prompt, tool descriptions, few-shot
//! examples). Replaying that prefix through the transformer on every request wastes
//! work. [`PrefixCache`] stores the `KvCache` produced after processing a token
//! sequence so a later request whose prompt *starts with* that sequence can restore
//! the cached K/V and only process the divergent suffix.
//!
//! Lookup performs longest-prefix matching: among cached entries whose key is a
//! prefix of the query, the longest wins. Entries are bounded by an LRU policy.

use crate::runtime::transformer::KvCache;

/// Cached K/V state plus the logits produced by the last token of the cached sequence.
/// The logits let an exact-match lookup skip recomputation of the final prompt step.
#[derive(Clone)]
pub struct CachedPrefix {
    /// K/V vectors for every layer, after processing the cached token sequence.
    pub kv: KvCache,
    /// Output logits of the last token in the cached sequence. Empty if the model
    /// produced no output head at insertion time.
    pub last_logits: Vec<f32>,
}

/// Result of a [`PrefixCache::lookup`] — the cloned cache state plus how much of the
/// query was matched.
#[derive(Clone)]
pub struct CacheLookup {
    /// Number of leading query tokens covered by the matched cached entry.
    pub matched_len: usize,
    /// Cloned K/V state for the matched prefix.
    pub kv: KvCache,
    /// Present only on an exact full-match (`matched_len == query.len()`), so the
    /// caller can reuse the stored logits instead of reprocessing the last token.
    pub last_logits: Option<Vec<f32>>,
}

/// Bounded LRU cache of prompt-prefix `KvCache` snapshots.
///
/// `entries` is ordered most-recently-used first; the tail is evicted when the cache
/// exceeds `max_entries`.
pub struct PrefixCache {
    entries: Vec<(Vec<u32>, CachedPrefix)>,
    max_entries: usize,
}

impl PrefixCache {
    /// New cache holding at most `max_entries` prefixes.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
        }
    }

    /// Find the longest cached token sequence that is a prefix of `tokens`.
    /// On a tie, the most-recently-inserted entry wins.
    pub fn lookup(&self, tokens: &[u32]) -> Option<CacheLookup> {
        let mut best: Option<(&Vec<u32>, &CachedPrefix)> = None;
        for (seq, cached) in &self.entries {
            if tokens.starts_with(seq)
                && best.is_none_or(|(b, _)| seq.len() > b.len())
            {
                best = Some((seq, cached));
            }
        }

        let (seq, cached) = best?;
        let matched_len = seq.len();
        let last_logits = (matched_len == tokens.len() && !cached.last_logits.is_empty())
            .then(|| cached.last_logits.clone());
        Some(CacheLookup {
            matched_len,
            kv: cached.kv.clone(),
            last_logits,
        })
    }

    /// Insert (or refresh) a cached prefix. Moves the entry to the front (MRU) and
    /// evicts the least-recently-used entry when over capacity.
    pub fn insert(&mut self, tokens: Vec<u32>, cached: CachedPrefix) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == tokens) {
            self.entries.remove(pos);
        }
        self.entries.insert(0, (tokens, cached));
        while self.entries.len() > self.max_entries {
            let last = self.entries.len() - 1;
            self.entries.remove(last);
        }
    }

    /// Number of cached prefixes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no prefixes.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.max_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::transformer::{KvCache, KvLayer};

    fn dummy_cache(n_positions: usize) -> KvCache {
        let mut kv = KvCache::new(1);
        let mut layer = KvLayer::new_fp32();
        for _ in 0..n_positions {
            layer.append(&[1.0, 2.0], &[3.0, 4.0]);
        }
        kv.layers[0] = Some(layer);
        kv
    }

    #[test]
    fn lookup_miss_returns_none() {
        let pc = PrefixCache::new(4);
        assert!(pc.lookup(&[1, 2, 3]).is_none());
        assert!(pc.is_empty());
    }

    #[test]
    fn exact_match_returns_logits_and_skips_reprocess() {
        let mut pc = PrefixCache::new(4);
        let tokens = vec![1, 2, 3];
        pc.insert(
            tokens.clone(),
            CachedPrefix {
                kv: dummy_cache(3),
                last_logits: vec![0.5, 0.2],
            },
        );

        let look = pc.lookup(&tokens).expect("exact match");
        assert_eq!(look.matched_len, 3);
        assert_eq!(look.last_logits, Some(vec![0.5, 0.2]));
    }

    #[test]
    fn prefix_match_returns_cloned_kv_without_logits() {
        let mut pc = PrefixCache::new(4);
        pc.insert(
            vec![1, 2],
            CachedPrefix {
                kv: dummy_cache(2),
                last_logits: vec![0.9],
            },
        );

        // Query extends the cached prefix — partial match, no logits reuse.
        let look = pc.lookup(&[1, 2, 3, 4]).expect("prefix match");
        assert_eq!(look.matched_len, 2);
        assert!(look.last_logits.is_none(), "partial match must not expose logits");
        // Cloned KV reflects the cached prefix length (2 positions, hidden=2 → 4 floats).
        let layer = look.kv.layers[0].as_ref().unwrap();
        assert_eq!(layer.len(), 2);
    }

    #[test]
    fn longest_prefix_wins() {
        let mut pc = PrefixCache::new(4);
        pc.insert(
            vec![1],
            CachedPrefix { kv: dummy_cache(1), last_logits: vec![] },
        );
        pc.insert(
            vec![1, 2, 3],
            CachedPrefix { kv: dummy_cache(3), last_logits: vec![] },
        );

        let look = pc.lookup(&[1, 2, 3, 4]).expect("match");
        assert_eq!(look.matched_len, 3, "longer cached prefix should win");
    }

    #[test]
    fn non_prefix_entry_is_ignored() {
        let mut pc = PrefixCache::new(4);
        pc.insert(
            vec![9, 9],
            CachedPrefix { kv: dummy_cache(2), last_logits: vec![] },
        );
        assert!(pc.lookup(&[1, 2, 9, 9]).is_none(), "entry must be a *prefix* of the query");
    }

    #[test]
    fn insert_refreshes_existing_key() {
        let mut pc = PrefixCache::new(4);
        pc.insert(
            vec![1, 2],
            CachedPrefix { kv: dummy_cache(2), last_logits: vec![0.1] },
        );
        pc.insert(
            vec![1, 2],
            CachedPrefix { kv: dummy_cache(2), last_logits: vec![0.9] },
        );
        assert_eq!(pc.len(), 1, "refresh should not duplicate");
        let look = pc.lookup(&[1, 2]).unwrap();
        assert_eq!(look.last_logits, Some(vec![0.9]), "refresh should overwrite");
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let mut pc = PrefixCache::new(2);
        pc.insert(vec![1], CachedPrefix { kv: dummy_cache(1), last_logits: vec![] });
        pc.insert(vec![2], CachedPrefix { kv: dummy_cache(1), last_logits: vec![] });
        pc.insert(vec![3], CachedPrefix { kv: dummy_cache(1), last_logits: vec![] });
        assert_eq!(pc.len(), 2);
        // [1] was evicted as LRU; [2] and [3] remain.
        assert!(pc.lookup(&[1, 2]).is_none());
        assert!(pc.lookup(&[2]).is_some());
        assert!(pc.lookup(&[3]).is_some());
    }

    #[test]
    fn capacity_minimum_is_one() {
        let pc = PrefixCache::new(0);
        assert_eq!(pc.capacity(), 1);
    }
}
