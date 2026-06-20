//! Byte-level BPE tokenizer for GPT-2 style models.
//!
//! Supports encoding text into token IDs and decoding back, using a vocabulary
//! of byte tokens (0–255) plus merge-derived tokens. This is the foundation for
//! real text-in/text-out transformer inference.

use std::collections::HashMap;

/// A byte-level BPE tokenizer.
pub struct BpeTokenizer {
    /// token_id → bytes
    vocab: Vec<Vec<u8>>,
    /// (first_token_id, second_token_id) → merged_token_id, in priority order
    merges: Vec<(u32, u32, u32)>,
    /// Fast lookup for merge rules: (a, b) → merged_id
    merge_map: HashMap<(u32, u32), u32>,
    /// single-byte → token_id (for initial encoding)
    byte_to_id: HashMap<u8, u32>,
}

impl BpeTokenizer {
    /// Create a tokenizer from an explicit vocab and ordered merge list.
    ///
    /// `vocab[i]` is the byte sequence for token ID `i`.  
    /// `merges` is a list of `(first_id, second_id, merged_id)` in priority order.
    pub fn new(vocab: Vec<Vec<u8>>, merges: Vec<(u32, u32, u32)>) -> Self {
        let merge_map = merges
            .iter()
            .map(|(a, b, m)| ((*a, *b), *m))
            .collect();
        let mut byte_to_id = HashMap::new();
        for (id, seq) in vocab.iter().enumerate() {
            if seq.len() == 1 {
                byte_to_id.insert(seq[0], id as u32);
            }
        }
        Self {
            vocab,
            merges,
            merge_map,
            byte_to_id,
        }
    }

    /// A minimal tokenizer with only the 256 byte tokens (no merges).
    /// Useful as a fallback or for testing.
    pub fn byte_fallback() -> Self {
        let vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        Self::new(vocab, Vec::new())
    }

    /// Encode `text` into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        // Start with each byte mapped to its vocab token ID.
        let mut tokens: Vec<u32> = bytes
            .iter()
            .map(|b| self.byte_to_id.get(b).copied().unwrap_or(*b as u32))
            .collect();

        // Greedily apply the highest-priority merge that exists in the current sequence.
        // Repeat until no more merges are possible.
        loop {
            let mut best_idx = None;
            let mut best_rank: Option<usize> = None;

            for (i, window) in tokens.windows(2).enumerate() {
                let key = (window[0], window[1]);
                if let Some(&merged) = self.merge_map.get(&key) {
                    // Find the rank (priority) of this merge.
                    let rank = self.merges.iter().position(|(a, b, _)| *a == key.0 && *b == key.1);
                    if let Some(r) = rank
                        && best_rank.is_none_or(|br| r < br)
                    {
                        best_rank = Some(r);
                        best_idx = Some((i, merged));
                    }
                }
            }

            if let Some((idx, merged)) = best_idx {
                tokens.splice(idx..=idx + 1, [merged]);
            } else {
                break;
            }
        }

        tokens
    }

    /// Decode token IDs back into a string.
    ///
    /// Invalid token IDs are skipped (they have no entry in the vocab).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if let Some(seq) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(seq);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size (number of distinct tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Build a `BpeTokenizer` from GGUF tokenizer metadata.
    ///
    /// Each vocab entry is converted to bytes via UTF-8. Merge rules are parsed
    /// as "first second" strings; their IDs are looked up in the vocab. The
    /// merged token is the concatenation of the two byte sequences, and its ID
    /// is found in the vocab (standard BPE vocab contains all merged tokens).
    pub fn from_gguf(meta: &crate::parsers::gguf::GgufTokenizerMetadata) -> anyhow::Result<Self> {
        use anyhow::Context;

        if meta.vocab.is_empty() {
            anyhow::bail!("GGUF tokenizer metadata has no vocab");
        }

        let vocab: Vec<Vec<u8>> = meta
            .vocab
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();

        // Build token string → ID map for merge lookups.
        let mut token_to_id: std::collections::HashMap<Vec<u8>, u32> =
            std::collections::HashMap::new();
        for (id, seq) in vocab.iter().enumerate() {
            token_to_id.insert(seq.clone(), id as u32);
        }

        let mut merges = Vec::new();
        for rule in &meta.merges {
            let mut parts = rule.splitn(2, ' ');
            let first_str = parts.next().context("empty merge rule")?;
            let second_str = parts.next().context("merge rule missing second token")?;

            let first_bytes = first_str.as_bytes().to_vec();
            let second_bytes = second_str.as_bytes().to_vec();

            let first_id = *token_to_id
                .get(&first_bytes)
                .with_context(|| format!("merge rule first token not in vocab: {first_str:?}"))?;
            let second_id = *token_to_id.get(&second_bytes).with_context(|| {
                format!("merge rule second token not in vocab: {second_str:?}")
            })?;

            let merged_bytes: Vec<u8> = first_bytes.iter().chain(&second_bytes).copied().collect();
            let merged_id = *token_to_id
                .get(&merged_bytes)
                .with_context(|| format!("merged token not in vocab: {first_str:?}+{second_str:?}"))?;

            merges.push((first_id, second_id, merged_id));
        }

        Ok(Self::new(vocab, merges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_fallback_roundtrip() {
        let tok = BpeTokenizer::byte_fallback();
        let text = "hello world 123 !@#";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn empty_string_encodes_to_empty() {
        let tok = BpeTokenizer::byte_fallback();
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn bpe_merge_combines_tokens() {
        // Vocab: 0='a', 1='b', 2='c', 3="ab" (merged)
        let vocab = vec![
            vec![b'a'],
            vec![b'b'],
            vec![b'c'],
            vec![b'a', b'b'],
        ];
        // Merge 'a' (0) + 'b' (1) → "ab" (3)
        let merges = vec![(0, 1, 3)];
        let tok = BpeTokenizer::new(vocab, merges);

        let ids = tok.encode("ab");
        assert_eq!(ids, vec![3]);

        let ids2 = tok.encode("abc");
        assert_eq!(ids2, vec![3, 2]);
    }

    #[test]
    fn bpe_multiple_merges_priority() {
        // Vocab: 0='a', 1='b', 2='c', 3="ab", 4="bc", 5="abc"
        let vocab = vec![
            vec![b'a'],
            vec![b'b'],
            vec![b'c'],
            vec![b'a', b'b'],
            vec![b'b', b'c'],
            vec![b'a', b'b', b'c'],
        ];
        // Priority: "ab" first, then "bc", then "abc"
        let merges = vec![(0, 1, 3), (1, 2, 4), (3, 2, 5)];
        let tok = BpeTokenizer::new(vocab, merges);

        let ids = tok.encode("abc");
        // "ab" merges first (priority 0) → [3, 2], then "abc" → 5 merges again.
        assert_eq!(ids, vec![5]);
    }

    #[test]
    fn invalid_token_id_skipped() {
        let tok = BpeTokenizer::byte_fallback();
        let decoded = tok.decode(&[0, 9999, 1]); // 9999 is out of range
        assert_eq!(decoded, "\x00\x01");
    }

    #[test]
    fn from_gguf_builds_tokenizer() {
        let meta = crate::parsers::gguf::GgufTokenizerMetadata {
            model: "gpt2".to_string(),
            vocab: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merges: vec!["a b".to_string()],
            bos_token_id: None,
            eos_token_id: None,
            chat_template: None,
        };
        let tok = BpeTokenizer::from_gguf(&meta).expect("from_gguf should succeed");
        assert_eq!(tok.vocab_size(), 3);
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![2]); // "ab" merged to token 2
    }

    #[test]
    fn from_gguf_fails_on_missing_vocab() {
        let meta = crate::parsers::gguf::GgufTokenizerMetadata::default();
        assert!(BpeTokenizer::from_gguf(&meta).is_err());
    }
}
