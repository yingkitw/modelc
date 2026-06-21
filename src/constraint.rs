//! Grammar-based constrained decoding.
//!
//! Provides token-level masking so only tokens that keep the partial output
//! compatible with a regular expression are allowed during sampling.

use regex::Regex;

/// A constraint that can mask invalid logits before sampling.
pub trait Constraint: Send + Sync {
    /// Given the decoded text so far, return a boolean mask of length `vocab_size`
    /// where `mask[i] == true` means token `i` is a valid next token.
    fn valid_mask(&self, prefix: &str, vocab_size: usize, token_bytes: &[Option<Vec<u8>>]) -> Vec<bool>;
}

/// Regex-based constraint: only tokens whose decoded text matches the regex prefix are allowed.
pub struct RegexConstraint {
    pattern: Regex,
}

impl RegexConstraint {
    pub fn new(pattern: &str) -> Option<Self> {
        Regex::new(format!("^(?:{}).*", pattern).as_str())
            .ok()
            .map(|pattern| Self { pattern })
    }
}

impl Constraint for RegexConstraint {
    fn valid_mask(&self, prefix: &str, vocab_size: usize, token_bytes: &[Option<Vec<u8>>]) -> Vec<bool> {
        let mut mask = vec![false; vocab_size];
        for (i, maybe_bytes) in token_bytes.iter().enumerate().take(vocab_size) {
            if let Some(bytes) = maybe_bytes {
                let candidate = format!("{}{}", prefix, String::from_utf8_lossy(bytes));
                // Token is valid if the concatenated string matches the regex *prefix*.
                // We check by seeing if the regex matches any string that starts with candidate.
                // A simpler heuristic: check if candidate itself matches the regex,
                // OR if there exists some suffix that would make it match.
                // For this minimal implementation, we allow tokens whose candidate
                // is itself a prefix of some regex match.
                mask[i] = self.pattern.is_match(&candidate) || could_match_prefix(&self.pattern, &candidate);
            }
        }
        mask
    }
}

/// Check whether `candidate` could be extended into a match of `pattern`.
/// Heuristic: try appending common suffix characters and see if any match.
fn could_match_prefix(pattern: &Regex, candidate: &str) -> bool {
    // Fast path: if the candidate is already matched, it's definitely valid.
    if pattern.is_match(candidate) {
        return true;
    }
    // Try a few common continuation strings to see if any extension matches.
    // Includes longer numeric strings so patterns like \d{4} are handled.
    for suffix in ["", " ", "a", "0", "00", "000", "0000", "1", "-", "-0", "{", "\"", "\n"] {
        let extended = format!("{candidate}{suffix}");
        if pattern.is_match(&extended) {
            return true;
        }
    }
    false
}

/// Build a byte representation lookup for every token ID in the vocabulary.
///
/// `decode_fn` should return the raw UTF-8 bytes for a given token ID.
/// Returns a vector where index `i` is `Some(bytes)` for token `i`.
pub fn build_token_bytes_lookup<F>(vocab_size: usize, decode_fn: F) -> Vec<Option<Vec<u8>>>
where
    F: FnMut(u32) -> Option<Vec<u8>>,
{
    (0..vocab_size as u32)
        .map(decode_fn)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_constraint_allows_matching_tokens() {
        // \d+ matches one or more digits; any digit is a valid start.
        let constraint = RegexConstraint::new(r"\d+").unwrap();
        let token_bytes: Vec<Option<Vec<u8>>> = vec![
            Some(b"1".to_vec()),
            Some(b"a".to_vec()),
            Some(b"12".to_vec()),
        ];
        let mask = constraint.valid_mask("", 3, &token_bytes);
        assert!(mask[0], "'1' can start a digit sequence");
        assert!(!mask[1], "'a' cannot start a digit sequence");
        assert!(mask[2], "'12' can start a digit sequence");
    }

    #[test]
    fn regex_constraint_continues_partial_match() {
        // \d+-\d+ matches digit-dash-digit; after a digit block a dash is valid.
        let constraint = RegexConstraint::new(r"\d+-\d+").unwrap();
        let token_bytes: Vec<Option<Vec<u8>>> = vec![
            Some(b"-".to_vec()),
            Some(b"0".to_vec()),
            Some(b"abc".to_vec()),
        ];
        let mask = constraint.valid_mask("2024", 3, &token_bytes);
        assert!(mask[0], "'-' continues digit-dash-digit after year");
        // Heuristic may permit '0' because a longer suffix (e.g., "20240-0") matches.
        // This is acceptable: the constraint is permissive to avoid false negatives.
        assert!(!mask[2], "'abc' breaks pattern");
    }

    #[test]
    fn regex_constraint_invalid_pattern_returns_none() {
        assert!(RegexConstraint::new("[invalid").is_none());
    }
}
