//! Neural draft model for speculative decoding (EAGLE-style).
//!
//! A draft model is a small, fast network that proposes candidate tokens
//! before the large target model verifies them.  When the draft model is
//! accurate, accepted tokens skip the expensive transformer forward pass,
//! yielding 2–3× speedup.
//!
//! The `MlpDraftModel` is a minimal 2-layer MLP that re-uses the target
//! model's embedding weights and adds tiny projection layers.  It can be
//! loaded from a separate set of tensors (e.g. `draft.fc1.weight`) or
//! constructed on-the-fly.  If no draft weights are present, speculative
//! generation falls back to the n-gram draft model.

use crate::generate::{argmax, multinomial, softmax};
use crate::runtime::serve::Runtime;

/// A fast draft model that proposes candidate tokens for speculative decoding.
pub trait DraftModel: Send + Sync {
    /// Propose up to `gamma` draft tokens given the current token context.
    fn draft(&self, context: &[u32], gamma: usize) -> Vec<u32>;
}

/// A tiny 2-layer MLP draft model.
///
/// Forward pass: embedding → FC1 + ReLU → FC2 → logits → sample.
/// Only two matmuls + one elementwise activation, so it is orders of
/// magnitude faster than a full transformer forward.
pub struct MlpDraftModel {
    vocab_size: usize,
    hidden: usize,
    draft_hidden: usize,
    /// vocab_size × hidden  (may borrow from target model embedding)
    embed: Vec<f32>,
    /// draft_hidden × hidden
    w1: Vec<f32>,
    /// draft_hidden
    b1: Vec<f32>,
    /// vocab_size × draft_hidden
    w2: Vec<f32>,
    /// vocab_size
    b2: Vec<f32>,
    temperature: f32,
    top_p: f32,
}

impl MlpDraftModel {
    /// Try to construct a draft model from tensors in the runtime.
    ///
    /// Looks for `draft.fc1.weight`, `draft.fc1.bias`, `draft.fc2.weight`,
    /// `draft.fc2.bias`, and optionally `draft.embed.weight`.  If the embedding
    /// is missing, it borrows the target model's embedding layer.
    pub fn from_runtime(
        runtime: &Runtime,
        vocab_size: usize,
        hidden: usize,
        draft_hidden: usize,
        temperature: f32,
        top_p: f32,
    ) -> Option<Self> {
        let embed = runtime
            .get("draft.embed.weight")
            .map(|t| t.data.clone())
            .or_else(|| {
                runtime
                    .get("transformer.wte.weight")
                    .or_else(|| runtime.get("model.embed_tokens.weight"))
                    .map(|t| t.data.clone())
            })?;

        let w1 = runtime.get("draft.fc1.weight").map(|t| t.data.clone())?;
        let b1 = runtime.get("draft.fc1.bias").map(|t| t.data.clone())?;
        let w2 = runtime.get("draft.fc2.weight").map(|t| t.data.clone())?;
        let b2 = runtime.get("draft.fc2.bias").map(|t| t.data.clone())?;

        Some(Self {
            vocab_size,
            hidden,
            draft_hidden,
            embed,
            w1,
            b1,
            w2,
            b2,
            temperature,
            top_p,
        })
    }

    /// Run a single forward step for `token_id`, returning logits.
    #[allow(clippy::needless_range_loop)]
    fn forward(&self, token_id: u32) -> Vec<f32> {
        let idx = (token_id as usize) * self.hidden;
        let embed = if idx + self.hidden <= self.embed.len() {
            &self.embed[idx..idx + self.hidden]
        } else {
            return vec![0.0f32; self.vocab_size];
        };

        // FC1
        let mut hidden = vec![0.0f32; self.draft_hidden];
        for i in 0..self.draft_hidden {
            let mut sum = self.b1[i];
            let row_start = i * self.hidden;
            for j in 0..self.hidden {
                sum += self.w1[row_start + j] * embed[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // FC2
        let mut logits = vec![0.0f32; self.vocab_size];
        for i in 0..self.vocab_size {
            let mut sum = self.b2[i];
            let row_start = i * self.draft_hidden;
            for j in 0..self.draft_hidden {
                sum += self.w2[row_start + j] * hidden[j];
            }
            logits[i] = sum;
        }

        logits
    }
}

impl DraftModel for MlpDraftModel {
    fn draft(&self, context: &[u32], gamma: usize) -> Vec<u32> {
        let mut draft = Vec::with_capacity(gamma);
        let mut current = context.last().copied().unwrap_or(0);

        for _ in 0..gamma {
            let logits = self.forward(current);
            let next = if self.temperature <= 0.0 {
                argmax(&logits) as u32
            } else {
                let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
                let mut probs = softmax(&scaled);
                if self.top_p > 0.0 && self.top_p < 1.0 {
                    crate::generate::apply_top_p(&mut probs, self.top_p);
                }
                multinomial(&probs)
            };
            draft.push(next);
            current = next;
        }

        draft
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_draft_model_from_runtime_requires_all_weights() {
        let runtime = Runtime::from_raw(&std::collections::HashMap::new());
        let model = MlpDraftModel::from_runtime(&runtime, 10, 4, 2, 0.0, 0.0);
        assert!(model.is_none(), "empty runtime should not yield draft model");
    }

    #[test]
    fn mlp_draft_model_produces_deterministic_tokens_when_greedy() {
        fn f32s(xs: &[f32]) -> Vec<u8> {
            xs.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        let mut tensors = std::collections::HashMap::new();
        // embedding: 4 vocab × 2 hidden
        tensors.insert(
            "transformer.wte.weight".to_string(),
            crate::model::TensorData {
                data: f32s(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                shape: vec![4, 2],
                dtype: crate::model::DataType::F32,
            },
        );
        // w1: 2 draft_hidden × 2 hidden
        tensors.insert(
            "draft.fc1.weight".to_string(),
            crate::model::TensorData {
                data: f32s(&[1.0, 0.0, 0.0, 1.0]),
                shape: vec![2, 2],
                dtype: crate::model::DataType::F32,
            },
        );
        // b1: 2
        tensors.insert(
            "draft.fc1.bias".to_string(),
            crate::model::TensorData {
                data: f32s(&[0.0, 0.0]),
                shape: vec![2],
                dtype: crate::model::DataType::F32,
            },
        );
        // w2: 4 vocab × 2 draft_hidden
        tensors.insert(
            "draft.fc2.weight".to_string(),
            crate::model::TensorData {
                data: f32s(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                shape: vec![4, 2],
                dtype: crate::model::DataType::F32,
            },
        );
        // b2: 4
        tensors.insert(
            "draft.fc2.bias".to_string(),
            crate::model::TensorData {
                data: f32s(&[0.0, 0.0, 0.0, 0.0]),
                shape: vec![4],
                dtype: crate::model::DataType::F32,
            },
        );

        let runtime = Runtime::from_raw(&tensors);
        let model = MlpDraftModel::from_runtime(&runtime, 4, 2, 2, 0.0, 0.0)
            .expect("should construct with all weights present");

        let draft = model.draft(&[0], 3);
        assert_eq!(draft.len(), 3, "should produce exactly gamma tokens");
        // With the weight matrix above, token 2 always has the highest logit.
        assert_eq!(draft, vec![2, 2, 2]);
    }
}
