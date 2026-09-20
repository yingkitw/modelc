//! Error types and handling for KV cache operations.
//!
//! This module provides robust error handling for key-value cache operations,
//! including validation, error recovery, and detailed error reporting for
//! prefix caching and transformer KV cache operations.

use std::fmt;

/// KV cache operation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum KvError {
    /// Invalid cache dimensions or shape mismatch.
    InvalidDimensions {
        expected: usize,
        actual: usize,
        operation: String,
    },
    /// Cache capacity exceeded.
    CapacityExceeded { requested: usize, capacity: usize },
    /// Invalid cache state or corruption.
    InvalidCacheState(String),
    /// Operation on empty cache.
    EmptyCache(String),
    /// Token sequence validation failed.
    InvalidTokenSequence {
        reason: String,
        position: Option<usize>,
    },
    /// Quantization error.
    QuantizationError(String),
    /// Layer index out of bounds.
    LayerOutOfBounds { index: usize, total_layers: usize },
    /// Position index out of bounds.
    PositionOutOfBounds {
        index: usize,
        total_positions: usize,
    },
    /// Mixed precision operation error.
    MixedPrecisionError(String),
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KvError::InvalidDimensions {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Invalid dimensions for '{}': expected {}, got {}",
                    operation, expected, actual
                )
            }
            KvError::CapacityExceeded {
                requested,
                capacity,
            } => {
                write!(
                    f,
                    "Cache capacity exceeded: requested {}, capacity {}",
                    requested, capacity
                )
            }
            KvError::InvalidCacheState(msg) => {
                write!(f, "Invalid cache state: {}", msg)
            }
            KvError::EmptyCache(operation) => {
                write!(f, "Cannot perform '{}' on empty cache", operation)
            }
            KvError::InvalidTokenSequence { reason, position } => {
                if let Some(pos) = position {
                    write!(f, "Invalid token sequence at position {}: {}", pos, reason)
                } else {
                    write!(f, "Invalid token sequence: {}", reason)
                }
            }
            KvError::QuantizationError(msg) => {
                write!(f, "Quantization error: {}", msg)
            }
            KvError::LayerOutOfBounds {
                index,
                total_layers,
            } => {
                write!(
                    f,
                    "Layer index {} out of bounds (total layers: {})",
                    index, total_layers
                )
            }
            KvError::PositionOutOfBounds {
                index,
                total_positions,
            } => {
                write!(
                    f,
                    "Position index {} out of bounds (total positions: {})",
                    index, total_positions
                )
            }
            KvError::MixedPrecisionError(msg) => {
                write!(f, "Mixed precision operation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for KvError {}

impl From<CacheValidationError> for KvError {
    fn from(err: CacheValidationError) -> Self {
        match err {
            CacheValidationError::EmptySequence => KvError::InvalidTokenSequence {
                reason: "Empty token sequence".to_string(),
                position: None,
            },
            CacheValidationError::SequenceTooLong { length, max_length } => {
                KvError::InvalidTokenSequence {
                    reason: format!("Sequence too long: {} (max {})", length, max_length),
                    position: None,
                }
            }
            CacheValidationError::InvalidTokenId { token_id, position } => {
                KvError::InvalidTokenSequence {
                    reason: format!("Invalid token ID {}", token_id),
                    position: Some(position),
                }
            }
            CacheValidationError::DimensionMismatch {
                layer,
                expected,
                actual,
            } => KvError::InvalidDimensions {
                expected,
                actual,
                operation: format!("layer {} KV cache", layer),
            },
        }
    }
}

/// Result type for KV cache operations.
pub type KvResult<T> = Result<T, KvError>;

/// Cache validation errors for prefix cache operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CacheValidationError {
    /// Token sequence is empty.
    EmptySequence,
    /// Token sequence too long for cache.
    SequenceTooLong { length: usize, max_length: usize },
    /// Invalid token ID.
    InvalidTokenId { token_id: u32, position: usize },
    /// KV state dimensions don't match.
    DimensionMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for CacheValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheValidationError::EmptySequence => {
                write!(f, "Token sequence cannot be empty")
            }
            CacheValidationError::SequenceTooLong { length, max_length } => {
                write!(
                    f,
                    "Token sequence too long: {} (max {})",
                    length, max_length
                )
            }
            CacheValidationError::InvalidTokenId { token_id, position } => {
                write!(f, "Invalid token ID {} at position {}", token_id, position)
            }
            CacheValidationError::DimensionMismatch {
                layer,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Dimension mismatch in layer {}: expected {}, got {}",
                    layer, expected, actual
                )
            }
        }
    }
}

impl std::error::Error for CacheValidationError {}

/// Result type for cache validation operations.
pub type ValidationResult<T> = Result<T, CacheValidationError>;

/// KV cache operation context for better error reporting.
#[derive(Debug, Clone)]
pub struct KvContext {
    pub operation: String,
    pub layer: Option<usize>,
    pub position: Option<usize>,
    pub sequence_length: Option<usize>,
}

impl KvContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            layer: None,
            position: None,
            sequence_length: None,
        }
    }

    pub fn with_layer(mut self, layer: usize) -> Self {
        self.layer = Some(layer);
        self
    }

    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    pub fn with_sequence_length(mut self, length: usize) -> Self {
        self.sequence_length = Some(length);
        self
    }

    pub fn dimension_error(&self, expected: usize, actual: usize) -> KvError {
        KvError::InvalidDimensions {
            expected,
            actual,
            operation: self.operation.clone(),
        }
    }

    pub fn out_of_bounds_error(&self, index: usize, total: usize, is_layer: bool) -> KvError {
        if is_layer {
            KvError::LayerOutOfBounds {
                index,
                total_layers: total,
            }
        } else {
            KvError::PositionOutOfBounds {
                index,
                total_positions: total,
            }
        }
    }
}
