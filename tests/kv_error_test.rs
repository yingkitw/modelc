//! Comprehensive tests for KV error handling.
//!
//! This test suite covers all error types, validation logic, and error recovery
//! scenarios for the KV cache system.

use modelc::kv_error::{CacheValidationError, KvContext, KvError, KvResult, ValidationResult};

// ===== KVError Tests =====

#[test]
fn test_kv_error_invalid_dimensions_formatting() {
    let ctx = KvContext::new("test_operation")
        .with_layer(2)
        .with_position(5);

    let err = ctx.dimension_error(128, 256);

    assert!(matches!(err, KvError::InvalidDimensions { .. }));

    let err_str = err.to_string();
    assert!(err_str.contains("Invalid dimensions"));
    assert!(err_str.contains("test_operation"));
    assert!(err_str.contains("128"));
    assert!(err_str.contains("256"));
}

#[test]
fn test_kv_error_capacity_exceeded() {
    let err = KvError::CapacityExceeded {
        requested: 50,
        capacity: 32,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("capacity exceeded"));
    assert!(err_str.contains("50"));
    assert!(err_str.contains("32"));
}

#[test]
fn test_kv_error_invalid_cache_state() {
    let err = KvError::InvalidCacheState("corrupted layer data".to_string());

    let err_str = err.to_string();
    assert!(err_str.contains("Invalid cache state"));
    assert!(err_str.contains("corrupted layer data"));
}

#[test]
fn test_kv_error_empty_cache() {
    let err = KvError::EmptyCache("lookup".to_string());

    let err_str = err.to_string();
    assert!(err_str.contains("Cannot perform 'lookup' on empty cache"));
}

#[test]
fn test_kv_error_invalid_token_sequence_without_position() {
    let err = KvError::InvalidTokenSequence {
        reason: "contains invalid UTF-8".to_string(),
        position: None,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Invalid token sequence"));
    assert!(err_str.contains("contains invalid UTF-8"));
}

#[test]
fn test_kv_error_invalid_token_sequence_with_position() {
    let err = KvError::InvalidTokenSequence {
        reason: "token out of vocabulary".to_string(),
        position: Some(42),
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Invalid token sequence at position 42"));
    assert!(err_str.contains("token out of vocabulary"));
}

#[test]
fn test_kv_error_quantization_error() {
    let err = KvError::QuantizationError("scale factor too small".to_string());

    let err_str = err.to_string();
    assert!(err_str.contains("Quantization error"));
    assert!(err_str.contains("scale factor too small"));
}

#[test]
fn test_kv_error_layer_out_of_bounds() {
    let err = KvError::LayerOutOfBounds {
        index: 24,
        total_layers: 12,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Layer index 24 out of bounds"));
    assert!(err_str.contains("total layers: 12"));
}

#[test]
fn test_kv_error_position_out_of_bounds() {
    let err = KvError::PositionOutOfBounds {
        index: 1000,
        total_positions: 512,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Position index 1000 out of bounds"));
    assert!(err_str.contains("total positions: 512"));
}

#[test]
fn test_kv_error_mixed_precision_error() {
    let err =
        KvError::MixedPrecisionError("cannot mix INT8 and FP32 in same operation".to_string());

    let err_str = err.to_string();
    assert!(err_str.contains("Mixed precision operation error"));
    assert!(err_str.contains("cannot mix INT8 and FP32"));
}

#[test]
fn test_kv_context_builders() {
    let ctx = KvContext::new("test")
        .with_layer(3)
        .with_position(7)
        .with_sequence_length(100);

    assert_eq!(ctx.operation, "test");
    assert_eq!(ctx.layer, Some(3));
    assert_eq!(ctx.position, Some(7));
    assert_eq!(ctx.sequence_length, Some(100));
}

#[test]
fn test_kv_context_layer_out_of_bounds() {
    let ctx = KvContext::new("test").with_layer(5);
    let err = ctx.out_of_bounds_error(10, 8, true);

    assert!(matches!(
        err,
        KvError::LayerOutOfBounds {
            index: 10,
            total_layers: 8
        }
    ));
}

#[test]
fn test_kv_context_position_out_of_bounds() {
    let ctx = KvContext::new("test").with_position(15);
    let err = ctx.out_of_bounds_error(1000, 512, false);

    assert!(matches!(
        err,
        KvError::PositionOutOfBounds {
            index: 1000,
            total_positions: 512
        }
    ));
}

// ===== CacheValidationError Tests =====

#[test]
fn test_cache_validation_error_empty_sequence() {
    let err = CacheValidationError::EmptySequence;

    let err_str = err.to_string();
    assert!(err_str.contains("Token sequence cannot be empty"));
}

#[test]
fn test_cache_validation_error_sequence_too_long() {
    let err = CacheValidationError::SequenceTooLong {
        length: 1000,
        max_length: 128,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Token sequence too long"));
    assert!(err_str.contains("1000"));
    assert!(err_str.contains("max 128"));
}

#[test]
fn test_cache_validation_error_invalid_token_id() {
    let err = CacheValidationError::InvalidTokenId {
        token_id: 999999,
        position: 50,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Invalid token ID 999999 at position 50"));
}

#[test]
fn test_cache_validation_error_dimension_mismatch() {
    let err = CacheValidationError::DimensionMismatch {
        layer: 2,
        expected: 64,
        actual: 128,
    };

    let err_str = err.to_string();
    assert!(err_str.contains("Dimension mismatch in layer 2"));
    assert!(err_str.contains("expected 64"));
    assert!(err_str.contains("got 128"));
}

// ===== Error Conversion Tests =====

#[test]
fn test_from_empty_sequence_to_kv_error() {
    let validation_err = CacheValidationError::EmptySequence;
    let kv_err: KvError = validation_err.into();

    assert!(matches!(kv_err, KvError::InvalidTokenSequence { 
        reason, 
        position: None 
    } if reason.contains("Empty")));
}

#[test]
fn test_from_sequence_too_long_to_kv_error() {
    let validation_err = CacheValidationError::SequenceTooLong {
        length: 500,
        max_length: 100,
    };
    let kv_err: KvError = validation_err.into();

    assert!(matches!(kv_err, KvError::InvalidTokenSequence { 
        reason, 
        position: None 
    } if reason.contains("500") && reason.contains("100")));
}

#[test]
fn test_from_invalid_token_id_to_kv_error() {
    let validation_err = CacheValidationError::InvalidTokenId {
        token_id: 88888,
        position: 25,
    };
    let kv_err: KvError = validation_err.into();

    assert!(matches!(kv_err, KvError::InvalidTokenSequence { 
        reason, 
        position: Some(25) 
    } if reason.contains("88888")));
}

#[test]
fn test_from_dimension_mismatch_to_kv_error() {
    let validation_err = CacheValidationError::DimensionMismatch {
        layer: 3,
        expected: 512,
        actual: 256,
    };
    let kv_err: KvError = validation_err.into();

    assert!(matches!(kv_err, KvError::InvalidDimensions { 
        expected: 512, 
        actual: 256, 
        operation 
    } if operation.contains("layer 3")));
}

// ===== KvResult Type Tests =====

#[test]
fn test_kv_result_ok() {
    let result: KvResult<i32> = Ok(42);
    assert_eq!(result, Ok(42));
}

#[test]
fn test_kv_result_err() {
    let result: KvResult<i32> = Err(KvError::EmptyCache("test".to_string()));
    assert!(result.is_err());
}

#[test]
fn test_kv_result_with_context() {
    let ctx = KvContext::new("test_operation").with_layer(1);
    let err = ctx.dimension_error(64, 32);

    assert!(matches!(err, KvError::InvalidDimensions { .. }));
}

// ===== Validation Result Tests =====

#[test]
fn test_validation_result_ok() {
    let result: ValidationResult<()> = Ok(());
    assert!(result.is_ok());
}

#[test]
fn test_validation_result_err() {
    let result: ValidationResult<()> = Err(CacheValidationError::EmptySequence);
    assert!(result.is_err());
}

#[test]
fn test_validation_result_with_different_errors() {
    let results: Vec<ValidationResult<()>> = vec![
        Err(CacheValidationError::EmptySequence),
        Err(CacheValidationError::SequenceTooLong {
            length: 200,
            max_length: 100,
        }),
        Err(CacheValidationError::InvalidTokenId {
            token_id: 12345,
            position: 10,
        }),
        Err(CacheValidationError::DimensionMismatch {
            layer: 0,
            expected: 64,
            actual: 32,
        }),
    ];

    assert_eq!(results.len(), 4);
    for result in results {
        assert!(result.is_err());
    }
}

// ===== Error Propagation Tests =====

#[test]
fn test_error_propagation_chain() {
    fn validate_layer(layer_idx: usize, total: usize) -> KvResult<()> {
        if layer_idx >= total {
            let ctx = KvContext::new("layer_access").with_layer(layer_idx);
            return Err(ctx.out_of_bounds_error(layer_idx, total, true));
        }
        Ok(())
    }

    fn validate_token(id: u32, vocab_size: usize) -> KvResult<()> {
        if (id as usize) >= vocab_size {
            return Err(KvError::InvalidTokenSequence {
                reason: format!("token {} exceeds vocab size {}", id, vocab_size),
                position: Some(0),
            });
        }
        Ok(())
    }

    // Valid case
    assert!(validate_layer(5, 10).is_ok());
    assert!(validate_token(100, 1000).is_ok());

    // Invalid cases
    assert!(validate_layer(15, 10).is_err());
    assert!(validate_token(2000, 1000).is_err());
}

#[test]
fn test_error_recovery_patterns() {
    fn attempt_operation(attempts: usize) -> KvResult<String> {
        let mut last_error = None;

        for i in 0..attempts {
            let ctx = KvContext::new("retry_operation").with_position(i);

            if i < 3 {
                last_error = Some(ctx.dimension_error(64, 32));
                continue;
            }

            return Ok(format!("succeeded on attempt {}", i));
        }

        Err(last_error.unwrap_or_else(|| KvError::EmptyCache("all attempts failed".to_string())))
    }

    // Should fail with 3 attempts
    assert!(attempt_operation(3).is_err());

    // Should succeed with 4 attempts
    let result = attempt_operation(4);
    assert!(result.is_ok());
    assert!(result.unwrap().contains("succeeded on attempt 3"));
}

#[test]
fn test_error_context_enrichment() {
    fn create_enriched_error(
        operation: &str,
        layer: usize,
        expected: usize,
        actual: usize,
    ) -> KvError {
        KvContext::new(operation)
            .with_layer(layer)
            .with_sequence_length(actual)
            .dimension_error(expected, actual)
    }

    let err = create_enriched_error("matrix_multiply", 2, 512, 256);

    assert!(matches!(err, KvError::InvalidDimensions { .. }));
    let err_str = err.to_string();
    assert!(err_str.contains("matrix_multiply"));
    assert!(err_str.contains("512"));
    assert!(err_str.contains("256"));
}

#[test]
fn test_error_display_comprehensive() {
    let errors = vec![
        KvError::InvalidDimensions {
            expected: 128,
            actual: 256,
            operation: "conv2d".to_string(),
        },
        KvError::CapacityExceeded {
            requested: 100,
            capacity: 50,
        },
        KvError::InvalidCacheState("corrupted header".to_string()),
        KvError::EmptyCache("read".to_string()),
        KvError::InvalidTokenSequence {
            reason: "invalid BPE merge".to_string(),
            position: Some(123),
        },
        KvError::QuantizationError("overflow during quantization".to_string()),
        KvError::LayerOutOfBounds {
            index: 50,
            total_layers: 24,
        },
        KvError::PositionOutOfBounds {
            index: 2048,
            total_positions: 1024,
        },
        KvError::MixedPrecisionError("FP16 not supported in this context".to_string()),
    ];

    for err in errors {
        let display = err.to_string();
        assert!(!display.is_empty());
        assert!(display.len() > 10); // Ensure reasonable error message length
    }
}

#[test]
fn test_multiple_error_conversions() {
    let validation_errors = vec![
        CacheValidationError::EmptySequence,
        CacheValidationError::SequenceTooLong {
            length: 999,
            max_length: 100,
        },
        CacheValidationError::InvalidTokenId {
            token_id: 77777,
            position: 1,
        },
        CacheValidationError::DimensionMismatch {
            layer: 1,
            expected: 768,
            actual: 384,
        },
    ];

    let kv_errors: Vec<KvError> = validation_errors.into_iter().map(Into::into).collect();

    assert_eq!(kv_errors.len(), 4);

    // Verify each conversion was successful
    assert!(matches!(kv_errors[0], KvError::InvalidTokenSequence { .. }));
    assert!(matches!(kv_errors[1], KvError::InvalidTokenSequence { .. }));
    assert!(matches!(kv_errors[2], KvError::InvalidTokenSequence { .. }));
    assert!(matches!(kv_errors[3], KvError::InvalidDimensions { .. }));
}

#[test]
fn test_error_in_nested_operations() {
    fn inner_operation(value: usize) -> ValidationResult<usize> {
        if value == 0 {
            return Err(CacheValidationError::EmptySequence);
        }
        if value > 100 {
            return Err(CacheValidationError::SequenceTooLong {
                length: value,
                max_length: 100,
            });
        }
        Ok(value * 2)
    }

    fn outer_operation(input: Vec<usize>) -> KvResult<Vec<usize>> {
        let mut results = Vec::new();
        for (idx, value) in input.iter().enumerate() {
            match inner_operation(*value) {
                Ok(result) => results.push(result),
                Err(_e) => {
                    let ctx = KvContext::new("outer_operation").with_position(idx);
                    return Err(ctx.out_of_bounds_error(*value, 100, false));
                }
            }
        }
        Ok(results)
    }

    // Valid input
    assert!(outer_operation(vec![1, 2, 3]).is_ok());

    // Empty sequence error
    assert!(outer_operation(vec![0]).is_err());

    // Sequence too long error
    assert!(outer_operation(vec![150]).is_err());
}

#[test]
fn test_error_messages_are_informative() {
    let kv_errors = vec![
        KvError::InvalidDimensions {
            expected: 128,
            actual: 256,
            operation: "convolution".to_string(),
        },
        KvError::EmptyCache("lookup".to_string()),
        KvError::InvalidTokenSequence {
            reason: "out of vocabulary".to_string(),
            position: Some(42),
        },
    ];

    for error in kv_errors {
        let error_str = error.to_string();

        // All error messages should be reasonably informative
        assert!(!error_str.is_empty());
        assert!(error_str.len() > 10);
        assert!(!error_str.contains("Error(")); // Not using default Debug formatting

        // Should contain relevant information
        assert!(
            error_str
                .chars()
                .all(|c| c.is_ascii() || c.is_alphanumeric())
        );
    }

    let validation_errors = vec![
        CacheValidationError::EmptySequence,
        CacheValidationError::SequenceTooLong {
            length: 1000,
            max_length: 100,
        },
    ];

    for error in validation_errors {
        let error_str = error.to_string();

        // All error messages should be reasonably informative
        assert!(!error_str.is_empty());
        assert!(error_str.len() > 10);
        assert!(!error_str.contains("Error(")); // Not using default Debug formatting
    }
}
