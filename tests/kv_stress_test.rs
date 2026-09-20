//! Stress tests and edge cases for KV catch capabilities.
//!
//! These tests ensure the error handling system is robust under extreme conditions
//! and handles edge cases correctly.

use modelc::kv_error::{CacheValidationError, KvContext, KvError};
use modelc::prefix_cache::{CachedPrefix, PrefixCache};
use modelc::runtime::transformer::{KvCache, KvLayer};

fn create_test_kv_cache(n_layers: usize, hidden: usize, n_positions: usize) -> KvCache {
    let mut kv = KvCache::new(n_layers);
    for layer_idx in 0..n_layers {
        let mut layer = KvLayer::new_fp32();
        for _ in 0..n_positions {
            let k: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
            let v: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.02).collect();
            layer.append(&k, &v);
        }
        kv.layers[layer_idx] = Some(layer);
    }
    kv
}

// ===== Stress Tests =====

#[test]
fn test_cache_handles_massive_insertion_load() {
    let mut cache = PrefixCache::new(1000);
    let successful_inserts = 50;

    for i in 0..successful_inserts {
        let tokens = vec![i as u32; 10];
        let result = cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 10),
                last_logits: vec![],
            },
        );
        assert!(result.is_ok(), "Insert {} should succeed", i);
    }

    assert_eq!(cache.len(), successful_inserts);

    let stats = cache.stats();
    assert_eq!(stats.entries, successful_inserts);
    assert_eq!(stats.total_tokens, successful_inserts * 10);
}

#[test]
fn test_cache_rapid_insertion_and_removal() {
    let mut cache = PrefixCache::new(100);
    let operations = 1000;

    for i in 0..operations {
        let tokens = vec![i as u32 % 50; 5]; // Cycle through 50 different keys

        if i % 3 == 0 {
            // Remove operation
            cache.remove(&tokens);
        } else {
            // Insert operation
            let _ = cache.insert(
                tokens.clone(),
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 5),
                    last_logits: vec![],
                },
            );
        }
    }

    // Cache should still be functional
    assert!(cache.len() <= 100);

    let lookup = cache.lookup(&[0, 0, 0, 0, 0]);
    assert!(lookup.is_ok() || matches!(lookup, Err(KvError::EmptyCache(_))));
}

#[test]
fn test_cache_under_constant_eviction_pressure() {
    let capacity = 10;
    let mut cache = PrefixCache::new(capacity);
    let iterations = 100;

    for i in 0..iterations {
        let tokens = vec![i as u32; 3];
        let result = cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 3),
                last_logits: vec![],
            },
        );

        // Some inserts may fail due to capacity, but that's OK
        if result.is_ok() {
            assert!(cache.len() <= capacity);
        }
    }

    // Final state should be consistent
    assert!(cache.len() <= capacity);
    let stats = cache.stats();
    assert!(stats.entries <= capacity);
}

#[test]
fn test_cache_concurrent_access_simulation() {
    let mut cache = PrefixCache::new(50);
    let threads = 10;
    let operations_per_thread = 100;

    for thread in 0..threads {
        for op in 0..operations_per_thread {
            let unique_id = (thread * operations_per_thread + op) as u32;
            let tokens = vec![unique_id; 5];

            // Simulate different operations
            match op % 4 {
                0 => {
                    // Insert
                    let _ = cache.insert(
                        tokens.clone(),
                        CachedPrefix {
                            kv: create_test_kv_cache(2, 128, 5),
                            last_logits: vec![],
                        },
                    );
                }
                1 => {
                    // Lookup
                    let _ = cache.lookup(&tokens);
                }
                2 => {
                    // Remove
                    cache.remove(&tokens);
                }
                3 => {
                    // Stats
                    let _ = cache.stats();
                }
                _ => unreachable!(),
            }
        }
    }

    // Cache should still be consistent
    assert!(cache.len() <= 50);
    let stats = cache.stats();
    assert!(stats.utilization <= 1.0);
}

// ===== Edge Case Tests =====

#[test]
fn test_cache_with_zero_capacity() {
    let mut cache = PrefixCache::new(0);

    // Should still function with minimum capacity of 1
    assert_eq!(cache.capacity(), 1);

    let tokens = vec![1_u32];
    let result = cache.insert(
        tokens,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 1),
            last_logits: vec![],
        },
    );

    // Insert should succeed since capacity is normalized to 1
    assert!(result.is_ok());
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_cache_with_very_large_capacity() {
    let large_capacity = 100_000;
    let mut cache = PrefixCache::new(large_capacity);

    assert_eq!(cache.capacity(), large_capacity);

    // Insert a reasonable number of entries
    for i in 0..100 {
        let tokens = vec![i as u32; 2];
        assert!(
            cache
                .insert(
                    tokens,
                    CachedPrefix {
                        kv: create_test_kv_cache(2, 128, 2),
                        last_logits: vec![],
                    },
                )
                .is_ok()
        );
    }

    assert_eq!(cache.len(), 100);
}

#[test]
fn test_cache_with_single_token_sequences() {
    let mut cache = PrefixCache::new(10);

    for i in 0..10 {
        let tokens = vec![i as u32];
        let result = cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 1),
                last_logits: vec![],
            },
        );
        assert!(result.is_ok());
    }

    assert_eq!(cache.len(), 10);

    // All single-token lookups should work
    for i in 0..10 {
        let lookup = cache.lookup(&[i as u32]);
        assert!(lookup.is_ok());
        assert_eq!(lookup.unwrap().matched_len, 1);
    }
}

#[test]
fn test_cache_with_very_long_token_sequences() {
    let mut cache = PrefixCache::new(5);

    let long_sequence: Vec<u32> = (0..100).collect();

    // This should fail due to capacity
    let result = cache.insert(
        long_sequence.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 100),
            last_logits: vec![],
        },
    );

    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(CacheValidationError::SequenceTooLong { .. })
    ));
}

#[test]
fn test_cache_with_duplicate_entries() {
    let mut cache = PrefixCache::new(10);

    let tokens = vec![1, 2, 3];

    // Insert same entry multiple times
    for _ in 0..5 {
        let result = cache.insert(
            tokens.clone(),
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 3),
                last_logits: vec![],
            },
        );
        assert!(result.is_ok());
    }

    // Should only have one entry
    assert_eq!(cache.len(), 1);

    // Lookup should work
    let lookup = cache.lookup(&tokens);
    assert!(lookup.is_ok());
}

#[test]
fn test_cache_with_very_large_token_ids() {
    let mut cache = PrefixCache::new(10);

    // Try to insert with unreasonably large token IDs
    let huge_tokens = vec![999_999_999, 1_000_000_000];
    let result = cache.insert(
        huge_tokens,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    );

    // Should fail due to invalid token IDs
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(CacheValidationError::InvalidTokenId { .. })
    ));
}

#[test]
fn test_cache_with_empty_kv_layers() {
    let mut cache = PrefixCache::new(10);

    let tokens = vec![1, 2, 3];
    let kv = KvCache::new(2);
    // Don't populate layers - leave them empty

    let result = cache.insert(
        tokens,
        CachedPrefix {
            kv,
            last_logits: vec![],
        },
    );

    // Empty KV layers should be OK
    assert!(result.is_ok());
}

#[test]
fn test_cache_with_mixed_layer_counts() {
    let mut cache = PrefixCache::new(10);

    // Entry with 2 layers
    let tokens1 = vec![1, 2];
    assert!(
        cache
            .insert(
                tokens1.clone(),
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 2),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    // Entry with 4 layers
    let tokens2 = vec![3, 4];
    assert!(
        cache
            .insert(
                tokens2.clone(),
                CachedPrefix {
                    kv: create_test_kv_cache(4, 128, 2),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    assert_eq!(cache.len(), 2);

    // Both should be accessible
    assert!(cache.lookup(&tokens1).is_ok());
    assert!(cache.lookup(&tokens2).is_ok());
}

// ===== Error Recovery Tests =====

#[test]
fn test_cache_recovers_from_failed_lookups() {
    let mut cache = PrefixCache::new(10);

    // Failed lookup on empty cache
    let result = cache.lookup(&[1, 2, 3]);
    assert!(matches!(result, Err(KvError::EmptyCache(_))));

    // Insert valid data
    assert!(
        cache
            .insert(
                vec![1, 2, 3],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 3),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    // Same lookup should now succeed
    let result = cache.lookup(&[1, 2, 3]);
    assert!(result.is_ok());
}

#[test]
fn test_cache_handles_interleaved_success_and_failure() {
    let mut cache = PrefixCache::new(10);

    // Valid insert
    assert!(
        cache
            .insert(
                vec![1, 2],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 2),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    // Invalid insert
    assert!(
        cache
            .insert(
                vec![],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 0),
                    last_logits: vec![],
                },
            )
            .is_err()
    );

    // Valid insert
    assert!(
        cache
            .insert(
                vec![3, 4],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 2),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    // Invalid insert
    assert!(
        cache
            .insert(
                vec![5; 20],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 20),
                    last_logits: vec![],
                },
            )
            .is_err()
    );

    // Should have exactly 2 valid entries
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_cache_statistics_under_error_conditions() {
    let mut cache = PrefixCache::new(10);

    // Initial stats
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.total_tokens, 0);
    assert_eq!(stats.utilization, 0.0);

    // Failed insert
    assert!(
        cache
            .insert(
                vec![],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 0),
                    last_logits: vec![],
                },
            )
            .is_err()
    );

    // Stats should remain unchanged
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.total_tokens, 0);

    // Successful insert
    assert!(
        cache
            .insert(
                vec![1, 2, 3],
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 3),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    // Stats should reflect the change
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.total_tokens, 3);
}

// ===== Performance and Memory Tests =====

#[test]
fn test_cache_lookup_performance_under_load() {
    let mut cache = PrefixCache::new(1000);

    // Populate cache
    for i in 0..100 {
        let tokens = vec![i as u32; 10];
        assert!(
            cache
                .insert(
                    tokens,
                    CachedPrefix {
                        kv: create_test_kv_cache(2, 128, 10),
                        last_logits: vec![],
                    },
                )
                .is_ok()
        );
    }

    // Perform many lookups
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let tokens = vec![(i % 100) as u32; 10];
        let _ = cache.lookup(&tokens);
    }
    let duration = start.elapsed();

    // Should complete reasonably quickly (< 1 second for 1000 lookups)
    assert!(
        duration.as_secs() < 1,
        "Cache lookup took too long: {:?}",
        duration
    );
}

#[test]
fn test_cache_memory_efficiency() {
    let mut cache = PrefixCache::new(100);

    // Add entries with different sizes
    for i in 0..50 {
        let tokens = vec![i as u32; (i % 10 + 1) as usize]; // Variable length
        assert!(
            cache
                .insert(
                    tokens,
                    CachedPrefix {
                        kv: create_test_kv_cache(2, 128, (i % 10 + 1) as usize),
                        last_logits: vec![],
                    },
                )
                .is_ok()
        );
    }

    let stats = cache.stats();

    // Verify memory calculations are reasonable
    assert!(stats.total_kv_bytes > 0);
    assert!(stats.total_kv_bytes < 100_000_000); // Should be less than 100MB

    // Verify utilization
    assert!(stats.utilization > 0.0);
    assert!(stats.utilization <= 0.5); // 50/100 = 0.5
}

// ===== Boundary Condition Tests =====

#[test]
fn test_cache_at_capacity_boundaries() {
    let capacity = 5;
    let mut cache = PrefixCache::new(capacity);

    // Fill exactly to capacity
    for i in 0..capacity {
        let tokens = vec![i as u32; 2];
        assert!(
            cache
                .insert(
                    tokens,
                    CachedPrefix {
                        kv: create_test_kv_cache(2, 128, 2),
                        last_logits: vec![],
                    },
                )
                .is_ok()
        );
    }

    assert_eq!(cache.len(), capacity);

    // Try to insert one more - should succeed and evict LRU
    let tokens = vec![99_u32; 2];
    assert!(
        cache
            .insert(
                tokens,
                CachedPrefix {
                    kv: create_test_kv_cache(2, 128, 2),
                    last_logits: vec![],
                },
            )
            .is_ok()
    );

    assert_eq!(cache.len(), capacity);
}

#[test]
fn test_cache_with_maximum_valid_token_ids() {
    let mut cache = PrefixCache::new(10);

    // Use maximum reasonable token ID (just under the validation threshold)
    let max_valid_token = 999_999;
    let tokens = vec![max_valid_token];

    let result = cache.insert(
        tokens,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 1),
            last_logits: vec![],
        },
    );

    // Maximum valid token should be accepted
    assert!(result.is_ok());
}

#[test]
fn test_cache_dimension_boundary_conditions() {
    let mut cache = PrefixCache::new(10);

    // Test exact dimension match
    let tokens = vec![1, 2, 3, 4, 5];
    let result = cache.insert(
        tokens.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 5), // Exact match
            last_logits: vec![],
        },
    );
    assert!(result.is_ok());

    // Test dimension slightly less than tokens (should be OK)
    let tokens2 = vec![6, 7, 8];
    let result2 = cache.insert(
        tokens2,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2), // 2 vs 3 tokens
            last_logits: vec![],
        },
    );
    assert!(result2.is_ok());
}

// ===== Error Message Quality Tests =====

#[test]
fn test_error_messages_are_informative() {
    let errors = vec![
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
        CacheValidationError::EmptySequence.into(),
        CacheValidationError::SequenceTooLong {
            length: 1000,
            max_length: 100,
        }
        .into(),
    ];

    for error in errors {
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
}

#[test]
fn test_context_enriched_errors() {
    let ctx = KvContext::new("test_operation")
        .with_layer(5)
        .with_position(10)
        .with_sequence_length(100);

    let err = ctx.dimension_error(256, 128);
    let err_str = err.to_string();

    assert!(err_str.contains("test_operation"));
    assert!(err_str.contains("256"));
    assert!(err_str.contains("128"));
}
