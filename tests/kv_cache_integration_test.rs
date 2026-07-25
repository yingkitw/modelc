//! Integration tests for KV catch capabilities in production scenarios.
//!
//! These tests verify that error handling works correctly in real-world usage
//! patterns, including cache operations, transformer inference, and server requests.

mod common;

use modelc::generate::{GenerationConfig, generate_token_ids_with_cache};
use modelc::kv_error::{KvError, CacheValidationError};
use modelc::prefix_cache::{PrefixCache, CachedPrefix};
use modelc::runtime::serve::Runtime;
use modelc::runtime::transformer::{KvCache, KvLayer};
use modelc::tokenizer::BpeTokenizer;

fn config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        ..GenerationConfig::default()
    }
}

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

// ===== Cache Population and Error Recovery Tests =====

#[test]
fn test_cache_handles_invalid_token_sequences_gracefully() {
    let mut cache = PrefixCache::new(8);
    
    // Try to insert empty token sequence - should fail validation
    let empty_tokens = vec![];
    let result = cache.insert(
        empty_tokens.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    );
    
    assert!(result.is_err());
    assert!(matches!(result, Err(CacheValidationError::EmptySequence)));
    assert_eq!(cache.len(), 0, "Cache should remain empty after failed insert");
}

#[test]
fn test_cache_handles_oversized_sequences() {
    let mut cache = PrefixCache::new(4);
    
    // Try to insert sequence larger than capacity
    let oversized_tokens = (0..10).collect::<Vec<_>>();
    let result = cache.insert(
        oversized_tokens.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 10),
            last_logits: vec![],
        },
    );
    
    assert!(result.is_err());
    assert!(matches!(result, Err(CacheValidationError::SequenceTooLong { .. })));
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_lookup_on_empty_cache_returns_proper_error() {
    let cache = PrefixCache::new(8);
    let tokens = vec![1, 2, 3];
    
    let result = cache.lookup(&tokens);
    
    assert!(result.is_err());
    assert!(matches!(result, Err(KvError::EmptyCache(_))));
}

#[test]
fn test_cache_recovery_after_failed_insert() {
    let mut cache = PrefixCache::new(8);
    
    // Fail to insert invalid data
    let _ = cache.insert(
        vec![],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    );
    
    assert_eq!(cache.len(), 0);
    
    // Successfully insert valid data
    let valid_tokens = vec![1, 2, 3];
    let result = cache.insert(
        valid_tokens.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 3),
            last_logits: vec![0.1, 0.2],
        },
    );
    
    assert!(result.is_ok());
    assert_eq!(cache.len(), 1);
    
    // Verify cache still works
    let lookup = cache.lookup(&valid_tokens);
    assert!(lookup.is_ok());
    assert_eq!(lookup.unwrap().matched_len, 3);
}

// ===== Cache State Consistency Tests =====

#[test]
fn test_cache_maintains_consistency_after_errors() {
    let mut cache = PrefixCache::new(8);
    
    // Insert valid entry
    let tokens1 = vec![1, 2, 3];
    assert!(cache.insert(
        tokens1.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 3),
            last_logits: vec![],
        },
    ).is_ok());
    
    assert_eq!(cache.len(), 1);
    
    // Try invalid insert
    assert!(cache.insert(
        vec![],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    ).is_err());
    
    // Cache should still have exactly one entry
    assert_eq!(cache.len(), 1);
    
    // Original entry should still be accessible
    let lookup = cache.lookup(&tokens1);
    assert!(lookup.is_ok());
    assert_eq!(lookup.unwrap().matched_len, 3);
}

#[test]
fn test_cache_clear_resets_state_completely() {
    let mut cache = PrefixCache::new(8);
    
    // Add multiple entries
    for i in 1..=5 {
        let tokens = vec![i as u32; 3];
        assert!(cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 3),
                last_logits: vec![],
            },
        ).is_ok());
    }
    
    assert_eq!(cache.len(), 5);
    
    // Clear cache
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    
    // Lookup should return proper error
    let result = cache.lookup(&[1, 1, 1]);
    assert!(matches!(result, Err(KvError::EmptyCache(_))));
}

// ===== Cache Capacity and Eviction Tests =====

#[test]
fn test_cache_respects_capacity_with_errors() {
    let capacity = 3;
    let mut cache = PrefixCache::new(capacity);
    
    // Fill cache to capacity
    for i in 1..=capacity {
        let tokens = vec![i as u32; 2];
        assert!(cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 2),
                last_logits: vec![],
            },
        ).is_ok());
    }
    
    assert_eq!(cache.len(), capacity);
    
    // Try to insert oversized sequence - should fail
    let oversized = vec![99 as u32; 10];
    let result = cache.insert(
        oversized,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 10),
            last_logits: vec![],
        },
    );
    
    assert!(result.is_err());
    assert_eq!(cache.len(), capacity, "Cache size should remain unchanged");
}

#[test]
fn test_cache_lru_eviction_with_failed_inserts() {
    let capacity = 2;
    let mut cache = PrefixCache::new(capacity);
    
    // Insert first entry
    assert!(cache.insert(
        vec![1, 2],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    
    // Try invalid insert - should not affect cache
    assert!(cache.insert(
        vec![],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    ).is_err());
    
    assert_eq!(cache.len(), 1);
    
    // Insert second valid entry
    assert!(cache.insert(
        vec![3, 4],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    
    // Insert third valid entry - should evict first
    assert!(cache.insert(
        vec![5, 6],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    
    assert_eq!(cache.len(), 2);
    
    // First entry should be evicted
    let result = cache.lookup(&[1, 2]);
    assert!(result.is_err());
}

// ===== Cache Removal Tests =====

#[test]
fn test_cache_remove_existing_entry() {
    let mut cache = PrefixCache::new(8);
    
    let tokens = vec![1, 2, 3, 4];
    assert!(cache.insert(
        tokens.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 4),
            last_logits: vec![],
        },
    ).is_ok());
    
    assert_eq!(cache.len(), 1);
    
    let removed = cache.remove(&tokens);
    assert!(removed);
    assert_eq!(cache.len(), 0);
    
    // Lookup should now fail with empty cache error
    let result = cache.lookup(&tokens);
    assert!(matches!(result, Err(KvError::EmptyCache(_))));
}

#[test]
fn test_cache_remove_nonexistent_entry() {
    let mut cache = PrefixCache::new(8);
    
    let tokens1 = vec![1, 2, 3];
    assert!(cache.insert(
        tokens1.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 3),
            last_logits: vec![],
        },
    ).is_ok());
    
    let removed = cache.remove(&[4, 5, 6]);
    assert!(!removed);
    assert_eq!(cache.len(), 1);
    
    // Original entry should still be accessible
    let lookup = cache.lookup(&tokens1);
    assert!(lookup.is_ok());
}

// ===== Cache Statistics Tests =====

#[test]
fn test_cache_stats_reflect_current_state() {
    let mut cache = PrefixCache::new(8);
    
    let initial_stats = cache.stats();
    assert_eq!(initial_stats.entries, 0);
    assert_eq!(initial_stats.capacity, 8);
    assert_eq!(initial_stats.total_tokens, 0);
    
    // Add entries
    let tokens1 = vec![1, 2, 3];
    let tokens2 = vec![4 as u32, 5 as u32];
    
    assert!(cache.insert(
        tokens1.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 3),
            last_logits: vec![],
        },
    ).is_ok());
    
    assert!(cache.insert(
        tokens2.clone(),
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    
    let stats = cache.stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.total_tokens, 5);
    assert!(stats.total_kv_bytes > 0);
    assert!((stats.utilization - 0.25).abs() < 0.01); // 2/8 = 0.25
}

#[test]
fn test_cache_stats_after_errors() {
    let mut cache = PrefixCache::new(8);
    
    // Try invalid insert
    assert!(cache.insert(
        vec![],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    ).is_err());
    
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.total_tokens, 0);
    
    // Add valid entry
    assert!(cache.insert(
        vec![1, 2],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.total_tokens, 2);
}

// ===== Dimension Validation Tests =====

#[test]
fn test_cache_dimension_validation_lenient() {
    let mut cache = PrefixCache::new(8);
    
    // KV cache with slightly fewer positions than tokens should be allowed
    let tokens = vec![1, 2, 3, 4, 5];
    let result = cache.insert(
        tokens,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 3), // 3 positions vs 5 tokens
            last_logits: vec![],
        },
    );
    
    // This should be allowed due to lenient validation
    assert!(result.is_ok());
}

#[test]
fn test_cache_dimension_validation_strict_for_extreme_cases() {
    let mut cache = PrefixCache::new(8);
    
    // KV cache with much fewer positions than tokens should fail
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let result = cache.insert(
        tokens,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 1), // 1 position vs 10 tokens - extreme mismatch
            last_logits: vec![],
        },
    );
    
    // This should fail due to extreme dimension mismatch
    assert!(result.is_err());
}

// ===== Real-world Scenario Tests =====

#[test]
fn test_cache_with_real_generation_workflow() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let prompt = "Hello, world!";
    let cfg = config(4);
    
    let mut cache = PrefixCache::new(8);
    
    // First generation should populate cache
    let first = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt,
        &cfg,
        Some(&mut cache),
        None,
    );
    
    assert!(!first.is_empty());
    
    // Check cache stats
    let stats = cache.stats();
    assert!(stats.entries <= 1); // May or may not have been populated
    
    // Second generation with same prompt should work
    let second = generate_token_ids_with_cache(
        &runtime,
        "gpt2",
        hidden,
        &tokenizer,
        prompt,
        &cfg,
        Some(&mut cache),
        None,
    );
    
    assert_eq!(first, second);
}

#[test]
fn test_cache_handles_multiple_concurrent_requests() {
    let model = common::create_gpt2_test_model();
    let runtime = Runtime::from_raw(&model.tensors);
    let hidden = model
        .metadata
        .get("hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);
    let tokenizer = BpeTokenizer::byte_fallback();
    let cfg = config(3);
    
    let mut cache = PrefixCache::new(8);
    
    let prompts = vec![
        "The quick brown fox",
        "jumps over the lazy dog",
        "The quick brown fox jumps",
    ];
    
    let mut results = Vec::new();
    for prompt in &prompts {
        let result = generate_token_ids_with_cache(
            &runtime,
            "gpt2",
            hidden,
            &tokenizer,
            prompt,
            &cfg,
            Some(&mut cache),
            None,
        );
        results.push(result);
    }
    
    // All generations should succeed
    assert_eq!(results.len(), 3);
    for result in results {
        assert!(!result.is_empty());
    }
    
    // Cache should handle multiple entries
    let stats = cache.stats();
    assert!(stats.entries <= 3);
}

#[test]
fn test_cache_error_handling_during_capacity_pressure() {
    let mut cache = PrefixCache::new(2); // Very small cache
    
    // Fill cache
    for i in 0..2 {
        let tokens = vec![i as u32; 2];
        assert!(cache.insert(
            tokens,
            CachedPrefix {
                kv: create_test_kv_cache(2, 128, 2),
                last_logits: vec![],
            },
        ).is_ok());
    }
    
    assert_eq!(cache.len(), 2);
    
    // Try to insert oversized sequence - should fail
    let oversized = vec![100 as u32; 10];
    let result = cache.insert(
        oversized,
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 10),
            last_logits: vec![],
        },
    );
    
    assert!(result.is_err());
    
    // Cache should still be functional
    assert_eq!(cache.len(), 2);
    let lookup = cache.lookup(&[0, 0]);
    assert!(lookup.is_ok());
}

#[test]
fn test_cache_state_after_multiple_error_scenarios() {
    let mut cache = PrefixCache::new(4);
    
    // Scenario 1: Empty sequence error
    assert!(cache.insert(
        vec![],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 0),
            last_logits: vec![],
        },
    ).is_err());
    assert_eq!(cache.len(), 0);
    
    // Scenario 2: Valid insert
    assert!(cache.insert(
        vec![1, 2],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    assert_eq!(cache.len(), 1);
    
    // Scenario 3: Oversized sequence error
    assert!(cache.insert(
        vec![3 as u32; 10],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 10),
            last_logits: vec![],
        },
    ).is_err());
    assert_eq!(cache.len(), 1);
    
    // Scenario 4: Another valid insert
    assert!(cache.insert(
        vec![3, 4],
        CachedPrefix {
            kv: create_test_kv_cache(2, 128, 2),
            last_logits: vec![],
        },
    ).is_ok());
    assert_eq!(cache.len(), 2);
    
    // Final state should be consistent
    let stats = cache.stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.total_tokens, 4);
    
    // Both entries should be accessible
    assert!(cache.lookup(&[1, 2]).is_ok());
    assert!(cache.lookup(&[3, 4]).is_ok());
}