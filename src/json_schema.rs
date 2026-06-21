//! Minimal JSON Schema validation for generated output.
//!
//! When a schema is provided, the generated text is parsed as JSON and validated
//! against the schema. If invalid, generation is retried up to a limit with a
//! slightly higher temperature to encourage diversity.

use serde_json::Value;

/// Validate a JSON string against a JSON Schema.
///
/// `schema` should be a valid JSON Schema object (parsed from a string).
/// Returns `true` if `text` parses as JSON and validates against the schema.
pub fn validate(schema: &Value, text: &str) -> bool {
    let Ok(doc) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    match jsonschema::validator_for(schema) {
        Ok(validator) => validator.is_valid(&doc),
        Err(_) => false,
    }
}

/// Generate text with JSON Schema validation, retrying up to `max_retries`.
///
/// `generate_fn` is a closure that takes a `&GenerationConfig` and returns text.
/// `schema` is the parsed JSON Schema.
/// `base_config` is the original generation config.
/// `max_retries` is the maximum number of retries (default 3).
///
/// Returns the first generated text that validates, or the last attempt if none do.
pub fn generate_with_schema<F>(
    mut generate_fn: F,
    schema: &Value,
    base_config: &crate::generate::GenerationConfig,
    max_retries: usize,
) -> String
where
    F: FnMut(&crate::generate::GenerationConfig) -> String,
{
    for attempt in 0..=max_retries {
        let mut cfg = base_config.clone();
        // Slightly increase temperature on retries to encourage diversity.
        if attempt > 0 && cfg.temperature <= 0.0 {
            cfg.temperature = 0.3 * attempt as f32;
        }
        let text = generate_fn(&cfg);
        if validate(schema, &text) {
            return text;
        }
    }
    // None validated — return the last attempt.
    generate_fn(base_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_valid_json_object() {
        let schema = serde_json::json!({"type": "object"});
        assert!(validate(&schema, r#"{"a": 1}"#));
    }

    #[test]
    fn validate_rejects_invalid_json() {
        let schema = serde_json::json!({"type": "object"});
        assert!(!validate(&schema, "not json"));
    }

    #[test]
    fn validate_rejects_wrong_type() {
        let schema = serde_json::json!({"type": "number"});
        assert!(!validate(&schema, r#""hello""#));
    }

    #[test]
    fn generate_with_schema_retries_until_valid() {
        let schema = serde_json::json!({"type": "object"});
        let mut attempt = 0;
        let result = generate_with_schema(
            |cfg| {
                attempt += 1;
                if cfg.temperature > 0.0 || attempt >= 2 {
                    r#"{"ok": true}"#.to_string()
                } else {
                    "not json".to_string()
                }
            },
            &schema,
            &crate::generate::GenerationConfig::default(),
            3,
        );
        assert_eq!(result, r#"{"ok": true}"#);
        assert_eq!(attempt, 2, "should have succeeded on second attempt");
    }
}
