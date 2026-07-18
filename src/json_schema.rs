//! Lightweight JSON validation for generated output.
//!
//! When a JSON Schema is provided, we don't run full schema validation (no external
//! dep). Instead we require the output to parse as JSON and, when the schema carries a
//! `type` field, match that top-level type. This catches the common failure mode
//! (non-JSON or wrong shape) without pulling in a schema validator.

use serde_json::Value;

/// Returns true when `text` parses as JSON and matches the top-level `type`
/// declared in `schema`, if any.
pub fn validate(schema: &Value, text: &str) -> bool {
    let Ok(doc) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    let Some(expected) = schema.get("type").and_then(Value::as_str) else {
        return true;
    };
    match expected {
        "object" => doc.is_object(),
        "array" => doc.is_array(),
        "string" => doc.is_string(),
        "number" => doc.is_number(),
        "integer" => doc.is_i64() || doc.is_u64(),
        "boolean" | "bool" => doc.is_boolean(),
        "null" => doc.is_null(),
        _ => true,
    }
}

/// Generate text with JSON validation, retrying up to `max_retries`.
///
/// `generate_fn` is a closure that takes a `&GenerationConfig` and returns text.
/// `schema` is the parsed JSON Schema (only the top-level `type` is enforced).
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
        if attempt > 0 && cfg.temperature <= 0.0 {
            cfg.temperature = 0.3 * attempt as f32;
        }
        let text = generate_fn(&cfg);
        if validate(schema, &text) {
            return text;
        }
    }
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
    fn validate_without_type_just_parses() {
        let schema = serde_json::json!({});
        assert!(validate(&schema, "{\"a\": 1}"));
        assert!(!validate(&schema, "not json"));
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
