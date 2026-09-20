//! Simple chat message formatting for transformer models.
//!
//! Renders a list of messages as a single prompt. We deliberately do not embed a
//! Jinja2 engine; for full chat-template support (e.g. Llama-3 special tokens),
//! render the prompt client-side. The fallback format used here is
//! `<role>: <content>\n` per message, suitable for base completion-style models
//! and for tests that only need a deterministic prompt string.

use serde::Serialize;

/// A single message in a chat conversation.
#[derive(Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render `messages` as a single prompt string.
///
/// The optional `template` argument is accepted for API stability but currently
/// ignored — the fallback format is always used.
pub fn apply_chat_template(_template: Option<&str>, messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_role_and_content() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let out = apply_chat_template(None, &messages);
        assert_eq!(out, "user: hello\n");
    }

    #[test]
    fn formats_multiple_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
        ];
        let out = apply_chat_template(None, &messages);
        assert_eq!(out, "system: You are helpful.\nuser: hi\n");
    }

    #[test]
    fn template_argument_is_ignored() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let with = apply_chat_template(
            Some("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"),
            &messages,
        );
        let without = apply_chat_template(None, &messages);
        assert_eq!(with, without);
    }
}
