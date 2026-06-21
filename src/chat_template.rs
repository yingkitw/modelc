//! Chat template rendering for transformer models.
//!
//! Reads Jinja2 chat templates from GGUF metadata (`tokenizer.chat_template`) and renders them
//! with a message list before tokenization. Falls back to a simple concatenation when no template
//! is available.

use minijinja::Environment;
use serde::Serialize;

/// A single message in a chat conversation.
#[derive(Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Apply a Jinja2 chat template to a list of messages.
///
/// If `template` is `None`, falls back to a simple `role: content\n` concatenation.
pub fn apply_chat_template(
    template: Option<&str>,
    messages: &[ChatMessage],
) -> anyhow::Result<String> {
    if let Some(tpl) = template {
        let mut env = Environment::new();
        env.add_template("chat", tpl)?;
        let tmpl = env.get_template("chat")?;

        let ctx = minijinja::context! {
            messages => messages,
            bos_token => "",
            eos_token => "",
            add_generation_prompt => true,
        };

        Ok(tmpl.render(ctx)?)
    } else {
        Ok(fallback_format(messages))
    }
}

fn fallback_format(messages: &[ChatMessage]) -> String {
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
    fn fallback_without_template() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let out = apply_chat_template(None, &messages).unwrap();
        assert_eq!(out, "user: hello\n");
    }

    #[test]
    fn simple_jinja_template() {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant:";
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let out = apply_chat_template(Some(template), &messages).unwrap();
        assert_eq!(out, "user: hi\nassistant:");
    }

    #[test]
    fn llama3_like_template() {
        let template = "<|begin_of_text|>{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
        ];
        let out = apply_chat_template(Some(template), &messages).unwrap();
        assert!(out.contains("system"));
        assert!(out.contains("user"));
        assert!(out.contains("assistant"));
        assert!(out.contains("<|begin_of_text|>"));
    }
}
