//! Prompt template and tokenization for MOSS-Image's Qwen3.5 text encoder
//! (ports `MOSSImagePipeline.encode_prompt`).

use super::MAX_PROMPT_TOKENS;
use anyhow::{Error as E, Result};
use tokenizers::Tokenizer;

/// A tokenized prompt ready for the text encoder.
pub struct EncodedPrompt {
    pub ids: Vec<u32>,
    /// Whether the prompt was cut to [`MAX_PROMPT_TOKENS`].
    pub truncated: bool,
}

/// The chat template's rendering of one user message, with
/// `add_generation_prompt=True, enable_thinking=True`. The template trims the
/// message; an empty one stays empty.
pub fn template(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        prompt.trim()
    )
}

/// Tokenizes `prompt`, cutting it to [`MAX_PROMPT_TOKENS`] like the
/// reference (which drops the end of the template too).
pub fn encode(tokenizer: &Tokenizer, prompt: &str) -> Result<EncodedPrompt> {
    let mut ids = tokenizer
        .encode(template(prompt).as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let truncated = ids.len() > MAX_PROMPT_TOKENS;
    ids.truncate(MAX_PROMPT_TOKENS);
    Ok(EncodedPrompt { ids, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_trims_the_message_and_opens_thinking() {
        assert_eq!(
            template("  A cat \n"),
            "<|im_start|>user\nA cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            template(""),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }
}
