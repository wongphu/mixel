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

    /// A tokenizer with one token per word (and per run of punctuation).
    fn word_tokenizer() -> Tokenizer {
        let json = r#"{"version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "word": 1}, "unk_token": "[UNK]"}}"#;
        json.parse().unwrap()
    }

    #[test]
    fn long_prompts_are_cut_to_the_limit_and_say_so() {
        let tokenizer = word_tokenizer();
        let short = encode(&tokenizer, "word word").unwrap();
        assert!(!short.truncated);
        assert!(short.ids.len() < MAX_PROMPT_TOKENS);
        // The template adds tokens of its own: a prompt of exactly the limit is cut.
        let long = vec!["word"; MAX_PROMPT_TOKENS].join(" ");
        let long = encode(&tokenizer, &long).unwrap();
        assert!(long.truncated);
        assert_eq!(long.ids.len(), MAX_PROMPT_TOKENS);
        // Like the reference, the cut drops the end, template included.
        assert_eq!(long.ids.last(), Some(&1));
    }

    #[test]
    #[ignore = "needs the MOSS-Image-SFT tokenizer from the Hugging Face cache; run with --ignored"]
    fn token_ids_match_the_reference() {
        let files = crate::nn::ModelFiles::new(super::super::REPO, None).unwrap();
        let tokenizer =
            Tokenizer::from_file(files.get("tokenizer/tokenizer.json").unwrap()).unwrap();
        // transformers' ids for the chat template (scripts/make_moss_reference.py).
        let fox = encode(
            &tokenizer,
            "A red fox sitting in fresh snow, wildlife photography",
        )
        .unwrap();
        assert_eq!(
            fox.ids,
            [
                248045, 846, 198, 32, 2438, 37550, 11362, 303, 7478, 11456, 11, 28346, 23009,
                248046, 198, 248045, 74455, 198, 248068, 198
            ]
        );
        let empty = encode(&tokenizer, "").unwrap();
        assert_eq!(
            empty.ids,
            [248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198]
        );
    }
}
