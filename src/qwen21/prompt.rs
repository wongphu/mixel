//! Prompt template, tokenization and M-RoPE positions for Qwen-Image-2.1's
//! Qwen3-VL text encoder (ports `QwenImage21Pipeline._get_qwen_prompt_embeds`
//! and transformers' `Qwen3VLModel.get_rope_index`).

use anyhow::{Error as E, Result};
use tokenizers::Tokenizer;

const SYSTEM: &str = "Comprehend and analyze the provided prompt.";
const IMAGE_PAD: &str = "<|image_pad|>";

/// A tokenized prompt ready for the text encoder.
pub struct EncodedPrompt {
    pub ids: Vec<u32>,
    /// (T, H, W) rotary position per token.
    pub positions: Vec<[i64; 3]>,
    /// Per reference image: index of its first `<|image_pad|>` token.
    pub image_starts: Vec<usize>,
    /// Leading system-message tokens whose hidden states are dropped.
    pub drop: usize,
}

impl EncodedPrompt {
    /// `true` at `<|image_pad|>` positions of the kept (post-`drop`) tokens.
    pub fn image_pad_mask(&self, image_pad_id: u32) -> Vec<bool> {
        self.ids[self.drop..]
            .iter()
            .map(|&i| i == image_pad_id)
            .collect()
    }
}

/// The token id of `<|image_pad|>`.
pub fn image_pad_id(tokenizer: &Tokenizer) -> Result<u32> {
    tokenizer
        .token_to_id(IMAGE_PAD)
        .ok_or_else(|| anyhow::anyhow!("tokenizer has no {IMAGE_PAD}"))
}

/// The prompt text sent to the tokenizer, with `<|image_pad|>` expanded to
/// `tokens` copies per image.
pub fn template(prompt: &str, image_tokens: &[usize]) -> String {
    // Qwen has no BOS token, so an empty prompt would leave nothing to read.
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let vision: String = image_tokens
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let sep = if i == 0 { "" } else { " " };
            format!(
                "{sep}<image{}><|vision_start|>{}<|vision_end|>",
                i + 1,
                IMAGE_PAD.repeat(n)
            )
        })
        .collect();
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{vision}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Tokenizes `prompt` with reference images whose merged vision grids are
/// `grids` (height, width in 32 px tokens), computing M-RoPE positions.
pub fn encode(
    tokenizer: &Tokenizer,
    prompt: &str,
    grids: &[(usize, usize)],
) -> Result<EncodedPrompt> {
    let image_tokens: Vec<usize> = grids.iter().map(|(h, w)| h * w).collect();
    let text = template(prompt, &image_tokens);
    let ids = tokenizer
        .encode(text.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let drop = tokenizer
        .encode(
            format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n").as_str(),
            true,
        )
        .map_err(E::msg)?
        .len();

    let pad = image_pad_id(tokenizer)?;
    let mut positions = Vec::with_capacity(ids.len());
    let mut image_starts = Vec::new();
    let mut grids = grids.iter();
    let mut pos: i64 = 0;
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != pad {
            positions.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let &(gh, gw) = grids
            .next()
            .ok_or_else(|| anyhow::anyhow!("more image placeholders than images"))?;
        image_starts.push(i);
        for r in 0..gh as i64 {
            for c in 0..gw as i64 {
                positions.push([pos, pos + r, pos + c]);
            }
        }
        i += gh * gw;
        pos += gh.max(gw) as i64;
    }
    anyhow::ensure!(
        grids.next().is_none(),
        "fewer image placeholders than images"
    );
    anyhow::ensure!(
        positions.len() == ids.len(),
        "image placeholder run was cut short"
    );
    Ok(EncodedPrompt {
        ids,
        positions,
        image_starts,
        drop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_reference_layout() {
        assert_eq!(
            template("a cat", &[]),
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        );
        let t = template("edit", &[2, 1]);
        assert!(t.contains(
            "user\n<image1><|vision_start|><|image_pad|><|image_pad|><|vision_end|> \
             <image2><|vision_start|><|image_pad|><|vision_end|>edit<|im_end|>"
        ));
        assert!(template("", &[]).contains("user\n <|im_end|>"));
    }
}
