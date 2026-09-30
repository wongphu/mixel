//! Qwen-Image-2.1's sampling schedule: diffusers' `FlowMatchEulerDiscreteScheduler`
//! with dynamic exponential shifting and `shift_terminal = 0.02`, fed
//! `sigmas = linspace(1, 1/n, n)` by the pipeline.

const BASE_SEQ_LEN: f64 = 256.0;
const MAX_SEQ_LEN: f64 = 8192.0;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 0.9;
const SHIFT_TERMINAL: f32 = 0.02;

/// `mu` for `image_seq_len` target tokens (linear in the sequence length).
pub fn calculate_shift(image_seq_len: usize) -> f64 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    image_seq_len as f64 * m + b
}

/// `num_steps + 1` sigmas (the last is 0), computed in f32 like the reference.
pub fn sigmas(num_steps: usize, image_seq_len: usize) -> Vec<f32> {
    let n = num_steps;
    let e_mu = calculate_shift(image_seq_len).exp() as f32;
    let mut s: Vec<f32> = (0..n)
        .map(|i| {
            // np.linspace(1.0, 1/n, n) in f64, then cast to f32.
            let v = if n == 1 {
                1.0
            } else {
                1.0 + (1.0 / n as f64 - 1.0) * i as f64 / (n - 1) as f64
            };
            v as f32
        })
        // Exponential time shift: e^mu / (e^mu + (1/t - 1)).
        .map(|t| e_mu / (e_mu + (1.0 / t - 1.0)))
        .collect();
    // Stretch so the schedule ends at shift_terminal. A single step has
    // nothing to stretch (its only sigma is 1, and 0 / 0 would be NaN).
    if n > 1 {
        let one_minus_last = 1.0 - s[n - 1];
        let scale = one_minus_last / (1.0 - SHIFT_TERMINAL);
        for v in &mut s {
            *v = 1.0 - (1.0 - *v) / scale;
        }
    }
    s.push(0.0);
    s
}

/// Index of the first step for img2img at `strength` (like diffusers).
pub fn start_index(num_steps: usize, strength: f64) -> usize {
    crate::zimage::scheduler::start_index(num_steps, strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_starts_at_one_and_ends_at_terminal() {
        let s = sigmas(8, 1024);
        assert_eq!(s.len(), 9);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[7] - SHIFT_TERMINAL).abs() < 1e-6);
        assert_eq!(s[8], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn single_step_goes_from_noise_to_image() {
        assert_eq!(sigmas(1, 4096), vec![1.0, 0.0]);
    }
}
