//! `MOSSImageScheduler`: flow-matching Euler over sigmas shifted by 5 and
//! then snapped to the nearest of the 1000 sigmas the model was trained on.
//! Everything is in f32, like the reference (torch on the CPU).

use crate::zimage::scheduler::linspace;
/// First step for img2img (`strength` < 1), like diffusers.
pub use crate::zimage::scheduler::start_index;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SHIFT: f32 = 5.0;

/// `shift * v / (1 + (shift - 1) * v)` over `v = linspace(1, lower, count)`,
/// where `lower` is the smallest training sigma after the same shift.
fn grid(count: usize, shift: f32) -> Vec<f32> {
    let total = NUM_TRAIN_TIMESTEPS as f64;
    let lower = shift as f64 / (total + shift as f64 - 1.0);
    linspace(1.0, lower as f32, count)
        .into_iter()
        .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
        .collect()
}

/// The sigmas for `num_steps` steps, then a final 0. Snapping merges steps
/// that land on the same training sigma, so with very many steps there can
/// be fewer than `num_steps`.
pub fn sigmas(num_steps: usize) -> Vec<f32> {
    // The training sigmas, ascending.
    let training: Vec<f32> = grid(NUM_TRAIN_TIMESTEPS, 1.0).into_iter().rev().collect();
    let mut out: Vec<f32> = Vec::with_capacity(num_steps + 1);
    for s in grid(num_steps, SHIFT) {
        // torch.searchsorted (left), then the nearer neighbour; ties go low.
        let upper = training
            .partition_point(|&t| t < s)
            .min(NUM_TRAIN_TIMESTEPS - 1);
        let lower = upper.saturating_sub(1);
        let nearest = if (s - training[lower]).abs() <= (training[upper] - s).abs() {
            training[lower]
        } else {
            training[upper]
        };
        if out.last() != Some(&nearest) {
            out.push(nearest);
        }
    }
    out.push(0.0);
    out
}

/// The transformer's time input at `sigma`: `sigma * 1000`, in f32.
pub fn timestep(sigma: f32) -> f32 {
    sigma * NUM_TRAIN_TIMESTEPS as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_start_at_one_fall_and_end_in_zero() {
        for n in [1, 4, 8, 50] {
            let s = sigmas(n);
            assert_eq!(s.len(), n + 1, "{n} steps: {s:?}");
            assert_eq!(s[0], 1.0);
            assert_eq!(*s.last().unwrap(), 0.0);
            assert!(s.windows(2).all(|w| w[0] > w[1]), "{s:?}");
        }
        assert_eq!(timestep(1.0), 1000.0);
    }

    #[test]
    fn matches_the_reference_schedules_bit_for_bit() {
        // `MOSSImageScheduler`'s sigmas (f32 bits), from
        // scripts/make_moss_reference.py (torch 2.14.1, Apple Silicon).
        let bits = |b: &[u32]| b.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>();
        let tail = [0x3cc49ba6, 0x0];
        assert_eq!(
            sigmas(4),
            bits(&[0x3f800000, 0x3f68f5c2, 0x3f378d50, tail[0], tail[1]])
        );
        assert_eq!(
            sigmas(8),
            bits(&[
                0x3f800000, 0x3f77ced9, 0x3f6d0e56, 0x3f5ef9db, 0x3f4a7efa, 0x3f2bc6a8, 0x3eed0e57,
                tail[0], tail[1],
            ])
        );
        assert_eq!(
            sigmas(50),
            bits(&[
                0x3f800000, 0x3f7ef9db, 0x3f7df3b6, 0x3f7cac08, 0x3f7ba5e3, 0x3f7a5e35, 0x3f791687,
                0x3f77ced9, 0x3f76872b, 0x3f74fdf4, 0x3f73b646, 0x3f722d0e, 0x3f70624e, 0x3f6ed916,
                0x3f6d0e56, 0x3f6b4395, 0x3f6978d5, 0x3f67ae14, 0x3f65a1cb, 0x3f6353f8, 0x3f6147ae,
                0x3f5ef9db, 0x3f5c6a7f, 0x3f59db23, 0x3f570a3d, 0x3f543958, 0x3f5126e9, 0x3f4e147b,
                0x3f4a7efa, 0x3f46e979, 0x3f43126e, 0x3f3ef9db, 0x3f3a9fbe, 0x3f360418, 0x3f3126e9,
                0x3f2bc6a8, 0x3f25e354, 0x3f1f7ced, 0x3f18d4fe, 0x3f116872, 0x3f09374b, 0x3f004189,
                0x3eed0e57, 0x3ed78d51, 0x3ebef9dc, 0x3ea45a1d, 0x3e85a1cb, 0x3e45a1cb, 0x3deb851f,
                tail[0], tail[1],
            ])
        );
    }

    #[test]
    fn every_sigma_is_a_training_sigma() {
        let training = grid(NUM_TRAIN_TIMESTEPS, 1.0);
        for s in &sigmas(50)[..50] {
            assert!(training.contains(s), "{s}");
        }
    }
}
