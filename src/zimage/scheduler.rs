//! FlowMatch Euler scheduler for Z-Image-Turbo, set up like diffusers'
//! `ZImagePipeline`: sigmas `linspace(1, 1/n, n)`, then the scheduler's static
//! shift `3s / (1 + 2s)` (the Turbo config has no dynamic shifting), then a
//! final 0. Everything is in f32, like the reference.
//!
//! (candle's port applies no shift, so its sigmas fall evenly from 1 and the
//! steps spend less time at high noise.)

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const SHIFT: f32 = 3.0;

pub struct Scheduler {
    /// `n + 1` sigmas, the last 0.
    sigmas: Vec<f32>,
    step_index: usize,
}

impl Scheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let n = num_inference_steps;
        // Python computes 1/n in f64 before torch rounds it to f32.
        let mut sigmas: Vec<f32> = linspace(1.0, (1.0 / n as f64) as f32, n)
            .into_iter()
            .map(|s| SHIFT * s / (1.0 + (SHIFT - 1.0) * s))
            .collect();
        sigmas.push(0.0);
        Self {
            sigmas,
            step_index: 0,
        }
    }

    /// For img2img: skips the steps a `strength` in (0, 1] leaves out, like
    /// diffusers' `get_timesteps`. Returns how many steps remain to run.
    pub fn skip_for_strength(&mut self, strength: f64) -> usize {
        let n = self.sigmas.len() - 1;
        self.step_index = start_index(n, strength);
        n - self.step_index
    }

    /// Model input time in [0, 1]: `(1000 - t) / 1000` with `t = 1000 * sigma`.
    pub fn current_timestep_normalized(&self) -> f32 {
        let t = self.sigmas[self.step_index] * NUM_TRAIN_TIMESTEPS;
        (NUM_TRAIN_TIMESTEPS - t) / NUM_TRAIN_TIMESTEPS
    }

    pub fn current_sigma(&self) -> f32 {
        self.sigmas[self.step_index]
    }

    /// Returns `dt = sigma_next - sigma` for the Euler update and advances.
    pub fn step_dt(&mut self) -> f32 {
        let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
        self.step_index += 1;
        dt
    }
}

/// `torch.linspace` for f32: the first half counts up from `start`, the
/// second down from `end`, so both ends are exact.
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n)
        .map(|i| {
            if i < n / 2 {
                start + step * i as f32
            } else {
                end - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

/// First step index for img2img: `int(n - min(n * strength, n))`, with
/// `strength` clamped to [0, 1] so the result is never past `num_steps`.
pub fn start_index(num_steps: usize, strength: f64) -> usize {
    let n = num_steps as f64;
    (n - n * strength.clamp(0.0, 1.0)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmas(n: usize) -> Vec<f32> {
        let mut s = Scheduler::new(n);
        (0..n)
            .map(|_| {
                let sigma = s.current_sigma();
                s.step_dt();
                sigma
            })
            .chain([0.0])
            .collect()
    }

    #[test]
    fn matches_diffusers_turbo_schedule_bit_for_bit() {
        // diffusers' sigmas for 9 steps (f32 bits), from
        // FlowMatchEulerDiscreteScheduler with Z-Image-Turbo's config.
        let expect: Vec<f32> = [
            0x3f800000u32,
            0x3f75c290,
            0x3f69bd39,
            0x3f5b6db6,
            0x3f4a1af3,
            0x3f34b4b5,
            0x3f199999,
            0x3eec4ec6,
            0x3e8ba2e9,
            0x0,
        ]
        .map(f32::from_bits)
        .to_vec();
        assert_eq!(sigmas(9), expect);
        assert_eq!(sigmas(4), [1.0, 0.9, 0.75, 0.5, 0.0]);
        assert_eq!(sigmas(1), [1.0, 0.0]);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // exact f32 values from diffusers
    fn model_time_is_one_minus_sigma_in_f32() {
        let mut s = Scheduler::new(9);
        let mut t = Vec::new();
        for _ in 0..9 {
            t.push(s.current_timestep_normalized());
            s.step_dt();
        }
        assert_eq!(t[0], 0.0);
        assert_eq!(t[1], 0.03999993950128555);
        assert_eq!(t[8], 0.7272726893424988);
    }

    #[test]
    fn strength_skips_leading_steps_like_diffusers() {
        assert_eq!(start_index(9, 1.0), 0);
        assert_eq!(start_index(9, 0.6), 3); // int(9 - 5.4)
        assert_eq!(start_index(9, 0.5), 4); // int(9 - 4.5)
        assert_eq!(start_index(9, 0.1), 8); // int(9 - 0.9)
        assert_eq!(start_index(9, 0.05), 8); // int(9 - 0.45)
        assert_eq!(start_index(9, 1.5), 0);

        let mut s = Scheduler::new(9);
        assert_eq!(s.skip_for_strength(0.6), 6);
        // Starts from the 4th sigma of the full schedule.
        let full = Scheduler::new(9);
        assert_eq!(s.current_sigma(), full.sigmas[3]);
    }
}
