//! FlowMatch Euler scheduler, ported from candle's `z_image::scheduler`.
//!
//! Note: with the Turbo config (`use_dynamic_shifting = false`) candle's
//! `set_timesteps(n, Some(mu))` applies no shift at all, so sigmas are linear
//! from 1.0 to the shifted minimum. This port keeps that behavior exactly so
//! results match candle.

const NUM_TRAIN_TIMESTEPS: f64 = 1000.0;
const SHIFT: f64 = 3.0;

pub struct Scheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    step_index: usize,
}

impl Scheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let shifted = |s: f64| SHIFT * s / (1.0 + (SHIFT - 1.0) * s);
        // Training schedule endpoints: t = 1000..1, sigma = t / 1000, shifted.
        let sigma_max = shifted(1.0);
        let sigma_min = shifted(1.0 / NUM_TRAIN_TIMESTEPS);

        let timesteps: Vec<f64> = (0..num_inference_steps)
            .map(|i| {
                let t = i as f64 / num_inference_steps as f64;
                (sigma_max * (1.0 - t) + sigma_min * t) * NUM_TRAIN_TIMESTEPS
            })
            .collect();
        let mut sigmas: Vec<f64> = timesteps.iter().map(|t| t / NUM_TRAIN_TIMESTEPS).collect();
        sigmas.push(0.0);
        Self {
            timesteps,
            sigmas,
            step_index: 0,
        }
    }

    /// Model input time in [0, 1]: (1000 - t) / 1000.
    pub fn current_timestep_normalized(&self) -> f64 {
        (NUM_TRAIN_TIMESTEPS - self.timesteps[self.step_index]) / NUM_TRAIN_TIMESTEPS
    }

    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// Returns `dt = sigma_next - sigma` for the Euler update and advances.
    pub fn step_dt(&mut self) -> f64 {
        let dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index];
        self.step_index += 1;
        dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_candle_turbo_schedule() {
        // Values printed by candle's pipeline for 9 steps.
        let mut s = Scheduler::new(9);
        let mut sigmas_after = Vec::new();
        let mut ts = Vec::new();
        for _ in 0..9 {
            ts.push(s.current_timestep_normalized());
            s.step_dt();
            sigmas_after.push(s.current_sigma());
        }
        let expect_t = [
            0.0, 0.1108, 0.2216, 0.3323, 0.4431, 0.5539, 0.6647, 0.7754, 0.8862,
        ];
        let expect_s = [
            0.8892, 0.7784, 0.6677, 0.5569, 0.4461, 0.3353, 0.2246, 0.1138, 0.0,
        ];
        for i in 0..9 {
            assert!((ts[i] - expect_t[i]).abs() < 1e-4, "t[{i}] = {}", ts[i]);
            assert!(
                (sigmas_after[i] - expect_s[i]).abs() < 1e-4,
                "sigma[{i}] = {}",
                sigmas_after[i]
            );
        }
    }
}
