//! Sigma-schedule helpers shared by the models' schedulers.

/// `torch.linspace` for f32 on the CPU: the first half counts up from
/// `start`, the second down from `end`, so both ends are exact. Each value is
/// one fused multiply-add, as torch's kernel compiles on Apple Silicon
/// (separate roundings differ in 239 of MOSS-Image's 1000 training sigmas,
/// and in Z-Image's schedules from 13 steps up).
pub(crate) fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n)
        .map(|i| {
            if i < n / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((n - 1 - i) as f32, end)
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
