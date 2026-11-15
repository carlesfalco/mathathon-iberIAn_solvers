use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Dense O(N^2) reference DFT with the NumPy/RustFFT negative-sign convention.
pub fn dft_naive(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|j| {
            input
                .iter()
                .enumerate()
                .map(|(k, &x)| {
                    let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                    x * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect()
}

/// Reusable FFT plan: planning is excluded from per-apply timings.
pub struct FftPlan {
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl FftPlan {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        let mut planner = FftPlanner::<f64>::new();
        Self { fft: planner.plan_fft_forward(n) }
    }

    pub fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(input.len(), self.fft.len());
        let mut buffer = input.to_vec();
        self.fft.process(&mut buffer);
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| Complex64::new((0.37 * i as f64).sin(), (0.19 * i as f64).cos()))
            .collect()
    }

    #[test]
    fn fft_matches_dense() {
        let x = input(64);
        let dense = dft_naive(&x);
        let fast = FftPlan::new(64).apply(&x);
        let error = dense
            .iter()
            .zip(&fast)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        assert!(error < 1e-12, "{error}");
    }

    #[test]
    fn parseval() {
        let x = input(64);
        let y = FftPlan::new(64).apply(&x);
        let left = x.iter().map(|a| a.norm_sqr()).sum::<f64>() * 64.0;
        let right = y.iter().map(|a| a.norm_sqr()).sum::<f64>();
        assert!((left - right).abs() / left < 1e-12);
    }
}
