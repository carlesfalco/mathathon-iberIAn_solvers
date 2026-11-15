use num_complex::Complex64;
use serde_json::json;
use std::hint::black_box;
use unified_rs::bench::{peak_rss_mb, time_apply};
use unified_rs::{dft_naive, FftPlan};

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    assert!(n.is_power_of_two());
    let input: Vec<_> = (0..n)
        .map(|i| Complex64::new((0.37 * i as f64).sin(), (0.19 * i as f64).cos()))
        .collect();
    let plan = FftPlan::new(n);
    let fast = plan.apply(&input);
    let dense = dft_naive(&input);
    let relative_error = norm_difference(&fast, &dense) / norm(&dense);
    let fast_time = time_apply(|| plan.apply(black_box(&input)));
    let dense_time = time_apply(|| dft_naive(black_box(&input)));
    println!("{}", serde_json::to_string_pretty(&json!({
        "problem": "DFT", "n": n, "relative_apply_error": relative_error,
        "dense_apply_ms_median": dense_time.median_ms,
        "fft_apply_ms_median": fast_time.median_ms,
        "dense_apply_ms_min": dense_time.min_ms,
        "fft_apply_ms_min": fast_time.min_ms,
        "dense_apply_ms_samples": dense_time.samples_ms,
        "fft_apply_ms_samples": fast_time.samples_ms,
        "apply_speedup": dense_time.median_ms / fast_time.median_ms,
        "apply_speedup_min": dense_time.min_ms / fast_time.min_ms,
        "dense_repeats": dense_time.repeats, "fft_repeats": fast_time.repeats,
        "peak_rss_mb": peak_rss_mb(),
        "note": "same Rust process; FFT plan reused; negative-sign unnormalised DFT"
    })).unwrap());
}

fn norm(x: &[Complex64]) -> f64 { x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() }
fn norm_difference(x: &[Complex64], y: &[Complex64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt()
}
