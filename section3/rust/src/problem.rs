use crate::bench::{peak_rss_mb, time_apply};
use crate::HMatrix;
use nalgebra::{DMatrix, DVector};
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct BenchmarkRecord {
    pub problem: String,
    pub n: usize,
    pub leaf: usize,
    pub eps: f64,
    pub dense_entries: usize,
    pub compressed_entries: usize,
    pub storage_ratio: f64,
    pub relative_apply_error: f64,
    pub dense_build_seconds: f64,
    pub compression_seconds: f64,
    pub qr_calls: usize,
    pub dense_blocks: usize,
    pub low_rank_blocks: usize,
    pub split_blocks: usize,
    pub max_rank: usize,
    pub dense_apply_ms_median: f64,
    pub compressed_apply_ms_median: f64,
    pub dense_apply_ms_min: f64,
    pub compressed_apply_ms_min: f64,
    pub dense_apply_ms_samples: Vec<f64>,
    pub compressed_apply_ms_samples: Vec<f64>,
    pub peak_rss_mb: Option<f64>,
    pub apply_speedup: f64,
    pub dense_apply_repeats: usize,
    pub compressed_apply_repeats: usize,
    pub construction_note: &'static str,
}

pub struct KernelProblem<F>
where
    F: Fn(f64, f64) -> f64,
{
    pub name: String,
    pub points: Vec<f64>,
    pub kernel: F,
}

impl<F> KernelProblem<F>
where
    F: Fn(f64, f64) -> f64,
{
    pub fn new(name: impl Into<String>, points: Vec<f64>, kernel: F) -> Self {
        assert!(!points.is_empty());
        Self { name: name.into(), points, kernel }
    }

    pub fn dense(&self) -> DMatrix<f64> {
        let n = self.points.len();
        DMatrix::from_fn(n, n, |i, j| (self.kernel)(self.points[i], self.points[j]))
    }

    pub fn benchmark(&self, leaf: usize, eps: f64) -> BenchmarkRecord {
        let started = Instant::now();
        let a = self.dense();
        let dense_build_seconds = started.elapsed().as_secs_f64();
        let h = HMatrix::from_dense(&a, leaf, eps);
        let x = test_vector(a.nrows());
        let exact = &a * &x;
        let approximate = h.apply(&x);
        let relative_apply_error = (&approximate - &exact).norm() / exact.norm();
        let dense_apply = time_apply(|| black_box(&a) * black_box(&x));
        let compressed_apply = time_apply(|| h.apply(black_box(&x)));
        BenchmarkRecord {
            problem: self.name.clone(),
            n: a.nrows(),
            leaf,
            eps,
            dense_entries: a.len(),
            compressed_entries: h.stored_entries(),
            storage_ratio: h.storage_ratio(),
            relative_apply_error,
            dense_build_seconds,
            compression_seconds: h.stats.seconds,
            qr_calls: h.stats.qr_calls,
            dense_blocks: h.stats.dense_blocks,
            low_rank_blocks: h.stats.low_rank_blocks,
            split_blocks: h.stats.split_blocks,
            max_rank: h.stats.max_rank,
            dense_apply_ms_median: dense_apply.median_ms,
            compressed_apply_ms_median: compressed_apply.median_ms,
            dense_apply_ms_min: dense_apply.min_ms,
            compressed_apply_ms_min: compressed_apply.min_ms,
            dense_apply_ms_samples: dense_apply.samples_ms.clone(),
            compressed_apply_ms_samples: compressed_apply.samples_ms.clone(),
            peak_rss_mb: peak_rss_mb(),
            apply_speedup: dense_apply.median_ms / compressed_apply.median_ms,
            dense_apply_repeats: dense_apply.repeats,
            compressed_apply_repeats: compressed_apply.repeats,
            construction_note: "dense-to-compressed oracle; not a fast hierarchical builder",
        }
    }
}

pub fn test_vector(n: usize) -> DVector<f64> {
    // Deterministic LCG, independent of an RNG crate.
    let mut state = 0x1234_5678_9abc_def0u64;
    DVector::from_fn(n, |_, _| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        2.0 * unit - 1.0
    })
}

pub fn grid(n: usize) -> Vec<f64> {
    assert!(n >= 2);
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

pub fn print_records(records: &[BenchmarkRecord]) {
    println!("{}", serde_json::to_string_pretty(records).expect("JSON serialization"));
}
