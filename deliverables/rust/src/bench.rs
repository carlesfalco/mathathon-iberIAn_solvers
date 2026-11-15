//! Small timing/memory helpers shared by examples, with raw samples retained.
use serde::Serialize;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    pub median_ms: f64,
    pub min_ms: f64,
    pub samples_ms: Vec<f64>,
    pub repeats: usize,
    pub warmup: usize,
}

/// One warm-up, then enough repeats for a stable short-kernel measurement.
/// Results exclude the warm-up and retain every timed sample.
pub fn time_apply<T>(mut apply: impl FnMut() -> T) -> Timing {
    let warmup = 1;
    let started = Instant::now();
    black_box(apply());
    let first = started.elapsed();
    let repeats = if first >= Duration::from_millis(30) {
        5
    } else {
        ((Duration::from_millis(30).as_secs_f64() / first.as_secs_f64().max(1e-9)).ceil() as usize)
            .clamp(5, 200)
    };
    let mut samples_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        black_box(apply());
        samples_ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let mut ordered = samples_ms.clone();
    ordered.sort_by(f64::total_cmp);
    Timing {
        median_ms: ordered[repeats / 2],
        min_ms: ordered[0],
        samples_ms,
        repeats,
        warmup,
    }
}

/// Linux high-water RSS, in MiB, or None on non-Linux systems.
pub fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kb = line.strip_prefix("VmHWM:")?.trim().strip_suffix(" kB")?.parse::<f64>().ok()?;
    Some(kb / 1024.0)
}
