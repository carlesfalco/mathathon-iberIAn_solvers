use unified_rs::problem::{grid, print_records};
use unified_rs::KernelProblem;

// One file per problem: the library owns trees, compression and timing.
fn main() {
    let n = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let problem = KernelProblem::new("log/Laplace, I + K", grid(n), |x, y| {
        let r = (x - y).abs();
        if r == 0.0 { 1.0 } else { -r.ln() / (2.0 * std::f64::consts::PI) }
    });
    print_records(&[problem.benchmark(32, 1e-6)]);
}
