use unified_rs::problem::{grid, print_records};
use unified_rs::KernelProblem;

fn main() {
    let n = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let length_scale = 0.08;
    let problem = KernelProblem::new("Gaussian covariance", grid(n), move |x, y| {
        (-0.5 * ((x - y) / length_scale).powi(2)).exp()
    });
    print_records(&[problem.benchmark(32, 1e-6)]);
}
