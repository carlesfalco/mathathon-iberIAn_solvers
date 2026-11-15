use unified_rs::problem::{grid, print_records};
use unified_rs::KernelProblem;

fn main() {
    let n = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let screening = 8.0;
    let problem = KernelProblem::new("exponential screening", grid(n), move |x, y| {
        (-screening * (x - y).abs()).exp()
    });
    print_records(&[problem.benchmark(32, 1e-6)]);
}
