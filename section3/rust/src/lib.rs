//! A deliberately small first Rust slice of the `unified` ideas.
//!
//! The kernel examples construct a dense matrix and then compress it.  That is
//! an oracle construction for checking representation, storage, and apply cost;
//! it is not an O(N log N) builder.  The FFT path is a genuine fast transform.

pub mod bench;
pub mod dft_butterfly;
pub mod fft;
pub mod hmatrix;
pub mod problem;

pub use fft::{dft_naive, FftPlan};
pub use hmatrix::HMatrix;
pub use problem::{BenchmarkRecord, KernelProblem};
