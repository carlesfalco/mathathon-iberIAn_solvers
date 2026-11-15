//! One-file example: the DFT *is* a rank-one butterfly, and that butterfly is the FFT.
//!
//!   cargo run --release --example dft_as_butterfly
//!
//! Everything else in this crate is machinery; this is the whole claim on one screen.
//!
//! For N = 2^L, F[j,k] = exp(+2 pi i j k / N). Put both trees in bit-reversed order, so a
//! depth-l node is the residue class {j = a mod 2^l}. Then for j = a + 2^l u, k = b + 2^(L-l) v,
//!
//!   jk/N = ab/N + a v / 2^l + b u / 2^(L-l) + u v,       u v an integer,
//!
//! so each complementary tile is (column vector in v) x (row vector in u): rank exactly 1.
//! The ID butterfly therefore keeps one skeleton column per tile and stores
//!
//!   N (W)  +  L * 2N (G_0..G_{L-1})  +  N (U)  =  2 N log2 N + 2 N
//!
//! entries, every one of them unimodular.

use unified_rs::dft_butterfly::butterfly::build_bitrev;
use unified_rs::dft_butterfly::c64::C64;
use unified_rs::dft_butterfly::fft::{fft_radix2, log2_exact};

fn main() {
    let n = 1024usize;
    let l = log2_exact(n);
    let g: Vec<C64> = (0..n)
        .map(|j| C64 {
            re: (0.7 * j as f64).sin(),
            im: (1.3 * j as f64).cos(),
        })
        .collect();

    let bf = build_bitrev(n, 1e-12);
    let f = bf.apply(&g);
    let reference = fft_radix2(&g);

    let err = f
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0_f64, f64::max)
        / reference.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

    println!("N = {n}, L = {l}");
    println!("ranks per stage      : {:?}", bf.ranks);
    println!("nnz                  : {}", bf.nnz());
    println!("2 N log2 N + 2 N     : {}", 2 * n * l + 2 * n);
    println!("nonzeros that are 1  : {}", bf.nnz() - bf.apply_mults());
    println!("relative error vs FFT: {err:.2e}");

    assert!(bf.ranks.iter().all(|&r| r == 1), "not every tile is rank one");
    assert_eq!(bf.nnz(), 2 * n * l + 2 * n, "storage count is not 2NL+2N");
    assert!(err < 1e-12, "the factorisation does not reproduce the DFT");
    println!("\nOK: the DFT's complementary tiles are rank one, with exact FFT-storage factors.");
}
