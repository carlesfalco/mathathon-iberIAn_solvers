//! A dependency-free reference implementation of the exact structural statement
//! that ties together the discrete Fourier transform, the interpolative butterfly
//! factorisation and the radix-2 FFT.
//!
//! The claim being implemented and checked:
//!
//! > With the DFT `F[j,k] = exp(+2 pi i j k / n)`, `n = 2^L`, the complementary
//! > pairing (row node at depth `l` of a row tree, column node at depth `L - l`
//! > of a column tree) has **exactly rank-one** tiles when both trees are the
//! > dyadic trees of the *bit-reversed* index order, i.e. when a depth-`l` node
//! > is the residue class `j = a (mod 2^l)`.  Consequently the ID butterfly has
//! > one skeleton column per tile, `L + 2` factors with
//! > `nnz = 2 n L + 2 n`, every nonzero unimodular, and it reproduces the DFT
//! > matrix exactly (to rounding).
//!
//! Modules:
//!   `c64`        complex arithmetic, `+2 pi i` twiddles
//!   `fft`        naive DFT, radix-2 FFT, general mixed-radix FFT
//!   `id`         column ID by pivoted modified Gram-Schmidt
//!   `butterfly`  the factorisation and its `nnz` / rank / unit-modulus counters

pub mod butterfly;
pub mod c64;
pub mod fft;
pub mod id;

#[cfg(test)]
mod smoke {
    use super::*;
    use c64::C64;

    #[test]
    fn naive_matches_radix2() {
        let n = 32;
        let x: Vec<C64> = (0..n)
            .map(|j| C64::from_polar(1.0 + (j % 7) as f64, 0.37 * j as f64))
            .collect();
        let a = fft::dft_naive(&x);
        let b = fft::fft_radix2(&x);
        let e = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (*p - *q).abs())
            .fold(0.0_f64, f64::max);
        assert!(e < 1e-12, "max abs diff {e:e}");
    }
}
