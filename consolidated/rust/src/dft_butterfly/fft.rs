//! Three independent routes to the same discrete Fourier transform, all in the
//! project's `+2*pi*i` convention:
//!
//!   `dft_naive`          O(n^2) definition, the ground truth for small n.
//!   `fft_radix2`         iterative decimation-in-time Cooley-Tukey, n = 2^L.
//!   `fft_mixed_radix`    general p-way decimation, n = prod(factors), any p.
//!
//! The three are cross-checked against each other in `tests/invariants.rs`; the
//! butterfly factorisation in `butterfly.rs` is cross-checked against all three.

use crate::dft_butterfly::c64::{twiddle, C64};

pub fn is_pow2(n: usize) -> bool {
    n != 0 && n & (n - 1) == 0
}

pub fn log2_exact(n: usize) -> usize {
    debug_assert!(is_pow2(n));
    n.trailing_zeros() as usize
}

/// Bit-reversal permutation of `0..n`, n = 2^L. `out[j] = rev_L(j)`.
pub fn bitrev_perm(n: usize) -> Vec<usize> {
    let l = log2_exact(n);
    (0..n)
        .map(|j| {
            let mut v = j;
            let mut r = 0usize;
            for _ in 0..l {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            r
        })
        .collect()
}

/// `out[j] = sum_k x[k] * exp(+2 pi i j k / n)`.
pub fn dft_naive(x: &[C64]) -> Vec<C64> {
    let n = x.len();
    let mut out = vec![C64::ZERO; n];
    for (j, o) in out.iter_mut().enumerate() {
        let mut s = C64::ZERO;
        for (k, &xk) in x.iter().enumerate() {
            s += xk * twiddle((j * k) as i64, n);
        }
        *o = s;
    }
    out
}

/// In-place iterative radix-2 decimation-in-time FFT. `x.len()` must be a power of two.
pub fn fft_radix2_inplace(x: &mut [C64]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    assert!(is_pow2(n), "fft_radix2 needs a power-of-two length, got {n}");
    let rev = bitrev_perm(n);
    for i in 0..n {
        if rev[i] > i {
            x.swap(i, rev[i]);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let mut i = 0;
        while i < n {
            for j in 0..half {
                let w = twiddle(j as i64, len);
                let u = x[i + j];
                let v = x[i + j + half] * w;
                x[i + j] = u + v;
                x[i + j + half] = u - v;
            }
            i += len;
        }
        len <<= 1;
    }
}

pub fn fft_radix2(x: &[C64]) -> Vec<C64> {
    let mut y = x.to_vec();
    fft_radix2_inplace(&mut y);
    y
}

/// Inverse of `fft_radix2`: `ifft(fft(x)) == x`, with the `1/n` on the forward-then-inverse side.
pub fn ifft_radix2(x: &[C64]) -> Vec<C64> {
    let n = x.len();
    let conj: Vec<C64> = x.iter().map(|z| z.conj()).collect();
    let mut y = fft_radix2(&conj);
    for z in y.iter_mut() {
        *z = z.conj().scale(1.0 / n as f64);
    }
    y
}

/// General mixed-radix Cooley-Tukey: `x.len() == factors.iter().product()`.
///
/// Write `n = p * m` with `p = factors[0]`, split `x[n1 + p*n2]`, and use
/// `k = k2 + m*k1`:
///   `X[k2 + m*k1] = sum_{n1<p} w_p^{n1 k1} * w_n^{n1 k2} * G_{n1}[k2]`,
/// where `G_{n1}` is the length-`m` transform of the strided subsequence
/// `x[n1::p]`. For `factors = [2,2,...]` this is the radix-2 algorithm with a
/// different internal index order; the output is in natural order.
pub fn fft_mixed_radix(x: &[C64], factors: &[usize]) -> Vec<C64> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    assert!(!factors.is_empty(), "factor list exhausted at n = {n}");
    let p = factors[0];
    assert!(p >= 2 && n % p == 0, "bad radix {p} for n = {n}");
    let m = n / p;
    let rest = &factors[1..];
    let subs: Vec<Vec<C64>> = (0..p)
        .map(|n1| {
            let seq: Vec<C64> = (0..m).map(|n2| x[n1 + p * n2]).collect();
            fft_mixed_radix(&seq, rest)
        })
        .collect();
    let mut out = vec![C64::ZERO; n];
    for k2 in 0..m {
        let y: Vec<C64> = (0..p)
            .map(|n1| subs[n1][k2] * twiddle((n1 * k2) as i64, n))
            .collect();
        for k1 in 0..p {
            let mut s = C64::ZERO;
            for (n1, &yn) in y.iter().enumerate() {
                s += yn * twiddle((n1 * k1) as i64, p);
            }
            out[k2 + m * k1] = s;
        }
    }
    out
}

/// Factor `n` into primes from `{2,3,5}` where possible; returns `None` otherwise.
pub fn small_factors(n: usize) -> Option<Vec<usize>> {
    let mut n = n;
    let mut out = Vec::new();
    for p in [2usize, 3, 5] {
        while n % p == 0 {
            out.push(p);
            n /= p;
        }
    }
    if n == 1 {
        Some(out)
    } else {
        None
    }
}
