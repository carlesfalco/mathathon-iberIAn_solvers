//! The invariants that pin down the DFT / butterfly / FFT correspondence.
//!
//! `cargo test --release` runs everything. Sizes are kept small enough that the
//! build stays under a second; the same properties are tabulated at larger `n`
//! by `src/bin/report.rs`.

use unified_rs::dft_butterfly::butterfly::{build, build_bitrev, build_natural};
use unified_rs::dft_butterfly::c64::{twiddle, C64};
use unified_rs::dft_butterfly::fft::{
    bitrev_perm, dft_naive, fft_mixed_radix, fft_radix2, ifft_radix2, log2_exact, small_factors,
};

fn gen_g(n: usize) -> Vec<C64> {
    (0..n)
        .map(|j| {
            let t = j as f64;
            C64 {
                re: (0.7 * t).sin() + 0.5 * (0.31 * t).cos(),
                im: (1.1 * t).cos() - 0.25 * (0.53 * t).sin(),
            }
        })
        .collect()
}

fn l2(a: &[C64]) -> f64 {
    a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

fn rel(a: &[C64], b: &[C64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0_f64, f64::max)
        / l2(b).max(1e-300)
}

fn dft_matrix(n: usize) -> Vec<C64> {
    let mut m = vec![C64::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = twiddle((i * j) as i64, n);
        }
    }
    m
}

// ---------------------------------------------------------------------------
// 1. The three FFT routes agree.
// ---------------------------------------------------------------------------

#[test]
fn radix2_equals_naive_dft() {
    for l in 0..=7 {
        let n = 1usize << l;
        let g = gen_g(n);
        let a = fft_radix2(&g);
        let b = dft_naive(&g);
        assert!(rel(&a, &b) < 1e-13, "n = {n}: rel = {:.2e}", rel(&a, &b));
    }
}

#[test]
fn mixed_radix_equals_naive_dft() {
    let mut n = 1usize;
    while n <= 500 {
        if let Some(f) = small_factors(n) {
            let g = gen_g(n);
            let a = fft_mixed_radix(&g, &f);
            let b = dft_naive(&g);
            assert!(rel(&a, &b) < 1e-13, "n = {n} factors {f:?}: rel {:.2e}", rel(&a, &b));
        }
        n += 1;
    }
}

#[test]
fn radix2_and_mixed_radix_agree() {
    // radix-2 is only defined for power-of-two lengths; mixed-radix is checked
    // against the naive DFT at every n in `mixed_radix_equals_naive_dft`.
    for l in 3..=8 {
        let n = 1usize << l;
        let g = gen_g(n);
        let a = fft_radix2(&g);
        let b = fft_mixed_radix(&g, &small_factors(n).unwrap());
        assert!(rel(&a, &b) < 1e-13, "n = {n}: rel {:.2e}", rel(&a, &b));
    }
}

// ---------------------------------------------------------------------------
// 2. Standard DFT invariants.
// ---------------------------------------------------------------------------

#[test]
fn parseval() {
    for &n in &[8usize, 64, 256, 1024] {
        let g = gen_g(n);
        let f = fft_radix2(&g);
        let lhs = f.iter().map(|z| z.norm_sqr()).sum::<f64>();
        let rhs = n as f64 * g.iter().map(|z| z.norm_sqr()).sum::<f64>();
        assert!(
            (lhs - rhs).abs() / rhs < 1e-13,
            "n = {n}: {:e} vs {:e}",
            lhs,
            rhs
        );
    }
}

#[test]
fn roundtrip() {
    for &n in &[4usize, 16, 64, 256, 512] {
        let g = gen_g(n);
        let back = ifft_radix2(&fft_radix2(&g));
        assert!(rel(&back, &g) < 1e-13, "n = {n}: {:.2e}", rel(&back, &g));
    }
}

#[test]
fn linearity_and_conjugation_symmetry() {
    let n = 32;
    let a = gen_g(n);
    let b: Vec<C64> = (0..n).map(|j| a[(j * 7 + 3) % n].conj()).collect();
    let fa = fft_radix2(&a);
    let fb = fft_radix2(&b);
    let sum: Vec<C64> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
    assert!(rel(&fft_radix2(&sum), &{
        let mut v = fa.clone();
        for (x, y) in v.iter_mut().zip(&fb) {
            *x = *x + *y;
        }
        v
    }) < 1e-13);
    // real input => conjugate-symmetric output about index 0 mod n
    let real: Vec<C64> = (0..n).map(|j| C64 { re: (0.3 * j as f64).sin(), im: 0.0 }).collect();
    let fr = fft_radix2(&real);
    for k in 1..n {
        assert!((fr[n - k] - fr[k].conj()).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// 3. The structural theorem: complementary tiles of the bit-reversed DFT are rank one.
// ---------------------------------------------------------------------------

/// Max residual of a rank-one least-squares fit of the tile `exp(2 pi i j k / n)`,
/// `j in rows`, `k in cols`, relative to the largest column norm.
fn tile_rank1_residual(rows: &[usize], cols: &[usize], n: usize) -> f64 {
    let ncols = cols.len();
    let mut m = vec![C64::ZERO; rows.len() * ncols];
    for (i, &j) in rows.iter().enumerate() {
        for (q, &k) in cols.iter().enumerate() {
            m[i * ncols + q] = twiddle((j * k) as i64, n);
        }
    }
    // reference column: the largest-norm one
    let mut best = 0usize;
    let mut bv = -1.0;
    for q in 0..ncols {
        let v: f64 = (0..rows.len()).map(|i| m[i * ncols + q].norm_sqr()).sum();
        if v > bv {
            bv = v;
            best = q;
        }
    }
    let r: Vec<C64> = (0..rows.len()).map(|i| m[i * ncols + best]).collect();
    let nn: f64 = r.iter().map(|z| z.norm_sqr()).sum();
    let mut worst = 0.0_f64;
    for q in 0..ncols {
        let c = (0..rows.len())
            .map(|i| r[i].conj() * m[i * ncols + q])
            .fold(C64::ZERO, |a, b| a + b);
        let t = c.div(C64 { re: nn, im: 0.0 });
        for i in 0..rows.len() {
            let resid = m[i * ncols + q] - r[i] * t;
            worst = worst.max(resid.abs());
        }
    }
    worst / nn.sqrt()
}

#[test]
fn bitreversed_tiles_are_exactly_rank_one() {
    for &n in &[8usize, 16, 32, 64] {
        let l = log2_exact(n);
        let p = bitrev_perm(n);
        let mut worst = 0.0_f64;
        for s in 0..=l {
            let nrow_boxes = 1usize << s;
            let ncol_boxes = 1usize << (l - s);
            for a in 0..nrow_boxes {
                let w = n >> s;
                let ru = &p[a * w..(a + 1) * w];
                for b in 0..ncol_boxes {
                    let wc = 1usize << s;
                    let ku = &p[b * wc..(b + 1) * wc];
                    worst = worst.max(tile_rank1_residual(ru, ku, n));
                }
            }
        }
        assert!(
            worst < 1e-12,
            "n = {n}: worst rank-one residual {worst:.2e}"
        );
    }
}

#[test]
fn natural_order_tiles_are_not_rank_one() {
    // Falsifier for the claim: without the residue-class ordering the middle tiles
    // are full rank, so the same assertion must fail.
    let n = 32;
    let l = log2_exact(n);
    let s = l / 2;
    let w = n >> s;
    let rows: Vec<usize> = (0..w).collect();
    let cols: Vec<usize> = (0..(1usize << s)).collect();
    let r = tile_rank1_residual(&rows, &cols, n);
    assert!(r > 1e-2, "expected a non-rank-one tile, residual {r:.2e}");
}

#[test]
fn bitreversed_boxes_are_residue_classes() {
    // What "bit-reversed order" means concretely: the depth-s box numbered `a` is a
    // residue class mod 2^s, with class index rev_s(a) (bit reversal on s bits, not
    // on L bits -- the two differ, which is an easy off-by-a-permutation).
    let rev = |a: usize, s: usize| -> usize {
        let mut v = a;
        let mut r = 0usize;
        for _ in 0..s {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        r
    };
    for &n in &[16usize, 64, 128] {
        let l = log2_exact(n);
        let p = bitrev_perm(n);
        for s in 0..=l {
            let w = n >> s;
            for a in 0..(1usize << s) {
                let set = &p[a * w..(a + 1) * w];
                let class = rev(a, s);
                for &j in set {
                    assert_eq!(
                        j % (1usize << s),
                        class,
                        "n {n} s {s} a {a}: j = {j}, class {class}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The factorisation reproduces the DFT, with the counted storage.
// ---------------------------------------------------------------------------

#[test]
fn bitrev_butterfly_is_the_dft() {
    for &n in &[8usize, 16, 32, 64, 128, 256] {
        let l = log2_exact(n);
        let bf = build_bitrev(n, 1e-12);
        assert!(
            bf.ranks.iter().all(|&r| r == 1),
            "n = {n}: ranks {:?}",
            bf.ranks
        );
        assert_eq!(bf.nnz(), 2 * n * l + 2 * n, "n = {n}: nnz {}", bf.nnz());
        assert!(bf.max_resid < 1e-12, "n = {n}: resid {:.2e}", bf.max_resid);
        for f in &bf.g {
            assert!(f.max_unit_dev() < 1e-12, "n = {n}: a G_l entry is not unimodular");
        }
        let g = gen_g(n);
        let f = bf.apply(&g);
        let r = dft_naive(&g);
        assert!(
            rel(&f, &r) < 1e-12,
            "n = {n}: apply error {:.2e}",
            rel(&f, &r)
        );
    }
}

#[test]
fn bitrev_butterfly_dense_matrix_is_the_dft_matrix() {
    for &n in &[8usize, 16, 32] {
        let bf = build_bitrev(n, 1e-13);
        let d = bf.dense();
        let m = dft_matrix(n);
        let e = d
            .iter()
            .zip(&m)
            .map(|(x, y)| (*x - *y).abs())
            .fold(0.0_f64, f64::max);
        assert!(e < 1e-12, "n = {n}: max entry error {e:.2e}");
    }
}

#[test]
fn natural_butterfly_is_exact_but_stores_about_dense() {
    // n >= 16 so that a middle tile is at least 4 x 4 and a rank above 2 is possible.
    for &n in &[16usize, 32, 64] {
        let l = log2_exact(n);
        let bf = build_natural(n, 1e-12);
        // still an exact DFT ...
        let g = gen_g(n);
        assert!(rel(&bf.apply(&g), &dft_naive(&g)) < 1e-12, "n = {n}");
        // ... but with no compression: storage is in the dense ballpark
        assert!(
            bf.nnz() > n * n,
            "n = {n}: nnz {} (expected dense-ish), 2nL+2n = {}",
            bf.nnz(),
            2 * n * l + 2 * n
        );
        assert!(
            bf.ranks.iter().any(|&r| r > 2),
            "n = {n}: expected ranks above 2, got {:?}",
            bf.ranks
        );
    }
}

#[test]
fn partial_permutations_do_not_give_rank_one() {
    // Reversing only one tree is the standard mistake; check it fails.
    let n = 64;
    let p = bitrev_perm(n);
    let ident: Vec<usize> = (0..n).collect();
    for (rp, cp, tag) in [(&p, &ident, "rows only"), (&ident, &p, "cols only")] {
        let bf = build(n, rp, cp, 1e-12);
        assert!(
            bf.ranks.iter().any(|&r| r > 1),
            "{tag}: expected a rank above 1, got {:?}",
            bf.ranks
        );
    }
}

#[test]
fn kernel_entry_count_is_quadratic() {
    // This builder evaluates every entry of every tile *on its candidate columns*:
    // stage 0 is n tiles x (n rows x 1 column) = n^2; stage s >= 1 is n tiles x
    // (n/2^s rows x 2 candidates) in the rank-one case, summing to 2n^2 - 2n.
    // Total 3n^2 - 2n = O(n^2).
    //
    // This is a property of *this* all-tile-row construction, not a lower bound for
    // entry-based construction: row-sampled IDBF (Pang-Ho-Yang, arXiv:1809.10573,
    // cited in butterfly_id.py) probes O(1) rows per tile and is O(n log n) to build.
    for &n in &[16usize, 64, 256] {
        let bf = build_bitrev(n, 1e-12);
        assert_eq!(bf.kernel_entries, 3 * n * n - 2 * n, "n = {n}");
    }
}
