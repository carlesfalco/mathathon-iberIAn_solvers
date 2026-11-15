//! Column interpolative decomposition by pivoted modified Gram-Schmidt.
//!
//! This is the Rust counterpart of
//! `updated/unified/core/compress.py::interp_decomp`, which uses
//! `scipy.linalg.qr(M, pivoting=True)` and stops at
//! `k = #{i : |R_ii| > eps * |R_00|}`. Pivoted MGS produces the same
//! `|R_ii|` sequence in exact arithmetic, so the same rank decision, and the
//! interpolation matrix `T = R11^{-1} R12` is the same least-squares solution
//! (it is unique once the skeleton set is fixed).

use crate::dft_butterfly::c64::C64;

pub struct IdResult {
    /// Local indices of the skeleton columns, in pivot order.
    pub skel: Vec<usize>,
    /// Local indices of the redundant columns, in increasing order.
    pub red: Vec<usize>,
    /// `T`, `skel.len() x red.len()`, row-major.
    pub t: Vec<C64>,
    /// Largest remaining residual column norm, relative to the largest initial
    /// column norm. Zero (to rounding) iff the returned rank is exact.
    pub resid: f64,
    /// The `|R_ii|` sequence, for reporting.
    pub pivots: Vec<f64>,
}

/// `M` is `nrows x ncols`, row-major. Returns `skel, red, T` with
/// `M[:, red] ~= M[:, skel] * T`.
pub fn interp_decomp(m: &[C64], nrows: usize, ncols: usize, eps: f64) -> IdResult {
    assert_eq!(m.len(), nrows * ncols);
    // residual copy of every column
    let mut res: Vec<Vec<C64>> = (0..ncols)
        .map(|j| (0..nrows).map(|i| m[i * ncols + j]).collect())
        .collect();
    let mut nrm: Vec<f64> = res
        .iter()
        .map(|col| col.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt())
        .collect();
    let nrm0 = nrm.iter().cloned().fold(0.0_f64, f64::max);

    let mut skel: Vec<usize> = Vec::new();
    let mut coef: Vec<Vec<C64>> = Vec::new(); // coef[k][j] = q_k^* M[:, j]
    let mut pivots: Vec<f64> = Vec::new();

    if nrm0 > 0.0 {
        loop {
            let mut best = usize::MAX;
            let mut bv = 0.0_f64;
            for j in 0..ncols {
                if !skel.contains(&j) && nrm[j] > bv {
                    bv = nrm[j];
                    best = j;
                }
            }
            if best == usize::MAX || !(bv > eps * nrm0) {
                break;
            }
            let q: Vec<C64> = res[best].iter().map(|z| z.scale(1.0 / bv)).collect();
            pivots.push(bv);
            skel.push(best);
            let mut row = vec![C64::ZERO; ncols];
            for j in 0..ncols {
                let mut cj = C64::ZERO;
                for i in 0..nrows {
                    cj += q[i].conj() * res[j][i];
                }
                row[j] = cj;
                for i in 0..nrows {
                    res[j][i] = res[j][i] - cj * q[i];
                }
                nrm[j] = res[j].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            }
            coef.push(row);
        }
    }

    let r = skel.len();
    let red: Vec<usize> = (0..ncols).filter(|j| !skel.contains(j)).collect();
    // T = R^{-1} C_red with R[k][i] = coef[k][skel[i]] (upper triangular).
    let mut t = vec![C64::ZERO; r * red.len()];
    for (jj, &j) in red.iter().enumerate() {
        let mut y = vec![C64::ZERO; r];
        for k in (0..r).rev() {
            let mut s = coef[k][j];
            for i in (k + 1)..r {
                s = s - coef[k][skel[i]] * y[i];
            }
            let d = coef[k][skel[k]];
            y[k] = if d.abs() > 0.0 {
                s.div(d)
            } else {
                C64::ZERO
            };
        }
        for (k, &yk) in y.iter().enumerate() {
            t[k * red.len() + jj] = yk;
        }
    }
    let resid = if nrm0 > 0.0 {
        red.iter().map(|&j| nrm[j] / nrm0).fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    IdResult {
        skel,
        red,
        t,
        resid,
        pivots,
    }
}
