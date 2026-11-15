//! The complementary-pairing interpolative butterfly factorization of the DFT,
//! and the invariants that pin it down.
//!
//! This is a faithful port of the construction in
//! `updated/unified/drivers/butterfly_id.py::butterfly_id`:
//!
//!   * one row tree at depth `l` and one column tree at depth `L - l` per stage,
//!   * stage 0:  column ID of `K(all rows, B)` for every column leaf `B`,
//!   * stage l -> l+1: candidate columns are the skeletons of the two parent
//!     tiles, and the ID of `K(child rows, candidates)` becomes a block row of `G_l`,
//!   * stage L: the final skeleton columns are evaluated exactly into `U`.
//!
//! Two things differ from the Python, both deliberate:
//!   1. the trees are the uniform dyadic trees of a virtual integer grid
//!      `0..n` (all leaves single points), so a level-`l` node is the contiguous
//!      block `[a * 2^(L-l), (a+1) * 2^(L-l))`.  Whether that block is a
//!      residue class of the *physical* DFT index is decided by the permutation
//!      `prow` / `pcol` passed in, which is exactly how
//!      `updated/unified/explore/e2_fft_replacements/fftswap.py::permuted_kernel`
//!      does it;
//!   2. no kernel function is called: the entries come from `twiddle` directly.
//!
//! Pass `prow = pcol = bitrev_perm(n)` for the residue-class (= FFT) ordering and
//! `prow = pcol = identity` for the natural ordering.  See `notes/` for what the
//! two give.

use crate::dft_butterfly::c64::{twiddle, C64};
use crate::dft_butterfly::fft::{bitrev_perm, log2_exact};
use crate::dft_butterfly::id::interp_decomp;

/// Compressed sparse row, the same factor type the Python driver uses (`scipy.sparse.csr_matrix`).
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    pub rowptr: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<C64>,
}

impl Csr {
    pub fn from_triplets(nrows: usize, ncols: usize, mut trips: Vec<(usize, usize, C64)>) -> Csr {
        trips.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        for w in trips.windows(2) {
            assert!(
                !(w[0].0 == w[1].0 && w[0].1 == w[1].1),
                "duplicate factor entry at ({}, {})",
                w[0].0,
                w[0].1
            );
        }
        let mut rowptr = vec![0usize; nrows + 1];
        for t in &trips {
            rowptr[t.0 + 1] += 1;
        }
        for i in 0..nrows {
            rowptr[i + 1] += rowptr[i];
        }
        Csr {
            nrows,
            ncols,
            rowptr,
            cols: trips.iter().map(|t| t.1).collect(),
            vals: trips.iter().map(|t| t.2).collect(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    pub fn matvec(&self, x: &[C64]) -> Vec<C64> {
        assert_eq!(x.len(), self.ncols, "matvec dimension mismatch");
        (0..self.nrows)
            .map(|i| {
                let mut s = C64::ZERO;
                for k in self.rowptr[i]..self.rowptr[i + 1] {
                    s += self.vals[k] * x[self.cols[k]];
                }
                s
            })
            .collect()
    }

    /// Entries that are exactly `1` cost no multiplication when applied.
    pub fn n_ones(&self) -> usize {
        self.vals.iter().filter(|v| v.is_one()).count()
    }

    pub fn max_unit_dev(&self) -> f64 {
        self.vals
            .iter()
            .map(|v| (v.abs() - 1.0).abs())
            .fold(0.0_f64, f64::max)
    }
}

pub struct Butterfly {
    pub n: usize,
    pub l: usize,
    pub w: Csr,
    pub g: Vec<Csr>,
    pub u: Csr,
    /// virtual row `u` is physical row `prow[u]`
    pub prow: Vec<usize>,
    /// virtual column `v` is physical column `pcol[v]`
    pub pcol: Vec<usize>,
    /// max skeleton size at each stage
    pub ranks: Vec<usize>,
    /// largest ID residual over all tiles, relative to the tile's largest column norm
    pub max_resid: f64,
    /// Kernel entries evaluated while building: `3 N^2 - 2 N` for this all-tile-row
    /// builder (stage 0 `N^2`, then `2N^2 - 2N`), i.e. `O(N^2)`.
    ///
    /// That is a property of *this* construction, not a lower bound for entry-based
    /// constructions: row-sampled variants (IDBF; Pang, Ho & Yang, arXiv:1809.10573,
    /// cited by `updated/unified/drivers/butterfly_id.py`) probe `O(1)` rows per tile
    /// and reduce construction to `O(N log N)`.
    pub kernel_entries: usize,
}

impl Butterfly {
    /// `nnz` of all `L + 2` factors, i.e. the storage count.
    pub fn nnz(&self) -> usize {
        self.w.nnz() + self.g.iter().map(|f| f.nnz()).sum::<usize>() + self.u.nnz()
    }

    /// Nonzeros that are not exactly `1`, i.e. the complex multiplications of one apply.
    pub fn apply_mults(&self) -> usize {
        self.w.nnz() - self.w.n_ones()
            + self.g.iter().map(|f| f.nnz() - f.n_ones()).sum::<usize>()
            + self.u.nnz()
            - self.u.n_ones()
    }

    /// `c = U G_{L-1} ... G_0 W g` on *virtual* indices. This is exactly what
    /// `IDButterfly.apply` does in the Python driver; the caller owns the permutation.
    pub fn apply_virtual(&self, g: &[C64]) -> Vec<C64> {
        let mut x = self.w.matvec(g);
        for f in &self.g {
            x = f.matvec(&x);
        }
        self.u.matvec(&x)
    }

    /// The physical operator: `f[prow[u]] = (apply_virtual g_v)[u]` with `g_v[v] = g[pcol[v]]`.
    ///
    /// The factorisation represents `K_virtual[u,v] = F[prow[u], pcol[v]]`, i.e.
    /// `K_virtual = P_row F P_col^T`, so `F = P_row^T K_virtual P_col`. This is the
    /// permutation handling of `updated/unified/showcase/models.py::check_dft`
    /// (`f[perm] = bf.apply(c.g[perm])`), and it is why the natural and the
    /// bit-reversed builds agree as operators even though their factors differ.
    pub fn apply(&self, g: &[C64]) -> Vec<C64> {
        let gv: Vec<C64> = self.pcol.iter().map(|&v| g[v]).collect();
        let fv = self.apply_virtual(&gv);
        let mut out = vec![C64::ZERO; self.n];
        for u in 0..self.n {
            out[self.prow[u]] = fv[u];
        }
        out
    }

    /// The dense `n x n` physical operator (column `j` = apply(e_j)).
    pub fn dense(&self) -> Vec<C64> {
        let n = self.n;
        let mut out = vec![C64::ZERO; n * n];
        for j in 0..n {
            let mut e = vec![C64::ZERO; n];
            e[j] = C64::ONE;
            let col = self.apply(&e);
            for i in 0..n {
                out[i * n + j] = col[i];
            }
        }
        out
    }
}

/// Build the factorisation for `K[u, v] = exp(+2 pi i * prow[u] * pcol[v] / n)`.
pub fn build(n: usize, prow: &[usize], pcol: &[usize], eps: f64) -> Butterfly {
    assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
    assert_eq!(prow.len(), n);
    assert_eq!(pcol.len(), n);
    let l = log2_exact(n);
    let kernel = |u: usize, v: usize| twiddle((prow[u] as i64) * (pcol[v] as i64), n);

    let row_range = |s: usize, a: usize| -> (usize, usize) {
        let w = n >> s;
        (a * w, (a + 1) * w)
    };
    // stage s pairs row depth s with column depth l - s: column boxes have width 2^s
    let col_ranges = |s: usize, b: usize| -> (usize, usize) {
        let w = 1usize << s;
        (b * w, (b + 1) * w)
    };
    let ncol_boxes = |s: usize| 1usize << (l - s);
    let tile = |s: usize, a: usize, b: usize| a * ncol_boxes(s) + b;

    let mut off: Vec<Vec<usize>> = vec![Vec::new(); l + 1];
    let mut size: Vec<usize> = vec![0; l + 1];
    let mut skel: Vec<Vec<Vec<usize>>> = vec![Vec::new(); l + 1];
    let mut ranks: Vec<usize> = vec![0; l + 1];
    let mut max_resid = 0.0_f64;
    let mut kernel_entries = 0usize;

    // ---- stage 0: tile (root, B) for every column leaf B = {b} -------------------
    let mut w_trips: Vec<(usize, usize, C64)> = Vec::new();
    let mut g_trips: Vec<Vec<(usize, usize, C64)>> = vec![Vec::new(); l];
    for b in 0..ncol_boxes(0) {
        let (c0, c1) = col_ranges(0, b);
        let cols: Vec<usize> = (c0..c1).collect();
        let rows: Vec<usize> = (0..n).collect();
        let mut m = vec![C64::ZERO; rows.len() * cols.len()];
        for (ii, &u) in rows.iter().enumerate() {
            for (jj, &v) in cols.iter().enumerate() {
                m[ii * cols.len() + jj] = kernel(u, v);
            }
        }
        kernel_entries += m.len();
        let res = interp_decomp(&m, rows.len(), cols.len(), eps);
        max_resid = max_resid.max(res.resid);
        let s = 0usize;
        let t = tile(s, 0, b);
        debug_assert_eq!(t, b);
        off[s].push(size[s]);
        size[s] += res.skel.len();
        ranks[s] = ranks[s].max(res.skel.len());
        skel[s].push(res.skel.iter().map(|&q| cols[q]).collect());
        // P = [I | T]: column slots in original order
        let r = res.skel.len();
        let mut p = vec![C64::ZERO; r * cols.len()];
        for (q, &sl) in res.skel.iter().enumerate() {
            p[q * cols.len() + sl] = C64::ONE;
        }
        for (qq, &rl) in res.red.iter().enumerate() {
            for q in 0..r {
                p[q * cols.len() + rl] = res.t[q * res.red.len() + qq];
            }
        }
        for q in 0..r {
            for (jj, _) in cols.iter().enumerate() {
                let v = p[q * cols.len() + jj];
                if v.re != 0.0 || v.im != 0.0 {
                    w_trips.push((off[s][t] + q, cols[jj], v));
                }
            }
        }
    }

    // ---- stages 1..=L ------------------------------------------------------------
    for s in 1..=l {
        let nrow_boxes = 1usize << s;
        let ncb = ncol_boxes(s);
        off[s] = Vec::with_capacity(nrow_boxes * ncb);
        skel[s] = Vec::with_capacity(nrow_boxes * ncb);
        for a in 0..nrow_boxes {
            for b in 0..ncb {
                let t = tile(s, a, b);
                off[s].push(size[s]);
                let (r0, r1) = row_range(s, a);
                let rows: Vec<usize> = (r0..r1).collect();
                // the two parent tiles at stage s-1: (a/2, 2b) and (a/2, 2b+1)
                let t1 = tile(s - 1, a / 2, 2 * b);
                let t2 = tile(s - 1, a / 2, 2 * b + 1);
                let n1 = skel[s - 1][t1].len();
                let n2 = skel[s - 1][t2].len();
                let mut cand: Vec<usize> = Vec::with_capacity(n1 + n2);
                cand.extend_from_slice(&skel[s - 1][t1]);
                cand.extend_from_slice(&skel[s - 1][t2]);
                let mut m = vec![C64::ZERO; rows.len() * cand.len()];
                for (ii, &u) in rows.iter().enumerate() {
                    for (jj, &v) in cand.iter().enumerate() {
                        m[ii * cand.len() + jj] = kernel(u, v);
                    }
                }
                kernel_entries += m.len();
                let res = interp_decomp(&m, rows.len(), cand.len(), eps);
                max_resid = max_resid.max(res.resid);
                size[s] += res.skel.len();
                ranks[s] = ranks[s].max(res.skel.len());
                skel[s].push(res.skel.iter().map(|&q| cand[q]).collect());

                let r = res.skel.len();
                let nc = cand.len();
                // P (r x nc) with P[:, skel] = I, P[:, red] = T
                let mut p = vec![C64::ZERO; r * nc];
                for (q, &sl) in res.skel.iter().enumerate() {
                    p[q * nc + sl] = C64::ONE;
                }
                for (qq, &rl) in res.red.iter().enumerate() {
                    for q in 0..r {
                        p[q * nc + rl] = res.t[q * res.red.len() + qq];
                    }
                }
                // place into G_{s-1}, splitting the columns by parent tile
                for q in 0..r {
                    for jj in 0..nc {
                        let v = p[q * nc + jj];
                        if v.re != 0.0 || v.im != 0.0 {
                            let (parent_tile, local) = if jj < n1 {
                                (t1, jj)
                            } else {
                                (t2, jj - n1)
                            };
                            g_trips[s - 1].push((
                                off[s][t] + q,
                                off[s - 1][parent_tile] + local,
                                v,
                            ));
                        }
                    }
                }
            }
        }
    }

    // ---- stage L: f_A = K(A, skel(A, root)) c_A for every single-row leaf A -------
    let mut u_trips: Vec<(usize, usize, C64)> = Vec::new();
    for a in 0..n {
        let t = tile(l, a, 0);
        let cols = &skel[l][t];
        for (q, &v) in cols.iter().enumerate() {
            u_trips.push((a, off[l][t] + q, kernel(a, v)));
        }
    }

    let w = Csr::from_triplets(size[0], n, w_trips);
    let mut g: Vec<Csr> = Vec::with_capacity(l);
    for s in 0..l {
        g.push(Csr::from_triplets(size[s + 1], size[s], g_trips[s].clone()));
    }
    let u = Csr::from_triplets(n, size[l], u_trips);

    Butterfly {
        n,
        l,
        w,
        g,
        u,
        prow: prow.to_vec(),
        pcol: pcol.to_vec(),
        ranks,
        max_resid,
        kernel_entries,
    }
}

/// Convenience wrappers used by the report and the tests.
pub fn build_bitrev(n: usize, eps: f64) -> Butterfly {
    let p = bitrev_perm(n);
    build(n, &p, &p, eps)
}

pub fn build_natural(n: usize, eps: f64) -> Butterfly {
    let p: Vec<usize> = (0..n).collect();
    build(n, &p, &p, eps)
}
