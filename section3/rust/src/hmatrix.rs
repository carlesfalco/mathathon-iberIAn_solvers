use nalgebra::{DMatrix, DVector};
use std::ops::Range;
use std::time::Instant;

#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub seconds: f64,
    pub qr_calls: usize,
    pub dense_blocks: usize,
    pub low_rank_blocks: usize,
    pub split_blocks: usize,
    pub max_rank: usize,
}

#[derive(Debug)]
enum Block {
    Dense {
        rows: Range<usize>,
        cols: Range<usize>,
        a: DMatrix<f64>,
    },
    LowRank {
        rows: Range<usize>,
        cols: Range<usize>,
        u: DMatrix<f64>,
        vt: DMatrix<f64>,
    },
    Split {
        children: Vec<Block>,
    },
}

impl Block {
    fn rows(&self) -> Range<usize> {
        match self {
            Self::Dense { rows, .. } | Self::LowRank { rows, .. } => rows.clone(),
            Self::Split { children } => {
                let first = children.first().expect("a split has children").rows();
                let last = children.last().expect("a split has children").rows();
                first.start..last.end
            }
        }
    }

    fn cols(&self) -> Range<usize> {
        match self {
            Self::Dense { cols, .. } | Self::LowRank { cols, .. } => cols.clone(),
            Self::Split { children } => {
                let first = children.first().expect("a split has children").cols();
                let last = children.last().expect("a split has children").cols();
                first.start..last.end
            }
        }
    }

    fn stored_entries(&self) -> usize {
        match self {
            Self::Dense { a, .. } => a.len(),
            Self::LowRank { u, vt, .. } => u.len() + vt.len(),
            Self::Split { children } => children.iter().map(Self::stored_entries).sum(),
        }
    }

    fn apply(&self, x: &DVector<f64>, y: &mut DVector<f64>) {
        match self {
            Self::Dense { rows, cols, a } => {
                let xb = x.rows(cols.start, cols.len());
                let z = a * xb;
                for i in 0..rows.len() {
                    y[rows.start + i] += z[i];
                }
            }
            Self::LowRank { rows, cols, u, vt } => {
                let xb = x.rows(cols.start, cols.len());
                if vt.ncols() > 0 && vt.nrows() > 0 {
                    let z = u * (vt * xb);
                    for i in 0..rows.len() {
                        y[rows.start + i] += z[i];
                    }
                }
            }
            Self::Split { children } => {
                for child in children {
                    child.apply(x, y);
                }
            }
        }
    }
}

/// A block hierarchical matrix made by recursive column-pivoted-QR compression of a dense oracle.
#[derive(Debug)]
pub struct HMatrix {
    n: usize,
    root: Block,
    pub stats: BuildStats,
}

impl HMatrix {
    pub fn from_dense(a: &DMatrix<f64>, leaf: usize, eps: f64) -> Self {
        assert_eq!(a.nrows(), a.ncols());
        assert!(leaf > 0);
        assert!(eps > 0.0 && eps < 1.0);
        let n = a.nrows();
        let started = Instant::now();
        let mut stats = BuildStats::default();
        let root = compress(a, 0..n, 0..n, leaf, eps, &mut stats);
        stats.seconds = started.elapsed().as_secs_f64();
        Self { n, root, stats }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn stored_entries(&self) -> usize {
        self.root.stored_entries()
    }

    pub fn storage_ratio(&self) -> f64 {
        self.stored_entries() as f64 / (self.n * self.n) as f64
    }

    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(x.len(), self.n);
        let mut y = DVector::zeros(self.n);
        self.root.apply(x, &mut y);
        y
    }

    pub fn dense_block(&self) -> (usize, usize) {
        (self.root.rows().len(), self.root.cols().len())
    }
}

fn store_dense(a: &DMatrix<f64>, rows: Range<usize>, cols: Range<usize>, stats: &mut BuildStats) -> Block {
    let data = a.view((rows.start, cols.start), (rows.len(), cols.len())).into_owned();
    if data.norm() == 0.0 {
        stats.low_rank_blocks += 1;
        return Block::LowRank {
            u: DMatrix::zeros(rows.len(), 0),
            vt: DMatrix::zeros(0, cols.len()),
            rows,
            cols,
        };
    }
    stats.dense_blocks += 1;
    Block::Dense { rows, cols, a: data }
}

fn split_block(
    a: &DMatrix<f64>,
    rows: Range<usize>,
    cols: Range<usize>,
    leaf: usize,
    eps: f64,
    stats: &mut BuildStats,
) -> Block {
    if (rows.len() <= leaf && cols.len() <= leaf) || rows.len() == 1 || cols.len() == 1 {
        return store_dense(a, rows, cols, stats);
    }
    stats.split_blocks += 1;
    let rm = rows.start + rows.len() / 2;
    let cm = cols.start + cols.len() / 2;
    let row_parts = [rows.start..rm, rm..rows.end];
    let col_parts = [cols.start..cm, cm..cols.end];
    let mut children = Vec::with_capacity(4);
    for rp in row_parts {
        for cp in col_parts.clone() {
            children.push(compress(a, rp.clone(), cp.clone(), leaf, eps, stats));
        }
    }
    Block::Split { children }
}

fn compress(
    a: &DMatrix<f64>,
    rows: Range<usize>,
    cols: Range<usize>,
    leaf: usize,
    eps: f64,
    stats: &mut BuildStats,
) -> Block {
    // Diagonal blocks contain the singular/near-field entries.  Refine them
    // directly instead of paying for a QR that is expected to be full rank.
    if rows == cols {
        return split_block(a, rows, cols, leaf, eps, stats);
    }
    if rows.len() <= leaf || cols.len() <= leaf {
        return store_dense(a, rows, cols, stats);
    }

    let block = a.view((rows.start, cols.start), (rows.len(), cols.len())).into_owned();
    stats.qr_calls += 1;
    let (q, r, p) = block.col_piv_qr().unpack();
    let diagonal = r.diagonal();
    if diagonal.len() == 0 || diagonal[0] == 0.0 {
        stats.low_rank_blocks += 1;
        return Block::LowRank {
            u: DMatrix::zeros(rows.len(), 0),
            vt: DMatrix::zeros(0, cols.len()),
            rows,
            cols,
        };
    }
    let rank = diagonal.iter().filter(|&&v| v.abs() > eps * diagonal[0].abs()).count();
    let m = rows.len();
    let n = cols.len();
    stats.max_rank = stats.max_rank.max(rank);
    if rank > 0 && rank < m.min(n) && rank * (m + n) < m * n {
        // The crate computes A P = Q R.  Unpermute the retained R rows,
        // giving A ~ Q_r R_r P^{-1} in the original column ordering.
        let u = q.columns(0, rank).into_owned();
        let mut vt = r.rows(0, rank).into_owned();
        p.inv_permute_columns(&mut vt);
        stats.low_rank_blocks += 1;
        Block::LowRank { rows, cols, u, vt }
    } else {
        split_block(a, rows, cols, leaf, eps, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_matrix_is_free() {
        let a = DMatrix::zeros(64, 64);
        let h = HMatrix::from_dense(&a, 16, 1e-8);
        assert_eq!(h.stored_entries(), 0);
        assert_eq!(h.apply(&DVector::from_element(64, 1.0)), DVector::zeros(64));
    }

    #[test]
    fn dense_diagonal_is_preserved() {
        let a = DMatrix::identity(64, 64);
        let h = HMatrix::from_dense(&a, 8, 1e-10);
        let x = DVector::from_fn(64, |i, _| (i as f64 + 1.0).sin());
        let err = (&h.apply(&x) - &x).norm() / x.norm();
        assert!(err < 1e-14, "{err}");
    }

    #[test]
    fn separable_rank_one_is_reconstructed() {
        let a = DMatrix::from_fn(64, 64, |i, j| ((i + 1) as f64).sin() * ((j + 2) as f64).cos());
        let h = HMatrix::from_dense(&a, 16, 1e-10);
        let x = DVector::from_fn(64, |i, _| (0.31 * i as f64).cos());
        let exact = &a * &x;
        let err = (&h.apply(&x) - &exact).norm() / (a.norm() * x.norm());
        assert!(err < 1e-12, "{err}");
    }

    #[test]
    fn smooth_log_kernel_meets_tolerance() {
        let n = 256;
        let a = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                1.0
            } else {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                -(x - y).abs().ln() / (2.0 * std::f64::consts::PI)
            }
        });
        let h = HMatrix::from_dense(&a, 32, 1e-8);
        let x = DVector::from_fn(n, |i, _| (0.17 * i as f64).sin());
        let exact = &a * &x;
        let err = (&h.apply(&x) - &exact).norm() / exact.norm();
        assert!(err < 1e-7, "{err}");
    }
}
