//! Representative Rust port of `updated/unified/{core,drivers}` (Python: 851 physical / 351 code lines).
//!
//! Scope, deliberately bounded: uniform 2^d-ary Tree, weak/strong admissibility, one column-ID by
//! pivoted Householder QR, exact and proxy compressors, the recursive-skeletonization factorization
//! with solve, and the 1-D interpolative butterfly. No crate dependencies: the QR/LU LAPACK calls of
//! the Python tree are replaced by hand-written pivoted QR (Businger-Golub) and partial-pivot LU, which
//! is exactly the substitution a production port has to decide about.
//!
//! `check`:  reproduce the numbers in `golden.txt` (produced by tools/golden.py from the Python tree).
//! `bench`:  time the same cases.
//!
//! Numerics are generic over `Scalar` (f64 / C64) because the Python code is dtype-polymorphic through
//! NumPy; the trait costs one bound per numeric function and no change to the algorithm bodies.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Instant;

const PI: f64 = std::f64::consts::PI;

// ============================== scalar ==============================================================
trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + std::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    fn abs(self) -> f64;
    fn conj(self) -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn abs(self) -> f64 {
        self.abs()
    }
    fn conj(self) -> Self {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        C64::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Div for C64 {
    type Output = C64;
    fn div(self, o: C64) -> C64 {
        let d = o.re * o.re + o.im * o.im;
        C64::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
}
impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

impl Scalar for C64 {
    fn zero() -> Self {
        C64::new(0.0, 0.0)
    }
    fn one() -> Self {
        C64::new(1.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        C64::new(x, 0.0)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn conj(self) -> Self {
        C64::new(self.re, -self.im)
    }
}

// ============================== dense matrix ========================================================
#[derive(Clone, Debug)]
struct Mat<T> {
    r: usize,
    c: usize,
    a: Vec<T>,
}

impl<T: Scalar> Mat<T> {
    fn zeros(r: usize, c: usize) -> Self {
        Mat { r, c, a: vec![T::zero(); r * c] }
    }
    fn from_rows(r: usize, c: usize, a: Vec<T>) -> Self {
        assert_eq!(a.len(), r * c);
        Mat { r, c, a }
    }
    fn at(&self, i: usize, j: usize) -> T {
        self.a[i * self.c + j]
    }
    fn set(&mut self, i: usize, j: usize, v: T) {
        self.a[i * self.c + j] = v;
    }
    fn add_at(&mut self, i: usize, j: usize, v: T) {
        let k = i * self.c + j;
        self.a[k] = self.a[k] + v;
    }
    /// NumPy `M[np.ix_(rows, cols)]`: a copy in the given index order.
    fn ix(&self, rows: &[usize], cols: &[usize]) -> Mat<T> {
        let mut out = Mat::zeros(rows.len(), cols.len());
        for (ii, &i) in rows.iter().enumerate() {
            for (jj, &j) in cols.iter().enumerate() {
                out.set(ii, jj, self.at(i, j));
            }
        }
        out
    }
    fn transpose(&self) -> Mat<T> {
        let mut out = Mat::zeros(self.c, self.r);
        for i in 0..self.r {
            for j in 0..self.c {
                out.set(j, i, self.at(i, j));
            }
        }
        out
    }
    fn vstack(blocks: &[Mat<T>]) -> Mat<T> {
        let c = blocks[0].c;
        let r: usize = blocks.iter().map(|b| b.r).sum();
        let mut out = Mat::zeros(r, c);
        let mut off = 0;
        for b in blocks {
            assert_eq!(b.c, c);
            for i in 0..b.r {
                for j in 0..b.c {
                    out.set(off + i, j, b.at(i, j));
                }
            }
            off += b.r;
        }
        out
    }
}

fn mmul<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    assert_eq!(a.c, b.r);
    let mut out = Mat::zeros(a.r, b.c);
    for i in 0..a.r {
        for k in 0..a.c {
            let aik = a.at(i, k);
            if aik == T::zero() {
                continue;
            }
            for j in 0..b.c {
                let v = out.at(i, j) + aik * b.at(k, j);
                out.set(i, j, v);
            }
        }
    }
    out
}

fn swap_rows<T: Scalar>(m: &mut Mat<T>, i: usize, j: usize) {
    if i == j {
        return;
    }
    for k in 0..m.c {
        let t = m.at(i, k);
        m.set(i, k, m.at(j, k));
        m.set(j, k, t);
    }
}

fn swap_cols<T: Scalar>(m: &mut Mat<T>, i: usize, j: usize) {
    if i == j {
        return;
    }
    for k in 0..m.r {
        let t = m.at(k, i);
        m.set(k, i, m.at(k, j));
        m.set(k, j, t);
    }
}

// ============================== pivoted QR =========================================================
// LAPACK dgeqp3 / zgeqp3 semantics at the level the drivers use: R (upper triangle of the working
// array), the column permutation, and diagonal magnitudes in decreasing order by construction.
struct Qr<T> {
    r: Mat<T>,       // m x n; R = upper triangle (R[k, j], j >= k)
    piv: Vec<usize>, // column permutation: column k of R is column piv[k] of the input
}

fn qr_pivot<T: Scalar>(m: &Mat<T>) -> Qr<T> {
    let (mm, n) = (m.r, m.c);
    let mut a = m.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    let minmn = mm.min(n);
    let mut colnrm: Vec<f64> = (0..n)
        .map(|j| (0..mm).map(|i| a.at(i, j).abs().powi(2)).sum())
        .collect();
    for k in 0..minmn {
        // Businger-Golub pivot: largest remaining column norm, first index on ties.
        let mut p = k;
        let mut best = colnrm[k];
        for j in (k + 1)..n {
            if colnrm[j] > best {
                best = colnrm[j];
                p = j;
            }
        }
        if p != k {
            swap_cols(&mut a, k, p);
            piv.swap(k, p);
            colnrm.swap(k, p);
        }
        // Householder reflector annihilating a[k+1.., k], with the complex phase that avoids cancellation.
        let len = mm - k;
        let mut v: Vec<T> = (0..len).map(|i| a.at(k + i, k)).collect();
        let nrm2: f64 = v.iter().map(|x| x.abs().powi(2)).sum();
        let nrm = nrm2.sqrt();
        if nrm == 0.0 {
            continue;
        }
        let x0 = v[0];
        let phase = if x0.abs() == 0.0 { T::one() } else { x0 / T::from_f64(x0.abs()) };
        let alpha = -(phase * T::from_f64(nrm));
        v[0] = x0 - alpha;
        let vn2: f64 = v.iter().map(|x| x.abs().powi(2)).sum();
        if vn2 > 0.0 {
            for j in k..n {
                let mut t = T::zero();
                for i in 0..len {
                    t = t + v[i].conj() * a.at(k + i, j);
                }
                let f = t * T::from_f64(2.0 / vn2);
                for i in 0..len {
                    let val = a.at(k + i, j) - v[i] * f;
                    a.set(k + i, j, val);
                }
            }
        }
        a.set(k, k, alpha);
        for j in (k + 1)..n {
            let d = a.at(k, j).abs();
            colnrm[j] = (colnrm[j] - d * d).max(0.0);
        }
    }
    Qr { r: a, piv }
}

/// ID: M[:, red] ~= M[:, skel] @ T.  Mirrors core/compress.py::interp_decomp, including the stopping
/// rule `k = #{ i : |R_ii| > eps*|R_00| }` (a *count*, not a first-crossing search).
struct IdResult<T> {
    skel: Vec<usize>,
    red: Vec<usize>,
    t: Mat<T>,
}

fn interp_decomp<T: Scalar>(m: &Mat<T>, eps: f64) -> IdResult<T> {
    let (mm, n) = (m.r, m.c);
    if n == 0 {
        return IdResult { skel: vec![], red: vec![], t: Mat::zeros(0, 0) };
    }
    if mm == 0 {
        return IdResult { skel: vec![], red: (0..n).collect(), t: Mat::zeros(0, n) };
    }
    let q = qr_pivot(m);
    let dmax = q.r.at(0, 0).abs();
    let minmn = mm.min(n);
    let k = if dmax > 0.0 {
        (0..minmn).filter(|&i| q.r.at(i, i).abs() > eps * dmax).count()
    } else {
        0
    };
    let skel: Vec<usize> = q.piv[..k].to_vec();
    let red: Vec<usize> = q.piv[k..].to_vec();
    let mut t = Mat::zeros(k, n - k);
    for i in (0..k).rev() {
        for j in k..n {
            let mut s = q.r.at(i, j);
            for l in (i + 1)..k {
                s = s - q.r.at(i, l) * t.at(l, j - k);
            }
            t.set(i, j - k, s / q.r.at(i, i));
        }
    }
    IdResult { skel, red, t }
}

// ============================== LU (partial pivoting) ==============================================
struct Lu<T> {
    a: Mat<T>,
    piv: Vec<usize>,
    n: usize,
}

fn lu_factor<T: Scalar>(m: &Mat<T>) -> Lu<T> {
    let n = m.r;
    assert_eq!(n, m.c, "lu_factor expects a square block");
    let mut a = m.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        let mut best = a.at(k, k).abs();
        for i in (k + 1)..n {
            if a.at(i, k).abs() > best {
                best = a.at(i, k).abs();
                p = i;
            }
        }
        if p != k {
            swap_rows(&mut a, k, p);
            piv[k] = p; // LAPACK ipiv convention: the row exchanged with k at step k
        }
        let akk = a.at(k, k);
        if akk.abs() == 0.0 {
            continue;
        }
        for i in (k + 1)..n {
            let f = a.at(i, k) / akk;
            a.set(i, k, f);
            for j in (k + 1)..n {
                let v = a.at(i, j) - f * a.at(k, j);
                a.set(i, j, v);
            }
        }
    }
    Lu { a, piv, n }
}

fn lu_solve<T: Scalar>(lu: &Lu<T>, b: &Mat<T>) -> Mat<T> {
    let n = lu.n;
    assert_eq!(b.r, n);
    let k = b.c;
    let mut x = b.clone();
    for i in 0..n {
        if lu.piv[i] != i {
            swap_rows(&mut x, i, lu.piv[i]);
        }
    }
    for j in 0..k {
        for i in 0..n {
            let mut s = x.at(i, j);
            for l in 0..i {
                s = s - lu.a.at(i, l) * x.at(l, j);
            }
            x.set(i, j, s);
        }
        for i in (0..n).rev() {
            let mut s = x.at(i, j);
            for l in (i + 1)..n {
                s = s - lu.a.at(i, l) * x.at(l, j);
            }
            x.set(i, j, s / lu.a.at(i, i));
        }
    }
    x
}

// ============================== tree ===============================================================
#[derive(Clone, Debug)]
struct Node {
    level: usize,
    coord: Vec<usize>,
    center: Vec<f64>,
    half_width: f64,
    idx: Vec<usize>,
    parent: Option<usize>,
    children: Vec<usize>,
}

struct Tree {
    n: usize,
    d: usize,
    size: f64,
    lo: Vec<f64>,
    depth: usize,
    levels: Vec<Vec<usize>>,
    nodes: Vec<Node>,
    by_key: HashMap<(usize, Vec<usize>), usize>,
    points: Vec<Vec<f64>>,
}

impl Tree {
    fn new(points: Vec<Vec<f64>>, depth: Option<usize>, max_leaf: usize) -> Tree {
        let n = points.len();
        let d = points[0].len();
        let mut lo = vec![f64::INFINITY; d];
        let mut hi = vec![f64::NEG_INFINITY; d];
        for p in &points {
            for k in 0..d {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let mut size = hi.iter().zip(&lo).map(|(h, l)| h - l).fold(0.0, f64::max) * (1.0 + 1e-9);
        if size == 0.0 {
            size = 1.0;
        }
        for k in 0..d {
            lo[k] -= 1e-12;
        }
        let mut tree = Tree {
            n,
            d,
            size,
            lo,
            depth: 0,
            levels: vec![],
            nodes: vec![],
            by_key: HashMap::new(),
            points,
        };
        let depth = match depth {
            Some(v) => v,
            None => {
                let mut dep = 0;
                while dep < 30 {
                    let mut seen: HashSet<Vec<usize>> = HashSet::new();
                    for c in tree.coords(dep) {
                        seen.insert(c);
                    }
                    if (n as f64) / (seen.len() as f64) <= max_leaf as f64 {
                        break;
                    }
                    dep += 1;
                }
                dep
            }
        };
        tree.depth = depth;
        for l in 0..=depth {
            let w = tree.size / (1u64 << l) as f64;
            let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
            for (i, c) in tree.coords(l).into_iter().enumerate() {
                groups.entry(c).or_default().push(i);
            }
            let mut level: Vec<usize> = vec![];
            for (coord, idx) in groups {
                let center: Vec<f64> =
                    (0..d).map(|k| tree.lo[k] + (coord[k] as f64 + 0.5) * w).collect();
                let id = tree.nodes.len();
                let parent = if l > 0 {
                    let pc: Vec<usize> = coord.iter().map(|&c| c / 2).collect();
                    let pid = tree.by_key[&(l - 1, pc)];
                    tree.nodes[pid].children.push(id);
                    Some(pid)
                } else {
                    None
                };
                tree.nodes.push(Node {
                    level: l,
                    coord: coord.clone(),
                    center,
                    half_width: w / 2.0,
                    idx,
                    parent,
                    children: vec![],
                });
                tree.by_key.insert((l, coord), id);
                level.push(id);
            }
            tree.levels.push(level);
        }
        tree
    }

    fn coords(&self, l: usize) -> Vec<Vec<usize>> {
        let w = self.size / (1u64 << l) as f64;
        let cap = (1u64 << l) - 1;
        (0..self.n)
            .map(|i| {
                (0..self.d)
                    .map(|k| {
                        let c = ((self.points[i][k] - self.lo[k]) / w).floor() as i64;
                        (c.max(0) as u64).min(cap) as usize
                    })
                    .collect()
            })
            .collect()
    }

    fn neighbors(&self, node_id: usize) -> Vec<usize> {
        let node = &self.nodes[node_id];
        let mut out = vec![];
        let mut offs = vec![0usize; self.d];
        loop {
            let mut coord: Vec<usize> = Vec::with_capacity(self.d);
            let mut ok = true;
            for k in 0..self.d {
                let c = node.coord[k] as i64 + offs[k] as i64 - 1;
                if c < 0 {
                    ok = false;
                    break;
                }
                coord.push(c as usize);
            }
            if ok && coord != node.coord {
                if let Some(&nb) = self.by_key.get(&(node.level, coord)) {
                    out.push(nb);
                }
            }
            // odometer, last coordinate fastest (matches np.ndindex)
            let mut k = self.d;
            loop {
                if k == 0 {
                    return out;
                }
                k -= 1;
                offs[k] += 1;
                if offs[k] < 3 {
                    break;
                }
                offs[k] = 0;
            }
        }
    }

    fn interaction_list(&self, node_id: usize) -> Vec<usize> {
        let node = &self.nodes[node_id];
        let Some(parent) = node.parent else { return vec![] };
        let mut cands: Vec<usize> = self.nodes[parent].children.clone();
        for nb in self.neighbors(parent) {
            cands.extend(self.nodes[nb].children.iter().copied());
        }
        cands
            .into_iter()
            .filter(|&c| {
                let o = &self.nodes[c];
                if c == node_id {
                    return false;
                }
                (0..self.d)
                    .map(|k| (o.coord[k] as i64 - node.coord[k] as i64).abs())
                    .max()
                    .unwrap_or(0)
                    > 1
            })
            .collect()
    }
}

// ============================== admissibility ======================================================
#[derive(Clone, Copy, PartialEq, Debug)]
enum Adm {
    Weak,
    Strong,
}

impl Adm {
    fn near_nodes(&self, tree: &Tree, node_id: usize) -> Vec<usize> {
        match self {
            Adm::Weak => vec![],
            Adm::Strong => tree.neighbors(node_id),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Adm::Weak => "weak",
            Adm::Strong => "strong",
        }
    }
}

// ============================== kernels ============================================================
fn gather(pts: &[Vec<f64>], idx: &[usize]) -> Vec<Vec<f64>> {
    idx.iter().map(|&i| pts[i].clone()).collect()
}

fn dist(x: &[Vec<f64>], y: &[Vec<f64>]) -> Mat<f64> {
    let d = x[0].len();
    let mut out = Mat::zeros(x.len(), y.len());
    for i in 0..x.len() {
        for j in 0..y.len() {
            let mut s = 0.0;
            for k in 0..d {
                let t = x[i][k] - y[j][k];
                s += t * t;
            }
            out.set(i, j, s.sqrt());
        }
    }
    out
}

fn laplace2d(x: &[Vec<f64>], y: &[Vec<f64>]) -> Mat<f64> {
    let r = dist(x, y);
    let mut k = Mat::zeros(r.r, r.c);
    for i in 0..r.r {
        for j in 0..r.c {
            let v = r.at(i, j);
            k.set(i, j, if v == 0.0 { 0.0 } else { -v.ln() / (2.0 * PI) });
        }
    }
    k
}

fn lap1d(x: &[f64], y: &[f64]) -> Mat<f64> {
    let mut k = Mat::zeros(x.len(), y.len());
    for i in 0..x.len() {
        for j in 0..y.len() {
            let r = (x[i] - y[j]).abs();
            k.set(i, j, if r == 0.0 { 0.0 } else { -0.5 * r });
        }
    }
    k
}

fn fio1d(t: f64, x: &[f64], xi: &[f64]) -> Mat<C64> {
    let mut k = Mat::zeros(x.len(), xi.len());
    for i in 0..x.len() {
        let cx = (2.0 + (2.0 * PI * x[i]).sin()) / 16.0;
        for j in 0..xi.len() {
            let ph = 2.0 * PI * (x[i] * xi[j] + t * cx * xi[j].abs());
            k.set(i, j, C64::new(ph.cos(), ph.sin()));
        }
    }
    k
}

/// Starfish arc-length column weight of the P1 problem, at the angles of the column argument.
fn starfish_weight(q: &[Vec<f64>], n_total: usize) -> Vec<f64> {
    q.iter()
        .map(|p| {
            let th = p[1].atan2(p[0]);
            let r = 1.0 + 0.3 * (5.0 * th).cos();
            let dr = -1.5 * (5.0 * th).sin();
            (r * r + dr * dr).sqrt() * 2.0 * PI / n_total as f64
        })
        .collect()
}

// ============================== compressors ========================================================
type Kern<T> = Box<dyn Fn(&[Vec<f64>], &[Vec<f64>]) -> Mat<T>>;

/// One compressor interface, with the working matrix explicit.
///
/// Python does `setattr(comp, "A", W)` on a `copy.copy` of the compressor inside
/// `drivers/rskel.py::_skeletonize`, so the constructor argument `A` is dead there and the *current*
/// Schur-complemented matrix reaches the compressor through an undeclared attribute. Passing `w`
/// restores what the paper means and removes the copy per box.
enum Comp<T: Scalar> {
    Exact { eps: f64 },
    Proxy { kernel: Kern<T>, points: Vec<Vec<f64>>, eps: f64, n_proxy: usize, radius_factor: f64 },
}

impl<T: Scalar> Comp<T> {
    fn eps(&self) -> f64 {
        match self {
            Comp::Exact { eps } => *eps,
            Comp::Proxy { eps, .. } => *eps,
        }
    }

    fn proxy_points(&self, node: &Node, d: usize) -> Vec<Vec<f64>> {
        match self {
            Comp::Exact { .. } => vec![],
            Comp::Proxy { n_proxy, radius_factor, .. } => {
                let r = radius_factor * node.half_width;
                if d == 1 {
                    vec![vec![node.center[0] - r], vec![node.center[0] + r]]
                } else {
                    (0..*n_proxy)
                        .map(|m| {
                            let th = 2.0 * PI * (m as f64) / (*n_proxy as f64);
                            vec![node.center[0] + r * th.cos(), node.center[1] + r * th.sin()]
                        })
                        .collect()
                }
            }
        }
    }

    /// `(box_idx, far_idx, node) -> IDResult`; rows of the ID matrix stack box->far and far->box.
    fn call(&self, w: &Mat<T>, box_idx: &[usize], far_idx: &[usize], node: &Node, q_extra: &[usize]) -> IdResult<T> {
        match self {
            Comp::Exact { eps } => {
                let m = Mat::vstack(&[w.ix(far_idx, box_idx), w.ix(box_idx, far_idx).transpose()]);
                interp_decomp(&m, *eps)
            }
            Comp::Proxy { kernel, points, eps, radius_factor, .. } => {
                let d = points[0].len();
                let r = radius_factor * node.half_width;
                let inside: Vec<usize> = far_idx
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let mut s = 0.0;
                        for k in 0..d {
                            let t = points[i][k] - node.center[k];
                            s += t * t;
                        }
                        s.sqrt() < r
                    })
                    .collect();
                let pp = self.proxy_points(node, d);
                let xb = gather(points, box_idx);
                let mut blocks = vec![kernel(&pp, &xb), kernel(&xb, &pp).transpose()];
                // far unknowns inside the proxy surface, plus FMM-LU's "modified far" set, stay explicit
                let expl: Vec<usize> = if q_extra.is_empty() {
                    inside.clone()
                } else {
                    let mut s: HashSet<usize> = inside.iter().copied().collect();
                    s.extend(q_extra.iter().copied());
                    let mut v: Vec<usize> = s.into_iter().collect();
                    v.sort_unstable();
                    v
                };
                if !expl.is_empty() {
                    blocks.push(w.ix(&expl, box_idx));
                    blocks.push(w.ix(box_idx, &expl).transpose());
                }
                interp_decomp(&Mat::vstack(&blocks), *eps)
            }
        }
    }
}

// ============================== recursive skeletonization ==========================================
#[derive(Clone, Copy, Debug, Default)]
struct LevelStat {
    boxes: usize,
    skel_max: usize,
    skel_total: usize,
    eliminated: usize,
}

struct Step<T> {
    skel: Vec<usize>,
    red: Vec<usize>,
    sn: Vec<usize>,
    t: Mat<T>,
    lu: Lu<T>,
    x_snr: Mat<T>,
    x_rsn: Mat<T>,
}

struct Factor<T> {
    n: usize,
    steps: Vec<Step<T>>,
    root: Vec<usize>,
    root_lu: Option<Lu<T>>,
    levels: BTreeMap<usize, LevelStat>,
    n_steps: usize,
}

fn rskel_factor<T: Scalar>(
    a: &Mat<T>,
    tree: &Tree,
    adm: Adm,
    comp: &Comp<T>,
    track_modified: bool,
) -> Factor<T> {
    let n = a.r;
    let mut w = a.clone();
    let mut active = vec![true; n];
    let mut steps: Vec<Step<T>> = vec![];
    let mut levels: BTreeMap<usize, LevelStat> = BTreeMap::new();
    for l in (1..=tree.depth).rev() {
        let nodes = tree.levels[l].clone();
        let mut modified: Vec<HashSet<usize>> = vec![HashSet::new(); tree.nodes.len()];
        let mut skel_sizes: Vec<usize> = vec![];
        let mut n_elim = 0usize;
        for &b_id in &nodes {
            let box_idx: Vec<usize> = tree.nodes[b_id].idx.iter().copied().filter(|&i| active[i]).collect();
            let near_nodes = adm.near_nodes(tree, b_id);
            let mut near: Vec<usize> = vec![];
            for &nb in &near_nodes {
                near.extend(tree.nodes[nb].idx.iter().copied().filter(|&i| active[i]));
            }
            let mut mask = active.clone();
            for &i in &box_idx {
                mask[i] = false;
            }
            for &i in &near {
                mask[i] = false;
            }
            let far: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            if box_idx.is_empty() || far.is_empty() {
                skel_sizes.push(box_idx.len());
                continue;
            }
            let mut q_extra: Vec<usize> = vec![];
            if track_modified {
                let near_set: HashSet<usize> = near_nodes.iter().copied().collect();
                let mut qs: Vec<usize> =
                    modified[b_id].iter().copied().filter(|q| !near_set.contains(q)).collect();
                qs.sort_unstable();
                for q in qs {
                    q_extra.extend(tree.nodes[q].idx.iter().copied().filter(|&i| active[i]));
                }
            }
            // 1. skeleton
            let id = comp.call(&w, &box_idx, &far, &tree.nodes[b_id], &q_extra);
            let skel: Vec<usize> = id.skel.iter().map(|&s| box_idx[s]).collect();
            let red: Vec<usize> = id.red.iter().map(|&r| box_idx[r]).collect();
            if std::env::var("DUMP").is_ok() {
                println!(
                    "step l={} c={} box={} near_box=? far={} q={} k={} red={}",
                    tree.nodes[b_id].level,
                    tree.nodes[b_id].coord.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("_"),
                    box_idx.len(),
                    far.len(),
                    q_extra.len(),
                    id.skel.len(),
                    id.red.len()
                );
            }
            skel_sizes.push(skel.len());
            if red.is_empty() {
                continue;
            }
            // 2. sparsify: columns first, then rows (the row update reads the already updated W)
            let mut loc: Vec<usize> = box_idx.clone();
            loc.extend(near.iter().copied());
            let w_loc_skel = w.ix(&loc, &skel);
            let update = mmul(&w_loc_skel, &id.t);
            for (ii, &i) in loc.iter().enumerate() {
                for (jj, &j) in red.iter().enumerate() {
                    let v = w.at(i, j) - update.at(ii, jj);
                    w.set(i, j, v);
                }
            }
            let t_t = id.t.transpose();
            let w_skel_loc = w.ix(&skel, &loc);
            let update_r = mmul(&t_t, &w_skel_loc);
            for (ii, &i) in red.iter().enumerate() {
                for (jj, &j) in loc.iter().enumerate() {
                    let v = w.at(i, j) - update_r.at(ii, jj);
                    w.set(i, j, v);
                }
            }
            if std::env::var("DUMP2").is_ok() {
                let mut ms = 0.0f64;
                for &i in &skel { for &j in &red { ms = ms.max(w.at(i, j).abs()); } }
                let mut mr = 0.0f64;
                for &i in &red { for &j in &skel { mr = mr.max(w.at(i, j).abs()); } }
                let mut mw = 0.0f64;
                for v in &w.a { mw = mw.max(v.abs()); }
                println!("INV l={} c={} max|W[S,R]|={:.2e} max|W[R,S]|={:.2e} max|W|={:.2e}",
                    tree.nodes[b_id].level,
                    tree.nodes[b_id].coord.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("_"),
                    ms, mr, mw);
            }
            // 3. eliminate R (Schur complement onto S u N)
            let mut sn: Vec<usize> = skel.clone();
            sn.extend(near.iter().copied());
            let lu = lu_factor(&w.ix(&red, &red));
            let x_snr = w.ix(&sn, &red);
            let x_rsn = w.ix(&red, &sn);
            let schur = mmul(&x_snr, &lu_solve(&lu, &x_rsn));
            for (ii, &i) in sn.iter().enumerate() {
                for (jj, &j) in sn.iter().enumerate() {
                    let v = w.at(i, j) - schur.at(ii, jj);
                    w.set(i, j, v);
                }
            }
            steps.push(Step { skel, red: red.clone(), sn, t: id.t, lu, x_snr, x_rsn });
            for &r in &red {
                active[r] = false;
            }
            n_elim += red.len();
            let mut group: Vec<usize> = vec![b_id];
            group.extend(near_nodes.iter().copied());
            for &p in &group {
                for &q in &group {
                    if q != p {
                        modified[p].insert(q);
                    }
                }
            }
        }
        let boxes = nodes.len();
        levels.insert(
            l,
            LevelStat {
                boxes,
                skel_max: skel_sizes.iter().copied().max().unwrap_or(0),
                skel_total: skel_sizes.iter().sum(),
                eliminated: n_elim,
            },
        );
    }
    let root: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let root_lu = if root.is_empty() { None } else { Some(lu_factor(&w.ix(&root, &root))) };
    Factor { n, n_steps: steps.len(), steps, root, root_lu, levels }
}

impl<T: Scalar> Factor<T> {
    fn solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();
        for s in &self.steps {
            // x_R -= T^T x_S
            let mut corr = vec![T::zero(); s.red.len()];
            for (i, &sk) in s.skel.iter().enumerate() {
                let xs = x[sk];
                for j in 0..s.red.len() {
                    corr[j] = corr[j] + s.t.at(i, j) * xs;
                }
            }
            for (j, &r) in s.red.iter().enumerate() {
                x[r] = x[r] - corr[j];
            }
            let rhs = Mat::from_rows(s.red.len(), 1, s.red.iter().map(|&r| x[r]).collect());
            let y = lu_solve(&s.lu, &rhs);
            for (j, &sni) in s.sn.iter().enumerate() {
                let mut acc = x[sni];
                for i in 0..s.red.len() {
                    acc = acc - s.x_snr.at(j, i) * y.at(i, 0);
                }
                x[sni] = acc;
            }
            for (j, &r) in s.red.iter().enumerate() {
                x[r] = y.at(j, 0);
            }
        }
        if let Some(rlu) = &self.root_lu {
            let rhs = Mat::from_rows(self.root.len(), 1, self.root.iter().map(|&r| x[r]).collect());
            let y = lu_solve(rlu, &rhs);
            for (j, &r) in self.root.iter().enumerate() {
                x[r] = y.at(j, 0);
            }
        }
        for s in self.steps.iter().rev() {
            let mut rhs = vec![T::zero(); s.red.len()];
            for i in 0..s.red.len() {
                let mut acc = T::zero();
                for (j, &sni) in s.sn.iter().enumerate() {
                    acc = acc + s.x_rsn.at(i, j) * x[sni];
                }
                rhs[i] = acc;
            }
            let y = lu_solve(&s.lu, &Mat::from_rows(s.red.len(), 1, rhs));
            for (i, &r) in s.red.iter().enumerate() {
                x[r] = x[r] - y.at(i, 0);
            }
            // x_S -= T x_R
            for (i, &sk) in s.skel.iter().enumerate() {
                let mut acc = T::zero();
                for j in 0..s.red.len() {
                    acc = acc + s.t.at(i, j) * x[s.red[j]];
                }
                x[sk] = x[sk] - acc;
            }
        }
        x
    }
}

// ============================== 1-D ID butterfly ===================================================
struct BfBlock<T> {
    row_off: usize,
    cols: Vec<usize>, // global column positions this block reads
    p: Mat<T>,
}

struct BfUBlock<T> {
    rows: Vec<usize>,
    col_off: usize,
    p: Mat<T>,
}

struct Butterfly<T> {
    n: usize,
    l: usize,
    w: Vec<BfBlock<T>>,
    g: Vec<Vec<BfBlock<T>>>,
    u: Vec<BfUBlock<T>>,
    ranks: Vec<usize>,
    nnz: usize,
}

/// Reserve this tile's slot inside stage `st`'s stacked weight vector; returns the row offset.
fn place(
    skel: &mut [HashMap<(usize, usize), Vec<usize>>],
    offset: &mut [HashMap<(usize, usize), usize>],
    size: &mut [usize],
    st: usize,
    tile: (usize, usize),
    cols: Vec<usize>,
) -> usize {
    let off = size[st];
    offset[st].insert(tile, off);
    size[st] += cols.len();
    skel[st].insert(tile, cols);
    off
}

fn butterfly_id<T: Scalar>(
    kernel: &dyn Fn(&[f64], &[f64]) -> Mat<T>,
    x: &[f64],
    xi: &[f64],
    l: usize,
    eps: f64,
) -> Butterfly<T> {
    let n = x.len();
    let row_tree = Tree::new(x.iter().map(|&v| vec![v]).collect(), Some(l), 1);
    let col_tree = Tree::new(xi.iter().map(|&v| vec![v]).collect(), Some(l), 1);
    let tiles: Vec<Vec<(usize, usize)>> = (0..=l)
        .map(|d| {
            let mut v = vec![];
            for &a in &row_tree.levels[d] {
                for &b in &col_tree.levels[l - d] {
                    v.push((a, b));
                }
            }
            v
        })
        .collect();
    let kblk = |rows: &[usize], cols: &[usize]| -> Mat<T> {
        let xr: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
        let xc: Vec<f64> = cols.iter().map(|&j| xi[j]).collect();
        kernel(&xr, &xc)
    };
    let mut skel: Vec<HashMap<(usize, usize), Vec<usize>>> = vec![HashMap::new(); l + 1];
    let mut offset: Vec<HashMap<(usize, usize), usize>> = vec![HashMap::new(); l + 1];
    let mut size: Vec<usize> = vec![0; l + 1];
    let mut ranks: Vec<usize> = vec![0; l + 1];
    let mut nnz = 0usize;

    // stage 0: compress each column box against all rows
    let mut w_blocks: Vec<BfBlock<T>> = vec![];
    let all_rows = row_tree.nodes[row_tree.levels[0][0]].idx.clone();
    for &(root, b) in &tiles[0] {
        let cols = col_tree.nodes[b].idx.clone();
        let id = interp_decomp(&kblk(&all_rows, &cols), eps);
        let skel_cols: Vec<usize> = id.skel.iter().map(|&s| cols[s]).collect();
        let p = interp_matrix(&id, &cols);
        let row_off = place(&mut skel, &mut offset, &mut size, 0, (root, b), skel_cols);
        nnz += p.r * p.c;
        w_blocks.push(BfBlock { row_off, cols, p });
    }

    // stages 1..L
    let mut g: Vec<Vec<BfBlock<T>>> = vec![];
    for st in 0..l {
        let mut blocks: Vec<BfBlock<T>> = vec![];
        for &(ac, pc) in &tiles[st + 1] {
            let a = row_tree.nodes[ac].parent.unwrap();
            let olds: Vec<(usize, usize)> = col_tree.nodes[pc].children.iter().map(|&b| (a, b)).collect();
            let mut cand: Vec<usize> = vec![];
            for &t in &olds {
                cand.extend(skel[st][&t].iter().copied());
            }
            let id = interp_decomp(&kblk(&row_tree.nodes[ac].idx, &cand), eps);
            let skel_cols: Vec<usize> = id.skel.iter().map(|&s| cand[s]).collect();
            let p = interp_matrix(&id, &cand);
            let row_off = size[st + 1];
            let mut col_pos: Vec<usize> = vec![];
            for &t in &olds {
                let o = offset[st][&t];
                let k = skel[st][&t].len();
                col_pos.extend(o..(o + k));
            }
            place(&mut skel, &mut offset, &mut size, st + 1, (ac, pc), skel_cols);
            nnz += p.r * p.c;
            blocks.push(BfBlock { row_off, cols: col_pos, p });
        }
        g.push(blocks);
    }

    // stage L: exact evaluation of the last skeleton columns, one leaf row box per tile
    let mut u: Vec<BfUBlock<T>> = vec![];
    for &(a, root) in &tiles[l] {
        let rows = row_tree.nodes[a].idx.clone();
        let cols = skel[l][&(a, root)].clone();
        let o = offset[l][&(a, root)];
        let p = kblk(&rows, &cols);
        nnz += p.r * p.c;
        u.push(BfUBlock { rows, col_off: o, p });
    }
    for st in 0..=l {
        ranks[st] = skel[st].values().map(|v| v.len()).max().unwrap_or(0);
    }
    Butterfly { n, l, w: w_blocks, g, u, ranks, nnz }
}

fn interp_matrix<T: Scalar>(id: &IdResult<T>, cols: &[usize]) -> Mat<T> {
    let k = id.skel.len();
    let n = cols.len();
    let mut p = Mat::zeros(k, n);
    for i in 0..k {
        p.set(i, id.skel[i], T::one());
    }
    for (j, &r) in id.red.iter().enumerate() {
        for i in 0..k {
            p.set(i, r, id.t.at(i, j));
        }
    }
    p
}

impl<T: Scalar> Butterfly<T> {
    fn apply(&self, gvec: &[T]) -> Vec<T> {
        let mut c = vec![T::zero(); self.w.iter().map(|b| b.row_off + b.p.r).max().unwrap_or(0)];
        for b in &self.w {
            for i in 0..b.p.r {
                let mut s = T::zero();
                for (j, &col) in b.cols.iter().enumerate() {
                    s = s + b.p.at(i, j) * gvec[col];
                }
                c[b.row_off + i] = s;
            }
        }
        for stage in &self.g {
            let sz = stage.iter().map(|b| b.row_off + b.p.r).max().unwrap_or(0);
            let mut c2 = vec![T::zero(); sz];
            for b in stage {
                for i in 0..b.p.r {
                    let mut s = T::zero();
                    for (j, &col) in b.cols.iter().enumerate() {
                        s = s + b.p.at(i, j) * c[col];
                    }
                    c2[b.row_off + i] = s;
                }
            }
            c = c2;
        }
        let mut f = vec![T::zero(); self.n];
        for b in &self.u {
            for (i, &row) in b.rows.iter().enumerate() {
                let mut s = T::zero();
                for j in 0..b.p.c {
                    s = s + b.p.at(i, j) * c[b.col_off + j];
                }
                f[row] = s;
            }
        }
        f
    }
}

// ============================== cases ==============================================================
mod cases;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("check");
    if mode == "lutest" {
        for n in [6usize, 17, 27, 35, 60] {
            cases::lutest(n);
        }
        return;
    }
    if mode == "diag" {
        cases::diag(args.get(2).and_then(|v| v.parse().ok()).unwrap_or(192),
                    args.get(3).and_then(|v| v.parse().ok()).unwrap_or(2));
        return;
    }
    if mode == "idfile" {
        cases::idfile(&args[2]);
        return;
    }
    if mode == "mini" {
        let d: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2);
        let nn: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(256);
        std::env::set_var("DUMP", "1");
        cases::mini(d, nn);
        return;
    }
    if mode == "selftest" {
        cases::selftest();
        return;
    }
    let golden = std::fs::read_to_string("golden.txt").unwrap_or_default();
    let g: HashMap<String, String> = golden
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
        .collect();
    let mut rep = cases::Report::new(&g);
    let t0 = Instant::now();
    cases::tree_cases(&mut rep);
    cases::id_cases(&mut rep);
    cases::rs_cases(&mut rep, mode == "bench");
    cases::butterfly_cases(&mut rep, mode == "bench");
    rep.finish(t0.elapsed().as_secs_f64());
}
