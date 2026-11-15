//! The golden-value cases: every number here has a matching key produced by `tools/golden.py` from the
//! Python tree, so `cargo run --release -- check` is a cross-language equivalence test of the port.
use crate::*;
use std::collections::HashMap;
use std::time::Instant;

pub struct Report<'a> {
    g: &'a HashMap<String, String>,
    pass: usize,
    fail: usize,
    rows: Vec<String>,
}

impl<'a> Report<'a> {
    pub fn new(g: &'a HashMap<String, String>) -> Self {
        Report { g, pass: 0, fail: 0, rows: vec![] }
    }
    fn exp(&self, k: &str) -> Option<&String> {
        self.g.get(k)
    }
    fn ok(&mut self, k: &str, got: String) {
        let want = self.exp(k).cloned().unwrap_or_else(|| "".to_string());
        if std::env::var("VERBOSE").is_ok() {
            println!("V {k} = {got}   (py {want})");
        }
        if got == want {
            self.pass += 1;
        } else {
            self.fail += 1;
            self.rows.push(format!("MISMATCH {k}: got {got} | python {want}"));
        }
    }
    fn num_ok(&mut self, k: &str, got: f64, tol: f64) {
        let want: f64 = self.exp(k).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN);
        if std::env::var("VERBOSE").is_ok() {
            println!("V {k} = {got:.6e}   (py {want:.6e})");
        }
        let denom = want.abs().max(1e-300);
        // residuals are roundoff: agree if both are at noise level, else require the stated tolerance
        let both_noise = got.abs() <= 1e-8 && want.abs() <= 1e-8;
        if (got - want).abs() <= tol * denom || both_noise || (got.is_nan() && want.is_nan()) {
            self.pass += 1;
        } else {
            self.fail += 1;
            self.rows.push(format!("MISMATCH {k}: got {got:.17e} | python {want:.17e}"));
        }
    }
    pub fn i(&mut self, k: &str, got: usize) {
        self.ok(k, got.to_string());
    }
    pub fn v(&mut self, k: &str, got: &[usize]) {
        self.ok(k, got.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    }
    pub fn f(&mut self, k: &str, got: f64, tol: f64) {
        self.num_ok(k, got, tol);
    }
    pub fn s(&mut self, k: &str, got: &str) {
        self.ok(k, got.to_string());
    }
    pub fn finish(&self, secs: f64) {
        for r in &self.rows {
            println!("{r}");
        }
        println!(
            "golden checks: {} passed, {} failed, {:.3} s total",
            self.pass, self.fail, secs
        );
        if self.fail > 0 {
            std::process::exit(1);
        }
    }
}

// ---- helpers ---------------------------------------------------------------------------------------
fn frob(m: &Mat<f64>) -> f64 {
    m.a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn sub_cols(m: &Mat<f64>, cols: &[usize]) -> Mat<f64> {
    m.ix(&(0..m.r).collect::<Vec<_>>(), cols)
}
fn axpy_sub(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let mut o = a.clone();
    for i in 0..o.a.len() {
        o.a[i] = a.a[i] - b.a[i];
    }
    o
}
fn matvec(m: &Mat<f64>, v: &[f64]) -> Vec<f64> {
    (0..m.r)
        .map(|i| (0..m.c).map(|j| m.at(i, j) * v[j]).sum())
        .collect()
}
fn weyl2d(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            vec![
                ((i * 7919) % 5003) as f64 / 5003.0,
                ((i * 104729) % 4999) as f64 / 4999.0,
            ]
        })
        .collect()
}
fn rank_mat(m: usize, n: usize, k: usize) -> Mat<f64> {
    let mut u = Mat::zeros(m, k);
    for i in 0..m {
        for l in 0..k {
            u.set(i, l, (0.3 * (i as f64) * (l as f64) + 0.1).sin());
        }
    }
    let mut v = Mat::zeros(k, n);
    for l in 0..k {
        for j in 0..n {
            v.set(l, j, (0.5 * (l as f64) * (j as f64) + 0.2).cos());
        }
    }
    mmul(&u, &v)
}
fn starfish(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = vec![];
    let mut w = vec![];
    for i in 0..n {
        let th = 2.0 * PI * (i as f64) / (n as f64);
        let r = 1.0 + 0.3 * (5.0 * th).cos();
        let dr = -1.5 * (5.0 * th).sin();
        x.push(vec![r * th.cos(), r * th.sin()]);
        w.push((r * r + dr * dr).sqrt() * 2.0 * PI / (n as f64));
    }
    (x, w)
}
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (3.0 * i as f64).sin() + 0.5).collect()
}
// ---- tree ------------------------------------------------------------------------------------------
pub fn tree_cases(rep: &mut Report) {
    let pts: Vec<Vec<f64>> = (0..64).map(|i| vec![i as f64 / 64.0]).collect();
    let t = Tree::new(pts, Some(6), 1);
    rep.v("tree1d_levels", &t.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    let mut leaf_sizes: Vec<usize> = t.levels[6].iter().map(|&id| t.nodes[id].idx.len()).collect();
    leaf_sizes.sort_unstable();
    leaf_sizes.dedup();
    rep.v("tree1d_leaf_sizes", &leaf_sizes);

    let t2 = Tree::new(weyl2d(500), None, 16);
    rep.i("tree2d_depth", t2.depth);
    rep.v("tree2d_occupied", &t2.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    let mid = t2.levels[2][t2.levels[2].len() / 2];
    rep.s("tree2d_mid_coord", &fmt_coords(&[t2.nodes[mid].coord.clone()]));
    let mut nb: Vec<Vec<usize>> = t2.neighbors(mid).iter().map(|&i| t2.nodes[i].coord.clone()).collect();
    nb.sort();
    rep.s("tree2d_neighbors_mid", &fmt_coords(&nb));
    let mut il: Vec<Vec<usize>> =
        t2.interaction_list(mid).iter().map(|&i| t2.nodes[i].coord.clone()).collect();
    il.sort();
    rep.s("tree2d_ilist_mid", &fmt_coords(&il));
}

fn fmt_coords(cs: &[Vec<usize>]) -> String {
    cs.iter()
        .map(|c| c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("_"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---- interpolative decomposition --------------------------------------------------------------------
pub fn id_cases(rep: &mut Report) {
    let m = rank_mat(80, 50, 7);
    for (tag, eps) in [("1e-10", 1e-10f64), ("0.001", 1e-3)] {
        let id = interp_decomp(&m, eps);
        rep.i(&format!("id_eps{tag}_k"), id.skel.len());
        rep.v(&format!("id_eps{tag}_skel"), &id.skel);
        let approx = mmul(&sub_cols(&m, &id.skel), &id.t);
        let diff = axpy_sub(&sub_cols(&m, &id.red), &approx);
        rep.f(&format!("id_eps{tag}_relres"), frob(&diff) / frob(&m), 1e-6);
    }
    let id = interp_decomp(&Mat::<f64>::zeros(5, 0), 1e-8);
    rep.i("id_n0_k", id.skel.len());
    let id = interp_decomp(&Mat::<f64>::zeros(0, 5), 1e-8);
    rep.i("id_m0_k", id.skel.len());
    rep.v("id_m0_red", &id.red);
    let id = interp_decomp(&rank_mat(30, 40, 7), 1e-12);
    rep.i("id_30x40_k", id.skel.len());
}

// ---- recursive skeletonization ----------------------------------------------------------------------
fn rs_case(rep: &mut Report, name: &str, tree: &Tree, a: &Mat<f64>, adm: Adm, comp: &Comp<f64>) {
    if std::env::var("DUMP").is_ok() {
        println!("STEPCASE {name}");
    }
    let f = rskel_factor(a, tree, adm, comp, true);
    rep.i(&format!("{name}_depth"), tree.depth);
    rep.v(&format!("{name}_levels"), &tree.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    let ls: Vec<&LevelStat> = f.levels.values().collect();
    rep.v(&format!("{name}_skelmax"), &ls.iter().map(|s| s.skel_max).collect::<Vec<_>>());
    rep.v(&format!("{name}_skeltotal"), &ls.iter().map(|s| s.skel_total).collect::<Vec<_>>());
    rep.v(&format!("{name}_eliminated"), &ls.iter().map(|s| s.eliminated).collect::<Vec<_>>());
    rep.v(&format!("{name}_boxes"), &ls.iter().map(|s| s.boxes).collect::<Vec<_>>());
    rep.i(&format!("{name}_root"), f.root.len());
    rep.i(&format!("{name}_nsteps"), f.n_steps);
    let b = rhs(a.r);
    let x = f.solve(&b);
    let r = matvec(a, &x);
    let diff: Vec<f64> = r.iter().zip(&b).map(|(ri, bi)| ri - bi).collect();
    rep.f(&format!("{name}_relres"), norm2(&diff) / norm2(&b), 1e-4);
}

pub fn rs_cases(rep: &mut Report, bench: bool) {
    let n = 256usize;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let k = lap1d(&xs, &xs);
    let w = vec![1.0 / n as f64; n];
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, k.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
        }
    }
    let tree = Tree::new(xs.iter().map(|&v| vec![v]).collect(), None, 8);
    let eps = 1e-10;
    rs_case(rep, "rs1d_weak", &tree, &a, Adm::Weak, &Comp::Exact { eps });
    rs_case(rep, "rs1d_strong", &tree, &a, Adm::Strong, &Comp::Exact { eps });

    // P1 curve: genuine 2-D ranks, the project's canonical rskel problem
    let n = 256usize;
    let (x, w) = starfish(n);
    let kern = laplace2d(&x, &x);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, kern.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
        }
    }
    let tree = Tree::new(x.clone(), None, 8);
    let eps = 1e-8;
    let proxy = || Comp::Proxy {
        kernel: p1_kernel(n),
        points: x.clone(),
        eps,
        n_proxy: 64,
        radius_factor: 2.5,
    };
    for (adm, comp, tag) in [
        (Adm::Weak, Comp::Exact { eps }, "p1_weak_exact"),
        (Adm::Weak, proxy(), "p1_weak_proxy"),
        (Adm::Strong, Comp::Exact { eps }, "p1_strong_exact"),
        (Adm::Strong, proxy(), "p1_strong_proxy"),
    ] {
        rs_case(rep, tag, &tree, &a, adm, &comp);
    }

    if bench {
        for n in [512usize, 1024] {
            let (x, w) = starfish(n);
            let kern = laplace2d(&x, &x);
            let mut a = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    a.set(i, j, kern.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
                }
            }
            let tree = Tree::new(x, None, 32);
            let t0 = Instant::now();
            let f = rskel_factor(&a, &tree, Adm::Strong, &Comp::Exact { eps: 1e-8 }, true);
            let dt = t0.elapsed().as_secs_f64();
            let b = rhs(n);
            let t1 = Instant::now();
            let x_hat = f.solve(&b);
            let st = t1.elapsed().as_secs_f64();
            let r = matvec(&a, &x_hat);
            let diff: Vec<f64> = r.iter().zip(&b).map(|(ri, bi)| ri - bi).collect();
            println!(
                "BENCH rust p1 strong exact N={n}: factor {dt:.3} s, solve {st:.4} s, root {}, relres {:.2e}",
                f.root.len(),
                norm2(&diff) / norm2(&b)
            );
        }
    }
}


// ---- butterfly ---------------------------------------------------------------------------------------
pub fn butterfly_cases(rep: &mut Report, bench: bool) {
    for (n, t, eps, tag) in [
        (64usize, 0.0f64, 1e-10f64, "bf_N64_t0_e1e-10"),
        (64, 3.0, 1e-8, "bf_N64_t3_e1e-08"),
        (128, 1.0, 1e-8, "bf_N128_t1_e1e-08"),
    ] {
        let tag = tag.to_string();
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let xi: Vec<f64> = (0..n).map(|i| i as f64 - (n / 2) as f64).collect();
        let l = (n as f64).log2().round() as usize;
        let kern = |p: &[f64], q: &[f64]| fio1d(t, p, q);
        let bf = butterfly_id(&kern, &x, &xi, l, eps);
        rep.i(&format!("{tag}_L"), bf.l);
        rep.v(&format!("{tag}_ranks"), &bf.ranks);
        let nz = |m: &Mat<C64>| m.a.iter().filter(|v| **v != C64::zero()).count();
        let nnz_w: usize = bf.w.iter().map(|b| nz(&b.p)).sum();
        let nnz_u: usize = bf.u.iter().map(|b| nz(&b.p)).sum();
        rep.v(&format!("{tag}_nnz_W_U"), &[nnz_w, nnz_u]);
        let nnz_g: Vec<usize> =
            bf.g.iter().map(|s| s.iter().map(|b| nz(&b.p)).sum()).collect();
        rep.v(&format!("{tag}_nnz_G"), &nnz_g);
        rep.i(&format!("{tag}_nnz"), nnz_w + nnz_u + nnz_g.iter().sum::<usize>());
        let gvec: Vec<C64> = (0..n)
            .map(|j| {
                C64::new(
                    (1.7 * j as f64).sin() + 0.3,
                    (0.9 * j as f64).cos() - 0.2,
                )
            })
            .collect();
        let f = bf.apply(&gvec);
        // exact reference: full kernel matrix times g
        let km = fio1d(t, &x, &xi);
        let mut fex = vec![C64::zero(); n];
        for i in 0..n {
            let mut s = C64::zero();
            for j in 0..n {
                s = s + km.at(i, j) * gvec[j];
            }
            fex[i] = s;
        }
        let num: f64 = f.iter().zip(&fex).map(|(a, b)| (*a - *b).abs().powi(2)).sum::<f64>().sqrt();
        let den: f64 = fex.iter().map(|b| b.abs().powi(2)).sum::<f64>().sqrt();
        rep.f(&format!("{tag}_relerr"), num / den, 1e-4);
    }
    if bench {
        let n = 512usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let xi: Vec<f64> = (0..n).map(|i| i as f64 - (n / 2) as f64).collect();
        let l = (n as f64).log2().round() as usize;
        let kern = |p: &[f64], q: &[f64]| fio1d(1.0, p, q);
        let t0 = Instant::now();
        let bf = butterfly_id(&kern, &x, &xi, l, 1e-8);
        let dt = t0.elapsed().as_secs_f64();
        println!("BENCH rust butterfly build N={n}: {dt:.3} s, nnz {}, ranks {:?}", bf.nnz, bf.ranks);
    }
}

/// P1's proxy kernel: Laplace Green's function times the starfish arc-length weight of the columns.
fn p1_kernel(n_total: usize) -> Kern<f64> {
    Box::new(move |p: &[Vec<f64>], q: &[Vec<f64>]| {
        let mut m = laplace2d(p, q);
        let wt = starfish_weight(q, n_total);
        for i in 0..m.r {
            for j in 0..m.c {
                m.set(i, j, m.at(i, j) * wt[j]);
            }
        }
        m
    })
}

/// Diagnostic: is the ID itself accurate on the 2-D problem, and does the algebra of one step close?
pub fn selftest() {
    let n = 256usize;
    let (x, w) = starfish(n);
    let kern = laplace2d(&x, &x);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, kern.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
        }
    }
    let tree = Tree::new(x, None, 8);
    println!("depth {} levels {:?}", tree.depth, tree.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    let l = tree.depth;
    let b_id = tree.levels[l][0];
    let box_idx = tree.nodes[b_id].idx.clone();
    let near_nodes = Adm::Weak.near_nodes(&tree, b_id);
    let mut near: Vec<usize> = vec![];
    for &nb in &near_nodes {
        near.extend(tree.nodes[nb].idx.iter().copied());
    }
    let far: Vec<usize> = (0..n).filter(|i| !box_idx.contains(i) && !near.contains(i)).collect();
    println!("box {} near {} far {}", box_idx.len(), near.len(), far.len());
    let comp = Comp::<f64>::Exact { eps: 1e-8 };
    let id = comp.call(&a, &box_idx, &far, &tree.nodes[b_id], &[]);
    let skel: Vec<usize> = id.skel.iter().map(|&s| box_idx[s]).collect();
    let red: Vec<usize> = id.red.iter().map(|&r| box_idx[r]).collect();
    println!("id: k={} red={}", skel.len(), red.len());
    // column relation on the true far block
    let m_far = a.ix(&far, &box_idx);
    let approx = mmul(&a.ix(&far, &skel), &id.t);
    let diff = axpy_sub(&m_far.ix(&(0..far.len()).collect::<Vec<_>>(), &id.red), &approx);
    println!(
        "ID col relerr {:.3e} (||block|| {:.3e})",
        frob(&diff) / frob(&m_far),
        frob(&m_far)
    );
    // row relation
    let m_row = a.ix(&box_idx, &far).transpose();
    let approx_r = mmul(&a.ix(&skel, &far).transpose(), &id.t);
    let diff_r = axpy_sub(&m_row.ix(&(0..far.len()).collect::<Vec<_>>(), &id.red), &approx_r);
    println!("ID row relerr {:.3e}", frob(&diff_r) / frob(&m_row));
    // one full step, then compare A against the reconstructed step
    let mut wmat = a.clone();
    let mut loc = box_idx.clone();
    loc.extend(near.iter().copied());
    let upd = mmul(&wmat.ix(&loc, &skel), &id.t);
    for (ii, &i) in loc.iter().enumerate() {
        for (jj, &j) in red.iter().enumerate() {
            let v = wmat.at(i, j) - upd.at(ii, jj);
            wmat.set(i, j, v);
        }
    }
    let tt = id.t.transpose();
    let upd_r = mmul(&tt, &wmat.ix(&skel, &loc));
    for (ii, &i) in red.iter().enumerate() {
        for (jj, &j) in loc.iter().enumerate() {
            let v = wmat.at(i, j) - upd_r.at(ii, jj);
            wmat.set(i, j, v);
        }
    }
    println!(
        "after sparsify: |W[far,red]| / |A[far,box]| = {:.3e}",
        frob(&wmat.ix(&far, &red)) / frob(&m_far)
    );
    let lu = lu_factor(&wmat.ix(&red, &red));
    println!("|W[red,red]| {:.3e}, lu diag min |{:e}|", frob(&wmat.ix(&red, &red)), lu.a.at(0, 0).abs());
}

/// Minimal 2-D RS case: p1 starfish, forced depth, weak/exact -> steps + residual (compare mini.py).
pub fn mini(depth: usize, n: usize) {
    let (x, w) = starfish(n);
    let kern = laplace2d(&x, &x);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, kern.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
        }
    }
    let tree = Tree::new(x, Some(depth), 8);
    println!("levels {:?}", tree.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
    let f = rskel_factor(&a, &tree, Adm::Weak, &Comp::Exact { eps: 1e-8 }, true);
    let b = rhs(n);
    let xh = f.solve(&b);
    let r = matvec(&a, &xh);
    let diff: Vec<f64> = r.iter().zip(&b).map(|(ri, bi)| ri - bi).collect();
    println!("root {} nsteps {} relres {:.3e}", f.root.len(), f.n_steps, norm2(&diff) / norm2(&b));
}

/// Read a matrix dumped by numpy, run the ported ID, report k / achieved error / pivots.
pub fn idfile(path: &str) {
    let txt = std::fs::read_to_string(path).unwrap();
    let rows: Vec<Vec<f64>> = txt
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    let (m, n) = (rows.len(), rows[0].len());
    let mut mat = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            mat.set(i, j, rows[i][j]);
        }
    }
    let id = interp_decomp(&mat, 1e-8);
    let approx = mmul(&sub_cols(&mat, &id.skel), &id.t);
    let diff = axpy_sub(&sub_cols(&mat, &id.red), &approx);
    println!(
        "rust: shape ({m}, {n}) k={} id_relerr={:.3e}",
        id.skel.len(),
        frob(&diff) / frob(&mat)
    );
    println!("pivots {:?}", &id.skel[..id.skel.len().min(12)]);
}

/// Localize a factorization defect: solve every unit vector and look at A X - I.
pub fn diag(n: usize, depth: usize) {
    let (x, w) = starfish(n);
    let kern = laplace2d(&x, &x);
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, kern.at(i, j) * w[j] + if i == j { 1.0 } else { 0.0 });
        }
    }
    let tree = Tree::new(x, Some(depth), 8);
    let f = rskel_factor(&a, &tree, Adm::Weak, &Comp::Exact { eps: 1e-8 }, true);
    for s in &f.steps {
        println!("step red={:?} sn={:?}", &s.red, &s.sn);
    }
    println!("root {:?}", &f.root[..f.root.len().min(20)]);
    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let xj = f.solve(&e);
        let r = matvec(&a, &xj);
        for i in 0..n {
            let d = if i == j { (r[i] - 1.0).abs() } else { r[i].abs() };
            if d > worst {
                worst = d;
                worst_at = (i, j);
            }
        }
    }
    println!("max |A X - I| = {worst:.3e} at (row {worst_at:?})");
}

/// LU sanity: does lu_factor/lu_solve invert a dense block at the sizes the drivers use?
pub fn lutest(n: usize) {
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, (0.7 * (i as f64) + 1.3 * (j as f64)).sin() + if i == j { 2.0 } else { 0.0 });
        }
    }
    let lu = lu_factor(&a);
    let b: Vec<f64> = (0..n).map(|i| (0.3 * i as f64).cos()).collect();
    let x = lu_solve(&lu, &Mat::from_rows(n, 1, b.clone()));
    let r = matvec(&a, &(0..n).map(|i| x.at(i, 0)).collect::<Vec<_>>());
    let diff: Vec<f64> = r.iter().zip(&b).map(|(ri, bi)| ri - bi).collect();
    println!("lutest n={n}: relres {:.3e}", norm2(&diff) / norm2(&b));
}
