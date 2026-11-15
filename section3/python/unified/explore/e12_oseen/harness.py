"""E12 harness: factor Oseen matrices with the unchanged rskel_factor under weak / strong / directional rules.

  * scalar area problem: ExactCompressor or the stock core ProxyCompressor (kernel = scalar Oseen G_lam)
  * tensor cylinder problem: ExactCompressor or `TensorProxy` (below), on a tree over interleaved dofs
    (each boundary node appears twice in the point list, so node.idx are dof indices)
Measured: backward/forward error, skeleton totals, storage, an elimination flop count, pivot health (E5 check).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.linalg as sla

from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak, interp_decomp
from unified.core.tree import Node
from unified.core.types import Admissibility, Compressor, IDResult, IndexArray, Matrix, Points, Vector
from unified.drivers.rskel import Factorization, rskel_factor

from .directional import Directional
from .oseen import AreaProblem, Cylinder, oseen_tensor, rel_err

type CaseResult = dict[str, Any]


def make_rule(name: str) -> Admissibility:
    """'weak', 'strong', or 'dir:a_down,a_up,a_cross'."""
    if name == "weak":
        return Weak()
    if name == "strong":
        return Strong()
    a, b, c = (int(v) for v in name.split(":")[1].split(","))
    return Directional(a, b, c)


class EllipseProxy(ProxyCompressor):
    """Stock ProxyCompressor whose proxy CURVE matches a Directional rule: an ellipse enclosing B and stretched over
    the near boxes along the stream axis, x in [-(2 a_up + 0.5) hw, (2 a_down + 0.5) hw], |y| <= (2 a_cross + 0.5) hw.
    (Exterior Dirichlet uniqueness holds for any closed curve, so the representation argument is unchanged.)
    The driver's "far points inside the proxy curve" test uses a CIRCLE of radius radius_factor * hw, so
    radius_factor is set to cover the whole ellipse: every far point inside the ellipse is then kept explicit
    (a superset; more rows, same correctness). Only proxy_points is overridden; the driver is unchanged."""

    def __init__(self, kernel: Any, points: Points, eps: float, A: Matrix, rule: Directional,
                 n_proxy: int = 64) -> None:
        self.rule = rule
        lo, hi = 2 * rule.a_up + 0.5, 2 * rule.a_down + 0.5
        self.cx, self.rx, self.ry = 0.5 * (hi - lo), 0.5 * (hi + lo), 2 * rule.a_cross + 0.5
        super().__init__(kernel, points, eps, A=A, n_proxy=n_proxy,
                         radius_factor=1.001 * max(abs(self.cx) + self.rx, self.ry))

    def proxy_points(self, node: Node) -> Points:
        hw = node.half_width
        m = int(np.ceil(self.n_proxy * max(self.rx, self.ry) / 2.5))   # keep the point density of the circle
        th = 2 * np.pi * np.arange(m) / m
        out: Points = node.center[None, :] + hw * np.c_[self.cx + self.rx * np.cos(th), self.ry * np.sin(th)]
        return out


class TensorProxy:
    """Proxy compressor for the 2x2 Oseenlet on interleaved dofs (not a core ProxyCompressor subclass: the core one
    calls kernel(points, points) and gets one row per point; a 2x2 kernel needs the dof component, cf. E6).
    Rows: S(p - y_box) (far velocity made by box forces; exterior Oseen Dirichlet problem is unique) and
    S(x_box - p)^T (box velocity made by far forces; the adjoint problem), on a circle of n_proxy points;
    plus explicit rows for far dofs inside the circle and for a conservative superset of the FMM-LU Q set:
    every active far dof in a same-level box that is near to one of B's near boxes (only such pairs can carry
    Schur-complement fill, as the driver's own Q bookkeeping assumes)."""

    def __init__(self, cyl: Cylinder, tree: Tree, rule: Admissibility, eps: float, A: Matrix,
                 n_proxy: int = 64, radius_factor: float = 2.5) -> None:
        self.cyl, self.tree, self.rule, self.eps, self.A = cyl, tree, rule, eps, A
        self.n_proxy, self.radius_factor = n_proxy, radius_factor
        self.dofX: Points = cyl.dof_points

    def __call__(self, box: IndexArray, far: IndexArray, node: Node) -> IDResult:
        r = self.radius_factor * node.half_width
        th = 2 * np.pi * np.arange(self.n_proxy) / self.n_proxy
        Pp: Points = node.center[None, :] + r * np.c_[np.cos(th), np.sin(th)]
        pts = np.unique(box // 2)
        loc = {p: i for i, p in enumerate(pts)}
        cols = np.array([2 * loc[d // 2] + d % 2 for d in box])
        Xb = self.cyl.X[pts]
        blocks = [oseen_tensor(self.cyl.k, Pp, Xb)[:, cols], oseen_tensor(self.cyl.k, Xb, Pp)[cols, :].T]
        inside = far[np.linalg.norm(self.dofX[far] - node.center, axis=1) < r]
        q_nodes = {q for nb in self.rule.near_nodes(self.tree, node) for q in self.rule.near_nodes(self.tree, nb)}
        q_dofs = np.concatenate([q.idx for q in q_nodes] + [np.zeros(0, np.intp)])
        expl = np.union1d(inside, np.intersect1d(far, q_dofs))
        if expl.size:
            blocks += [self.A[np.ix_(expl, box)], self.A[np.ix_(box, expl)].T]
        return interp_decomp(np.vstack(blocks), self.eps)


def pivot_health(F: Factorization) -> tuple[float, float]:
    """(worst cond(X_RR), worst min eig of Herm(X_RR) / ||X_RR||) over all eliminated pivot blocks (E5's check).
    X_RR is rebuilt from scipy's packed LU and its row-swap vector."""
    worst_c, worst_h = 1.0, 1.0
    for s in F.steps:
        lu, piv = s.lu
        n = lu.shape[0]
        L = np.tril(lu, -1) + np.eye(n)
        U = np.triu(lu)
        X = L @ U
        for i in range(n - 1, -1, -1):              # undo the row swaps (applied in order 0..n-1)
            if piv[i] != i:
                X[[i, piv[i]]] = X[[piv[i], i]]
        sv = np.linalg.svd(X, compute_uv=False)
        worst_c = max(worst_c, float(sv[0] / sv[-1]))
        worst_h = min(worst_h, float(np.linalg.eigvalsh(0.5 * (X + X.T))[0] / sv[0]))
    return worst_c, worst_h


def cost(F: Factorization) -> tuple[float, float]:
    """(numbers stored, elimination flops): per step |T| + |X_snr| + |X_rsn| + |R|^2, flops ~ 2/3|R|^3 +
    2 |SN| |R|^2 + 2 |SN|^2 |R| (LU of X_RR, the solves, the Schur update); plus the dense root LU."""
    st, fl = 0.0, 0.0
    for s in F.steps:
        R, SN = len(s.red), len(s.sn)
        st += s.T.size + s.X_snr.size + s.X_rsn.size + R * R
        fl += 2 / 3 * R ** 3 + 2 * SN * R ** 2 + 2 * SN ** 2 * R
    m = F.root.size
    return st + m * m, fl + 2 / 3 * m ** 3


def norm2_est(A: Matrix, iters: int = 30) -> float:
    """||A||_2 by power iteration on A^T A (enough for a normwise backward error)."""
    v: Vector = np.random.default_rng(3).standard_normal((int(A.shape[1]),))
    for _ in range(iters):
        v = A.T @ (A @ v)
        v = v / float(np.linalg.norm(v))
    return float(np.linalg.norm(A @ v))


def _summary(F: Factorization, A: Matrix, b: Vector, x: Vector, x_dense: Vector, tf: float,
             health: bool) -> CaseResult:
    lv = F.stats["levels"]
    st, fl = cost(F)
    out: CaseResult = dict(
        err=rel_err(x, x_dense),
        backward=float(np.linalg.norm(A @ x - b) / (norm2_est(A) * np.linalg.norm(x) + np.linalg.norm(b))),
        skel_total=int(sum(v["skel_total"] for v in lv.values())),
        skel_max={int(l): int(v["skel_max"]) for l, v in lv.items()},
        root=int(F.stats["root_size"]), stored=st, flops=fl, factor_s=tf)
    if health:
        out["pivot_cond"], out["pivot_herm_min"] = pivot_health(F)
    return out


def area_case(prob: AreaProblem, rule: str, comp: str, x_dense: Vector, eps: float = 1e-8, max_leaf: int = 32,
              n_proxy: int = 64, health: bool = False) -> CaseResult:
    tree = Tree(prob.X, max_leaf=max_leaf)
    R = make_rule(rule)
    c: Compressor
    if comp == "exact":
        c = ExactCompressor(prob.A, eps)
    elif comp == "ellipse" and isinstance(R, Directional):
        c = EllipseProxy(prob.kernel, prob.X, eps, prob.A, R, n_proxy=n_proxy)
    else:
        c = ProxyCompressor(prob.kernel, prob.X, eps, A=prob.A, n_proxy=n_proxy)
    t0 = time.perf_counter()
    F = rskel_factor(prob.A, tree, R, c)
    tf = time.perf_counter() - t0
    x = F.solve(prob.b)
    out = dict(lam=prob.lam, N=len(prob.X), rule=rule, comp=comp, n_proxy=n_proxy, depth=tree.depth)
    out.update(_summary(F, prob.A, prob.b, x, x_dense, tf, health))
    return out


def cylinder_case(cyl: Cylinder, rule: str, comp: str, f_dense: Vector, eps: float = 1e-8,
                  max_leaf: int = 64) -> CaseResult:
    tree = Tree(cyl.dof_points, max_leaf=max_leaf)
    R = make_rule(rule)
    c: Any = ExactCompressor(cyl.A, eps) if comp == "exact" else TensorProxy(cyl, tree, R, eps, cyl.A)
    t0 = time.perf_counter()
    F = rskel_factor(cyl.A, tree, R, c)
    tf = time.perf_counter() - t0
    f = F.solve(cyl.rhs)
    out = dict(k=cyl.k, n=cyl.n, rule=rule, comp=comp, depth=tree.depth, drag=cyl.drag(f))
    out.update(_summary(F, cyl.A, cyl.rhs, f, f_dense, tf, True))
    out["err_mod_null"] = rel_err(project_out_normal(cyl, f), project_out_normal(cyl, f_dense))
    return out


def project_out_normal(cyl: Cylinder, f: Vector) -> Vector:
    """Remove the component along the normal density n: for Stokes AND Oseen, a uniform normal force density on a
    closed curve is balanced by pressure and makes no velocity, so n spans the (near-)null space of the
    single-layer matrix and the dense and fast densities may differ along it without any effect on the flow or
    the drag (int n_1 ds = 0)."""
    nvec = (cyl.X / cyl.a).ravel()
    nvec = nvec / np.linalg.norm(nvec)
    out: Vector = f - (nvec @ f) * nvec
    return out


def similarity_route(prob: AreaProblem, eps: float, x_dense: Vector) -> CaseResult:
    """Alternative to a new admissibility: A = D (I + K0/N) D^{-1}, D = diag(e^{lam x_1}) [proved from the
    factorisation G = e^{lam x_1} K0 e^{-lam y_1}]. Factor the SYMMETRIC Yukawa matrix (strong, stock proxy), solve
    (I + K0/N) y = D^{-1} b, return x = D y. Cheap ranks, but cond(D) = e^{lam * width}: measures the price."""
    from scipy.special import k0 as K0fun

    X = prob.X
    lam = prob.lam

    def yuk(P: Points, Q: Points) -> Matrix:
        r = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
        out = np.zeros_like(r)
        nz = r > 0
        out[nz] = K0fun(lam * r[nz]) / (2 * np.pi)
        return out

    B: Matrix = np.eye(len(X)) + yuk(X, X) / len(X)
    d = np.exp(lam * (X[:, 0] - X[:, 0].min()))
    tree = Tree(X, max_leaf=32)
    F = rskel_factor(B, tree, Strong(), ProxyCompressor(yuk, X, eps, A=B))
    x = d * F.solve(prob.b / d)
    return dict(lam=lam, N=len(X), rule="strong", comp="similarity+yukawa", err=rel_err(x, x_dense),
                skel_total=int(sum(v["skel_total"] for v in F.stats["levels"].values())),
                cond_D=float(d.max() / d.min()),
                residual=float(np.linalg.norm(prob.A @ x - prob.b) / np.linalg.norm(prob.b)))


def dense_solve(A: Matrix, b: Vector) -> Vector:
    x: Vector = sla.solve(A, b)
    return x
