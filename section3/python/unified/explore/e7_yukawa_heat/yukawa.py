"""E7: the modified Helmholtz (Yukawa) kernel G_κ(x, y) = K0(κ|x − y|)/(2π) on the starfish, plus the two
small compressor wrappers that test the admissibility / tolerance questions.

Everything the harness needs is a kernel function and a dense matrix; the drivers are untouched.

    slp_kernel(κ)              K0(κ r)/(2π)                         (dial-free swap of P1's kernel)
    dlp_problem(κ, n)          A = −½ I + D_κ w  (interior Dirichlet, second kind; the physics problem)
    AbsTolExact(A, tau)        exact ID with an ABSOLUTE threshold tau instead of ε·|R11|
    Truncated(base, X, rcut)   drop far points farther than rcut (screening truncation) before the ID

Sign conventions: n_y is the outward normal, D_κ(x, y) = ∂_{n_y} G_κ(x, y) = −κ K1(κr) (y − x)·n_y / (2π r).
The interior limit of the double layer is −½σ + D_κσ (the same jump as Laplace, since κK1(κr) = 1/r + O(r log r)).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, kve

from unified.bench.problems import Problem, dense_identity_plus, starfish
from unified.core import ProxyCompressor, interp_decomp
from unified.core.tree import Node, Tree
from unified.core.types import Admissibility, IDResult, IndexArray, Kernel, KernelMatrix, Matrix, Points, RealArray, Vector

type Screening = float          # κ = 1/sqrt(dt): inverse screening length (Debye length 1/κ)


# ---- starfish geometry: r(θ) = 1 + 0.3 cos 5θ ----------------------------------------------------------
def _gamma_derivs(th: RealArray) -> tuple[RealArray, RealArray]:
    """γ'(θ) and γ''(θ) for γ(θ) = r(θ)(cos θ, sin θ)."""
    r, rp, rpp = 1 + 0.3 * np.cos(5 * th), -1.5 * np.sin(5 * th), -7.5 * np.cos(5 * th)
    c, s = np.cos(th), np.sin(th)
    d1 = np.c_[rp * c - r * s, rp * s + r * c]
    d2 = np.c_[(rpp - r) * c - 2 * rp * s, (rpp - r) * s + 2 * rp * c]
    return d1, d2


def normals_curvature(Y: Points) -> tuple[RealArray, RealArray, NDArray[np.bool_]]:
    """For points on the starfish (θ = atan2): outward unit normal, signed curvature, and an on-curve mask."""
    Y = np.atleast_2d(Y)
    th = np.arctan2(Y[:, 1], Y[:, 0])
    on: NDArray[np.bool_] = np.abs(np.hypot(Y[:, 0], Y[:, 1]) - (1 + 0.3 * np.cos(5 * th))) < 1e-9
    d1, d2 = _gamma_derivs(th)
    sp = np.hypot(d1[:, 0], d1[:, 1])
    nrm: RealArray = np.c_[d1[:, 1], -d1[:, 0]] / sp[:, None]
    curv: RealArray = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp ** 3
    return nrm, curv, on


def _diff(X: Points, Y: Points) -> tuple[RealArray, RealArray]:
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    dv: RealArray = Y[None, :, :] - X[:, None, :]            # y − x, shape (m, n, 2)
    return dv, np.linalg.norm(dv, axis=2)


# ---- kernels ---------------------------------------------------------------------------------------------
def slp_kernel(kappa: Screening) -> Kernel:
    """G_κ = K0(κ r)/(2π); zero on the diagonal (core convention). kve keeps large κr finite until underflow."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        _, r = _diff(X, Y)
        out: RealArray = np.zeros(r.shape)
        nz = r > 0
        z = kappa * r[nz]
        out[nz] = kve(0, z) * np.exp(-z) / (2 * np.pi)
        return out
    return K


def history_coef(p: int, z: RealArray) -> RealArray:
    """(z/2)^p / p! · K_{p−1}(z), computed in logs (p ≥ 1). It is the radial factor of the Rothe history kernel L_p."""
    out: RealArray = np.exp(p * np.log(z / 2) - gammaln(p + 1) - z) * kve(p - 1, z)
    return out


def monopole_p(kappa: Screening, p: int, r: RealArray) -> RealArray:
    """M_p(r) = (z/2)^p K_p(z) / (2π p!), z = κr: the point-source field after p extra backward-Euler steps,
    (I − dtΔ) M_p = M_{p−1}, M_0 = G_κ (κ² = 1/dt)."""
    z = kappa * r
    out: RealArray = np.exp(p * np.log(z / 2) - gammaln(p + 1) - z) * kve(p, z) / (2 * np.pi)
    return out


def dlp_kernel(kappa: Screening, n: int, p: int = 0) -> Kernel:
    """Weighted double layer (p = 0) or Rothe history layer L_p = (−1/dt)^p/p! ∂^p_{κ²} D_κ (p ≥ 1):

        L_p(x, y) w(y) = −(κ/2π) (z/2)^p/p! K_{p−1}(z) (y − x)·n_y / r · w(y),   z = κ r,  K_{−1} = K_1.

    Columns y ON the starfish get the normal and the trapezoid weight recovered from θ = atan2 (P1's trick).
    Columns OFF the curve (proxy points used as SOURCES) get the plain monopole G_κ: by Graf's theorem a ring
    of monopoles spans the regular solutions I_m(κρ)e^{imφ} inside the ring for every κ > 0, which is all the
    proxy compressor needs; a dipole with a made-up normal would also work but has no meaning. Diagonal = 0."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        Y = np.atleast_2d(Y)
        nrm, _, on = normals_curvature(Y)
        th = np.arctan2(Y[:, 1], Y[:, 0])
        d1, _ = _gamma_derivs(th)
        w = np.hypot(d1[:, 0], d1[:, 1]) * 2 * np.pi / n
        dv, r = _diff(X, Y)
        out: RealArray = np.zeros(r.shape)
        nz = r > 0
        z = kappa * r[nz]
        radial = kve(1, z) * np.exp(-z) if p == 0 else history_coef(p, z)
        dotn = np.einsum("ijk,jk->ij", dv, nrm)[nz]
        out[nz] = -kappa / (2 * np.pi) * radial * dotn / r[nz]
        out *= w[None, :]
        if not on.all():                                   # proxy points as sources: monopoles
            out[:, ~on] = slp_kernel(kappa)(X, Y[~on])
        return out
    return K


def dlp_problem(kappa: Screening, n: int) -> Problem:
    """Interior Dirichlet problem (−Δ + κ²)u = 0 in the starfish, u = g on Γ, with u = D_κ[σ]:
        (−½ I + D_κ w) σ = g.   Diagonal: −½ + (−curv/(4π)) w  (the Laplace limit of the double-layer kernel)."""
    X, w = starfish(n)
    K = dlp_kernel(kappa, n)
    A = dense_identity_plus(K, X)
    _, curv, _ = normals_curvature(X)
    A[np.diag_indices(n)] += -1.5 - curv * w / (4 * np.pi)   # dense_identity_plus added +1; we want −½
    return Problem(f"Y2-DLP(N={n},kappa={kappa:g})", X, A, K)


def slp_problem(kappa: Screening, n: int) -> Problem:
    """Y1: P1 with ONLY the kernel swapped, A = I + S_κ w (the one-line change)."""
    from unified.bench.problems import _weighted
    X, _ = starfish(n)
    K = _weighted(slp_kernel(kappa), n)
    return Problem(f"Y1-SLP(N={n},kappa={kappa:g})", X, dense_identity_plus(K, X), K)


def dlp_eval(kappa: Screening, n: int, targets: Points, sigma: Vector, p: int = 0) -> Vector:
    """u(x) = L_p[σ](x) at interior targets (smooth trapezoid rule; targets must be ≫ h from Γ)."""
    X, _ = starfish(n)
    out: Vector = dlp_kernel(kappa, n, p)(targets, X) @ sigma
    return out


# ---- compressor wrappers (dials 2 and 4) -----------------------------------------------------------------
def abs_id(M: Matrix, tau: float) -> IDResult:
    """Column ID keeping pivots |R_kk| > tau (absolute), via the core ID: its threshold is eps·|R_11| and
    |R_11| is the largest column norm, so eps = tau / max-column-norm. Rank 0 if every column is below tau."""
    n = M.shape[1]
    c = float(np.max(np.linalg.norm(M, axis=0))) if M.size else 0.0
    if c <= tau:
        return IDResult(np.zeros(0, np.intp), np.arange(n, dtype=np.intp), np.zeros((0, n)))
    return interp_decomp(M, tau / c)


@dataclass
class AbsTolExact:
    """ExactCompressor with an absolute tolerance tau (= ε × matrix scale). `A` is reset by the driver."""
    A: Matrix
    tau: float

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        M = np.vstack([self.A[np.ix_(far_idx, box_idx)], self.A[np.ix_(box_idx, far_idx)].T])
        return abs_id(M, self.tau)


@dataclass
class Truncated:
    """Screening truncation: before the ID, drop far unknowns farther than `rcut` from the box (box radius
    √2·hw added, so the test is conservative). If nothing is left, the box has rank 0 (all redundant).
    With a ProxyCompressor base and every kept far point inside the proxy ring, the ring is skipped and the
    ID uses only those explicit rows. Weak admissibility only for proxy bases (the driver's FMM-LU Q set is
    handled inside rskel for bare ProxyCompressors only)."""
    base: AbsTolExact | ProxyCompressor
    points: Points
    rcut: float
    A: Matrix | None = None

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        d = np.linalg.norm(self.points[far_idx] - node.center, axis=1)
        keep: IndexArray = far_idx[d < self.rcut + np.sqrt(2) * node.half_width]
        if keep.size == 0:
            n = box_idx.size
            return IDResult(np.zeros(0, np.intp), np.arange(n, dtype=np.intp), np.zeros((0, n)))
        b = copy.copy(self.base)
        b.A = self.A if self.A is not None else b.A
        if isinstance(b, ProxyCompressor):
            ring = b.radius_factor * node.half_width
            if np.all(np.linalg.norm(self.points[keep] - node.center, axis=1) < ring):
                W = b.A
                assert W is not None
                return interp_decomp(np.vstack([W[np.ix_(keep, box_idx)], W[np.ix_(box_idx, keep)].T]), b.eps)
        return b(box_idx, keep, node)


def cutoff_radius(kappa: Screening, eps: float, n: int, wmax: float, scale: float = 0.5) -> float:
    """Smallest R with  max_entry(r ≥ R) · n ≤ eps·scale, where max_entry(r) bounds |D_κ w| and |G_κ w| (both
    ≤ (κK1(κr) + K0(κr))/(2π) · wmax): then the dropped part of every row has 1-norm ≤ eps·scale. Bisection."""
    def bound(r: float) -> float:
        z = kappa * r
        return float((kappa * kve(1, z) + kve(0, z)) * np.exp(-z) / (2 * np.pi) * wmax * n)
    lo, hi = 1e-6, 50.0
    if bound(hi) > eps * scale:
        return np.inf
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if bound(mid) > eps * scale else (lo, mid)
    return hi


class CenteredProxy(ProxyCompressor):
    """κ → 0 fix for the monopole proxy rows. K0(κr) = −log r + c(κ) + o(1) with c(κ) = log(2/κ) − γ → ∞, so
    every ring row K0(κ|x − p|) carries a huge common constant and the relative ID only resolves the useful
    −log variation to ε·c(κ). Replace each monopole row by its deviation from the value at the box centre
    and add ONE constant row: span{rows} ⊆ span{centred rows, 1} exactly, and the scales are now O(1)."""

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        X = self.points
        r = self.radius_factor * node.half_width
        inside: IndexArray = far_idx[np.linalg.norm(X[far_idx] - node.center, axis=1) < r] if far_idx.size else far_idx
        Pp = self.proxy_points(node)
        S = self.kernel(X[box_idx], Pp).T                               # monopole rows (n_p, |B|)
        S_c = S - self.kernel(node.center[None, :], Pp).T               # minus the value at the box centre
        blocks: list[Matrix] = [self.kernel(Pp, X[box_idx]), S_c, np.full((1, box_idx.size), float(np.max(np.abs(S_c))))]
        if inside.size:
            assert self.A is not None
            blocks += [self.A[np.ix_(inside, box_idx)], self.A[np.ix_(box_idx, inside)].T]
        return interp_decomp(np.vstack(blocks), self.eps)


@dataclass(eq=False)
class NearTruncated:
    """Near-field truncation (dial 2): the base rule's near list (Strong: touching boxes) minus every
    neighbour N that is negligible or cheap to compress, judged on the ORIGINAL blocks
    M = [A[N,B]; A[B,N]ᵀ]:

        rank_frac = 0   (norm criterion):  demote if max(‖A[B,N]‖_F, ‖A[N,B]‖_F) ≤ tau
        rank_frac > 0   (rank criterion):  demote if #{σ_i(M) > tau} ≤ rank_frac · min(|B|, |N|)

    A demoted neighbour joins the far field, so the box is compressed against it instead of being
    Schur-coupled to it. Accuracy does not rest on this heuristic: the compressor (AbsTolExact, reading the
    CURRENT matrix W) sees the demoted neighbour's actual entries, including Schur fill-in, and keeps
    whatever rank they need at the absolute tolerance. The heuristic only decides what stays exact (cost).
    Every neighbour demoted: the rule IS Weak; none demoted: it IS Strong."""
    base: Admissibility
    A: Matrix
    tau: float
    rank_frac: float = 0.0
    name: str = "strong-nt"
    kept: int = 0
    dropped: int = 0
    _cache: dict[tuple[int, int], bool] = field(default_factory=dict)

    def _negligible(self, B: Node, N: Node) -> bool:
        key = (id(B), id(N))
        if key not in self._cache:
            AB, AN = self.A[np.ix_(B.idx, N.idx)], self.A[np.ix_(N.idx, B.idx)]
            if self.rank_frac == 0.0:
                self._cache[key] = max(float(np.linalg.norm(AB)), float(np.linalg.norm(AN))) <= self.tau
            else:
                s = np.linalg.svd(np.vstack([AN, AB.T]), compute_uv=False)
                self._cache[key] = int(np.sum(s > self.tau)) <= self.rank_frac * min(B.idx.size, N.idx.size)
        return self._cache[key]

    def near_nodes(self, tree: Tree, node: Node) -> list[Node]:
        nbs = self.base.near_nodes(tree, node)
        out = [nb for nb in nbs if not self._negligible(node, nb)]
        self.kept += len(out)
        self.dropped += len(nbs) - len(out)
        return out
