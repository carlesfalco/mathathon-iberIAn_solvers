"""E9 solver variants: everything here REUSES unified.core / unified.drivers unchanged.

    exact      ExactCompressor (stacked ID of [A_FB; A_BFᵀ] read from the current matrix)
    realified  ID of [Re M; Im M] (E5's fix: T real, so the driver's Tᵀ row step is a congruence)
    proxy2     ProxyCompressor with the two-sided LS kernel (d at unknowns, 1 at proxy points)
    proxyN     ProxyCompressor with the naive LS kernel (contrast evaluated at proxy points too)

plus diagnostics read off a finished Factorization (pivot conditioning, Hermitian part, storage) and
per-box ranks of the one-sided blocks vs the stacked matrix (hypothesis H2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak, interp_decomp
from unified.core.tree import Node
from unified.core.types import Admissibility, Compressor, IDResult, IndexArray, Matrix, Vector
from unified.drivers.rskel import Factorization, rskel_factor

from .ls_problem import LSProblem, ProxyMode, ls_kernel

type VariantName = Literal["exact", "realified", "proxy2", "proxyN"]
type RuleName = Literal["weak", "strong"]
RULES: dict[RuleName, Admissibility] = {"weak": Weak(), "strong": Strong()}


class RealifiedCompressor:
    """Exact far field, but the ID is taken of [Re M; Im M] so the interpolation matrix T is real (E5).
    Has an `.A` attribute because rskel_factor points every compressor at its current matrix."""

    def __init__(self, A: Matrix, eps: float) -> None:
        self.A, self.eps = A, eps

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        M = np.vstack([self.A[np.ix_(far_idx, box_idx)], self.A[np.ix_(box_idx, far_idx)].T])
        return interp_decomp(np.vstack([M.real, M.imag]), self.eps)


def n_proxy_for(prob: LSProblem, tree: Tree, radius_factor: float = 2.5) -> int:
    """Proxy points for the largest compressed box: ~1.5 kR + 16 modes (E5 r9), at least 64."""
    hw = max(nd.half_width for nd in tree.levels[1]) if tree.depth >= 1 else tree.root.half_width
    return max(64, int(1.5 * prob.k * radius_factor * hw + 16))


def make_compressor(name: VariantName, prob: LSProblem, tree: Tree, eps: float) -> Compressor:
    if name == "exact":
        return ExactCompressor(prob.A, eps)
    if name == "realified":
        return RealifiedCompressor(prob.A, eps)
    mode: ProxyMode = "two_sided" if name == "proxy2" else "naive"
    return ProxyCompressor(ls_kernel(prob, mode), prob.X, eps, A=prob.A, n_proxy=n_proxy_for(prob, tree))


@dataclass
class RunResult:
    variant: str
    rule: str
    err: float               # ||x - x_dense|| / ||x_dense||
    relres: float            # ||A x - b|| / ||b||
    skel_total: int
    root: int
    skel_max_by_level: dict[int, int]
    storage: int             # complex numbers stored by the factorization
    cond_rr_max: float       # worst cond(X_RR) over eliminated pivot blocks
    herm_min: float          # min over boxes of lambda_min(Herm X_RR)
    t_factor: float
    x: Vector                # the fast solution


def pivot_blocks(f: Factorization) -> list[Matrix]:
    """Reconstruct every eliminated pivot block X_RR = P L U from its packed LU (scipy lu_factor format)."""
    out: list[Matrix] = []
    for s in f.steps:
        lu, piv = s.lu
        n = lu.shape[0]
        LU = (np.tril(lu, -1) + np.eye(n)) @ np.triu(lu)
        perm = np.arange(n)
        for i, p in enumerate(piv):
            perm[i], perm[p] = perm[p], perm[i]
        X = np.empty_like(LU)
        X[perm] = LU
        out.append(X)
    return out


def storage(f: Factorization) -> int:
    tot = sum(s.T.size + s.lu[0].size + s.X_snr.size + s.X_rsn.size for s in f.steps)
    return int(tot + f.root.size ** 2)


def run(prob: LSProblem, tree: Tree, rule: RuleName, variant: VariantName, eps: float, b: Vector,
        x_ref: Vector, diagnostics: bool = True) -> RunResult:
    """Factor A (copy) with rskel_factor, solve A x = b, compare with the dense solution x_ref."""
    comp = make_compressor(variant, prob, tree, eps)
    t0 = time.perf_counter()
    f = rskel_factor(prob.A, tree, RULES[rule], comp)
    t1 = time.perf_counter() - t0
    x = f.solve(b)
    err = float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    relres = float(np.linalg.norm(prob.A @ x - b) / np.linalg.norm(b))
    lv = f.stats["levels"]
    cmax, hmin = float("nan"), float("nan")
    if diagnostics and f.steps:
        blocks = pivot_blocks(f)
        cmax = max(float(np.linalg.cond(X)) for X in blocks)
        hmin = min(float(np.linalg.eigvalsh((X + X.conj().T) / 2)[0]) for X in blocks)
    return RunResult(variant, rule, err, relres, sum(v["skel_total"] for v in lv.values()), f.stats["root_size"],
                     {int(l): int(v["skel_max"]) for l, v in lv.items()}, storage(f), cmax, hmin, t1, x)


@dataclass
class StackRanks:
    level: int
    boxes: int
    col: float               # mean rank of A_FB          (columns scaled by d_B)
    row: float               # mean rank of A_BFᵀ         (= D_F K_FB: relations of K_FB)
    stack: float             # mean rank of [A_FB; A_BFᵀ] (what the driver's single ID must capture)
    ratio_max: float         # max over boxes of stack / max(col, row)


def stack_ranks(prob: LSProblem, tree: Tree, rule: RuleName, eps: float, level: int) -> StackRanks:
    """Ranks of the one-sided far blocks vs the stacked matrix, on the ORIGINAL A (no elimination)."""
    A, cols, rows, stacks, ratios = prob.A, [], [], [], []
    for B in tree.levels[level]:
        near = np.concatenate([nb.idx for nb in RULES[rule].near_nodes(tree, B)] + [np.zeros(0, np.intp)])
        mask = np.ones(prob.n, bool); mask[B.idx] = False; mask[near] = False
        far = np.flatnonzero(mask)
        if far.size == 0 or B.idx.size == 0:
            continue
        c = interp_decomp(A[np.ix_(far, B.idx)], eps).skel.size
        r = interp_decomp(A[np.ix_(B.idx, far)].T, eps).skel.size
        s = interp_decomp(np.vstack([A[np.ix_(far, B.idx)], A[np.ix_(B.idx, far)].T]), eps).skel.size
        cols.append(c); rows.append(r); stacks.append(s); ratios.append(s / max(c, r, 1))
    return StackRanks(level, len(cols), float(np.mean(cols)), float(np.mean(rows)), float(np.mean(stacks)),
                      float(np.max(ratios)))


# ---- symmetrized LS (Ho-Ying 2016 HIF-IE, Example 3): removes the one-sided penalty of H2 ----------
def symmetrized(prob: LSProblem) -> tuple[Matrix, Matrix, Vector]:
    """Â = I + D^½ K D^½ (complex SYMMETRIC, so A_BFᵀ = A_FB and one ID serves rows and columns exactly),
    the kernel matrix K, and √d. Solve Â y = √d ∘ f, then recover u = f - K (√d ∘ y) by one kernel apply
    (an FMM in practice; dense here). No division by √d, so contrast -> 0 at the edge of a bump is safe."""
    sd: Vector = np.sqrt(prob.d)
    K: Matrix = (prob.A - np.eye(prob.n)) / prob.d[None, :]      # every unknown has q > 0
    Ah: Matrix = sd[:, None] * K * sd[None, :]
    Ah[np.diag_indices(prob.n)] += 1
    return Ah, K, sd


def run_symmetrized(prob: LSProblem, tree: Tree, rule: RuleName, eps: float, b: Vector, x_ref: Vector,
                    realify: bool = False) -> RunResult:
    """Exact-ID RS on Â, error measured on the recovered u."""
    Ah, K, sd = symmetrized(prob)
    comp: Compressor = RealifiedCompressor(Ah, eps) if realify else ExactCompressor(Ah, eps)
    t0 = time.perf_counter()
    f = rskel_factor(Ah, tree, RULES[rule], comp)
    t1 = time.perf_counter() - t0
    y = f.solve(sd * b)
    x: Vector = b - K @ (sd * y)
    err = float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    relres = float(np.linalg.norm(prob.A @ x - b) / np.linalg.norm(b))
    lv = f.stats["levels"]
    blocks = pivot_blocks(f)
    cmax = max((float(np.linalg.cond(X)) for X in blocks), default=1.0)
    hmin = min((float(np.linalg.eigvalsh((X + X.conj().T) / 2)[0]) for X in blocks), default=1.0)
    return RunResult("sym-real" if realify else "sym", rule, err, relres, sum(v["skel_total"] for v in lv.values()),
                     f.stats["root_size"], {int(l): int(v["skel_max"]) for l, v in lv.items()}, storage(f),
                     cmax, hmin, t1, x)
