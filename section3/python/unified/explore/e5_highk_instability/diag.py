"""E5 instrumentation: per-box diagnostics of rskel_factor, and one `run_case` for every compressor variant.

For each compressed box we rebuild the sparsified pivot exactly as the driver forms it,
    X_RR = Pi^T W_BB Pi,   Pi = [I_R; -T]  (rows R then S of the box),
from the CURRENT working matrix W (after all earlier Schur complements), and record its condition number, the
smallest eigenvalue of its Hermitian part, the same for W_BB, ||W_BB - I||, and how complex T is.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

import numpy as np

import unified.drivers.rskel as rskel_mod
from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak
from unified.core.tree import Node
from unified.core.types import Admissibility, Compressor, IDResult, IndexArray, InterpMatrix, Matrix, Vector
from unified.drivers.rskel import rskel_factor

from ._deps import Problem
from .congruent import congruent_factor
from .fixes import (AbsTolCompressor, HybridCompressor, OversampledCompressor, RealifiedCompressor,
                    SRRQRCompressor)

type RuleName = Literal["weak", "strong"]
type Variant = Literal["exact", "proxy", "realified", "congruent", "hybrid", "oversample8", "srrqr", "abstol"]
type Wavenumber = float
type CaseResult = dict[str, Any]          # flat JSON-able record of one factor + solve
type Basis = Matrix                       # Pi = [I; -T] in box coordinates, columns span the redundant space

VARIANTS: tuple[Variant, ...] = ("exact", "proxy", "realified", "congruent", "hybrid", "oversample8", "srrqr",
                                 "abstol")
RULES: dict[RuleName, type[Weak] | type[Strong]] = {"weak": Weak, "strong": Strong}


class BoxRecord(TypedDict):
    level: int
    half_width: float
    k_hw: float              # k * half-width (box size in radians / 2)
    nb: int                  # active unknowns of the box
    ns: int                  # skeletons kept
    cond_x: float            # cond(X_RR)
    herm_min_x: float        # lambda_min((X_RR + X_RR^H) / 2)
    herm_min_wbb: float      # lambda_min((W_BB + W_BB^H) / 2)
    dev_wbb: float           # ||W_BB - I||_2
    t_norm: float            # ||T||_2
    t_max: float             # max |T_ij| (largest skeleton weight)
    t_cond: float            # cond(T) = sigma_max / sigma_min of the k x |R| interpolation matrix
    t_imag: float            # ||Im T||_F / ||T||_F  (0 = real T: the sparsification is a congruence)


def redundant_basis(nb: int, skel_loc: IndexArray, red_loc: IndexArray, T: InterpMatrix) -> Basis:
    """Pi with Pi[red] = I, Pi[skel] = -T (box-local coordinates): W_BB-sparsified X_RR = Pi^T W_BB Pi."""
    Pi: Basis = np.zeros((nb, len(red_loc)), complex)
    Pi[red_loc] = np.eye(len(red_loc))
    Pi[skel_loc] = -T
    return Pi


def _cond(T: InterpMatrix) -> float:
    if T.size == 0:
        return 1.0
    s = np.linalg.svd(T, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def box_record(W: Matrix, box: IndexArray, res: IDResult, node: Node, k: Wavenumber) -> BoxRecord:
    sk, rd, T = res
    WBB: Matrix = W[np.ix_(box, box)]
    Pi = redundant_basis(len(box), sk, rd, T)
    X: Matrix = Pi.T @ WBB @ Pi
    tn = float(np.linalg.norm(T))
    return BoxRecord(level=node.level, half_width=node.half_width, k_hw=k * node.half_width, nb=len(box),
                     ns=len(sk), cond_x=float(np.linalg.cond(X)),
                     herm_min_x=float(np.linalg.eigvalsh((X + X.conj().T) / 2)[0]),
                     herm_min_wbb=float(np.linalg.eigvalsh((WBB + WBB.conj().T) / 2)[0]),
                     dev_wbb=float(np.linalg.norm(WBB - np.eye(len(box)), 2)),
                     t_norm=float(np.linalg.norm(T, 2)), t_max=float(np.abs(T).max()) if T.size else 0.0,
                     t_cond=_cond(T),
                     t_imag=float(np.linalg.norm(np.imag(T)) / tn) if tn > 0 else 0.0)


@contextlib.contextmanager
def instrument(records: list[BoxRecord], k: Wavenumber) -> Iterator[list[BoxRecord]]:
    """Wrap rskel._skeletonize; append one BoxRecord per box that eliminates something. Restores on exit."""
    orig = rskel_mod._skeletonize

    def wrapped(compressor: Any, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
                q_extra: IndexArray) -> IDResult:
        res: IDResult = orig(compressor, W, box, far, node, q_extra)
        if len(res.red):
            records.append(box_record(W, box, res, node, k))
        return res

    rskel_mod._skeletonize = wrapped
    try:
        yield records
    finally:
        rskel_mod._skeletonize = orig


def make_comp(kind: Variant, prob: Problem, eps: float, k: Wavenumber) -> Compressor:
    A = prob.A
    if kind == "exact":
        return ExactCompressor(A, eps)
    if kind == "proxy":
        return ProxyCompressor(prob.kernel, prob.X, eps, A=A)
    if kind == "realified":
        return RealifiedCompressor(A, eps)
    if kind == "hybrid":
        return HybridCompressor(A, eps, prob.kernel, prob.X, k)
    if kind == "oversample8":
        return OversampledCompressor(A, eps, p=8)
    if kind == "srrqr":
        return SRRQRCompressor(A, eps)
    if kind == "abstol":
        return AbsTolCompressor(A, eps)
    raise ValueError(kind)


def rhs(n: int) -> Vector:
    rng = np.random.default_rng(1)
    b: Vector = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return b


def run_case(prob: Problem, rule: RuleName, kind: Variant, eps: float, k: Wavenumber, x_dense: Vector,
             records: list[BoxRecord] | None = None, max_leaf: int = 32) -> CaseResult:
    """Factor + solve; err = ||x - x_dense|| / ||x_dense||, worst cond(X_RR), total skeletons.
    If `records` is given, per-box diagnostics are appended (not available for 'congruent')."""
    n = len(prob.X)
    tree = Tree(prob.X, max_leaf=max_leaf)
    adm: Admissibility = RULES[rule]()
    b = rhs(n)
    if kind == "congruent":
        G = congruent_factor(prob.A, tree, adm, eps)
        x, cond, skel = G.solve(b), G.cond_rr_max(), G.skel_total
    else:
        comp = make_comp(kind, prob, eps, k)
        with contextlib.ExitStack() as st:
            if records is not None:
                st.enter_context(instrument(records, k))
            F = rskel_factor(prob.A, tree, adm, comp)
        x = F.solve(b)
        cond = max((float(np.linalg.cond(np.triu(s.lu[0]))) for s in F.steps), default=1.0)
        skel = int(sum(v["skel_total"] for v in F.stats["levels"].values()))
    err = float(np.linalg.norm(x - x_dense) / np.linalg.norm(x_dense))
    return dict(rule=rule, comp=kind, eps=eps, k=k, N=n, depth=tree.depth, err=err, cond_rr_max=cond,
                skel_total=skel)
