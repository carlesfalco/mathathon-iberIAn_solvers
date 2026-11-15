"""One measurement: factor a Problem with rskel_factor + a chosen compressor, solve, compare to dense."""
from __future__ import annotations
import time
from typing import Any, Literal

import numpy as np
from ._deps import Problem
from unified.core import Tree, Weak, Strong, ExactCompressor, ProxyCompressor
from unified.core.types import Admissibility, Compressor, Vector
from unified.drivers.rskel import rskel_factor, LevelStats
from .compressors import RandomizedCompressor, MultiRingProxy, instrument, cost_summary, SkeletonRecord

type RuleName = Literal["weak", "strong"]
type CompKind = Literal["exact", "proxy", "proxy2", "randomized"]
type Wavenumber = float
type CaseResult = dict[str, Any]     # flat JSON-able record of one factor + solve

RULES: dict[RuleName, type[Weak] | type[Strong]] = {"weak": Weak, "strong": Strong}


def make_comp(kind: CompKind, prob: Problem, eps: float, k_wave: Wavenumber = 0.0, seed: int = 0) -> Compressor:
    """Build the compression engine to plug into rskel_factor."""
    if kind == "exact":
        return ExactCompressor(prob.A, eps)
    if kind == "proxy":
        return ProxyCompressor(prob.kernel, prob.X, eps, A=prob.A)
    if kind == "proxy2":   # two rings + n_proxy growing with kR
        return MultiRingProxy(prob.kernel, prob.X, eps, A=prob.A, rings=2, k=k_wave)
    if kind == "randomized":
        return RandomizedCompressor(prob.A, eps, seed=seed)
    raise ValueError(kind)


def _rhs(prob: Problem) -> Vector:
    n = len(prob.X)
    rng = np.random.default_rng(1)
    b: Vector = rng.standard_normal(n)
    if np.iscomplexobj(prob.A):
        b = b + 1j * rng.standard_normal(n)
    return b


def case(prob: Problem, rule: RuleName, kind: CompKind, eps: float = 1e-8, max_leaf: int = 32,
         k_wave: Wavenumber = 0.0, x_dense: Vector | None = None, seed: int = 0) -> CaseResult:
    """Returns: residual ||Ax-b||/||b||, err ||x-x_dense||/||x_dense||, skeleton sizes, cost, worst cond(X_RR)."""
    n = len(prob.X)
    tree = Tree(prob.X, max_leaf=max_leaf)
    comp = make_comp(kind, prob, eps, k_wave, seed)
    adm: Admissibility = RULES[rule]()
    recs: list[SkeletonRecord] = []
    t0 = time.perf_counter()
    with instrument(recs):
        F = rskel_factor(prob.A, tree, adm, comp)
    tf = time.perf_counter() - t0
    b = _rhs(prob)
    x = F.solve(b)
    if x_dense is None:
        x_dense = np.linalg.solve(prob.A, b)
    res = float(np.linalg.norm(prob.A @ x - b) / np.linalg.norm(b))
    err = float(np.linalg.norm(x - x_dense) / np.linalg.norm(x_dense))
    lv: dict[int, LevelStats] = F.stats["levels"]
    # worst condition number of an eliminated (sparsified) redundant block X_RR: the pivot of each step
    cond_rr = max((float(np.linalg.cond(np.triu(s.lu[0]))) for s in F.steps), default=1.0)
    cost = cost_summary(recs, n)
    return dict(problem=prob.name, rule=rule, comp=kind, eps=eps, N=n, depth=tree.depth,
                residual=res, err=err, factor_s=tf,
                skel_max={int(lvl): int(v["skel_max"]) for lvl, v in lv.items()},
                skel_total=int(sum(v["skel_total"] for v in lv.values())),
                root=int(F.stats["root_size"]),
                rank_sum=int(sum(r["k"] for r in recs)), cond_rr_max=cond_rr, **cost)


def dense_ref(prob: Problem) -> Vector:
    """Dense LU solution for the same right-hand side `case` uses."""
    x: Vector = np.linalg.solve(prob.A, _rhs(prob))
    return x
