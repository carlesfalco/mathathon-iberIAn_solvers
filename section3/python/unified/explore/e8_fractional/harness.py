"""One measurement: factor a FracProblem with rskel_factor + a chosen compressor, solve, compare to dense.

The only things that vary between runs are the kernel (the problem) and dial 4 (the compressor).
Tree (dial 1), admissibility (dial 2: Weak / Strong), the ID representation (dial 3) and the factor
algebra (dial 5) are the frozen core / driver code, imported read-only.
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

import numpy as np

import unified.drivers.rskel as rskel_mod
from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak
from unified.core.tree import Node
from unified.core.types import Compressor, IDResult, IndexArray, Matrix, Vector
from unified.drivers.rskel import rskel_factor

from .fractional import FracProblem, rel_err
from .proxies import AnnulusProxy, LiftedProxy

type RuleName = Literal["weak", "strong"]
type CompKind = Literal["exact", "plane", "lifted", "annulus"]
type CaseResult = dict[str, Any]

COMPS: tuple[CompKind, ...] = ("exact", "plane", "lifted", "annulus")
RULES: dict[RuleName, type[Weak] | type[Strong]] = {"weak": Weak, "strong": Strong}


class BoxRecord(TypedDict):
    level: int
    nb: int          # active unknowns of the box
    k: int           # skeletons kept (the rank)
    rows: int        # rows of the ID input matrix (cost driver: ID flops ~ 2 rows nb^2)
    n_proxy: int


def make_comp(kind: CompKind, prob: FracProblem, eps: float, **opts: Any) -> Compressor:
    if kind == "exact":
        return ExactCompressor(prob.A, eps)
    if kind == "plane":
        return ProxyCompressor(prob.kernel, prob.X, eps, A=prob.A, **opts)
    if kind == "lifted":
        return LiftedProxy(prob.kernel, prob.X, eps, A=prob.A, **opts)
    if kind == "annulus":
        return AnnulusProxy(prob.kernel, prob.X, eps, A=prob.A, **opts)
    raise ValueError(kind)


@contextlib.contextmanager
def instrument(records: list[BoxRecord]) -> Iterator[list[BoxRecord]]:
    """Temporarily wrap rskel._skeletonize to log one BoxRecord per compressed box (driver file untouched)."""
    orig = rskel_mod._skeletonize

    def wrapped(comp: Any, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
                q_extra: IndexArray) -> IDResult:
        res: IDResult = orig(comp, W, box, far, node, q_extra)
        if isinstance(comp, ProxyCompressor):
            inside = far[np.linalg.norm(comp.points[far] - node.center, axis=1) < comp.radius_factor * node.half_width]
            npx = len(comp.proxy_points(node))
            rows = 2 * (npx + np.union1d(inside, q_extra).size)
        else:
            npx, rows = 0, 2 * far.size
        records.append(BoxRecord(level=node.level, nb=int(box.size), k=int(len(res.skel)), rows=int(rows),
                                 n_proxy=int(npx)))
        return res

    setattr(rskel_mod, "_skeletonize", wrapped)
    try:
        yield records
    finally:
        setattr(rskel_mod, "_skeletonize", orig)


def dense_solve(prob: FracProblem) -> Vector:
    x: Vector = np.linalg.solve(prob.A, prob.b)
    return x


def case(prob: FracProblem, rule: RuleName, kind: CompKind, eps: float = 1e-8, max_leaf: int = 32,
         x_dense: Vector | None = None, **opts: Any) -> CaseResult:
    """err = ||x - x_dense|| / ||x_dense||, residual, per-level max rank, total skeletons, ID rows (cost)."""
    n = len(prob.X)
    tree = Tree(prob.X, max_leaf=max_leaf)
    comp = make_comp(kind, prob, eps, **opts)
    recs: list[BoxRecord] = []
    t0 = time.perf_counter()
    with instrument(recs):
        F = rskel_factor(prob.A, tree, RULES[rule](), comp)
    tf = time.perf_counter() - t0
    x = F.solve(prob.b)
    xd = dense_solve(prob) if x_dense is None else x_dense
    lv = F.stats["levels"]
    out: CaseResult = dict(
        problem=prob.name, family=prob.family, d=prob.d, s=prob.s, p=prob.p, N=n, rule=rule, comp=kind,
        eps=eps, depth=tree.depth, err=rel_err(x, xd),
        residual=float(np.linalg.norm(prob.A @ x - prob.b) / np.linalg.norm(prob.b)),
        rank_max={int(l): int(max((r["k"] for r in recs if r["level"] == l), default=0)) for l in lv},
        skel_total=int(sum(v["skel_total"] for v in lv.values())), root=int(F.stats["root_size"]),
        id_rows=int(sum(r["rows"] for r in recs)),
        id_flops=float(sum(2.0 * r["rows"] * r["nb"] ** 2 for r in recs)),
        n_proxy_max=int(max((r["n_proxy"] for r in recs), default=0)), factor_s=tf, **{k: v for k, v in opts.items()})
    if prob.u_exact is not None:
        out["err_vs_exact"] = rel_err(x, prob.u_exact)
        out["dense_err_vs_exact"] = rel_err(xd, prob.u_exact)
    return out
