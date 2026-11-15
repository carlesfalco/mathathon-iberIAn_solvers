"""One measurement: build A for a geometry, factor it with the UNCHANGED rskel_factor, solve, compare to dense LU."""
from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

from unified.core import ExactCompressor, Strong, Tree, Weak
from unified.core.types import Admissibility, Compressor, Matrix, Vector
from unified.drivers.rskel import Factorization, rskel_factor

from .compressor import BlockProxyCompressor, ProxySources
from .stokes import Geometry, laplace_matrix, stokes_matrix

type Physics = Literal["stokes", "laplace"]
type RuleName = Literal["weak", "strong"]
type CompKind = Literal["exact", "proxy-sl", "proxy-dl", "proxy-both", "proxy-norigid"]
type CaseResult = dict[str, Any]


def build(g: Geometry, physics: Physics) -> tuple[Matrix, Tree, int]:
    """Dense A, the tree over its unknowns (dof points for Stokes), and the number of components."""
    depth = Tree(g.X, max_leaf=32).depth          # same geometric boxes for both physics
    if physics == "stokes":
        return stokes_matrix(g), Tree(g.dof_points(), depth=depth), 2
    return laplace_matrix(g), Tree(g.X, depth=depth), 1


def make_comp(kind: CompKind, g: Geometry, A: Matrix, tree: Tree, nc: int, rule: RuleName, eps: float,
              n_proxy: int = 64, radius_factor: float = 2.5) -> Compressor:
    if kind == "exact":
        return ExactCompressor(A, eps)
    sources: dict[str, ProxySources] = {"proxy-sl": "sl", "proxy-dl": "dl", "proxy-both": "both", "proxy-norigid": "sl"}
    src = sources[kind]
    return BlockProxyCompressor(g, eps, A, tree, ncomp=nc, sources=src, n_proxy=n_proxy,
                                radius_factor=radius_factor, rigid_rows=kind != "proxy-norigid",
                                strong=rule == "strong")


def storage(F: Factorization) -> int:
    """Numbers stored by the factorization (T, LU of X_RR, the two off-diagonal blocks, root LU)."""
    s = sum(st.T.size + st.lu[0].size + st.X_snr.size + st.X_rsn.size for st in F.steps)
    return int(s + (F.root_lu[0].size if F.root_lu is not None else 0))


def case(g: Geometry, physics: Physics, rule: RuleName, kind: CompKind, eps: float = 1e-10,
         b: Vector | None = None, A: Matrix | None = None, x_dense: Vector | None = None,
         n_proxy: int = 64, radius_factor: float = 2.5, keep: bool = False) -> CaseResult:
    """Factor + solve; returns err vs dense, residual, skeleton sizes per level, storage, times."""
    A0, tree, nc = build(g, physics)
    if A is None:
        A = A0
    rhs: Vector = np.random.default_rng(1).standard_normal(size=(int(A.shape[0]),)) if b is None else b
    comp = make_comp(kind, g, A, tree, nc, rule, eps, n_proxy, radius_factor)
    adm: Admissibility = Weak() if rule == "weak" else Strong()
    t0 = time.perf_counter()
    F = rskel_factor(A, tree, adm, comp)
    tf = time.perf_counter() - t0
    x = F.solve(rhs)
    if x_dense is None:
        x_dense = np.linalg.solve(A, rhs)
    lv = F.stats["levels"]
    out: CaseResult = dict(
        physics=physics, rule=rule, comp=kind, eps=eps, N_pts=g.n, N_dof=A.shape[0], depth=tree.depth,
        err=float(np.linalg.norm(x - x_dense) / np.linalg.norm(x_dense)),
        residual=float(np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)),
        skel_mean={int(l): float(v["skel_mean"]) for l, v in lv.items()},
        skel_max={int(l): int(v["skel_max"]) for l, v in lv.items()},
        skel_total=int(sum(v["skel_total"] for v in lv.values())), root=int(F.stats["root_size"]),
        storage=storage(F), factor_s=tf)
    if keep:
        out["_F"] = F
        out["_x"] = x
    return out
