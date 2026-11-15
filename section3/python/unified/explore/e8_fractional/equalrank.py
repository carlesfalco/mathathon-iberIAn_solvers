"""E8 follow-up: compare compressors at EQUAL TOLERANCE and at EQUAL RANK (ranks copied box by box from the
exact-ID run), measuring the true far-field ID error of every box, the solve error and the cost.

Mechanics (no file outside this directory is edited; everything is restored on exit):
  * `instrument_far` wraps rskel._skeletonize. Before each box it publishes the box key (level, coord) for
    the fixed-rank ID; after it, it measures the ID error against the TRUE far block of the current
    (Schur-complemented) matrix W:  M = [W[F, B]; W[B, F]^T],  err = ||M_R - M_S T||_F / ||M||_F.
  * `fixed_ranks(schedule)` replaces interp_decomp (in unified.core.compress and in the driver's namespace)
    by a pivoted-QR ID truncated at schedule[(level, coord)] instead of at eps. Because every child keeps the
    same number of skeletons, every parent sees the same active-set sizes as in the reference run, so the
    schedule transfers exactly.
  * `XCProxy` is a Xing-Chow-style selected volume proxy: per tree level, dense random candidates in the far
    region (radii log-uniform in [r1, r_max]) are reduced by a pivoted QR of K(candidates, box samples) at
    tolerance eps/10 (Xing & Chow 2020, Alg. 5.1, without their "densification" step).
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

import numpy as np
import scipy.linalg as sla
from numpy.typing import ArrayLike

import unified.core.compress as compress_mod
import unified.drivers.rskel as rskel_mod
from unified.core import ProxyCompressor, Tree
from unified.core.tree import Node
from unified.core.types import Compressor, GridCoord, IDResult, IndexArray, Kernel, Matrix, Points, Vector
from unified.drivers.rskel import rskel_factor

from .fractional import FracProblem, rel_err
from .harness import RULES, RuleName, make_comp

type BoxKey = tuple[int, GridCoord]
type RankSchedule = dict[BoxKey, int]
type EKind = Literal["exact", "plane", "lifted", "annulus", "xc"]
ECOMPS: tuple[EKind, ...] = ("exact", "plane", "lifted", "annulus", "xc")
LIFTED_LAT = 4      # latitudes of the lifted hemisphere (R3: 2 suffice for hyper, 3 for Riesz at 1e-8)


class XCProxy(ProxyCompressor):
    """Selected volume proxy (Xing-Chow style). Selection is done once per box size and cached; its flops
    are reported in `stats["select_flops"]`."""

    def __init__(self, kernel: Kernel, points: ArrayLike, eps: float, A: Matrix | None = None,
                 radius_factor: float = 2.5, n_cand: int = 3000, n_box: int = 400, seed: int = 0) -> None:
        super().__init__(kernel, points, eps, A=A, radius_factor=radius_factor)
        P = self.points
        self.r_max = float(np.linalg.norm(P.max(0) - P.min(0)))
        self.n_cand, self.n_box, self.seed = n_cand, n_box, seed
        self.cache: dict[float, Points] = {}
        self.stats: dict[str, float] = {"select_flops": 0.0}   # shared by the driver's per-box shallow copies

    def _select(self, hw: float) -> Points:
        rng = np.random.default_rng(self.seed)
        d = self.points.shape[1]
        r1 = self.radius_factor * hw
        rad = np.exp(rng.uniform(np.log(r1), np.log(max(self.r_max, 1.01 * r1)), self.n_cand))
        if d == 1:
            sign = np.where(rng.random(self.n_cand) < 0.5, -1.0, 1.0)
            cand: Points = (rad * sign)[:, None]
            box: Points = np.linspace(-hw, hw, min(self.n_box, 100))[:, None]
        else:
            th = rng.uniform(0, 2 * np.pi, self.n_cand)
            cand = np.c_[rad * np.cos(th), rad * np.sin(th)]
            g = np.linspace(-hw, hw, int(np.sqrt(self.n_box)))
            box = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        K = self.kernel(cand, box)                       # (n_cand, n_box): rows = candidate proxy points
        _, R, piv = sla.qr(K.T, mode="economic", pivoting=True)
        dg = np.abs(np.diag(R))
        k = int(np.sum(dg > 0.1 * self.eps * dg[0]))
        self.stats["select_flops"] += 4.0 * K.shape[0] * K.shape[1] * min(K.shape)
        sel: Points = cand[piv[:k]]
        return sel

    def proxy_points(self, node: Node) -> Points:
        hw = round(node.half_width, 12)
        if hw not in self.cache:
            self.cache[hw] = self._select(node.half_width)
        out: Points = node.center[None, :] + self.cache[hw]
        return out


def make_ecomp(kind: EKind, prob: FracProblem, eps: float) -> Compressor:
    if kind == "xc":
        return XCProxy(prob.kernel, prob.X, eps, A=prob.A)
    return make_comp(kind, prob, eps, **({"n_lat": LIFTED_LAT} if kind == "lifted" else {}))


# ---- fixed-rank ID --------------------------------------------------------------------------------------
_STATE: dict[str, Any] = {"key": None, "schedule": None, "amax": 1.0}


def fixed_rank_id(M: Matrix, eps: float) -> IDResult:
    """Pivoted-QR column ID truncated at the scheduled rank of the current box (eps ignored)."""
    sched: RankSchedule | None = _STATE["schedule"]
    key: BoxKey | None = _STATE["key"]
    if sched is None or key is None or key not in sched:
        raise KeyError(f"no scheduled rank for box {key}")
    m, n = M.shape
    _, R, piv = sla.qr(M, mode="economic", pivoting=True)
    k = min(sched[key], n, m)
    skel, red = piv[:k].astype(np.intp), piv[k:].astype(np.intp)
    T: Matrix = sla.solve_triangular(R[:k, :k], R[:k, k:]) if k else np.zeros((0, n))
    return IDResult(skel, red, T)


@contextlib.contextmanager
def fixed_ranks(schedule: RankSchedule) -> Iterator[None]:
    orig_c, orig_r = compress_mod.interp_decomp, getattr(rskel_mod, "interp_decomp")
    _STATE["schedule"] = schedule
    setattr(compress_mod, "interp_decomp", fixed_rank_id)
    setattr(rskel_mod, "interp_decomp", fixed_rank_id)
    try:
        yield
    finally:
        setattr(compress_mod, "interp_decomp", orig_c)
        setattr(rskel_mod, "interp_decomp", orig_r)
        _STATE["schedule"] = None


class FarRecord(TypedDict):
    key: BoxKey
    nb: int
    k: int
    far_err: float     # ||M_R - M_S T||_F / ||M||_F against the true current far block (nan if nothing eliminated)
    far_abs: float     # ||M_R - M_S T||_F / max|A_ij|: the absolute perturbation the elimination drops
    rows: int          # rows of the compressor's ID input
    n_proxy: int


@contextlib.contextmanager
def instrument_far(records: list[FarRecord]) -> Iterator[list[FarRecord]]:
    orig = rskel_mod._skeletonize

    def wrapped(comp: Any, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
                q_extra: IndexArray) -> IDResult:
        key: BoxKey = (node.level, node.coord)
        _STATE["key"] = key
        res: IDResult = orig(comp, W, box, far, node, q_extra)
        M = np.vstack([W[np.ix_(far, box)], W[np.ix_(box, far)].T])
        nrm = float(np.linalg.norm(M))
        res_n = float(np.linalg.norm(M[:, res.red] - M[:, res.skel] @ res.T)) if res.red.size else 0.0
        err = res_n / nrm if nrm > 0 and res.red.size else float("nan")
        if isinstance(comp, ProxyCompressor):
            inside = far[np.linalg.norm(comp.points[far] - node.center, axis=1) < comp.radius_factor * node.half_width]
            npx = len(comp.proxy_points(node))
            rows = 2 * (npx + int(np.union1d(inside, q_extra).size))
        else:
            npx, rows = 0, 2 * int(far.size)
        records.append(FarRecord(key=key, nb=int(box.size), k=int(res.skel.size), far_err=err,
                                 far_abs=res_n / _STATE["amax"], rows=rows,
                                 n_proxy=npx))
        return res

    setattr(rskel_mod, "_skeletonize", wrapped)
    try:
        yield records
    finally:
        setattr(rskel_mod, "_skeletonize", orig)
        _STATE["key"] = None


def ecase(prob: FracProblem, rule: RuleName, kind: EKind, x_dense: Vector, eps: float = 1e-8,
          schedule: RankSchedule | None = None, max_leaf: int = 32) -> tuple[dict[str, Any], RankSchedule]:
    """One factor + solve. With `schedule`, every box keeps exactly the scheduled number of skeletons."""
    tree = Tree(prob.X, max_leaf=max_leaf)
    comp = make_ecomp(kind, prob, eps)
    recs: list[FarRecord] = []
    _STATE["amax"] = float(np.max(np.abs(prob.A)))
    t0 = time.perf_counter()
    with instrument_far(recs), (fixed_ranks(schedule) if schedule is not None else contextlib.nullcontext()):
        F = rskel_factor(prob.A, tree, RULES[rule](), comp)
    tf = time.perf_counter() - t0
    x = F.solve(prob.b)
    errs = np.array([r["far_err"] for r in recs if not np.isnan(r["far_err"])])
    row: dict[str, Any] = dict(
        family=prob.family, d=prob.d, s=prob.s, p=prob.p, N=len(prob.X), rule=rule, comp=kind,
        mode="equal_rank" if schedule is not None else "equal_tol", eps=eps,
        err=rel_err(x, x_dense), far_err_max=float(errs.max()) if errs.size else 0.0,
        far_err_median=float(np.median(errs)) if errs.size else 0.0,
        far_abs_max=float(max((r["far_abs"] for r in recs), default=0.0)),
        skel_compressed=int(sum(r["k"] for r in recs)),   # skeletons of compressed boxes only rank_max=int(max((r["k"] for r in recs), default=0)),
        root=int(F.stats["root_size"]), n_proxy_max=int(max((r["n_proxy"] for r in recs), default=0)),
        id_rows=int(sum(r["rows"] for r in recs)),
        id_flops=float(sum(2.0 * r["rows"] * r["nb"] ** 2 for r in recs)),
        select_flops=float(comp.stats["select_flops"]) if isinstance(comp, XCProxy) else 0.0, factor_s=tf)
    sched: RankSchedule = {r["key"]: r["k"] for r in recs}
    return row, sched
