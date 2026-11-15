"""E11 harness: one measurement = factor a GPProblem with the unchanged rskel_factor and one compressor, then
solve, log-determinant, posterior mean, and a per-box check of the TRUE far-field ID error.

Compressor names:
  exact     core ExactCompressor (true far block)                         -- kernel-agnostic
  rand      E3's RandomizedCompressor (Gaussian sketch of the far block)  -- kernel-agnostic, imported read-only
  plane1    circle, main kernel only                                      -- the stock proxy
  planeM    circle, beta kernel orders  (B3's multi-kernel ring)
  lift1     hemisphere in R^3, main kernel only  (E8's lift)
  liftM     hemisphere, beta kernel orders; falls back to the circle when D = 2 (integer nu): the NEW proxy
  annulus   8 rings to the domain diameter, main kernel
  xc        Xing-Chow-style selected volume proxy, main kernel
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

import numpy as np

import scipy.linalg as sla

import unified.core.compress as compress_mod
import unified.drivers.rskel as rskel_mod
import unified.explore.e3_compression_swaps.compressors as e3c
from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak
from unified.core.tree import Node
from unified.core.types import Compressor, IDResult, IndexArray, Matrix, Vector
from unified.drivers.rskel import Factorization, rskel_factor
from unified.explore.e3_compression_swaps.compressors import RandomizedCompressor

from .matern import GPProblem, StackedKernel, lift_order, multi_orders, posterior_mean, rel_err
from .proxies import MaternProxy

type BoxKey = tuple[int, tuple[int, ...]]
type RankSchedule = dict[BoxKey, int]
_STATE: dict[str, Any] = {"key": None, "schedule": None}
type CompName = Literal["exact", "rand", "plane1", "planeM", "lift1", "liftM", "annulus", "xc"]
type RuleName = Literal["weak", "strong"]
COMPS: tuple[CompName, ...] = ("exact", "rand", "plane1", "planeM", "lift1", "liftM", "annulus", "xc")
RULES: dict[RuleName, type[Weak] | type[Strong]] = {"weak": Weak, "strong": Strong}


def make_comp(name: CompName, prob: GPProblem, eps: float, n_lat: int = 4, n_proxy: int = 64) -> Compressor:
    """The ONLY dial that changes between compressor runs (dial 4, sampling)."""
    k0 = prob.kernel
    if name == "exact":
        return ExactCompressor(prob.A, eps)
    if name == "rand":
        return RandomizedCompressor(prob.A, eps)
    multi = name.endswith("M") and k0.kind == "matern"
    ker = StackedKernel(k0.nu, k0.ell, multi_orders(k0.nu) if multi else [], kind=k0.kind)
    geo: Any = {"plane1": "plane", "planeM": "plane", "lift1": "lifted", "liftM": "lifted",
                "annulus": "annulus", "xc": "xc"}[name]
    if name == "liftM" and k0.kind == "matern" and lift_order(k0.nu)[1] == 2.0:
        geo = "plane"                                   # integer nu: C_nu is a Green's function of R^2 itself
    return MaternProxy(ker, prob.X, eps, geo, A=prob.A, n_lat=n_lat, n_proxy=n_proxy)


class BoxRecord(TypedDict):
    level: int
    nb: int
    k: int
    rows: int          # rows of the ID input (cost driver: ID flops ~ 2 rows nb^2)
    n_proxy: int
    far_err: float     # ||M_R - M_S T||_F / ||M||_F, M = [W_FB; W_BF^T] the TRUE current far block
    key: BoxKey


@contextlib.contextmanager
def instrument(records: list[BoxRecord], far_check: bool = True) -> Iterator[list[BoxRecord]]:
    """Wrap rskel._skeletonize (driver file untouched) to log one BoxRecord per compressed box."""
    orig = rskel_mod._skeletonize

    def wrapped(comp: Any, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
                q_extra: IndexArray) -> IDResult:
        _STATE["key"] = (node.level, tuple(node.coord))
        res: IDResult = orig(comp, W, box, far, node, q_extra)
        if isinstance(comp, ProxyCompressor):
            inside = far[np.linalg.norm(comp.points[far] - node.center, axis=1) < comp.radius_factor * node.half_width]
            npx = len(comp.proxy_points(node))
            rows = 2 * (npx * (len(getattr(comp.kernel, "orders", [])) or 1) + np.union1d(inside, q_extra).size)
        elif isinstance(comp, RandomizedCompressor):
            npx, rows = 0, int(comp.log[-1]["p"]) if comp.log else 0
        else:
            npx, rows = 0, 2 * far.size
        fe = float("nan")
        if far_check and len(res.red):
            M = np.vstack([W[np.ix_(far, box)], W[np.ix_(box, far)].T])
            nm = np.linalg.norm(M)
            fe = float(np.linalg.norm(M[:, res.red] - M[:, res.skel] @ res.T) / nm) if nm > 0 else 0.0
        records.append(BoxRecord(level=node.level, nb=int(box.size), k=int(len(res.skel)), rows=int(rows),
                                 n_proxy=int(npx), far_err=fe, key=(node.level, tuple(node.coord))))
        return res

    setattr(rskel_mod, "_skeletonize", wrapped)
    try:
        yield records
    finally:
        setattr(rskel_mod, "_skeletonize", orig)


def fixed_rank_id(M: Matrix, eps: float) -> IDResult:
    """Pivoted-QR column ID truncated at the scheduled rank of the current box (eps ignored); E8 R6's recipe."""
    sched: RankSchedule = _STATE["schedule"]
    k = min(sched[_STATE["key"]], *M.shape)
    _, R, piv = sla.qr(M, mode="economic", pivoting=True)
    skel, red = piv[:k].astype(np.intp), piv[k:].astype(np.intp)
    T: Matrix = sla.solve_triangular(R[:k, :k], R[:k, k:]) if k else np.zeros((0, M.shape[1]))
    return IDResult(skel, red, T)


@contextlib.contextmanager
def fixed_ranks(schedule: RankSchedule) -> Iterator[None]:
    """Equal-rank mode: every ID (core compressors, the driver's Q-set path, E3's randomized sketch) is truncated
    at the rank the reference run kept for the same box. Restores the originals on exit."""
    mods = [compress_mod, rskel_mod, e3c]
    orig = [getattr(m, "interp_decomp") for m in mods]
    _STATE["schedule"] = schedule
    for m in mods:
        setattr(m, "interp_decomp", fixed_rank_id)
    try:
        yield
    finally:
        for m, o in zip(mods, orig):
            setattr(m, "interp_decomp", o)
        _STATE["schedule"] = None


def schedule_of(records: list[BoxRecord]) -> RankSchedule:
    return {r["key"]: r["k"] for r in records}


def logdet(F: Factorization) -> float:
    """log|det A| from the factorization: the sparsifying row/column operations are unit triangular (det 1), so
    det A = prod det(X_RR) * det(root block). [proved: block elimination; exact up to the dropped O(eps) blocks]"""
    tot = 0.0
    for s in F.steps:
        tot += float(np.sum(np.log(np.abs(np.diag(s.lu[0])))))
    if F.root_lu is not None:
        tot += float(np.sum(np.log(np.abs(np.diag(F.root_lu[0])))))
    return tot


type CaseResult = dict[str, Any]


class DenseRef(TypedDict):
    norm2: float       # ||A||_2 (power iteration), for the normwise backward error
    x: Vector
    logdet: float
    mean: Vector
    cond: float


def dense_ref(prob: GPProblem, with_cond: bool = True) -> DenseRef:
    x: Vector = np.linalg.solve(prob.A, prob.y)
    ld = float(np.linalg.slogdet(prob.A)[1])
    cond = float("nan")
    if with_cond:
        ev = np.linalg.eigvalsh(prob.A) if np.allclose(prob.A, prob.A.T) else np.abs(np.linalg.eigvals(prob.A))
        cond = float(np.max(np.abs(ev)) / np.min(np.abs(ev)))
    mean = posterior_mean(prob, x) if prob.kernel.kind == "matern" else x[:0]
    v = np.random.default_rng(1).standard_normal(len(x))
    for _ in range(50):
        v = prob.A.T @ (prob.A @ v); v /= np.linalg.norm(v)
    norm2 = float(np.linalg.norm(prob.A @ v))
    return DenseRef(norm2=norm2, x=x, logdet=ld, mean=mean, cond=cond)


def case(prob: GPProblem, rule: RuleName, comp_name: CompName, ref: DenseRef, eps: float = 1e-8,
         max_leaf: int = 32, far_check: bool = True, schedule: RankSchedule | None = None,
         records_out: list[BoxRecord] | None = None, **opts: Any) -> CaseResult:
    tree = Tree(prob.X, max_leaf=max_leaf)
    comp = make_comp(comp_name, prob, eps, **opts)
    recs: list[BoxRecord] = []
    t0 = time.perf_counter()
    with instrument(recs, far_check), (fixed_ranks(schedule) if schedule is not None else contextlib.nullcontext()):
        F = rskel_factor(prob.A, tree, RULES[rule](), comp)
    if records_out is not None:
        records_out.extend(recs)
    tf = time.perf_counter() - t0
    x = F.solve(prob.y)
    lv = F.stats["levels"]
    fes = [r["far_err"] for r in recs if not np.isnan(r["far_err"])]
    out: CaseResult = dict(
        problem=prob.name, kind=prob.kernel.kind, nu=prob.kernel.nu, ell=prob.kernel.ell, sigma2=prob.sigma2,
        N=len(prob.X), rule=rule, comp=comp_name, eps=eps, depth=tree.depth,
        err=rel_err(x, ref["x"]),
        residual=float(np.linalg.norm(prob.A @ x - prob.y) / np.linalg.norm(prob.y)),
        backward=float(np.linalg.norm(prob.A @ x - prob.y)
                       / (ref["norm2"] * np.linalg.norm(x) + np.linalg.norm(prob.y))),
        cond=ref["cond"],
        far_err_max=float(max(fes, default=0.0)),
        far_err_median=float(np.median(fes)) if fes else 0.0,
        equal_rank=schedule is not None,
        yalpha=float(np.real(prob.y @ x)),
        logdet_fast=logdet(F), logdet_dense=ref["logdet"], logdet_abs_err=abs(logdet(F) - ref["logdet"]),
        rank_max=int(max((r["k"] for r in recs), default=0)),
        rank_max_level={int(l): int(max((r["k"] for r in recs if r["level"] == l), default=0)) for l in lv},
        skel_total=int(sum(v["skel_total"] for v in lv.values())), root=int(F.stats["root_size"]),
        id_rows=int(sum(r["rows"] for r in recs)),
        id_flops=float(sum(2.0 * r["rows"] * r["nb"] ** 2 for r in recs)),
        n_proxy_max=int(max((r["n_proxy"] for r in recs), default=0)), factor_s=tf, **opts)
    if prob.kernel.kind == "matern":
        out["mean_err"] = rel_err(posterior_mean(prob, x), ref["mean"])
    return out
