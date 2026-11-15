"""E3: swappable compression engines for unified/drivers/rskel.py, plus cost instrumentation.

    RandomizedCompressor   kernel-agnostic: sketch the far block M = [A[F,B]; A[B,F]^T] with Gaussian test
                           vectors (p restricted matvecs with A^T and p with A), then ONE interp_decomp on the
                           p x |B| sketch.  Adaptive p (doubling) until rank <= p - oversample.
    MultiRingProxy         ProxyCompressor with several concentric proxy circles and n_proxy growing with kR.
                           (Fix for kernels whose proxy representation degenerates at isolated kR.)
    instrument(...)        context manager that wraps rskel._skeletonize and records, per box, the sizes the
                           ID saw, so every compressor's cost is counted with one formula set.

Cost model (per box; |B| = active box unknowns, |F| = far unknowns, k = skeleton size):
    Exact      : entries read 2|F||B|,              ID flops ~ 2 (2|F|) |B|^2
    Proxy      : kernel evals (2 P + 2 I)|B|        ID flops ~ 2 (2P + 2I) |B|^2      (P proxy pts, I explicit)
    Randomized : sketch flops 2 (2|F|) |B| p_tot    ID flops ~ 2 p |B|^2,  matvecs 2 p_tot (restricted)
"""
from __future__ import annotations
import contextlib
from collections.abc import Iterator
from typing import Any, TypedDict, NotRequired

import numpy as np
from unified.core import ProxyCompressor, ExactCompressor, interp_decomp
from unified.core.tree import Node
from unified.core.types import IDResult, IndexArray, Kernel, Matrix, Points
import unified.drivers.rskel as rskel_mod

type Wavenumber = float
type SketchSize = int          # p: number of Gaussian test vectors per direction
type FlopCount = float


class SketchLog(TypedDict):
    nb: int
    nf: int
    p: SketchSize
    k: int


class SkeletonRecord(TypedDict):
    """Sizes seen by one ID call (one box), enough to evaluate the cost model above."""
    kind: str
    level: int
    nb: int
    nf: int
    nq: int
    k: int
    n_explicit: NotRequired[int]
    n_proxy: NotRequired[int]
    p: NotRequired[SketchSize]


class RandomizedCompressor:
    """Randomized ID (Halko-Martinsson-Tropp 2011, Sec. 5.2 / Martinsson-Rokhlin-Tygert ID).

    Uses ONLY products of the current matrix with vectors supported on the far unknowns:
        Y = A[B, F] G2 + A[F, B]^T G1        (p "A" matvecs + p "A^T" matvecs, restricted to F -> B)
    and runs the shared interp_decomp on the sketch S = Y^T (p x |B|). The driver sets `.A` to its current
    working matrix before every call, so Schur-complement fill (FMM-LU's Q set) is seen automatically."""

    def __init__(self, A: Matrix, eps: float, p0: SketchSize = 16, oversample: int = 10, seed: int = 0,
                 log: list[SketchLog] | None = None) -> None:
        self.A, self.eps, self.p0, self.oversample = A, eps, p0, oversample
        self.rng = np.random.default_rng(seed)
        self.log: list[SketchLog] = log if log is not None else []

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        nb, nf = len(box_idx), len(far_idx)
        Afb: Matrix = self.A[np.ix_(far_idx, box_idx)]   # stands in for "apply A^T to far-supported vectors"
        Abf: Matrix = self.A[np.ix_(box_idx, far_idx)]   # stands in for "apply A to far-supported vectors"
        rows: list[Matrix] = []
        p_tot: SketchSize = 0
        p_new: SketchSize = min(self.p0, nb)
        while True:
            G1 = self.rng.standard_normal((nf, p_new))
            G2 = self.rng.standard_normal((nf, p_new))
            rows.append((Afb.T @ G1 + Abf @ G2).T)       # p_new more sketch rows
            p_tot += p_new
            S: Matrix = np.vstack(rows)
            res = interp_decomp(S, self.eps)
            if len(res.skel) <= p_tot - self.oversample or p_tot >= nb:
                break
            p_new = min(p_tot, nb - p_tot)               # double the sketch
        self.log.append(SketchLog(nb=nb, nf=nf, p=p_tot, k=len(res.skel)))
        return res


class MultiRingProxy(ProxyCompressor):
    """Proxy compressor with `rings` concentric circles (radii radius_factor * (1, 1.37, ...) * half-width) and
    n_proxy per ring = max(n_proxy, ceil(ppw * k * R) + 16) for wave kernels (k = wavenumber, 0 if static).
    Overriding proxy_points is enough: rskel._skeletonize calls proxy_points for the strong/Q-set path too."""

    def __init__(self, kernel: Kernel, points: Points, eps: float, A: Matrix | None = None, n_proxy: int = 64,
                 radius_factor: float = 2.5, rings: int = 2, k: Wavenumber = 0.0, ppw: float = 1.5) -> None:
        super().__init__(kernel, points, eps, A=A, n_proxy=n_proxy, radius_factor=radius_factor)
        self.rings, self.k, self.ppw = rings, k, ppw

    def proxy_points(self, node: Node) -> Points:
        out: list[Points] = []
        for j in range(self.rings):
            r = self.radius_factor * node.half_width * (1 + 0.37 * j)
            m = max(self.n_proxy, int(np.ceil(self.ppw * self.k * r)) + 16)
            th = 2 * np.pi * (np.arange(m) + 0.5 * j) / m
            out.append(node.center[None, :] + r * np.c_[np.cos(th), np.sin(th)])
        return np.vstack(out)


def kind_of(comp: object) -> str:
    if isinstance(comp, RandomizedCompressor):
        return "randomized"
    if isinstance(comp, ProxyCompressor):
        return "proxy"
    if isinstance(comp, ExactCompressor):
        return "exact"
    return type(comp).__name__


@contextlib.contextmanager
def instrument(records: list[SkeletonRecord]) -> Iterator[list[SkeletonRecord]]:
    """Wrap rskel._skeletonize; append one SkeletonRecord per compressed box to `records`.
    Restores the original function on exit (no file outside this directory is edited)."""
    orig = rskel_mod._skeletonize

    def wrapped(compressor: Any, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
                q_extra: IndexArray) -> IDResult:
        res: IDResult = orig(compressor, W, box, far, node, q_extra)
        rec = SkeletonRecord(kind=kind_of(compressor), level=node.level, nb=int(box.size), nf=int(far.size),
                             nq=int(q_extra.size), k=int(len(res.skel)))
        if isinstance(compressor, ProxyCompressor):
            X = compressor.points
            dist = np.linalg.norm(X[far] - node.center, axis=1)
            inside: IndexArray = far[dist < compressor.radius_factor * node.half_width]
            rec["n_explicit"] = int(np.union1d(inside, q_extra).size)
            rec["n_proxy"] = int(len(compressor.proxy_points(node)))
        if isinstance(compressor, RandomizedCompressor):
            rec["p"] = compressor.log[-1]["p"]
        records.append(rec)
        return res

    rskel_mod._skeletonize = wrapped
    try:
        yield records
    finally:
        rskel_mod._skeletonize = orig


def cost_summary(records: list[SkeletonRecord], n: int) -> dict[str, float]:
    """Totals over boxes. `entries` = matrix/kernel entries read or evaluated to build the ID input;
    `flops` = sketch + ID (pivoted-QR) flops; `matvec_equiv` = flops / (2 n^2) i.e. in units of one dense
    matvec with A; `restricted_matvecs` (randomized only) = number of restricted products 2 * sum p."""
    entries: float = 0.0
    flops: FlopCount = 0.0
    rmv: float = 0.0
    for r in records:
        nb = r["nb"]
        if r["kind"] == "exact":
            m = 2 * r["nf"]; entries += m * nb; flops += 2 * m * nb * nb
        elif r["kind"] == "proxy":
            m = 2 * (r.get("n_proxy", 0) + r.get("n_explicit", 0)); entries += m * nb; flops += 2 * m * nb * nb
        elif r["kind"] == "randomized":
            m = 2 * r["nf"]; p = r.get("p", 0)
            flops += 2 * m * nb * p + 2 * p * nb * nb; rmv += 2 * p
            entries += m * nb  # each far-block entry is touched once per sketch pass (dense model)
    return dict(entries=entries, flops=flops, matvec_equiv=flops / (2.0 * n * n), restricted_matvecs=rmv)
