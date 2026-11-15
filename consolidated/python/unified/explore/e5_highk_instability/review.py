"""E5 reviewer diagnostics.

    FixedRankCompressor  every compressor variant at a PRESCRIBED rank per box (equal-rank comparison), recording
                         the ID's error against the TRUE far block of the current matrix.
    RankRecorder         wraps a compressor and records the rank it chose per box (level, coord).
    proxy_hypothesis     geometric test on W = A: is the exact far row space inside the proxy row space
                         (Green's representation), and is each row space closed under complex conjugation?
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg as sla

from unified.core import ProxyCompressor
from unified.core.tree import Node, Tree
from unified.core.types import Compressor, GridCoord, IDResult, IndexArray, Kernel, Matrix, Points

from .fixes import far_block, id_with_rank, realify

type FixedKind = Literal["exact", "proxy", "randomized", "realified"]
type BoxKey = tuple[int, GridCoord]
type RankMap = dict[BoxKey, int]


def key(node: Node) -> BoxKey:
    return (node.level, node.coord)


class RankRecorder:
    """Pass-through compressor that records the rank chosen per box. Forwards `.A` (set by the driver)."""

    def __init__(self, inner: Compressor) -> None:
        self.inner = inner
        self.ranks: RankMap = {}

    @property
    def A(self) -> Matrix:
        a: Matrix = getattr(self.inner, "A")
        return a

    @A.setter
    def A(self, W: Matrix) -> None:
        setattr(self.inner, "A", W)

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        res = self.inner(box_idx, far_idx, node)
        self.ranks[key(node)] = len(res.skel)
        return res


def proxy_rows(kernel: Kernel, X: Points, W: Matrix, box: IndexArray, far: IndexArray, node: Node,
               radius_factor: float = 2.5, n_proxy: int = 64) -> Matrix:
    """Exactly the matrix ProxyCompressor feeds to the ID (weak case: no Q set)."""
    pc = ProxyCompressor(kernel, X, 1.0, A=W, n_proxy=n_proxy, radius_factor=radius_factor)
    r = radius_factor * node.half_width
    inside = far[np.linalg.norm(X[far] - node.center, axis=1) < r]
    Pp = pc.proxy_points(node)
    blocks: list[Matrix] = [kernel(Pp, X[box]), kernel(X[box], Pp).T]
    if inside.size:
        blocks += [W[np.ix_(inside, box)], W[np.ix_(box, inside)].T]
    M: Matrix = np.vstack(blocks)
    return M


class FixedRankCompressor:
    """ID of the `kind` matrix at rank ranks[(level, coord)]; `far_errs` = per-box relative error of the
    resulting T on the TRUE far block [W_FB; W_BF^T] of the current matrix (what sparsification drops)."""

    def __init__(self, kind: FixedKind, A: Matrix, kernel: Kernel, X: Points, ranks: RankMap, seed: int = 0,
                 oversample: int = 10) -> None:
        self.kind, self.A, self.kernel, self.X, self.ranks = kind, A, kernel, X, ranks
        self.rng = np.random.default_rng(seed)
        self.oversample = oversample
        self.far_errs: list[float] = []      # shared by the driver's shallow copies

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        Mt = far_block(self.A, box_idx, far_idx)
        r = min(self.ranks[key(node)], len(box_idx))
        if self.kind == "exact":
            M = Mt
        elif self.kind == "realified":
            M = realify(Mt)
        elif self.kind == "proxy":
            M = proxy_rows(self.kernel, self.X, self.A, box_idx, far_idx, node)
        else:   # randomized: Gaussian sketch of the true far block, p = r + oversample rows
            p = r + self.oversample
            G = self.rng.standard_normal((p, Mt.shape[0])) + 1j * self.rng.standard_normal((p, Mt.shape[0]))
            M = G @ Mt
        res = id_with_rank(M, r)
        if len(res.red):
            err = np.linalg.norm(Mt[:, res.red] - Mt[:, res.skel] @ res.T) / max(np.linalg.norm(Mt), 1e-300)
            self.far_errs.append(float(err))
        return res


def _rowspace(M: Matrix, eps: float) -> Matrix:
    """Orthonormal basis (as columns, in box coordinates) of the eps-numerical row space of M."""
    _, s, Vh = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > eps * s[0]))
    V: Matrix = Vh[:r].conj().T
    return V


def proxy_hypothesis(kernel: Kernel, X: Points, A: Matrix, tree: Tree, level: int, k: float, eps: float = 1e-8
                     ) -> dict[str, float]:
    """Weak far field, W = A, all boxes of one level. Reports (max over boxes):
    - contain: ||M_exact (I - P_proxy)|| / ||M_exact||  (Green's representation: should be ~eps)
    - conj_exact, conj_proxy, conj_ring (proxy circle rows only): sin of the largest principal angle between the eps-row space and its conjugate
    - rank_exact, rank_proxy: eps-numerical ranks (mean over boxes).
    Proxy rows use n_proxy = max(64, 1.5 k R + 16) so the circle resolves the field."""
    out = dict(contain=0.0, conj_exact=0.0, conj_proxy=0.0, conj_ring=0.0, rank_exact=0.0, rank_proxy=0.0)
    nodes = tree.levels[level]
    for node in nodes:
        box = node.idx
        far = np.setdiff1d(np.arange(len(X)), box)
        Me = far_block(A, box, far)
        R = 2.5 * node.half_width
        Mp = proxy_rows(kernel, X, A, box, far, node, n_proxy=max(64, int(1.5 * k * R) + 16))
        n_ring = 2 * max(64, int(1.5 * k * R) + 16)          # ring rows come first in proxy_rows
        Ve, Vp, Vr = _rowspace(Me, eps), _rowspace(Mp, eps), _rowspace(Mp[:n_ring], eps)
        resid = Me - (Me @ Vp) @ Vp.conj().T
        out["contain"] = max(out["contain"], float(np.linalg.norm(resid, 2) / np.linalg.norm(Me, 2)))
        for name, V in (("conj_exact", Ve), ("conj_proxy", Vp), ("conj_ring", Vr)):
            out[name] = max(out[name], float(np.sin(sla.subspace_angles(V, V.conj()).max())))
        out["rank_exact"] += Ve.shape[1] / len(nodes)
        out["rank_proxy"] += Vp.shape[1] / len(nodes)
    return out
