"""One interpolative decomposition (ID) and swappable ways to build the matrix it is applied to.

ID: for a matrix M with columns indexed by a box's active unknowns, choose skeleton columns `skel`
and an interpolation matrix T with   M[:, red] ~= M[:, skel] @ T   (relative accuracy eps).
"Each redundant unknown acts on the far field like a fixed combination of skeleton unknowns."

Compressors build M for a box. They all have the same call signature (the `Compressor` protocol):
    compressor(box_idx, far_idx, node) -> IDResult(skel, red, T)      (skel/red index into box_idx)
where box_idx are the box's active unknowns and far_idx the active unknowns it must be compressed
against (as decided by the admissibility rule). M stacks both directions (box -> far and far -> box),
so one ID serves rows and columns, which keeps nonsymmetric problems correct.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from numpy.typing import ArrayLike

from .tree import Node
from .types import IDResult, IndexArray, Kernel, Matrix, Points


def interp_decomp(M: Matrix, eps: float) -> IDResult:
    """Column ID by pivoted QR. Returns IDResult(skel, red, T) with M[:, red] ~= M[:, skel] @ T."""
    m, n = M.shape
    if n == 0:
        return IDResult(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros((0, 0)))
    if m == 0:  # nothing to be accurate against: every column is redundant (rank 0)
        return IDResult(np.zeros(0, np.intp), np.arange(n, dtype=np.intp), np.zeros((0, n)))
    _, R, piv = sla.qr(M, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    k = int(np.sum(d > eps * d[0])) if d.size and d[0] > 0 else 0
    skel, red = piv[:k].astype(np.intp), piv[k:].astype(np.intp)
    T: Matrix = sla.solve_triangular(R[:k, :k], R[:k, k:]) if k else np.zeros((0, n))
    return IDResult(skel, red, T)


class ExactCompressor:
    """Uses the true matrix entries A[far, box] and A[box, far]. Simple, exact, O(N) per box.
    Good for testing and for any matrix (no kernel or PDE knowledge needed)."""

    def __init__(self, A: Matrix, eps: float) -> None:
        self.A, self.eps = A, eps

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        M = np.vstack([self.A[np.ix_(far_idx, box_idx)], self.A[np.ix_(box_idx, far_idx)].T])
        return interp_decomp(M, self.eps)


class ProxyCompressor:
    """Green's representation (Huygens) trick: instead of all far unknowns, compress against
    n_proxy points on a circle (2D) / two points (1D) of radius radius_factor * box half-width,
    plus the far unknowns that happen to lie inside that circle (kept explicitly).
    Valid when matrix entries outside the box are kernel evaluations K(x_i, x_j) of a PDE kernel."""

    def __init__(self, kernel: Kernel, points: ArrayLike, eps: float, A: Matrix | None = None,
                 n_proxy: int = 64, radius_factor: float = 2.5) -> None:
        P: Points = np.asarray(points, float)
        self.kernel, self.points, self.eps, self.A = kernel, (P[:, None] if P.ndim == 1 else P), eps, A
        self.n_proxy, self.radius_factor = n_proxy, radius_factor

    def proxy_points(self, node: Node) -> Points:
        """Points on the proxy circle (2D) or the two proxy points (1D) around `node`."""
        r = self.radius_factor * node.half_width
        if self.points.shape[1] == 1:
            pts: Points = node.center[None, :] + np.array([[-r], [r]])
            return pts
        th = 2 * np.pi * np.arange(self.n_proxy) / self.n_proxy
        circle: Points = node.center[None, :] + r * np.c_[np.cos(th), np.sin(th)]
        return circle

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        X = self.points
        r = self.radius_factor * node.half_width
        far_idx = np.asarray(far_idx, np.intp)
        inside: IndexArray = (far_idx[np.linalg.norm(X[far_idx] - node.center, axis=1) < r]
                              if far_idx.size else far_idx)
        Pp = self.proxy_points(node)
        blocks: list[Matrix] = [self.kernel(Pp, X[box_idx]), self.kernel(X[box_idx], Pp).T]
        if inside.size:  # far unknowns inside the proxy circle: use the actual matrix entries
            if self.A is not None:
                blocks += [self.A[np.ix_(inside, box_idx)], self.A[np.ix_(box_idx, inside)].T]
            else:
                blocks += [self.kernel(X[inside], X[box_idx]), self.kernel(X[box_idx], X[inside]).T]
        return interp_decomp(np.vstack(blocks), self.eps)
