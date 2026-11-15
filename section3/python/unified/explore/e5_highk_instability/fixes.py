"""E5: compressors that test candidate fixes for the high-k pivot breakdown of RS on P3.

All share the core call signature compressor(box_idx, far_idx, node) -> IDResult and read the CURRENT working
matrix from `.A` (rskel._skeletonize sets it), so Schur-complement fill is seen like ExactCompressor sees it.

    RealifiedCompressor   ID of [Re M; Im M]: T is REAL. The driver's row op T^T is then T^H, so every
                          sparsification is a congruence U^H W U  (the fix; see findings.md, claim 3).
    HybridCompressor      exact far rows + rows from a proxy circle (fix (a) of the task).
    OversampledCompressor keep rank + p skeletons (fix (b)).
    SRRQRCompressor       pivoted QR followed by Gu-Eisenstat swaps until max |T_ij| <= f (fix (c)).
    AbsTolCompressor      ID with an ABSOLUTE threshold |R_jj| > tol instead of eps |R_11| (fix (e)).
Fix (d) "larger far field" is measured by weak vs strong admissibility (weak's far field strictly contains
strong's) and needs no new compressor.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from unified.core import interp_decomp
from unified.core.tree import Node
from unified.core.types import IDResult, IndexArray, InterpMatrix, Kernel, Matrix, Points

type Tolerance = float
type GrowthBound = float        # f in strong RRQR: |T_ij| <= f
type Wavenumber = float


def far_block(A: Matrix, box: IndexArray, far: IndexArray) -> Matrix:
    """The matrix every exact-type ID compresses: M = [A[F, B]; A[B, F]^T] (both directions, one T)."""
    M: Matrix = np.vstack([A[np.ix_(far, box)], A[np.ix_(box, far)].T])
    return M


def realify(M: Matrix) -> Matrix:
    """[Re M; Im M]: an ID of this real matrix has a real T, and M[:, red] ~= M[:, skel] T still holds."""
    out: Matrix = np.vstack([M.real, M.imag]) if np.iscomplexobj(M) else M
    return out


def id_with_rank(M: Matrix, rank: int) -> IDResult:
    """Column ID of fixed rank via pivoted QR: the first `rank` pivots are skeletons."""
    n = M.shape[1]
    rank = min(rank, n, M.shape[0])
    _, R, piv = sla.qr(M, mode="economic", pivoting=True)
    skel, red = piv[:rank].astype(np.intp), piv[rank:].astype(np.intp)
    T: InterpMatrix = sla.solve_triangular(R[:rank, :rank], R[:rank, rank:]) if rank else np.zeros((0, n - rank))
    return IDResult(skel, red, T)


class _Base:
    def __init__(self, A: Matrix, eps: Tolerance) -> None:
        self.A, self.eps = A, eps


class RealifiedCompressor(_Base):
    """Exact far field, but the ID is taken of [Re M; Im M] so that T is real."""

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        return interp_decomp(realify(far_block(self.A, box_idx, far_idx)), self.eps)


class OversampledCompressor(_Base):
    """Exact ID, then keep p extra skeletons (rank + p)."""

    def __init__(self, A: Matrix, eps: Tolerance, p: int = 8) -> None:
        super().__init__(A, eps)
        self.p = p

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        M = far_block(self.A, box_idx, far_idx)
        r = len(interp_decomp(M, self.eps).skel)
        return id_with_rank(M, r + self.p)


class AbsTolCompressor(_Base):
    """ID truncated at |R_jj| > eps * scale with a FIXED scale (default 1 = ||A - I|| order), not eps |R_11|."""

    def __init__(self, A: Matrix, eps: Tolerance, scale: float = 1.0) -> None:
        super().__init__(A, eps)
        self.scale = scale

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        M = far_block(self.A, box_idx, far_idx)
        if M.size == 0:
            return interp_decomp(M, self.eps)
        R = sla.qr(M, mode="r", pivoting=True)[0]
        d = np.abs(np.diag(R))
        return id_with_rank(M, int(np.sum(d > self.eps * self.scale)))


class SRRQRCompressor(_Base):
    """Rank from the standard ID; then swap a skeleton with a redundant column while some |T_ij| > f
    (Gu-Eisenstat 1996 strong RRQR criterion, simplified: we do not track the R_11 column norms)."""

    def __init__(self, A: Matrix, eps: Tolerance, f: GrowthBound = 1.01, max_swaps: int = 200) -> None:
        super().__init__(A, eps)
        self.f, self.max_swaps = f, max_swaps
        self.swaps = 0

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node | None = None) -> IDResult:
        M = far_block(self.A, box_idx, far_idx)
        skel, red, T = interp_decomp(M, self.eps)
        for _ in range(self.max_swaps):
            if T.size == 0:
                break
            i, j = np.unravel_index(int(np.argmax(np.abs(T))), T.shape)
            if abs(T[i, j]) <= self.f:
                break
            skel, red = skel.copy(), red.copy()
            skel[i], red[j] = red[j], skel[i]
            T = np.linalg.lstsq(M[:, skel], M[:, red], rcond=None)[0]
            self.swaps += 1
        return IDResult(skel, red, T)


class HybridCompressor(_Base):
    """Exact far rows PLUS n_proxy rows of the kernel on a circle of radius radius_factor * half-width
    (n_proxy = max(64, 1.5 k r + 16) so the circle resolves the field at wavenumber k)."""

    def __init__(self, A: Matrix, eps: Tolerance, kernel: Kernel, points: Points, k: Wavenumber,
                 radius_factor: float = 2.5) -> None:
        super().__init__(A, eps)
        self.kernel, self.points, self.k, self.radius_factor = kernel, points, k, radius_factor

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        r = self.radius_factor * node.half_width
        m = max(64, int(np.ceil(1.5 * self.k * r)) + 16)
        th = 2 * np.pi * np.arange(m) / m
        Pp: Points = node.center[None, :] + r * np.c_[np.cos(th), np.sin(th)]
        Xb = self.points[box_idx]
        M = np.vstack([far_block(self.A, box_idx, far_idx), self.kernel(Pp, Xb), self.kernel(Xb, Pp).T])
        return interp_decomp(M, self.eps)
