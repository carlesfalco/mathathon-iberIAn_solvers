"""E6 dial 4 (sampling): a proxy-surface compressor for BLOCK (vector-valued) kernels on interleaved dofs.

Why core.ProxyCompressor cannot be reused unchanged (each is a finding, see findings.md):
  1. It calls kernel(points, points) and gets ONE row per proxy point. A 2x2 kernel needs the dof's
     component and the source normal, which a coordinate array does not carry. Here kernels are called on
     the box's POINTS and the result is sliced to the box's dofs.
  2. The double-layer matrix is not symmetric, so the two directions need different proxy "sources":
       column side  A[far, box]   (far velocity made by box densities): DL kernel evaluated AT proxy points;
       row side     A[box, far]^T (box velocity made by far densities): proxy SOURCES on the circle, which may
                    be Stokeslets ("sl"), stresslets with the circle normal ("dl"), or both ("both").
  3. A = 1/2 I + K W + P: the rigid-body projector P is not a kernel value, so the Green's representation does
     not see it. Its far blocks have row space spanned by V_box^T and (W V)_box^T (3 + 3 rows per body that
     touches the box); `rigid_rows=True` adds them. False shows what goes wrong without them.
  4. rskel_factor gives its FMM-LU "Q set" (far unknowns whose entries an earlier Schur complement changed)
     only to subclasses of core.ProxyCompressor, and through the scalar kernel call. This class is not a
     subclass, so for STRONG admissibility it takes a conservative superset of Q itself: every active far dof
     in the same-level boxes at Chebyshev distance 2 (Q can only contain neighbours of neighbours).
The ID itself (core.interp_decomp) is unchanged.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from unified.core import interp_decomp
from unified.core.tree import Node, Tree
from unified.core.types import IDResult, IndexArray, Matrix, Points

from .stokes import Geometry, Normals, laplace_dl, rigid_basis, stokeslet, stresslet_dl

type ProxySources = Literal["sl", "dl", "both"]
type DLKernel = Callable[[Points, Points, Normals], Matrix]
type SLKernel = Callable[[Points, Points], Matrix]


def _laplace_sl(X: Points, Y: Points) -> Matrix:
    r = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    out: Matrix = -np.log(np.where(r > 0, r, 1.0)) / (2 * np.pi)
    return out


class BlockProxyCompressor:
    """comp(box_dofs, far_dofs, node) -> IDResult, for A = c I + (DL kernel) W + (low-rank projector)."""

    def __init__(self, g: Geometry, eps: float, A: Matrix, tree: Tree, ncomp: int = 2,
                 sources: ProxySources = "sl", n_proxy: int = 64, radius_factor: float = 2.5,
                 rigid_rows: bool = True, strong: bool = False, matrix_kernel: Literal["dl", "sl"] = "dl") -> None:
        self.g, self.eps, self.A, self.tree, self.ncomp = g, eps, A, tree, ncomp
        self.sources, self.n_proxy, self.radius_factor = sources, n_proxy, radius_factor
        self.rigid_rows, self.strong = rigid_rows, strong
        self.matrix_kernel = matrix_kernel     # "sl": A = I + G W (area problem), column side uses G at proxies
        self.dl: DLKernel = stresslet_dl if ncomp == 2 else laplace_dl
        self.sl: SLKernel = stokeslet if ncomp == 2 else _laplace_sl
        self.dof_xy: Points = np.repeat(g.X, ncomp, axis=0)
        self.wdof = np.repeat(g.w, ncomp)
        # per body: its dofs, and the two row blocks spanning the projector's far blocks, scaled like P itself
        self.bodies: list[tuple[IndexArray, Matrix, Matrix]] = []
        for k in range(g.K):
            pts = np.flatnonzero(g.body == k)
            dofs: IndexArray = np.stack([ncomp * pts + a for a in range(ncomp)], 1).ravel()
            V = rigid_basis(g, k) if ncomp == 2 else np.ones((pts.size, 1))
            Wd = np.repeat(g.proj_w(), ncomp)[dofs]
            Z: Matrix = np.linalg.solve(V.T @ (Wd[:, None] * V), (V * Wd[:, None]).T)   # P_k = V Z
            self.bodies.append((dofs, V * float(np.abs(Z).max()), Z.T))

    def _q_superset(self, node: Node, far: IndexArray) -> IndexArray:
        """Active far dofs in same-level boxes at Chebyshev distance exactly 2 (superset of the FMM-LU Q set)."""
        boxes = [self.tree.node(node.level, (node.coord[0] + i, node.coord[1] + j))
                 for i in range(-2, 3) for j in range(-2, 3) if max(abs(i), abs(j)) == 2]
        pts = np.concatenate([b.idx for b in boxes if b is not None] + [np.zeros(0, np.intp)])
        out: IndexArray = np.intersect1d(far, pts)
        return out

    def __call__(self, box_idx: IndexArray, far_idx: IndexArray, node: Node) -> IDResult:
        c = nc = self.ncomp
        far_idx = np.asarray(far_idx, np.intp)
        R = self.radius_factor * node.half_width
        th = 2 * np.pi * np.arange(self.n_proxy) / self.n_proxy
        nrm: Normals = np.c_[np.cos(th), np.sin(th)]
        Pp: Points = node.center[None, :] + R * nrm
        pw = 2 * np.pi * R / self.n_proxy                       # proxy quadrature weight (row scaling)
        pts = np.unique(box_idx // nc)                          # boundary points owning the box dofs
        cols = np.searchsorted(pts, box_idx // nc) * c + box_idx % nc   # box dofs inside the point blocks
        Xb, Nb = self.g.X[pts], self.g.normal[pts]
        wb = self.wdof[box_idx]
        colside = self.sl(Pp, Xb) if self.matrix_kernel == "sl" else self.dl(Pp, Xb, Nb)
        blocks: list[Matrix] = [colside[:, cols] * wb[None, :]]                     # column side: field at proxies
        if self.sources in ("sl", "both"):                                         # row side: proxy sources
            blocks.append(pw * self.sl(Xb, Pp)[cols, :].T)
        if self.sources in ("dl", "both"):
            blocks.append(pw * self.dl(Xb, Pp, nrm)[cols, :].T)
        explicit = far_idx[np.linalg.norm(self.dof_xy[far_idx] - node.center, axis=1) < R] if far_idx.size else far_idx
        if self.strong:
            explicit = np.union1d(explicit, self._q_superset(node, far_idx))
        if explicit.size:
            blocks += [self.A[np.ix_(explicit, box_idx)], self.A[np.ix_(box_idx, explicit)].T]
        if self.rigid_rows:     # P[far, box] = V_far Z_box  and  P[box, far]^T = Z_far^T V_box^T
            for k in np.unique(self.g.body[pts]):
                bdofs, Vs, Zt = self.bodies[int(k)]
                pos = np.searchsorted(bdofs, box_idx)
                hit = (pos < bdofs.size) & (bdofs[np.minimum(pos, bdofs.size - 1)] == box_idx)
                rows = np.zeros((box_idx.size, 2 * Vs.shape[1]))
                rows[hit] = np.c_[Vs[pos[hit]], Zt[pos[hit]]]
                blocks.append(rows.T)
        return interp_decomp(np.vstack(blocks), self.eps)
