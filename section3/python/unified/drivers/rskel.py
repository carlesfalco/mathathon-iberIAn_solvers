"""A1: ONE recursive-skeletonization (RS) factorization for weak AND strong admissibility.

    rskel_factor(A, tree, admissibility, compressor) -> Factorization   (.solve(b), .stats)

Weak admissibility (`Weak`, near list empty) gives HBS-style RS (Ho-Greengard 2012).
Strong admissibility (`Strong`, near list = touching boxes) gives strong RS / FMM-LU
(Minden-Ho-Damle-Ying 2017; Sushnikova-Greengard-O'Neil-Rachh 2023). The code path is identical;
weak is literally "strong with an empty near list".

Per box B, leaves to root (unknowns ordered R = redundant, S = skeleton, N = near, F = far):
  1. skeleton   : ID of B's active unknowns against the far field F (the compressor decides how F is
                  sampled):  A[F, R] ~= A[F, S] T  and  A[R, F] ~= T^T A[S, F].
  2. sparsify   : column op A[:, R] -= A[:, S] T and row op A[R, :] -= T^T A[S, :]; the R-F blocks
                  become O(eps) and are dropped. Only rows/cols in B u N are touched.
  3. eliminate  : block Gaussian elimination of R; Schur complement
                  A[SuN, SuN] -= A[SuN, R] A[R, R]^{-1} A[R, SuN].
After all boxes of a level, the surviving skeletons of the children are the parent's active unknowns.
At the top, one dense LU of what is left ("root block").

Key subtlety (strong case only). Step 3 changes A[N1, N2] for every pair of B's neighbours. Two
neighbours of B can be two boxes apart, i.e. FAR from each other, so later, when N1 is compressed, its
interaction with N2 is no longer a kernel value. A proxy compressor (which replaces the far field by
kernel evaluations on a circle) would miss it. We record these "modified far" pairs (FMM-LU's Q set)
and append the current entries of those unknowns explicitly to the ID matrix. The ExactCompressor is
simply pointed at the current matrix, so it sees them automatically. Weak admissibility never creates
such pairs (its Schur complement stays inside B).

Storage: the current matrix is a dense copy of A (clarity over memory; fine for N <= 16k on a 24 GB
machine: 2 GB for real float64 at N = 16384). Cost is therefore O(N^2) memory; the arithmetic is the
O(N) (strong) / O(N^{3/2}) (weak, 2D area) RS work plus O(N) bookkeeping per box.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as sla

from unified.core import ProxyCompressor, interp_decomp
from unified.core.tree import Node, Tree
from unified.core.types import (Admissibility, Compressor, IDResult, IndexArray, InterpMatrix, Matrix,
                                RedundantIdx, SkeletonIdx, Vector)

# ---- vocabulary of this driver -----------------------------------------------------------------------
type NearIdx = IndexArray            # active unknowns of the near field N (touching boxes; empty for Weak)
type FarIdx = IndexArray             # active unknowns of the far field F (what the box is compressed against)
type SkelNearIdx = IndexArray        # S followed by N: the unknowns a Schur complement updates
type ActiveMask = NDArray[np.bool_]  # (N,) True while an unknown has not been eliminated
type LUFactors = tuple[Matrix, NDArray[np.int32]]   # scipy.linalg.lu_factor output (packed LU, pivots)
type QSet = set[Node]                # FMM-LU "modified far" boxes: same-level boxes with non-kernel blocks


class LevelStats(TypedDict):
    """Per-level statistics of the factorization (skeleton sizes after the ID)."""
    boxes: int
    skel_max: int
    skel_mean: float
    skel_total: int
    eliminated: int


class FactorStats(TypedDict, total=False):
    """`Factorization.stats`: a plain dict at runtime; `solve_time` appears after the first solve."""
    admissibility: str
    n: int
    depth: int
    levels: dict[int, LevelStats]
    root_size: int
    n_steps: int
    factor_time: float
    solve_time: float


@dataclass
class _Step:
    """One box elimination: skeleton S, redundant R, near N (global indices), interpolation matrix T,
    LU of the sparsified X_RR, and the off-diagonal blocks X_{(S u N) R}, X_{R (S u N)}."""
    skel: SkeletonIdx
    red: RedundantIdx
    sn: SkelNearIdx           # S followed by N (the unknowns touched by the Schur complement)
    T: InterpMatrix
    lu: LUFactors             # LU of X_RR
    X_snr: Matrix             # X_{(S u N) R}
    X_rsn: Matrix             # X_{R (S u N)}


@dataclass
class Factorization:
    """A ~= (sparsify/eliminate factors) * root LU * (...). `solve` applies the inverse in O(stored)."""
    n: int
    steps: list[_Step]
    root: IndexArray          # unknowns left at the top (the "root block")
    root_lu: LUFactors | None
    dtype: np.dtype[np.inexact]
    stats: FactorStats = field(default_factory=lambda: FactorStats())

    def solve(self, b: Vector) -> Vector:
        """Solve A x = b: forward sweep (sparsify rows: b_R -= T^T b_S; eliminate R: b_SN -= X_SN,R X_RR^-1 b_R),
        dense root solve, backward sweep (eliminate: x_R -= X_RR^-1 X_R,SN x_SN; un-sparsify: x_S -= T x_R)."""
        t0 = time.perf_counter()
        x: Vector = np.array(b, dtype=np.result_type(self.dtype, np.asarray(b).dtype), copy=True)
        for s in self.steps:                                   # forward: V^{-1} = L E
            x[s.red] = x[s.red] - s.T.T @ x[s.skel]
            y: Vector = sla.lu_solve(s.lu, x[s.red])
            x[s.sn] = x[s.sn] - s.X_snr @ y
            x[s.red] = y
        if self.root.size and self.root_lu is not None:        # root block: dense LU
            x[self.root] = sla.lu_solve(self.root_lu, x[self.root])
        for s in reversed(self.steps):                         # backward: W^{-1} = F^{-1}... U^{-1}
            x[s.red] = x[s.red] - sla.lu_solve(s.lu, s.X_rsn @ x[s.sn])
            x[s.skel] = x[s.skel] - s.T @ x[s.red]
        self.stats["solve_time"] = time.perf_counter() - t0
        return x


def _lu_factor(M: Matrix) -> LUFactors:
    """scipy.linalg.lu_factor for a real OR complex block (the stubs only overload concrete dtypes)."""
    return cast(LUFactors, sla.lu_factor(cast(Any, M)))


def _skeletonize(compressor: Compressor, W: Matrix, box: IndexArray, far: FarIdx, node: Node,
                 q_extra: IndexArray) -> IDResult:
    """Step 1 (skeleton / redundant / T). The compressor reads explicit entries from the CURRENT matrix W.
    For a ProxyCompressor with a nonempty Q set `q_extra` (far unknowns whose interaction with this box
    was modified by an earlier Schur complement), those unknowns are added as explicit rows/columns,
    next to the proxy rows and the far unknowns inside the proxy circle."""
    comp = copy.copy(compressor)
    setattr(comp, "A", W)   # every compressor reads explicit entries from `.A`; the protocol does not declare it
    if not isinstance(comp, ProxyCompressor) or q_extra.size == 0:
        return comp(box, far, node)
    X = comp.points
    inside: FarIdx = far[np.linalg.norm(X[far] - node.center, axis=1) < comp.radius_factor * node.half_width]
    expl: IndexArray = np.union1d(inside, q_extra)
    Pp = comp.proxy_points(node)
    M: Matrix = np.vstack([comp.kernel(Pp, X[box]), comp.kernel(X[box], Pp).T,
                   W[np.ix_(expl, box)], W[np.ix_(box, expl)].T])
    return interp_decomp(M, comp.eps)


def rskel_factor(A: Matrix, tree: Tree, admissibility: Admissibility, compressor: Compressor,
                 overwrite_a: bool = False, track_modified: bool = True) -> Factorization:
    """Recursive skeletonization factorization (weak = HBS-style RS, strong = FMM-LU-style strong RS).

    A: dense (N, N) matrix (real or complex, need not be symmetric), ordered like tree.points.
    admissibility: Weak() or Strong()  (only `near_nodes` is used).
    compressor: ExactCompressor or ProxyCompressor; called as compressor(box_idx, far_idx, node).
    overwrite_a: use A itself as the working matrix (saves one N x N copy).
    track_modified: keep the FMM-LU Q set (default). False is ONLY for demonstrating that strong RS with
        proxy compression breaks without it.

    stats: levels[l] = {boxes, skel_max, skel_mean, skel_total, eliminated} (skeleton sizes after the ID;
    a box with an EMPTY far field, e.g. every level-1 box under strong admissibility in 2D, is not
    compressed and reports its full active size), root_size, n_steps, factor_time, solve_time (last solve).
    """
    t0 = time.perf_counter()
    W: Matrix = A if overwrite_a else np.array(A, copy=True)
    n = W.shape[0]
    active: ActiveMask = np.ones(n, bool)
    steps: list[_Step] = []
    levels: dict[int, LevelStats] = {}
    for l in range(tree.depth, 0, -1):                # leaves -> level 1; the root (level 0) is dense
        nodes = tree.levels[l]
        modified: dict[Node, QSet] = {nd: set() for nd in nodes}        # Q set: same-level boxes with modified (non-kernel) blocks
        skel_sizes: list[int] = []
        n_elim = 0
        for B in nodes:
            box: IndexArray = B.idx[active[B.idx]]
            near_nodes = admissibility.near_nodes(tree, B)
            near: NearIdx = np.concatenate([nb.idx[active[nb.idx]] for nb in near_nodes] + [np.zeros(0, int)])
            mask = active.copy(); mask[box] = False; mask[near] = False
            far: FarIdx = np.flatnonzero(mask)
            if box.size == 0 or far.size == 0:        # nothing to compress against: keep all (goes up a level)
                skel_sizes.append(box.size)
                continue
            q_nodes: QSet = modified[B] - set(near_nodes) if track_modified else set()
            q_extra: IndexArray = np.concatenate([m.idx[active[m.idx]] for m in q_nodes] + [np.zeros(0, int)])
            # 1. skeleton: ID against the far field
            sk, rd, T = _skeletonize(compressor, W, box, far, B, q_extra)
            skel: SkeletonIdx = box[sk]
            red: RedundantIdx = box[rd]
            skel_sizes.append(skel.size)
            if red.size == 0:
                continue
            # 2. sparsify with T (columns, then rows); far entries of R are now O(eps) and dropped
            loc: IndexArray = np.concatenate([box, near])
            LR, RL = np.ix_(loc, red), np.ix_(red, loc)
            W[LR] = W[LR] - W[np.ix_(loc, skel)] @ T
            W[RL] = W[RL] - T.T @ W[np.ix_(skel, loc)]
            # 3. eliminate R: Schur complement onto S u N
            sn: SkelNearIdx = np.concatenate([skel, near])
            lu = _lu_factor(W[np.ix_(red, red)])
            X_snr: Matrix = W[np.ix_(sn, red)]
            X_rsn: Matrix = W[np.ix_(red, sn)]
            SS = np.ix_(sn, sn)
            W[SS] = W[SS] - X_snr @ sla.lu_solve(lu, X_rsn)
            steps.append(_Step(skel, red, sn, T, lu, X_snr, X_rsn))
            active[red] = False
            n_elim += red.size
            group: list[Node] = [B] + list(near_nodes)  # record fill-in among B and its neighbours
            for P in group:
                modified[P].update(Q for Q in group if Q is not P)
        levels[l] = LevelStats(boxes=len(nodes), skel_max=max(skel_sizes, default=0),
                               skel_mean=float(np.mean(skel_sizes)) if skel_sizes else 0.0,
                               skel_total=int(np.sum(skel_sizes)), eliminated=n_elim)
    root: IndexArray = np.flatnonzero(active)
    root_lu = _lu_factor(W[np.ix_(root, root)]) if root.size else None
    stats = FactorStats(admissibility=admissibility.name, n=n, depth=tree.depth, levels=levels,
                        root_size=int(root.size), n_steps=len(steps), factor_time=time.perf_counter() - t0)
    return Factorization(n, steps, root, root_lu, W.dtype, stats)
