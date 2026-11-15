"""E5 fix (g): recursive skeletonization whose every sparsification is a CONGRUENCE  W -> U^H W U.

Identical to unified/drivers/rskel.py (weak or strong, exact far field read from the current matrix) except:
  * the ID is taken of M = [W[F, B]; W[B, F]^H]  (conjugate transpose, not plain transpose), so that
        W[F, R] ~= W[F, S] T     and     W[R, F] ~= T^H W[S, F];
  * the row operation is  W[R, :] -= T^H W[S, :]  (the driver uses T^T).
Then X_RR = Pi^H W_BB Pi with Pi = [I; -T], and Herm(X_RR) >= lambda_min(Herm W_BB) I (findings.md, claim 3).
Kept small and separate because the shared driver may not be edited (EXPLORE.md rules).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from unified.core import interp_decomp
from unified.core.tree import Tree
from unified.core.types import Admissibility, IndexArray, InterpMatrix, Matrix, RedundantIdx, SkeletonIdx, Vector
from unified.drivers.rskel import LUFactors, _lu_factor

type SkelNearIdx = IndexArray


@dataclass
class CStep:
    """One box elimination of the congruent factorization."""
    skel: SkeletonIdx
    red: RedundantIdx
    sn: SkelNearIdx
    T: InterpMatrix
    lu: LUFactors
    X_snr: Matrix
    X_rsn: Matrix


@dataclass
class CongruentFactorization:
    n: int
    steps: list[CStep]
    root: IndexArray
    root_lu: LUFactors | None
    skel_total: int

    def solve(self, b: Vector) -> Vector:
        x: Vector = np.array(b, dtype=complex, copy=True)
        for s in self.steps:
            x[s.red] = x[s.red] - s.T.conj().T @ x[s.skel]
            y: Vector = sla.lu_solve(s.lu, x[s.red])
            x[s.sn] = x[s.sn] - s.X_snr @ y
            x[s.red] = y
        if self.root.size and self.root_lu is not None:
            x[self.root] = sla.lu_solve(self.root_lu, x[self.root])
        for s in reversed(self.steps):
            x[s.red] = x[s.red] - sla.lu_solve(s.lu, s.X_rsn @ x[s.sn])
            x[s.skel] = x[s.skel] - s.T @ x[s.red]
        return x

    def cond_rr_max(self) -> float:
        return max((float(np.linalg.cond(np.triu(s.lu[0]))) for s in self.steps), default=1.0)


def congruent_factor(A: Matrix, tree: Tree, admissibility: Admissibility, eps: float) -> CongruentFactorization:
    W: Matrix = np.array(A, dtype=complex, copy=True)
    n = W.shape[0]
    active = np.ones(n, bool)
    steps: list[CStep] = []
    skel_total = 0
    for lvl in range(tree.depth, 0, -1):
        for B in tree.levels[lvl]:
            box: IndexArray = B.idx[active[B.idx]]
            near_nodes = admissibility.near_nodes(tree, B)
            near: IndexArray = np.concatenate([nb.idx[active[nb.idx]] for nb in near_nodes] + [np.zeros(0, np.intp)])
            mask = active.copy(); mask[box] = False; mask[near] = False
            far: IndexArray = np.flatnonzero(mask)
            if box.size == 0 or far.size == 0:
                skel_total += box.size
                continue
            M: Matrix = np.vstack([W[np.ix_(far, box)], W[np.ix_(box, far)].conj().T])
            sk, rd, T = interp_decomp(M, eps)
            skel, red = box[sk], box[rd]
            skel_total += skel.size
            if red.size == 0:
                continue
            loc = np.concatenate([box, near])
            W[np.ix_(loc, red)] = W[np.ix_(loc, red)] - W[np.ix_(loc, skel)] @ T
            W[np.ix_(red, loc)] = W[np.ix_(red, loc)] - T.conj().T @ W[np.ix_(skel, loc)]
            sn = np.concatenate([skel, near])
            lu = _lu_factor(W[np.ix_(red, red)])
            X_snr, X_rsn = W[np.ix_(sn, red)], W[np.ix_(red, sn)]
            W[np.ix_(sn, sn)] = W[np.ix_(sn, sn)] - X_snr @ sla.lu_solve(lu, X_rsn)
            steps.append(CStep(skel, red, sn, T, lu, X_snr, X_rsn))
            active[red] = False
    root = np.flatnonzero(active)
    root_lu = _lu_factor(W[np.ix_(root, root)]) if root.size else None
    return CongruentFactorization(n, steps, root, root_lu, skel_total)
