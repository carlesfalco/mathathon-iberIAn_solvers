"""Storage of one dense kernel matrix K under the three admissibility rules of unified/core.

All three use the same two `Tree`s (rows, columns; same depth L) and the same eps-rank rule
(singular values sigma_i > eps * sigma_1 of the block or tile).

  weak (HODLR)        block tree: a pair of same-level boxes is compressed iff their coords differ
                      (the `Weak` rule: no near field). Leaf pairs with equal coords are dense.
  strong (H)          block tree: a pair is compressed iff the boxes do not touch (the `Strong` rule:
                      neighbours stay exact). Touching leaf pairs are dense.
  complementary (BF)  butterfly tiles: row box at depth l x column box at depth L - l, for every l
                      (the `Complementary` rule). Nested orthonormal bases, as in code/butterfly.py,
                      generalised to any pair of trees (1D binary or 2D quadtree).

Storage = numbers stored.  weak/strong: sum over compressed blocks of min(r (m + n), m n) plus m n for
dense blocks (flat, non-nested H storage).  complementary: entries of the L + 2 butterfly factors
W, G_0..G_{L-1}, U (same count as `nnz` in code/butterfly.py).
"""
from __future__ import annotations
from typing import Any, cast

import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from unified.core import Tree, Weak, Strong, Complementary
from unified.core.tree import Node
from unified.core.types import (ComplexArray, Field, GridCoord, IndexArray, KernelMatrix, Matrix,
                                RealArray, Spectrum)

# ---- domain aliases ---------------------------------------------------------------------------------------
type StorageCount = int                      # numbers stored by a compressed representation
type Rank = int                              # eps-rank of one block / tile
type Tolerance = float                       # relative eps of the rank rule
type SingularValues = RealArray              # (min(m, n),) descending
type Ordering = str                          # "natural" | "bitrev"
type BlockPair = tuple[Node, Node]           # (row box, column box) at the same level
type TileKey = tuple[GridCoord, GridCoord]   # (row box coord, column box coord) of a butterfly tile
type TileRanks = list[dict[TileKey, Rank]]   # R[l][tile] for every stage l = 0..L
type TileBases = list[dict[TileKey, Matrix]] # U[l][tile]: orthonormal column basis of the tile
type Coeffs = dict[TileKey, ComplexArray]    # c^(l): butterfly coefficients of every tile at one stage
type Report = dict[str, Any]                 # JSON-ready measurement record


def parent(node: Node) -> Node:
    """The parent box (every node below the root has one)."""
    assert node.parent is not None
    return node.parent


# ---- the eps-rank of a block -----------------------------------------------------------------------
def eps_rank(s: SingularValues, eps: Tolerance) -> Rank:
    """Numerical rank of a block with singular values s: #{sigma_i > eps * sigma_1}."""
    return int(np.sum(s > eps * s[0])) if s.size and s[0] > 0 else 0


def svals(B: Matrix) -> SingularValues:
    """Singular values of a block (no vectors)."""
    sv: SingularValues = sla.svd(cast(ComplexArray, B), compute_uv=False, check_finite=False, lapack_driver="gesdd")
    return sv


# ---- 1D trees on an ORDERING coordinate --------------------------------------------------------------
def bitrev(n: int) -> IndexArray:
    """Bit-reversal permutation of 0..n-1 (n a power of two)."""
    L = int(round(np.log2(n)))
    j: IndexArray = np.arange(n)
    r: IndexArray = np.zeros(n, np.intp)
    for b in range(L):
        r |= ((j >> b) & 1) << (L - 1 - b)
    return r


def order_tree(n: int, depth: int, order: Ordering = "natural") -> Tree:
    """A binary tree whose boxes are contiguous blocks of the chosen ordering of 0..n-1.
    natural: blocks are index intervals.  bitrev: point j sits at position bitrev(j), so a depth-l box is a
    residue class j = a (mod 2^l) - exactly the index sets the radix-2 FFT recurses on."""
    pos: IndexArray = np.arange(n) if order == "natural" else bitrev(n)
    u: RealArray = (pos + 0.5) / n          # ordering coordinate in (0, 1)
    return Tree(u, depth=depth, bbox=([0.0], [1.0]))


# ---- weak / strong: flat H block partition -------------------------------------------------------------
def _admissible(rule: Weak | Strong, s: Node, t: Node) -> bool:
    """Weak: coords differ. Strong: boxes do not touch. (Same-level boxes of trees with equal depth.)"""
    d = max(abs(a - b) for a, b in zip(s.coord, t.coord))
    return d > 0 if isinstance(rule, Weak) else d > 1


def block_partition(rtree: Tree, ctree: Tree, rule: Weak | Strong) -> tuple[list[BlockPair], list[BlockPair]]:
    """H block tree from the root pair down. Returns (lowrank_pairs, dense_pairs) of (row node, col node).
    Every matrix entry lies in exactly one block (tested in test_structures.py)."""
    assert rtree.depth == ctree.depth
    low: list[BlockPair] = []
    dense: list[BlockPair] = []
    stack: list[BlockPair] = [(rtree.root, ctree.root)]
    while stack:
        s, t = stack.pop()
        if s.is_leaf or t.is_leaf:
            dense.append((s, t))
            continue
        for sc in s.children:
            for tc in t.children:
                (low if _admissible(rule, sc, tc) else stack).append((sc, tc))
    return low, dense


def hmatrix_storage(K: KernelMatrix, rtree: Tree, ctree: Tree, rule: Weak | Strong,
                    eps: Tolerance) -> Report:
    """Flat (non-nested) H storage of K under `rule` (Weak or Strong), eps-ranks by SVD.
    Returns dict: storage, max_rank, rel. Frobenius error (exact, from discarded singular values)."""
    low, dense = block_partition(rtree, ctree, rule)
    storage: StorageCount = 0
    max_rank: Rank = 0
    err2 = 0.0                                   # squared Frobenius norm of everything discarded
    for s, t in dense:
        storage += len(s.idx) * len(t.idx)
    for s, t in low:
        m, n = len(s.idx), len(t.idx)
        sv = svals(K[np.ix_(s.idx, t.idx)])
        r = eps_rank(sv, eps)
        if r * (m + n) < m * n:
            storage += r * (m + n)
            err2 += float(np.sum(sv[r:] ** 2))
        else:
            storage += m * n
        max_rank = max(max_rank, r)
    return dict(storage=int(storage), max_rank=int(max_rank), n_lowrank=len(low), n_dense=len(dense),
                err=float(np.sqrt(err2) / np.linalg.norm(K)))


# ---- complementary: butterfly with nested SVD bases -----------------------------------------------------
def basis(B: Matrix, eps: Tolerance) -> Matrix:
    """Orthonormal basis of the eps-numerical column space of a tile (at least one column)."""
    if B.size == 0:
        empty: Matrix = np.zeros((B.shape[0], 0), B.dtype)
        return empty
    U: Matrix
    s: SingularValues
    U, s, _ = sla.svd(cast(ComplexArray, B), full_matrices=False, check_finite=False, lapack_driver="gesdd")
    return U[:, :max(1, eps_rank(s, eps))]


class Butterfly:
    """Butterfly of K over two trees of equal depth L under `Complementary` pairs.
    Stage l tiles (A, B): A at depth l of the row tree, B at depth L-l of the column tree.
        c(root, B) = U^H K(:, B) g_B                                   (factor W)
        c(Ac, P)   = sum_{B child of P} U(Ac,P)^H U(A,B)[rows of Ac] c(A, B),  A = parent(Ac)   (G_l)
        f_A        = U(A, root) c(A, root)                             (factor U)
    """

    K: KernelMatrix
    rt: Tree
    ct: Tree
    L: int
    U: TileBases
    max_rank_per_stage: list[Rank]
    pos: dict[int, IndexArray]          # id(child row box) -> its rows inside the parent box

    def __init__(self, K: KernelMatrix, rtree: Tree, ctree: Tree, eps: Tolerance) -> None:
        assert rtree.depth == ctree.depth
        self.K, self.rt, self.ct, self.L = K, rtree, ctree, rtree.depth
        L, cp = self.L, Complementary()
        self.U = [{(a.coord, b.coord): basis(K[np.ix_(a.idx, b.idx)], eps) for a, b in cp.pairs(rtree, ctree, l)}
                  for l in range(L + 1)]
        self.max_rank_per_stage = [max(u.shape[1] for u in Ul.values()) for Ul in self.U]
        # where each child's rows sit inside its parent's index list (Tree.idx is sorted)
        self.pos = {id(c): np.searchsorted(parent(c).idx, c.idx) for lv in rtree.levels[1:] for c in lv}

    def ranks(self) -> TileRanks:
        return [{key: u.shape[1] for key, u in Ul.items()} for Ul in self.U]

    def storage(self) -> StorageCount:
        return bf_storage(self.ranks(), self.rt, self.ct)

    def apply(self, g: Spectrum) -> Field:
        L, U, K = self.L, self.U, self.K
        rroot = self.rt.root
        c: Coeffs = {(rroot.coord, b.coord): U[0][(rroot.coord, b.coord)].conj().T @ (K[:, b.idx] @ g[b.idx])
             for b in self.ct.levels[L]}
        for l in range(L):
            cn: Coeffs = {}
            for ac in self.rt.levels[l + 1]:
                A, rows = parent(ac), self.pos[id(ac)]
                for P in self.ct.levels[L - l - 1]:
                    Un = U[l + 1][(ac.coord, P.coord)]
                    acc: ComplexArray = np.zeros(Un.shape[1], complex)
                    for B in P.children:
                        acc += Un.conj().T @ (U[l][(A.coord, B.coord)][rows] @ c[(A.coord, B.coord)])
                    cn[(ac.coord, P.coord)] = acc
            c = cn
        f: Field = np.zeros(K.shape[0], complex)
        croot = self.ct.root
        for a in self.rt.levels[L]:
            f[a.idx] = U[L][(a.coord, croot.coord)] @ c[(a.coord, croot.coord)]
        return f


def butterfly_storage(K: KernelMatrix, rtree: Tree, ctree: Tree, eps: Tolerance, n_probe: int = 4,
                      seed: int = 0) -> Report:
    """Butterfly storage, max tile rank, and relative Frobenius error estimated with Gaussian probes
    (E||(K - B) g||^2 = ||K - B||_F^2)."""
    bf = Butterfly(K, rtree, ctree, eps)
    G: RealArray = np.random.default_rng(seed).standard_normal((K.shape[1], n_probe))
    E: Matrix = np.column_stack([bf.apply(G[:, j]) for j in range(n_probe)]) - K @ G
    err = np.linalg.norm(E) / np.sqrt(n_probe) / np.linalg.norm(K)
    return dict(storage=bf.storage(), max_rank=int(max(bf.max_rank_per_stage)),
                rank_per_stage=[int(r) for r in bf.max_rank_per_stage], err=float(err))


def bf_storage(R: TileRanks, rtree: Tree, ctree: Tree) -> StorageCount:
    """Entries of W, G_0..G_{L-1}, U given the rank R[l][(row coord, col coord)] of every tile
    (the count code/butterfly.py reports as nnz). Depends only on tile ranks and box sizes."""
    L = rtree.depth
    st: StorageCount = sum(R[0][(rtree.root.coord, b.coord)] * len(b.idx) for b in ctree.levels[L])
    for l in range(L):
        for ac in rtree.levels[l + 1]:
            for P in ctree.levels[L - l - 1]:
                st += R[l + 1][(ac.coord, P.coord)] * sum(R[l][(parent(ac).coord, B.coord)] for B in P.children)
    st += sum(len(a.idx) * R[L][(a.coord, ctree.root.coord)] for a in rtree.levels[L])
    return int(st)


def tile_ranks(K: KernelMatrix, rtree: Tree, ctree: Tree, eps: Tolerance) -> TileRanks:
    """eps-rank (>= 1, as in `basis`) of every complementary tile: R[l][(row coord, col coord)].
    No factorization; used for the ordering test."""
    cp = Complementary()
    return [{(a.coord, b.coord): max(1, eps_rank(svals(K[np.ix_(a.idx, b.idx)]), eps))
             for a, b in cp.pairs(rtree, ctree, l)} for l in range(rtree.depth + 1)]
