"""1D interpolative butterfly (ID butterfly), built from the shared core: one `Tree` over the positions
x, one `Tree` over the frequencies xi, `Complementary` pairs, and the SAME `interp_decomp` that the
recursive-skeletonization drivers use. No SVDs anywhere.

Style: O'Neil, Woolfe & Rokhlin 2010 (skeleton columns, "equivalent weights") and the column-ID half of
Pang, Ho & Yang's IDBF (arXiv:1809.10573). [cited]

Notation (N = 2^L points, trees of depth L):
  * A tile at stage l is a pair (A, B): row box A at depth l of the x-tree, column box B at depth L - l of
    the xi-tree. Every tile covers one phase-space cell, |A| * |B| = N entries, so it has small
    numerical rank r (complementary low-rank property).                                    [cited]
  * The column ID of a tile picks SKELETON columns sk(A, B) ⊂ B and an interpolation matrix T with
        K(A, B)[:, red] ≈ K(A, B)[:, skel] @ T,   i.e.   K(A, B) ≈ K(A, sk(A, B)) @ P_AB
    where P_AB = [I | T] (columns in their original order) is the full r x |B| interpolation matrix.
  * The butterfly keeps "equivalent weights" c_AB (length r) with
        K(A, B) g_B ≈ K(A, sk(A, B)) c_AB.
    Weights of all tiles at stage l, stacked, form the vector c^(l).

Factors (same shape as the packaged reference `unified.reference.butterfly`):
    c^(0) = W g                  W: tile (root, B):  c = P_{root,B} g_B         (ID of K(all x, B))
    c^(l+1) = G_l c^(l)          G_l: tile (Ac, P) merges tiles (A, B1), (A, B2), A = parent(Ac),
                                      B1, B2 = children(P). Candidate columns S = sk(A,B1) ∪ sk(A,B2):
                                      K(Ac, P) g_P ≈ K(Ac, S) [c1; c2] ≈ K(Ac, sk(Ac,P)) P_new [c1; c2]
                                      so the block row of G_l is P_new (the ID of K(Ac, S)).
    f = U c^(L)                  U: tile (A, root) with A a leaf:  f_A = K(A, sk(A, root)) c_A.
    =>  K ≈ U G_{L-1} ... G_0 W,   L + 2 scipy.sparse factors, applied by `apply(g)`.

Construction here uses all rows of each tile (exact entries, O(N^2) work, simple to read); IDBF reduces
it to O(N log N) by sampling rows. Apply is O(r N log N).                       [computed / cited]
"""
from __future__ import annotations
from dataclasses import dataclass, field
import importlib.util
import pathlib
import time
from types import ModuleType
from typing import Callable, NamedTuple, TypedDict, cast

import numpy as np
import scipy.sparse as sp

from unified.core import Tree, Complementary, interp_decomp, kernels
from unified.core.admissibility import TilePair
from unified.core.kernels import ButterflyKernel
from unified.core.types import (ComplexArray, Field, FrequencyGrid, IndexArray, InterpMatrix, KernelMatrix,
                                PositionGrid, RedundantIdx, SkeletonIdx, Spectrum)

# ---- vocabulary of this driver -----------------------------------------------------------------------
type Tile = TilePair                        # (row box A at depth l, column box B at depth L - l)
type FrequencyIdx = IndexArray              # global indices into the frequency grid xi (skeleton columns)
type FullInterpMatrix = ComplexArray        # P = [I | T] in original column order, shape (r, |B|)
type EquivalentWeights = ComplexArray       # c^(l): the stacked equivalent weights of all tiles at stage l
type SparseFactor = sp.csr_matrix           # one butterfly factor (W, G_l or U), CSR
type TileBlock = Callable[[IndexArray, IndexArray], KernelMatrix]   # K(x[rows], xi[cols]): one tile's entries


class ButterflyStats(TypedDict, total=False):
    """`IDButterfly.stats` (a plain dict at runtime; empty until `butterfly_id` fills it)."""
    N: int
    L: int
    eps: float
    ranks: list[int]
    nnz: int
    build_time: float


class B1Problem(NamedTuple):
    """Spec problem B1 (unpacks as `kernel, x, xi = b1_problem(N, t)`)."""
    kernel: ButterflyKernel
    x: PositionGrid
    xi: FrequencyGrid


class TreePair(NamedTuple):
    """The two butterfly trees (unpacks as `row_tree, col_tree = build_trees(x, xi)`)."""
    row_tree: Tree      # over positions x (rows of K)
    col_tree: Tree      # over frequencies xi (columns of K)


class SVDComparison(TypedDict):
    """Result of `compare_with_svd`: error, storage, ranks and build time of both butterflies. [computed]"""
    N: int
    t: float
    eps: float
    id_err: float
    id_nnz: int
    id_ranks: list[int]
    id_time: float
    svd_err: float
    svd_nnz: int
    svd_ranks: list[int]
    svd_time: float


# ---- small helpers ---------------------------------------------------------------------------------
def interp_matrix(skel: SkeletonIdx, red: RedundantIdx, T: InterpMatrix, n: int) -> FullInterpMatrix:
    """Full interpolation matrix P (k x n) of a column ID: P[:, skel] = I, P[:, red] = T,
    so that M ≈ M[:, skel] @ P."""
    P: FullInterpMatrix = np.zeros((len(skel), n), dtype=complex)
    P[np.arange(len(skel)), skel] = 1.0
    P[:, red] = T
    return P


def build_trees(x: PositionGrid, xi: FrequencyGrid, depth: int | None = None) -> TreePair:
    """The two trees of the butterfly: positions x (rows) and frequencies xi (columns), same depth.
    Default depth = log2(N), i.e. one point per leaf."""
    N = len(x)
    assert len(xi) == N, "the 1D butterfly here is square"
    if depth is None:
        depth = int(round(np.log2(N)))
    return TreePair(Tree(np.asarray(x, float), depth=depth), Tree(np.asarray(xi, float), depth=depth))


@dataclass
class IDButterfly:
    """K ≈ U @ G[L-1] @ ... @ G[0] @ W, all scipy.sparse CSR. `ranks[l]` = max skeleton size at stage l."""
    W: SparseFactor           # c^(0) = W g
    G: list[SparseFactor]     # c^(l+1) = G[l] c^(l)
    U: SparseFactor           # f = U c^(L)
    ranks: list[int]
    build_time: float = 0.0
    stats: ButterflyStats = field(default_factory=lambda: ButterflyStats())   # filled in by butterfly_id

    def apply(self, g: Spectrum) -> Field:
        """f = U G_{L-1} ... G_0 W g: L + 2 sparse matrix-vector products (works for g of shape (N,) or (N, m))."""
        c = cast(EquivalentWeights, self.W @ g)
        for Gl in self.G:
            c = cast(EquivalentWeights, Gl @ c)
        return cast(Field, self.U @ c)

    @property
    def nnz(self) -> int:
        """Storage = number of stored nonzeros over all sparse factors."""
        return self.W.nnz + sum(Gl.nnz for Gl in self.G) + self.U.nnz


# ---- the butterfly ---------------------------------------------------------------------------------
def butterfly_id(kernel: ButterflyKernel, x: PositionGrid, xi: FrequencyGrid, eps: float = 1e-8,
                 row_tree: Tree | None = None, col_tree: Tree | None = None) -> IDButterfly:
    """Interpolative butterfly factorization of K[i, j] = kernel(x_i, xi_j), from kernel entries only.

    Steps (one per stage l = 0..L of the complementary pairing):
      1. stage 0, tiles (root, B): ID of the tile K(all x, B) -> skeleton columns sk(root, B), and the
         interpolation matrix P goes into W.
      2. stage l -> l+1, tiles (Ac, P): candidate columns = skeletons of the two parent tiles (A, B1),
         (A, B2); ID of K(Ac, candidates) -> new skeleton + interpolation matrix = block row of G_l
         (the butterfly "X": each new tile reads two old tiles).
      3. stage L, tiles (leaf A, root): the final skeleton columns are evaluated exactly, K(A, sk), into U.
    """
    t0 = time.perf_counter()
    x, xi = np.asarray(x, float), np.asarray(xi, float)
    N = len(x)
    if row_tree is None or col_tree is None:
        row_tree, col_tree = build_trees(x, xi)
    L = row_tree.depth
    rule = Complementary()
    tiles: list[list[Tile]] = [rule.pairs(row_tree, col_tree, l) for l in range(L + 1)]
    Kblk: TileBlock = lambda rows, cols: kernel(x[rows], xi[cols])   # entry access: one tile block at a time

    skel: list[dict[Tile, FrequencyIdx]] = [dict() for _ in range(L + 1)]   # skel[l][(A, B)] = global xi indices of the skeleton columns
    offset: list[dict[Tile, int]] = [dict() for _ in range(L + 1)]  # where tile (A, B)'s weights sit inside c^(l)
    size: list[int] = [0] * (L + 1)                                 # length of c^(l)

    def place(l: int, tile: Tile, cols: FrequencyIdx) -> None:
        skel[l][tile], offset[l][tile] = cols, size[l]
        size[l] += len(cols)

    # 1. Stage 0: compress each column box B against all rows.
    W_blocks: list[tuple[int, FrequencyIdx, FullInterpMatrix]] = []   # (row offset in c^(0), columns, P)
    for root, B in tiles[0]:
        s, r, T = interp_decomp(Kblk(row_tree.root.idx, B.idx), eps)
        place(0, (root, B), B.idx[s])
        W_blocks.append((offset[0][root, B], B.idx, interp_matrix(s, r, T, len(B.idx))))
    W = sp.lil_matrix((size[0], N), dtype=complex)
    for o, cols, P in W_blocks:
        W[o:o + P.shape[0], cols] = P

    # 2. Stages 1..L: merge the two parent tiles' skeletons, re-skeletonize on the child row box.
    G: list[SparseFactor] = []
    for l in range(L):
        blocks: list[tuple[Tile, list[Tile], FullInterpMatrix]] = []   # (new tile, its two parent tiles, P_new)
        for Ac, Pc in tiles[l + 1]:
            A = Ac.parent
            assert A is not None   # Ac sits at depth l + 1 >= 1
            olds: list[Tile] = [(A, B) for B in Pc.children]
            cand: FrequencyIdx = np.concatenate([skel[l][t] for t in olds])          # candidate skeleton columns S
            s, r, T = interp_decomp(Kblk(Ac.idx, cand), eps)
            place(l + 1, (Ac, Pc), cand[s])
            blocks.append(((Ac, Pc), olds, interp_matrix(s, r, T, len(cand))))
        Gl = sp.lil_matrix((size[l + 1], size[l]), dtype=complex)
        for new, olds, P in blocks:
            i, j0 = offset[l + 1][new], 0
            for t in olds:                                               # split P column-wise by parent tile
                k = len(skel[l][t])
                Gl[i:i + P.shape[0], offset[l][t]:offset[l][t] + k] = P[:, j0:j0 + k]
                j0 += k
        G.append(Gl.tocsr())

    # 3. Stage L: f_A = K(A, sk(A, root)) c_A for every leaf A of the x-tree.
    U = sp.lil_matrix((N, size[L]), dtype=complex)
    for A, root in tiles[L]:
        cols, o = skel[L][A, root], offset[L][A, root]
        U[np.ix_(A.idx, np.arange(o, o + len(cols)))] = Kblk(A.idx, cols)

    ranks: list[int] = [max(len(c) for c in skel[l].values()) for l in range(L + 1)]
    bf = IDButterfly(W.tocsr(), G, U.tocsr(), ranks, time.perf_counter() - t0)
    bf.stats = ButterflyStats(N=N, L=L, eps=eps, ranks=ranks, nnz=bf.nnz, build_time=bf.build_time)
    return bf


# ---- B1 test problem and comparison with the SVD reference ----------------------------------------
def b1_problem(N: int, t: float) -> B1Problem:
    """Spec problem B1: K = fio1d(t), x = arange(N)/N, xi = arange(N) - N/2."""
    xi = cast(FrequencyGrid, np.arange(N) - N // 2)   # integer-valued; every consumer applies np.asarray(xi, float)
    return B1Problem(kernels.fio1d(t), np.arange(N) / N, xi)


def exact_apply(kernel: ButterflyKernel, x: PositionGrid, xi: FrequencyGrid, g: Spectrum,
                chunk: int = 512) -> Field:
    """Reference f = K g by evaluating K a block of rows at a time (no N x N array kept)."""
    f: Field = np.concatenate([kernel(x[i:i + chunk], xi) @ g for i in range(0, len(x), chunk)])
    return f


def load_svd_reference() -> ModuleType:
    """Import the packaged, read-only SVD butterfly used by the comparison tests."""
    return importlib.import_module("unified.reference.butterfly")


def compare_with_svd(N: int, t: float, eps: float = 1e-8, seed: int = 0) -> SVDComparison:
    """Error and storage (nnz) of the ID butterfly vs the SVD reference on B1. [computed]"""
    kernel, x, xi = b1_problem(N, t)
    g: Spectrum = np.random.default_rng(seed).standard_normal(N) + 1j * np.random.default_rng(seed + 1).standard_normal(N)
    f = exact_apply(kernel, x, xi, g)
    bf = butterfly_id(kernel, x, xi, eps)
    ref = load_svd_reference()
    t0 = time.perf_counter()
    W, G, Uf, ranks_svd = ref.butterfly_factorize(kernel(x, xi), eps=eps)
    t_svd = time.perf_counter() - t0
    f_svd = ref.butterfly_apply(W, G, Uf, g)
    rel: Callable[[Field], float] = lambda h: float(np.linalg.norm(h - f) / np.linalg.norm(f))
    return SVDComparison(N=N, t=t, eps=eps,
                         id_err=rel(bf.apply(g)), id_nnz=bf.nnz, id_ranks=bf.ranks, id_time=bf.build_time,
                         svd_err=rel(f_svd), svd_nnz=W.nnz + sum(Gl.nnz for Gl in G) + Uf.nnz,
                         svd_ranks=ranks_svd, svd_time=t_svd)


if __name__ == "__main__":
    print(f"{'N':>5} {'t':>2} | {'ID err':>8} {'ID nnz':>10} {'ID max r':>8} | "
          f"{'SVD err':>8} {'SVD nnz':>10} {'SVD max r':>9} | nnz ratio ID/SVD")
    for N in (256, 1024, 4096):
        for t in (0, 1, 3):
            r = compare_with_svd(N, t)
            print(f"{N:5d} {t:2d} | {r['id_err']:8.1e} {r['id_nnz']:10,d} {max(r['id_ranks']):8d} | "
                  f"{r['svd_err']:8.1e} {r['svd_nnz']:10,d} {max(r['svd_ranks']):9d} | "
                  f"{r['id_nnz'] / r['svd_nnz']:.2f}")
