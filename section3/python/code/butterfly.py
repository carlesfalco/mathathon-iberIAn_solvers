"""SVD butterfly reference (orthonormal bases) for the 1D butterfly test problem B1.

Partner of `unified/drivers/butterfly_id.py`: the ID butterfly keeps *skeleton columns* of the matrix,
this reference keeps *truncated SVD bases*. Same phase-space tiling, same factor layout, same storage
accounting, so the two `nnz` numbers are directly comparable.

Style: the 1D butterfly factorization of Michielssen & Bojarski as presented by O'Neil, Woolfe &
Rokhlin 2010 (Section 3, nested bases) and Martinsson 2019, *Fast Direct Solvers for Elliptic PDEs*,
Chapter 12 (butterfly factorization). [cited]

Notation (N = 2^L; K is the N x N matrix K[i, j] = kernel(x_i, xi_j), rows ordered by x and columns by
xi, as in spec problem B1):

  * A tile at stage l is a pair (A, B): row box A at depth l of the x-tree, column box B at depth
    L - l of the xi-tree. Both grids are monotone, so a depth-d box is the contiguous index interval
    [b * d-width, ... + d-width) and stage l is just the dyadic index split: |A| = N/2^l, |B| = 2^l.
    Every tile covers |A| * |B| = N entries, so each stage has N tiles and partitions the N x N matrix
    (compare Complementary().pairs in unified/core/admissibility.py).
  * Complementary low-rank property: since |A| * |B| = N fixes the phase x * xi over a tile, every
    tile has small numerical rank r, independent of N.                                    [cited]

Factors. For every tile the reference carries an orthonormal basis U_{A,B} (|A| x r) of the tile
K(A, B) and weights c_{A,B} (length r) with K(A, B) g_B ~= U_{A,B} c_{A,B}; c^(l) stacks c_{A,B} over
the tiles of stage l. The three steps are the three steps of the ID driver:

    U_{A,B}         the leading k left singular vectors of the tile K(A, B), k = eps-rank of the tile
                    (see `truncate`). Stage 0 and stage L are the degenerate ends: a single column
                    (k = 1) and a single row (k = 1).
    c^(0) = W g     tile (root, {j}) has a SINGLE column, so its 1 x 1 block is the singular value
                    sigma_j = ||K[:, j]||_2: c_j = sigma_j g_j, and K(x, {j}) g_j = U_j c_j holds
                    exactly with U_j = K[:, j] / sigma_j.  =>  W = diag(sigma_j): N stored numbers.
    c^(l+1) = G_l c^(l)
                    tile (Ac, Pc): Ac a child row box of A, Pc the parent column box of B1, B2.
                        block row of G_l = U_{Ac,Pc}^H [ U_{A,B1}[Ac, :] | U_{A,B2}[Ac, :] ]
                                                                     (k x (r1 + r2))
                    so that K(Ac, Pc) g_{Pc} ~= U_{Ac,Pc} (G_l block) [c1; c2].
    f = U c^(L)     tile ({i}, root) has a SINGLE row, so its 1 x r block is U_{{i},root} and
                    f_i = U_{{i},root} c_{{i},root}.              =>  N stored numbers.

    =>  K ~= U G_{L-1} ... G_0 W,   L + 2 scipy.sparse factors, applied by `butterfly_apply`.

Why this is a factorization of K at all (induction on the stage): the stage-0 identity is exact; the
merge step is K(Ac, Pc) g = K(Ac, B1) g_{B1} + K(Ac, B2) g_{B2} ~= U_{A,B1}[Ac] c1 + U_{A,B2}[Ac] c2,
and the block row is exactly the projection onto the next basis. The stage-(l+1) bases are the only
thing the next stage needs, so nothing but W, the G_l and U has to be stored: the storage is
sum_l N * r_l * (r_{l-1} + r_{l-1}) = O(r^2 N log N), the same accounting as the ID butterfly, whose
bases are columns of K and are evaluated at the last stage.

Two construction choices that a reader should check, because both change the storage that claim 12 of
the README compares:
  1. Bases are the SVD of the tile itself, not the SVD of the merged parent bases. Deriving them from
     the parents instead is also a valid butterfly, but the truncation noise of the earlier stages sits
     in the parents' bases and inflates the eps-rank: measured on B1 at N = 1024, t = 0, eps = 1e-8,
     ranks per stage come out 1,2,4,8,12,20,16,8,4,2,1 instead of the tile ranks 1,2,4,8,9,9,9,8,4,2,1,
     i.e. 2.1x the storage of this file. Both are accurate to ~1e-9; only this one reports the storage
     of an SVD butterfly, so only this one is a fair baseline for the ID's nnz.        [computed]
  2. The truncation is relative to sigma_1 *of the tile*, `sigma_i > eps * sigma_1`, which is the SVD
     analogue of the ID's `interp_decomp(M, eps)` (a pivoted QR that drops |R_ii| <= eps |R_11|).
     Ranks are a function of eps, not constants: at eps = 1e-8 on B1 they are min(2^l, 2^(L-l), 9) for
     t = 0 and min(2^l, 2^(L-l), 6) in the middle for t = 3.                            [computed]

Scope. Square N = 2^L and the B1 ordering of both grids (monotone grids are what make a tree box an
index interval). Nothing else: `butterfly_factorize` reads entries of K, not a kernel. Cost: N small
SVDs per stage, all entries of K read once per stage; peak memory is the input K plus two stages of
bases, O(N^2), the same order as K. This is a reference for bench sizes (N <= 4096), not a solver.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

type SparseFactor = sp.csr_matrix            # one butterfly factor (W, G_l or U), CSR
type ComplexArray = np.ndarray              # dense complex array
type Vector = np.ndarray                    # length N, or (N, m) for a block of right-hand sides
type StageRanks = list[int]                 # max rank over the tiles of stage l, l = 0 .. L
type Basis = dict[tuple[int, int], ComplexArray]    # (row box, column box) -> orthonormal basis


class SVDButterfly(NamedTuple):
    """K ~= U @ G[L-1] @ ... @ G[0] @ W. Unpacks as `W, G, Uf, ranks = butterfly_factorize(K, eps)`."""
    W: SparseFactor           # c^(0) = W g             (N x N, diagonal)
    G: list[SparseFactor]     # c^(l+1) = G[l] c^(l)    (N x N each, block rows)
    U: SparseFactor           # f = U c^(L)             (N x N, one row per point)
    ranks: StageRanks         # max rank over the tiles of each stage

    @property
    def nnz(self) -> int:
        """Storage: stored numbers over all factors of the three groups."""
        return self.W.nnz + sum(Gl.nnz for Gl in self.G) + self.U.nnz


def truncate(sigma: Vector, eps: float) -> int:
    """Number of singular values kept: #{sigma_i > eps * sigma_1}, at least 1 when sigma_1 > 0."""
    if sigma.size == 0 or not sigma[0] > 0:
        return 0
    return max(1, int(np.count_nonzero(sigma > eps * sigma[0])))


def _stage_bases(K: ComplexArray, l: int, eps: float) -> tuple[Basis, int]:
    """Truncated SVD bases of every tile of stage l, and the largest rank of the stage.

    Stage l has 2^l row boxes of |A| = N/2^l rows and N/2^l column boxes of |B| = 2^l columns.
    """
    N = K.shape[0]
    bs, bw = N >> l, 1 << l
    basis: Basis = {}
    rank = 0
    for a in range(1 << l):
        rows = slice(a * bs, (a + 1) * bs)
        for b in range(N >> l):
            U, sigma, _ = np.linalg.svd(K[rows, b * bw:(b + 1) * bw], full_matrices=False)
            k = truncate(sigma, eps)
            basis[(a, b)] = np.ascontiguousarray(U[:, :k])
            rank = max(rank, k)
    return basis, rank


def butterfly_factorize(K: ComplexArray, eps: float = 1e-8) -> SVDButterfly:
    """SVD butterfly factorization of a dense N x N matrix K (N a power of two, B1 row/column order).

    Works on tile blocks only: no temporary bigger than one stage of bases is created.
    """
    K = np.asarray(K, dtype=complex)
    N = K.shape[0]
    if N == 0 or N & (N - 1) or K.shape != (N, N):
        raise ValueError(f"butterfly_factorize needs a square 2^L x 2^L matrix, got {K.shape}")
    L = N.bit_length() - 1

    # ---- stage 0: one single-column tile per column, basis = normalized column --------------------
    basis, rank0 = _stage_bases(K, 0, eps)              # each tile is K[:, j]: sigma_1 = ||K[:, j]||
    sigma0: Vector = np.array([np.linalg.norm(K[:, j]) for j in range(N)])
    W: SparseFactor = sp.diags(sigma0.astype(complex)).tocsr()      # c_j = sigma_j g_j, N numbers
    ranks: StageRanks = [rank0]

    # ---- stages 1..L: merge the two parent tiles' bases onto the child row box ---------------------
    G: list[SparseFactor] = []
    for l in range(L):
        new_basis, rank_new = _stage_bases(K, l + 1, eps)
        bs_new = N >> (l + 1)                          # rows per row box at stage l + 1
        # offset of each tile's weights inside c^(l); tiles are in (row box, column box) order
        off_old: dict[tuple[int, int], int] = {}
        n_old = 0
        for t, B in basis.items():
            off_old[t] = n_old
            n_old += B.shape[1]
        off_new: dict[tuple[int, int], int] = {}
        n_new = 0
        for t, B in new_basis.items():
            off_new[t] = n_new
            n_new += B.shape[1]
        Gl = sp.lil_matrix((n_new, n_old), dtype=complex)
        for a_new in range(1 << (l + 1)):              # row box Ac at depth l + 1
            a, half = divmod(a_new, 2)                 # parent row box A, and which half Ac occupies
            rows = slice(half * bs_new, (half + 1) * bs_new)      # Ac's rows inside A
            for b_new in range(N >> (l + 1)):          # column box Pc at depth L - l - 1
                Ac, Pc = (a_new, b_new), (a, 2 * b_new)          # (tile, first parent tile)
                Pc2 = (a, 2 * b_new + 1)
                M = np.concatenate([basis[Pc][rows, :], basis[Pc2][rows, :]], axis=1)
                P = new_basis[Ac].conj().T @ M                     # k x (r1 + r2)
                o, o1, o2 = off_new[Ac], off_old[Pc], off_old[Pc2]
                r1, r2 = basis[Pc].shape[1], basis[Pc2].shape[1]
                Gl[o:o + P.shape[0], o1:o1 + r1] = P[:, :r1]
                Gl[o:o + P.shape[0], o2:o2 + r2] = P[:, r1:]
        G.append(Gl.tocsr())
        ranks.append(rank_new)
        basis = new_basis

    # ---- stage L: one single-row tile per row, f_i = U_{{i},root} c_{{i},root} ---------------------
    U = sp.lil_matrix((N, sum(B.shape[1] for B in basis.values())), dtype=complex)
    o = 0
    for (a, b), B in basis.items():
        assert b == 0, "stage L has exactly one column box (the root)"
        assert B.shape[0] == 1, "stage L row boxes are single points"
        U[a, o:o + B.shape[1]] = B[0, :]
        o += B.shape[1]
    return SVDButterfly(W, G, U.tocsr(), ranks)


def butterfly_apply(W: SparseFactor, G: list[SparseFactor], Uf: SparseFactor, g: Vector) -> Vector:
    """f = U G_{L-1} ... G_0 W g: L + 2 sparse products (g of shape (N,) or (N, m))."""
    c = W @ np.asarray(g)
    for Gl in G:
        c = Gl @ c
    return Uf @ c


def reconstruct(bf: SVDButterfly) -> ComplexArray:
    """Dense product U G_{L-1} ... G_0 W (verification only: allocates N x N)."""
    M = bf.W
    for Gl in bf.G:
        M = Gl @ M
    return (bf.U @ M).toarray()


# ---- self-check: python butterfly.py ------------------------------------------------------------
def _b1_matrix(N: int, t: float) -> ComplexArray:
    """Spec problem B1: K[i, j] = exp(2 pi i (x_i xi_j + t c(x_i) |xi_j|)), x = i/N, xi = j - N/2.
    Mirrors unified.core.kernels.fio1d(t) with its default speed profile c(x) = (2 + sin 2 pi x)/16."""
    x = np.arange(N) / N
    xi = np.arange(N) - N // 2
    c = (2 + np.sin(2 * np.pi * x)) / 16
    return np.exp(2j * np.pi * (np.outer(x, xi) + t * np.outer(c, np.abs(xi))))


if __name__ == "__main__":
    print(f"{'N':>5} {'t':>2} {'eps':>6} | {'rel err':>9} {'nnz':>10} {'nnz/N':>7} {'max rank':>8}"
          f" | ranks per stage")
    for N_ in (256, 1024):
        for t_ in (0.0, 1.0, 3.0):
            K_ = _b1_matrix(N_, t_)
            bf_ = butterfly_factorize(K_, eps=1e-8)
            g_ = (np.random.default_rng(0).standard_normal(N_)
                  + 1j * np.random.default_rng(1).standard_normal(N_))
            err_ = np.linalg.norm(butterfly_apply(bf_.W, bf_.G, bf_.U, g_) - K_ @ g_) / np.linalg.norm(K_ @ g_)
            print(f"{N_:5d} {t_:2.0f} {1e-8:6.0e} | {err_:9.1e} {bf_.nnz:10,d} {bf_.nnz / N_:7.1f} "
                  f"{max(bf_.ranks):8d} | {bf_.ranks}")
