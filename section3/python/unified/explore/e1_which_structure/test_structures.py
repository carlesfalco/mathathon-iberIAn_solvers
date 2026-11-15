"""Correctness checks for the E1 storage measurements."""
import numpy as np
import pytest

from unified.core.types import KernelMatrix, Points

from unified.core import Tree, Weak, Strong, kernels
from unified.explore.e1_which_structure.structures import (
    block_partition, bitrev, order_tree, tile_ranks, bf_storage, butterfly_storage, hmatrix_storage)


def _coverage(rt: Tree, ct: Tree, rule: Weak | Strong, n: int, m: int) -> np.ndarray:
    """How many blocks of the partition contain each matrix entry."""
    low, dense = block_partition(rt, ct, rule)
    C = np.zeros((n, m), int)
    for s, t in low + dense:
        C[np.ix_(s.idx, t.idx)] += 1
    return C


@pytest.mark.parametrize("rule", [Weak(), Strong()])
def test_block_partition_covers_every_entry_once(rule: Weak | Strong) -> None:
    T = Tree(np.random.default_rng(1).random((300, 2)), max_leaf=8)
    assert (_coverage(T, T, rule, 300, 300) == 1).all()
    r, c = order_tree(128, 4), order_tree(128, 4, "bitrev")
    assert (_coverage(r, c, rule, 128, 128) == 1).all()


def test_bitrev_is_a_permutation_and_residue_classes() -> None:
    p = bitrev(64)
    assert sorted(p) == list(range(64))
    T = order_tree(64, 6, "bitrev")
    for node in T.levels[3]:            # depth-3 boxes are residue classes mod 8
        assert len(set(node.idx % 8)) == 1


def test_dft_bitreversed_tiles_have_exact_rank_one() -> None:
    N = 256
    K = kernels.fio1d(0.0)(np.arange(N) / N, np.arange(N, dtype=float) - N // 2)
    rt, ct = order_tree(N, 8, "bitrev"), order_tree(N, 8, "bitrev")
    R = tile_ranks(K, rt, ct, 1e-13)
    assert max(max(Rl.values()) for Rl in R) == 1
    assert bf_storage(R, rt, ct) == (2 * 8 + 2) * N      # the radix-2 FFT: 2 entries per row per stage
    Rn = tile_ranks(K, order_tree(N, 8), order_tree(N, 8), 1e-6)
    assert max(max(Rl.values()) for Rl in Rn) > 1        # natural order: small but not 1


@pytest.mark.parametrize("dim", [1, 2])
def test_butterfly_apply_is_accurate(dim: int) -> None:
    K: KernelMatrix
    if dim == 1:
        N = 256
        K = kernels.fio1d(1.0)(np.arange(N) / N, np.arange(N, dtype=float) - N // 2)
        rt, ct = order_tree(N, 6), order_tree(N, 6)
    else:
        X: Points = np.random.default_rng(0).random((400, 2))
        K = kernels.helmholtz2d(5.0)(X, X)
        rt = ct = Tree(X, max_leaf=8)
    r = butterfly_storage(K, rt, ct, 1e-8)
    assert r["err"] < 1e-6


def test_hmatrix_error_matches_dense_truncation() -> None:
    X: Points = np.random.default_rng(2).random((200, 2))
    K = kernels.laplace2d(X, X)
    T = Tree(X, max_leaf=16)
    r = hmatrix_storage(K, T, T, Strong(), 1e-4)
    assert 0 < r["err"] < 1e-4 and r["storage"] < 200 * 200
