"""Tests for deliverable A2: the 1D interpolative (ID) butterfly on spec problem B1."""
from functools import lru_cache

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from unified.core import Complementary
from unified.core.types import Field, Spectrum
from unified.drivers.butterfly_id import (IDButterfly, b1_problem, build_trees, butterfly_id, exact_apply,
                                          load_svd_reference)

EPS = 1e-8
SIZES = [256, 1024, 4096]


@lru_cache(maxsize=None)
def built(N: int, t: float, eps: float = EPS) -> IDButterfly:
    """Build each (N, t, eps) butterfly once per test session."""
    kernel, x, xi = b1_problem(N, t)
    return butterfly_id(kernel, x, xi, eps)


def rand_vec(N: int, seed: int = 0) -> Spectrum:
    rng = np.random.default_rng(seed)
    g: Spectrum = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return g


def rel_err(f: Field, f_ref: Field) -> float:
    err: float = np.linalg.norm(f - f_ref) / np.linalg.norm(f_ref)
    return err


def test_trees_and_complementary_tiles_cover_phase_space() -> None:
    """Every stage has N tiles of N entries each; tiles at a stage partition the N x N matrix."""
    N = 64
    _, x, xi = b1_problem(N, 1)
    rt, ct = build_trees(x, xi)
    assert rt.depth == ct.depth == 6
    for l in range(rt.depth + 1):
        tiles = Complementary().pairs(rt, ct, l)
        assert len(tiles) == N and all(len(A.idx) * len(B.idx) == N for A, B in tiles)
        cover = np.zeros((N, N), int)
        for A, B in tiles:
            cover[np.ix_(A.idx, B.idx)] += 1
        assert (cover == 1).all()


@pytest.mark.parametrize("N", SIZES)
def test_t0_matches_numpy_fft(N: int) -> None:
    """t = 0: K is a shifted inverse DFT, K g = (-1)^j N ifft(g). Acceptance: error <= 10 eps."""
    g = rand_vec(N)
    f_fft = (-1.0) ** np.arange(N) * N * np.fft.ifft(g)
    assert rel_err(built(N, 0).apply(g), f_fft) <= 10 * EPS


@pytest.mark.parametrize("t", [1, 3])
@pytest.mark.parametrize("N", SIZES)
def test_fio_matches_exact_product(N: int, t: float) -> None:
    """t = 1, 3 (Fourier integral operator): compare with K g evaluated from kernel entries."""
    kernel, x, xi = b1_problem(N, t)
    g = rand_vec(N, seed=1)
    assert rel_err(built(N, t).apply(g), exact_apply(kernel, x, xi, g)) <= 10 * EPS


def test_explicit_sparse_factors_multiply_to_K() -> None:
    """Factors are scipy.sparse; U G_{L-1} ... G_0 W reproduces K entrywise (N = 256), and apply() takes blocks."""
    N = 256
    kernel, x, xi = b1_problem(N, 1)
    bf = built(N, 1)
    assert len(bf.G) == bf.U.shape[0].bit_length() - 1 == 8
    for F in [bf.W, bf.U, *bf.G]:
        assert sp.issparse(F)
    Kapprox = bf.U
    for Gl in reversed(bf.G):
        Kapprox = Kapprox @ Gl
    Kapprox = (Kapprox @ bf.W).toarray()
    K = kernel(x, xi)
    assert np.linalg.norm(Kapprox - K, 2) <= 10 * EPS * np.linalg.norm(K, 2)
    G = np.random.default_rng(2).standard_normal((N, 3))
    assert np.allclose(bf.apply(G), Kapprox @ G)


def test_uses_interp_decomp_not_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole repo shares ONE compression primitive: building the butterfly must not call any SVD."""
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("SVD called")
    monkeypatch.setattr(np.linalg, "svd", boom)
    monkeypatch.setattr(scipy.linalg, "svd", boom)
    kernel, x, xi = b1_problem(128, 3)
    butterfly_id(kernel, x, xi, EPS)


def test_ranks_bounded_and_storage_n_log_n() -> None:
    """Complementary low-rank property [computed]: max skeleton size does not grow with N, and every
    sparse factor stores O(N) numbers (L + 2 factors => O(r N log N) storage)."""
    for t in (0, 1, 3):
        r = [max(built(N, t).ranks) for N in SIZES]
        assert max(r) <= min(r) + 2, (t, r)
        per = [max(F.nnz for F in [built(N, t).W, built(N, t).U, *built(N, t).G]) / N for N in SIZES]
        assert max(per) <= 1.5 * min(per), (t, per)


@pytest.mark.parametrize("t", [0, 1, 3])
@pytest.mark.parametrize("N", [256, 1024])
def test_storage_and_accuracy_vs_svd_reference(N: int, t: float) -> None:
    """Same eps: the ID butterfly is as accurate as the SVD butterfly in code/butterfly.py and stores
    fewer numbers (each interpolation matrix contains an identity block). [computed]"""
    ref = load_svd_reference()
    kernel, x, xi = b1_problem(N, t)
    g = rand_vec(N, seed=3)
    f = exact_apply(kernel, x, xi, g)
    W, G, Uf, _ = ref.butterfly_factorize(kernel(x, xi), eps=EPS)
    svd_nnz = W.nnz + sum(Gl.nnz for Gl in G) + Uf.nnz
    bf = built(N, t)
    assert rel_err(ref.butterfly_apply(W, G, Uf, g), f) <= 10 * EPS
    assert rel_err(bf.apply(g), f) <= 10 * EPS
    assert bf.nnz < svd_nnz
    print(f"N={N} t={t}: ID nnz {bf.nnz:,} vs SVD nnz {svd_nnz:,} ({bf.nnz / svd_nnz:.2f}x)")


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-10])
def test_eps_sweep(eps: float) -> None:
    """Error tracks the requested ID tolerance: <= 10 eps for every eps."""
    N = 256
    kernel, x, xi = b1_problem(N, 3)
    g = rand_vec(N, seed=4)
    assert rel_err(built(N, 3, eps).apply(g), exact_apply(kernel, x, xi, g)) <= 10 * eps


def test_shallower_trees_leaf_size_4() -> None:
    """Leaves with several points (depth log2 N - 2): W and U become small dense blocks, still accurate."""
    N = 1024
    kernel, x, xi = b1_problem(N, 1)
    rt, ct = build_trees(x, xi, depth=8)
    assert all(len(A.idx) == 4 for A in rt.leaves())
    bf = butterfly_id(kernel, x, xi, EPS, rt, ct)
    assert len(bf.G) == 8
    g = rand_vec(N, seed=5)
    assert rel_err(bf.apply(g), exact_apply(kernel, x, xi, g)) <= 10 * EPS
