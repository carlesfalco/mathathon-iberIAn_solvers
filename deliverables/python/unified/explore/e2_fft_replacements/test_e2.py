"""Correctness checks for E2 (run: .venv/bin/python -m pytest -q unified/explore/e2_fft_replacements)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repository root, for `unified`
from unified.explore.e2_fft_replacements.fftswap import (MEDIA, bitrev, permuted_kernel, LowRankPhaseFFT, fft_cost,  # noqa: E402
                                         b1_kernel, exact_apply, LowRank)
from unified.explore.e2_fft_replacements._deps import butterfly_id  # noqa: E402
from unified.core.types import ComplexArray, FrequencyGrid, PositionGrid, Spectrum  # noqa: E402


def _grids(N: int) -> tuple[PositionGrid, FrequencyGrid]:
    return np.arange(N) / N, (np.arange(N) - N // 2).astype(float)


def _g(N: int) -> Spectrum:
    r = np.random.default_rng(1)
    g: ComplexArray = r.standard_normal(N) + 1j * r.standard_normal(N)
    return g


def test_bitrev_is_involution() -> None:
    p = bitrev(64)
    assert sorted(p) == list(range(64)) and np.all(p[p] == np.arange(64))


@pytest.mark.parametrize("N", [64, 256])
def test_bitrev_id_butterfly_is_an_fft(N: int) -> None:
    """t = 0, both orderings bit-reversed: every tile rank 1, nnz = 2N log2 N + 2N, unimodular twiddles."""
    x, xi = _grids(N)
    k, br, u = b1_kernel(0.0, "sine"), bitrev(N), np.arange(N, dtype=float)
    bf = butterfly_id(permuted_kernel(k, x, xi, br, br), u, u, 1e-6)
    assert bf.ranks == [1] * (len(bf.ranks))
    assert bf.nnz == fft_cost(N)
    vals = np.concatenate([G.data for G in bf.G])
    assert np.allclose(np.abs(vals), 1)
    g = _g(N)
    f: ComplexArray = np.empty(N, complex)
    f[br] = bf.apply(g[br])
    j = np.arange(N)
    assert np.allclose(f, (-1.0) ** j * N * np.fft.ifft(g), atol=1e-10 * N)


def test_constant_medium_is_one_fft() -> None:
    N = 256
    lr = LowRankPhaseFFT(N, 3.0, MEDIA["constant"], 1e-6)
    assert lr.R == 1
    x, xi = _grids(N)
    g = _g(N)
    f = exact_apply(b1_kernel(3.0, "constant"), x, xi, g)
    assert np.linalg.norm(lr.apply(g) - f) / np.linalg.norm(f) < 1e-12


@pytest.mark.parametrize("med", ["weak", "sine", "lens"])
def test_lowrank_fft_accuracy_and_rank_growth(med: str) -> None:
    N, eps = 256, 1e-6
    x, xi = _grids(N)
    g = _g(N)
    Rs: list[LowRank] = []
    for t in (0.01, 0.3, 3.0):
        lr = LowRankPhaseFFT(N, t, MEDIA[med], eps)
        f = exact_apply(b1_kernel(t, med), x, xi, g)
        assert np.linalg.norm(lr.apply(g) - f) / np.linalg.norm(f) < 10 * eps
        Rs.append(lr.R)
    assert Rs[0] <= Rs[1] <= Rs[2] and Rs[2] > Rs[0]
