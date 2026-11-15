"""Correctness checks for the E3 compressors (small N, < 30 s)."""
from __future__ import annotations

import numpy as np
from scipy.special import yn_zeros

from unified.core import Tree, interp_decomp
from . import _deps as P
from . import e3_problems as E
from .compressors import RandomizedCompressor
from .harness import case, dense_ref


def test_randomized_id_matches_rank_and_accuracy() -> None:
    """Rank-12 perturbation: the stacked [A_FB; A_BF^T] has rank 24 (12 per direction); sketch ID matches it."""
    rng = np.random.default_rng(3)
    n = 200
    A = np.eye(n) + rng.standard_normal((n, 12)) @ rng.standard_normal((12, n))
    box, far = np.arange(40), np.arange(40, n)
    skel, red, T = RandomizedCompressor(A, 1e-10)(box, far)
    M = np.vstack([A[np.ix_(far, box)], A[np.ix_(box, far)].T])
    assert len(skel) == 24 == len(interp_decomp(M, 1e-10).skel)
    assert np.linalg.norm(M[:, red] - M[:, skel] @ T) < 1e-8 * np.linalg.norm(M)


def test_all_compressors_solve_laplace() -> None:
    prob = P.p1_curve(512)
    xd = dense_ref(prob)
    for rule in ("weak", "strong"):
        for comp in ("exact", "proxy", "randomized", "proxy2"):
            assert case(prob, rule, comp, x_dense=xd)["err"] < 1e-7


def test_randomized_kernel_agnostic_proxy_fails_on_rbf() -> None:
    """Hypothesis check: proxy needs a PDE kernel; randomized does not."""
    prob = E.rbf(1024, 0.2)
    xd = dense_ref(prob)
    assert case(prob, "strong", "randomized", x_dense=xd)["err"] < 1e-7
    assert case(prob, "strong", "proxy", x_dense=xd)["err"] > 1e-6


def test_y0_proxy_fails_at_zero_and_two_rings_fix_it() -> None:
    """Y0 kernel with k R_2 = y_{0,1}: single-ring proxy loses the m = 0 mode; a second ring restores it."""
    X = np.random.default_rng(0).random((1024, 2))
    R2 = 2.5 * Tree(X, max_leaf=32).levels[2][0].half_width
    k = float(yn_zeros(0, 1)[0]) / R2
    prob = E.helm_area(1024, k, "Y0")
    xd = dense_ref(prob)
    assert case(prob, "strong", "proxy", x_dense=xd, k_wave=k)["err"] > 1e-6
    assert case(prob, "strong", "proxy2", x_dense=xd, k_wave=k)["err"] < 1e-8
