"""Correctness checks for E7 (Yukawa kernel, double-layer BVP, compressor wrappers, Rothe heat solver)."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.special import k0

from unified.bench.problems import starfish
from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak
from unified.core.types import Admissibility, Compressor
from unified.drivers.rskel import rskel_factor

from .heat import continuous_exact, rothe_heat, semidiscrete_exact
from .yukawa import (AbsTolExact, CenteredProxy, NearTruncated, Truncated, abs_id, cutoff_radius, dlp_eval, dlp_kernel, dlp_problem,
                     monopole_p, slp_kernel)

SRC = np.array([[1.36, 0.0], [0.0, 1.06]])
TGT = np.array([[1.2, 0.0], [0.0, 0.9]])


def test_slp_kernel_matches_k0() -> None:
    X = np.array([[0.0, 0.0]]); Y = np.array([[0.3, 0.4]])
    assert slp_kernel(2.0)(X, Y)[0, 0] == pytest.approx(k0(1.0) / (2 * np.pi), rel=1e-13)


def test_dlp_gauss_law_laplace_limit() -> None:
    """κ → 0: interior limit of D[1] is −1 (Gauss), so (−½I + D w) 1 = −1 on Γ."""
    P = dlp_problem(1e-6, 512)
    assert np.allclose(P.A @ np.ones(512), -1.0, atol=1e-8)


def test_bvp_analytic() -> None:
    kappa, n = 10.0, 1024
    P = dlp_problem(kappa, n)
    g = slp_kernel(kappa)(P.X, SRC).sum(1)
    u = dlp_eval(kappa, n, TGT, np.linalg.solve(P.A, g))
    uex = slp_kernel(kappa)(TGT, SRC).sum(1)
    assert np.max(np.abs(u - uex)) / np.max(np.abs(uex)) < 1e-5


def test_proxy_kernel_uses_monopoles_off_curve() -> None:
    X, _ = starfish(64)
    off = np.array([[3.0, 0.1]])
    assert np.allclose(dlp_kernel(5.0, 64)(X[:3], off), slp_kernel(5.0)(X[:3], off))


@pytest.mark.parametrize("kappa", [1.0, 30.0, 300.0])
def test_fast_solvers_match_dense(kappa: float) -> None:
    n = 1024
    P = dlp_problem(kappa, n)
    tree = Tree(P.X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    xd = np.linalg.solve(P.A, b)
    _, w = starfish(n)
    rc = cutoff_radius(kappa, 1e-8, n, float(w.max()))
    cases: list[tuple[Admissibility, Compressor]] = [(Weak(), ExactCompressor(P.A, 1e-8)), (Strong(), ProxyCompressor(P.kernel, P.X, 1e-8)),
                      (Strong(), AbsTolExact(P.A, 5e-9)), (Strong(), Truncated(AbsTolExact(P.A, 5e-9), P.X, rc)),
                      (Weak(), Truncated(ProxyCompressor(P.kernel, P.X, 1e-8), P.X, rc))]
    for adm, comp in cases:
        x = rskel_factor(P.A, tree, adm, comp).solve(b)
        assert np.linalg.norm(x - xd) / np.linalg.norm(xd) < 1e-7


def test_abs_id_rank_zero_below_tau() -> None:
    M = 1e-12 * np.random.default_rng(0).standard_normal((20, 5))
    r = abs_id(M, 1e-9)
    assert r.skel.size == 0 and r.red.size == 5


def test_relative_id_breaks_on_subnormal_far_field_abs_does_not() -> None:
    """Negative result: at κ = 3000 the strong far blocks are subnormal and the relative ID returns a
    non-finite T (scipy raises in lu_factor); the absolute tolerance gives rank 0 and stays exact."""
    n = 1024
    P = dlp_problem(3000.0, n)
    tree = Tree(P.X, max_leaf=32)
    with pytest.raises(ValueError), np.errstate(all="ignore"):
        rskel_factor(P.A, tree, Strong(), ProxyCompressor(P.kernel, P.X, 1e-8))
    b = np.ones(n)
    x = rskel_factor(P.A, tree, Strong(), AbsTolExact(P.A, 5e-9)).solve(b)
    assert np.linalg.norm(x - np.linalg.solve(P.A, b)) < 1e-10


def test_monopole_recursion() -> None:
    """(I − dtΔ) M_p = M_{p−1} away from the source, checked with a 5-point Laplacian."""
    kappa, p, hh = 7.0, 3, 1e-3
    dt = 1 / kappa ** 2
    x = np.array([0.31, 0.12])
    def M(q: int, y: np.ndarray) -> float:
        return float(monopole_p(kappa, q, np.atleast_1d(np.linalg.norm(y)))[0])
    lap = (sum(M(p, x + d) for d in hh * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])) - 4 * M(p, x)) / hh ** 2
    assert M(p, x) - dt * lap == pytest.approx(M(p - 1, x), rel=1e-5)


def test_rothe_heat_semidiscrete_exact_and_continuous() -> None:
    n, kappa, steps = 512, 10.0, 6
    P = dlp_problem(kappa, n)
    lu = sla.lu_factor(cast(Any, P.A))
    x0 = np.array([1.34, 0.0]); tg = np.array([[1.2, 0.0], [1.1, 0.0]])
    ex = semidiscrete_exact(kappa, x0)
    run = rothe_heat(kappa, n, steps, ex, tg, lambda r: sla.lu_solve(lu, r))
    uex = np.array([ex(tg, t) for t in run.t])
    assert np.max(np.abs(run.u - uex)) / np.max(np.abs(uex)) < 1e-4
    c = continuous_exact(x0)(tg, run.t[-1])                  # backward Euler is only O(dt) accurate
    assert np.max(np.abs(run.u[-1] - c)) / np.max(np.abs(c)) < 0.2


def test_centred_proxy_fixes_small_kappa() -> None:
    """κ → 0: plain monopole proxy loses ~log(1/κ) digits in weak RS; the centred ring restores ε."""
    n = 1024
    P = dlp_problem(1e-9, n)
    tree = Tree(P.X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    xd = np.linalg.solve(P.A, b)
    e_plain = np.linalg.norm(rskel_factor(P.A, tree, Weak(), ProxyCompressor(P.kernel, P.X, 1e-8)).solve(b) - xd)
    e_cent = np.linalg.norm(rskel_factor(P.A, tree, Weak(), CenteredProxy(P.kernel, P.X, 1e-8)).solve(b) - xd)
    assert e_cent / np.linalg.norm(xd) < 2e-8 < e_plain / np.linalg.norm(xd)


@pytest.mark.parametrize("frac", [0.0, 0.25])
def test_near_truncation_matches_dense_and_lowers_cost(frac: float) -> None:
    """Strong minus negligible (norm) / low-rank (rank) neighbours: still ε-accurate vs dense, and at large κ
    it demotes neighbours and factors with fewer flops than plain strong."""
    from .run import factor_cost
    n, kappa = 1024, 3000.0
    P = dlp_problem(kappa, n)
    tree = Tree(P.X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    xd = np.linalg.solve(P.A, b)
    nt = NearTruncated(Strong(), P.A, 5e-9, frac)
    F = rskel_factor(P.A, tree, nt, AbsTolExact(P.A, 5e-9))
    assert np.linalg.norm(F.solve(b) - xd) / np.linalg.norm(xd) < 1e-7
    assert nt.dropped > 0
    assert factor_cost(F)[0] < 0.5 * factor_cost(rskel_factor(P.A, tree, Strong(), AbsTolExact(P.A, 5e-9)))[0]


def test_near_truncation_keeps_everything_when_nothing_negligible() -> None:
    """κ = 1: no touching block is below τ in norm, so nt-norm reproduces Strong exactly."""
    n = 512
    P = dlp_problem(1.0, n)
    tree = Tree(P.X, max_leaf=32)
    nt = NearTruncated(Strong(), P.A, 5e-9)
    rskel_factor(P.A, tree, nt, AbsTolExact(P.A, 5e-9))
    assert nt.dropped == 0 and nt.kept > 0
