"""Correctness checks for E10 (run: .venv/bin/python -m pytest -q unified/explore/e10_sht_butterfly)."""
from __future__ import annotations

import numpy as np
from scipy.special import factorial, jv, lpmv, spherical_jn

from unified.explore.e10_sht_butterfly.sht import (alf_table, bitrev, build, cheb_problem, gauss_nodes,
                                                   sht_problem, tile_ranks, turning_class, _legendre_pn)


def test_alf_matches_scipy_small_degree() -> None:
    th, _ = gauss_nodes(40)
    for m in (0, 2, 7):
        A = alf_table(th, m, 30)
        ref = np.stack([lpmv(m, l, np.cos(th)) * np.sqrt((2 * l + 1) / 2 * factorial(l - m) / factorial(l + m))
                        * (-1) ** m for l in range(m, 31)], 1)
        assert np.abs(A - ref).max() < 1e-12


def test_alf_survives_underflow_of_start_value() -> None:
    """P̃_m^m = c sin^m θ underflows for m = 1500 at θ = 0.3, but P̃_l^m is O(1) past the turning point
    l ≈ m / sin θ ≈ 5079; orthonormality at n = 6000 would fail without the rescaling."""
    th, w = gauss_nodes(512)
    A = alf_table(th, 200, 511)
    assert np.abs(A.T @ (w[:, None] * A) - np.eye(312)).max() < 1e-12
    v = alf_table(np.array([0.3]), 1500, 6000)[0]
    assert np.isfinite(v).all() and np.abs(v).max() > 0.1 and v[0] == 0.0


def test_gauss_quadrature_orthonormal() -> None:
    for n, m in ((128, 0), (128, 40), (256, 255)):
        p = sht_problem(n, m)
        assert np.abs(p.A.T @ (p.w[:, None] * p.A) - np.eye(n - m)).max() < 1e-12


def test_newton_derivative_sign_improves_a_root() -> None:
    theta, _ = gauss_nodes(128)
    n, k = 128, 64
    theta0 = theta[k] + 1e-5
    x0 = np.cos(theta0)
    pn0, pm0 = _legendre_pn(x0, n)
    dpdth0 = n * (x0 * pn0 - pm0) / np.sin(theta0)
    theta1 = theta0 - pn0 / dpdth0
    pn1, _ = _legendre_pn(np.cos(theta1), n)
    assert abs(pn1) < abs(pn0)


def test_butterfly_roundtrip_legendre() -> None:
    rng = np.random.default_rng(0)
    for m in (0, 37):
        p = sht_problem(256, m, clip=1e-16)
        B = build(p, 1e-10)
        a = rng.standard_normal(256 - m)
        f = B.synth(a)
        assert np.linalg.norm(f - p.A @ a) < 1e-9 * np.linalg.norm(p.A @ a)
        assert np.linalg.norm(B.analyze(f) - a) < 1e-9 * np.linalg.norm(a)


def test_plane_wave_analytic() -> None:
    """Analysis of the m-th Fourier mode of a plane wave gives 2 i^l j_l(k) P̃_l^m(cos α)."""
    n, k, al, m = 128, 30.0, 0.7, 5
    th, _ = gauss_nodes(n)
    B = build(sht_problem(n, m, clip=1e-16), 1e-10)
    fm = (1j) ** m * jv(m, k * np.sin(al) * np.sin(th)) * np.exp(1j * k * np.cos(al) * np.cos(th))
    l = np.arange(m, n)
    exact = 2 * (1j) ** l * spherical_jn(l, k) * alf_table(np.array([al]), m, n - 1)[0]
    assert np.linalg.norm(B.analyze(fm) - exact) < 1e-9 * np.linalg.norm(exact)


def test_chebyshev_bitrev_rank_four() -> None:
    """Proved in findings.md: bit-reversed DCT-II tiles have rank <= 4 (rank 2 per exponential, x2 for Re)."""
    p = cheb_problem(256, "bitrev")
    B = build(p, 1e-12)
    assert max(B.bf.ranks) <= 4
    assert sorted(bitrev(16).tolist()) == list(range(16)) and bitrev(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_subnormal_tiles_have_a_safe_clipped_path() -> None:
    """The clipped path stays accurate on this large-subnormal tile.

    The raw path can return either finite zeros or NaN depending on the BLAS/NumPy build.
    Test the safety guard itself, never that one platform reproduces the NaN.
    """
    a = np.random.default_rng(0).standard_normal(64)
    _raw = build(sht_problem(512, 448), 1e-10).synth(a)
    fixed = sht_problem(512, 448, clip=1e-16)
    good = build(fixed, 1e-10).synth(a)
    assert np.isfinite(good).all()
    assert np.linalg.norm(good - fixed.A @ a) < 1e-9 * np.linalg.norm(fixed.A @ a)


def test_turning_class_and_tile_ranks() -> None:
    p = sht_problem(128, 32, clip=1e-16)
    c = turning_class(p.theta, p.ls, p.m)
    assert c[0, 0] == 0 and c[64, -1] == 1                      # pole, low l: evanescent; equator, high l: osc
    tr = tile_ranks(p, 3, 1e-8)
    assert len(tr) > 0 and all(t["rank"] <= t["size"] for t in tr)


def test_absolute_tolerance_wrapper_restores_driver_and_handles_subnormals() -> None:
    import importlib
    from typing import Any
    drv: Any = importlib.import_module("unified.drivers.butterfly_id")
    from unified.explore.e10_sht_butterfly.sht import absolute_tolerance
    orig = drv.interp_decomp
    p = sht_problem(512, 448)                                   # raw kernel: NaN with the relative ID
    with absolute_tolerance(float(np.abs(p.A).max())):
        B = build(p, 1e-10)
    assert drv.interp_decomp is orig
    a = np.random.default_rng(1).standard_normal(64)
    assert np.linalg.norm(B.synth(a) - p.A @ a) < 1e-9 * np.linalg.norm(p.A @ a)
