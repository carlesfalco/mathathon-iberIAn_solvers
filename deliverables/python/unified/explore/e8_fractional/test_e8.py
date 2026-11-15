"""E8 correctness checks (small N, a few seconds)."""
from __future__ import annotations

import numpy as np
from scipy.special import zeta

from unified.core import Tree
from unified.explore.e8_fractional.fractional import (ball_lattice, dirichlet_beta, exact_u, frac_poisson,
                                                      lattice_zeta, power_kernel, rel_err, riesz_second_kind,
                                                      taylor_coef)
from unified.explore.e8_fractional.harness import case, dense_solve
from unified.explore.e8_fractional.proxies import LiftedProxy


def test_lattice_constants() -> None:
    m = 400
    g = np.arange(-m, m + 1)
    Z = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    r = np.linalg.norm(Z[np.any(Z != 0, axis=1)], axis=1)
    r = r[r <= m]                                                       # disk cutoff
    s = 0.5
    brute = np.sum(r ** (-(2 + 2 * s))) + 2 * np.pi / (2 * s * m ** (2 * s))   # + integral tail outside the disk
    assert abs(brute - lattice_zeta(2, s)) < 1e-5
    assert abs(lattice_zeta(1, s) - 2 * zeta(2.0)) < 1e-14
    assert abs(dirichlet_beta(1.0) - np.pi / 4) < 1e-13 and abs(dirichlet_beta(0.0) - 0.5) < 1e-13
    assert taylor_coef(1, 0.5) == -0.5 and taylor_coef(2, 0.5) < 0


def test_exact_solution_constant() -> None:
    """d = 1, s = 1/2: (-Lap)^{1/2} sqrt(1 - x^2) = 1 (the semicircle), so the prefactor is exactly 1."""
    X = np.array([[0.0], [0.6]])
    assert np.allclose(exact_u(X, 0.5), [1.0, 0.8])


def test_lifted_kernel_restricts_to_plane() -> None:
    X, _ = ball_lattice(2, 64)
    K = power_kernel(2.6, 1.0)
    Xl = np.c_[X, np.zeros(len(X))]
    assert np.array_equal(K(X, X[:5]), K(Xl, X[:5]))


def test_lifted_points_on_hemisphere() -> None:
    X, _ = ball_lattice(2, 256)
    node = Tree(X, max_leaf=32).levels[2][3]
    comp = LiftedProxy(power_kernel(2.0, 1.0), X, 1e-8)
    P = comp.proxy_points(node)
    r = np.linalg.norm(P - np.r_[node.center, 0.0], axis=1)
    assert P.shape[1] == 3 and np.allclose(r, 2.5 * node.half_width) and np.all(P[:, 2] >= 0)


def test_fractional_poisson_converges_to_exact() -> None:
    e = [rel_err(dense_solve(P), P.u_exact) for P in (frac_poisson(1, 0.5, 128), frac_poisson(1, 0.5, 512))
         if P.u_exact is not None]
    assert e[1] < e[0] / 2 and e[1] < 3e-3


def test_matrix_is_spd() -> None:
    for P in (frac_poisson(2, 0.75, 300), riesz_second_kind(2, 0.5, 300)):
        assert np.allclose(P.A, P.A.T) and np.linalg.eigvalsh(P.A)[0] > 0


def test_plane_proxy_fails_and_lifted_proxy_works() -> None:
    """The headline result at small N: strong RS, hypersingular 2D kernel."""
    P = frac_poisson(2, 0.5, 600)
    xd = dense_solve(P)
    assert case(P, "strong", "plane", x_dense=xd, max_leaf=16)["err"] > 1e-5
    assert case(P, "strong", "lifted", x_dense=xd, max_leaf=16)["err"] < 1e-8
    assert case(P, "strong", "exact", x_dense=xd, max_leaf=16)["err"] < 1e-8


def test_equal_rank_schedule_is_reproduced() -> None:
    """Copying the exact run's per-box ranks into another compressor gives identical ranks box by box,
    and the interp_decomp patch is undone afterwards."""
    import unified.core.compress as compress_mod
    from unified.explore.e8_fractional.equalrank import ecase
    P = frac_poisson(2, 0.5, 600)
    xd = dense_solve(P)
    ref, sched = ecase(P, "strong", "exact", xd, max_leaf=16)
    r, sched2 = ecase(P, "strong", "lifted", xd, schedule=sched, max_leaf=16)
    assert sched2 == sched and r["skel_compressed"] == ref["skel_compressed"]
    assert compress_mod.interp_decomp.__name__ == "interp_decomp"
    assert ref["far_err_max"] < 1e-7          # exact ID: per-box far error at the tolerance


def test_xc_proxy_selects_few_points_and_works() -> None:
    from unified.explore.e8_fractional.equalrank import ecase
    P = frac_poisson(2, 0.5, 600)
    xd = dense_solve(P)
    r, _ = ecase(P, "strong", "xc", xd, max_leaf=16)
    assert r["err"] < 1e-7 and 0 < r["n_proxy_max"] < 400 and r["select_flops"] > 0
