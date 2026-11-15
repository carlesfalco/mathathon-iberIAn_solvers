"""E12 correctness checks (fast)."""
from __future__ import annotations

import numpy as np
import pytest

from unified.core import Tree
from unified.explore.e12_oseen.directional import Directional
from unified.explore.e12_oseen.harness import area_case, cylinder_case, dense_solve
from unified.explore.e12_oseen.oseen import (ScalarOseen, area_problem, cylinder, lamb_drag, near_constant,
                                             oseen_tensor_blocks, stokeslet_blocks)


def test_scalar_kernel_solves_pde() -> None:
    lam, h = 3.0, 1e-3
    G = ScalarOseen(lam)
    o = np.zeros((1, 2))
    for x0 in ([0.4, 0.3], [-0.5, 0.2], [0.1, -0.7]):
        p = np.array([x0])
        def f(q: np.ndarray) -> float:
            return float(G(q, o)[0, 0])
        e = np.eye(2) * h
        lap = sum(f(p + e[i]) + f(p - e[i]) - 2 * f(p) for i in range(2)) / h ** 2
        gx = (f(p + e[0]) - f(p - e[0])) / (2 * h)
        assert abs(-lap + 2 * lam * gx) < 1e-4 * abs(lap)


def test_scalar_kernel_is_scaled_yukawa() -> None:
    """G(x - y) = e^{lam x1} K0(lam r)/(2 pi) e^{-lam y1}: blocks have the Yukawa block's exact rank."""
    from scipy.special import k0
    rng = np.random.default_rng(0)
    X, Y = rng.random((20, 2)), rng.random((30, 2)) + 2
    lam = 2.0
    r = np.linalg.norm(X[:, None] - Y[None], axis=2)
    ref = np.exp(lam * X[:, :1]) * k0(lam * r) / (2 * np.pi) * np.exp(-lam * Y[:, 0])[None, :]
    assert np.allclose(ScalarOseen(lam)(X, Y), ref, rtol=1e-12)


def test_oseen_tensor_pde() -> None:
    """mu(-Lap + 2k d1) u + grad p = 0 and div u = 0 away from the source, p_j = d_j Phi (Phi = log r / 2 pi)."""
    k, h = 0.7, 1e-3
    def S(x: np.ndarray) -> np.ndarray:
        a, b, c = oseen_tensor_blocks(k, np.array([x[0]]), np.array([x[1]]))
        return np.array([[a[0], b[0]], [b[0], c[0]]])
    x0 = np.array([0.6, -0.4])
    e = np.eye(2) * h
    lap = sum(S(x0 + e[i]) + S(x0 - e[i]) - 2 * S(x0) for i in range(2)) / h ** 2
    d1 = (S(x0 + e[0]) - S(x0 - e[0])) / (2 * h)
    d2 = (S(x0 + e[1]) - S(x0 - e[1])) / (2 * h)
    div = d1[0, :] + d2[1, :]
    r2 = x0 @ x0
    gradp = (np.eye(2) * r2 - 2 * np.outer(x0, x0)) / (2 * np.pi * r2 ** 2)   # d_i d_j Phi
    mom = -lap + 2 * k * d1 + gradp
    assert np.max(np.abs(div)) < 1e-5 and np.max(np.abs(mom)) < 1e-4


def test_near_constant() -> None:
    for k in (0.1, 1.0):
        for th in (0.0, 0.9, 2.5):
            r = 1e-5
            dx, dy = np.array([r * np.cos(th)]), np.array([r * np.sin(th)])
            S = np.array(oseen_tensor_blocks(k, dx, dy)).ravel()
            St = np.array(stokeslet_blocks(dx, dy)).ravel()
            C = near_constant(k)
            assert np.allclose(S - St, [C[0, 0], 0, C[1, 1]], atol=1e-4 * k + 1e-9)


def test_drag_matches_lamb_at_small_ka() -> None:
    c = cylinder(0.01, 128)
    f = dense_solve(c.A, c.rhs)
    assert abs(c.drag(f) / lamb_drag(0.01) - 1) < 5e-4


def test_directional_contains_neighbours() -> None:
    X = np.random.default_rng(0).random((2000, 2))
    t = Tree(X, max_leaf=32)
    node = t.levels[-1][len(t.levels[-1]) // 2]
    nb = set(t.neighbors(node))
    near = set(Directional(2, 2, 1).near_nodes(t, node))
    assert nb <= near and all(abs(q.coord[1] - node.coord[1]) <= 1 for q in near)


@pytest.mark.parametrize("rule,comp", [("strong", "proxy"), ("dir:2,2,1", "ellipse"), ("weak", "exact")])
def test_area_fast_matches_dense(rule: str, comp: str) -> None:
    p = area_problem(1024, 20.0)
    xd = dense_solve(p.A, p.b)
    r = area_case(p, rule, comp, xd, health=True)
    assert r["err"] < 1e-7 and r["pivot_cond"] < 1e4


def test_cylinder_fast_drag() -> None:
    c = cylinder(1.0, 256)
    fd = dense_solve(c.A, c.rhs)
    r = cylinder_case(c, "strong", "proxy", fd)
    assert abs(r["drag"] / c.drag(fd) - 1) < 1e-9 and r["err_mod_null"] < 1e-6
