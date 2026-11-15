"""E11 correctness checks (fast: N <= 1024)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from unified.explore.e11_matern.harness import case, dense_ref, schedule_of, BoxRecord
from unified.explore.e11_matern.matern import (StackedKernel, gp_problem, lift_order, matern_cov, multi_orders,
                                               radial_rK)


def test_matern_special_cases() -> None:
    r = np.linspace(0.01, 2, 50)
    assert np.allclose(matern_cov(0.5, 0.3, r), np.exp(-r / 0.3))
    k = math.sqrt(3) / 0.3
    assert np.allclose(matern_cov(1.5, 0.3, r), (1 + k * r) * np.exp(-k * r))
    assert matern_cov(0.7, 0.2, np.zeros(1))[0] == 1.0


@pytest.mark.parametrize("nu", [0.5, 0.7, 1.0, 1.5, 2.5])
def test_lift_order(nu: float) -> None:
    beta, D = lift_order(nu)
    assert beta == math.ceil(nu + 1 - 1e-12) and D >= 2 and abs(beta - D / 2 - nu) < 1e-12
    assert multi_orders(nu)[0] == nu and len(multi_orders(nu)) == beta


def _L(mu: float, D: float, kappa: float, r: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """(kappa^2 - Delta_D) applied to the radial function r^mu K_mu(kappa r) (finite differences)."""
    def f(s: np.ndarray) -> np.ndarray:
        return radial_rK(mu, kappa, s)
    lap = (f(r + h) - 2 * f(r) + f(r - h)) / h ** 2 + (D - 1) / r * (f(r + h) - f(r - h)) / (2 * h)
    out: np.ndarray = kappa ** 2 * f(r) - lap
    return out


@pytest.mark.parametrize("mu,D", [(0.5, 3.0), (0.5, 2.0), (1.5, 2.0), (0.7, 2.6), (0.7, 2.0)])
def test_family_identity_any_dimension(mu: float, D: float) -> None:
    """(kappa^2 - Delta_D)[r^mu K_mu] = c_D r^{mu-1} K_{mu-1} for r > 0 in EVERY D (only c_D depends on D)."""
    r = np.linspace(0.3, 2.0, 30)
    lhs, rhs = _L(mu, D, 2.0, r), radial_rK(mu - 1, 2.0, r)
    c = np.dot(lhs, rhs) / np.dot(rhs, rhs)
    assert np.max(np.abs(lhs - c * rhs)) < 1e-5 * np.max(np.abs(lhs))


@pytest.mark.parametrize("nu", [0.5, 0.7, 1.0, 1.5, 2.5])
def test_chain_terminates_only_in_lift_dimension(nu: float) -> None:
    """The last kernel of the chain, mu = nu - (beta - 1) = 1 - D/2, is the Green's function of kappa^2 - Delta_D:
    annihilated for r > 0 in D = 2 (beta - nu), and NOT in the plane D = 2 unless nu is an integer. This is why the
    multi-kernel proxy needs the lift for non-integer nu [the termination makes the beta-step uniqueness argument
    work]."""
    beta, D = lift_order(nu)
    mu_last = nu - (beta - 1)
    r = np.linspace(0.3, 2.0, 30)
    scale = np.max(np.abs(radial_rK(mu_last, 2.0, r))) * 4.0
    assert np.max(np.abs(_L(mu_last, D, 2.0, r))) < 1e-5 * scale
    if D != 2.0:
        assert np.max(np.abs(_L(mu_last, 2.0, 2.0, r))) > 1e-2 * scale


def test_stacked_kernel_shapes() -> None:
    k = StackedKernel(1.5, 0.1, multi_orders(1.5))
    P = np.c_[np.random.rand(5, 2), np.ones(5)]
    X = np.random.rand(7, 2)
    assert k(P, X).shape == (15, 7) and k(X, P).shape == (7, 15) and k(X, X).shape == (7, 7)


@pytest.fixture(scope="module")
def prob15():  # type: ignore[no-untyped-def]
    p = gp_problem(1024, 1.5, 0.1, 1e-2)
    return p, dense_ref(p)


def test_lifted_multi_reaches_eps_and_logdet(prob15) -> None:  # type: ignore[no-untyped-def]
    p, ref = prob15
    r = case(p, "strong", "liftM", ref, 1e-8)
    assert r["backward"] < 1e-9 and r["far_err_max"] < 1e-7
    assert r["logdet_abs_err"] < 1e-8 and r["mean_err"] < 1e-7


def test_stock_circle_fails(prob15) -> None:  # type: ignore[no-untyped-def]
    p, ref = prob15
    assert case(p, "strong", "plane1", ref, 1e-8)["far_err_max"] > 1e-4


def test_equal_rank_schedule_is_reproduced(prob15) -> None:  # type: ignore[no-untyped-def]
    p, ref = prob15
    recs: list[BoxRecord] = []
    case(p, "strong", "exact", ref, 1e-8, records_out=recs)
    recs2: list[BoxRecord] = []
    case(p, "strong", "xc", ref, 1e-8, schedule=schedule_of(recs), records_out=recs2)
    assert schedule_of(recs) == schedule_of(recs2)
