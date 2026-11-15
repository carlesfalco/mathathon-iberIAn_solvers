"""Correctness checks for experiment E4 (fast: ~10 s)."""
from __future__ import annotations
import numpy as np

from unified.explore.e4_butterfly_in_solver.arcs import butterfly, compare, far_block, lowrank_id
from unified.explore.e4_butterfly_in_solver.fio_solve import ButterflyOp, fio_kernel, gmres, grids, solve_case


def test_far_block_wavelength_sampling() -> None:
    """10 points per wavelength: k * h = 2 pi / 10 on the parametrised arc."""
    _, s, _, k = far_block("opposite_arcs", 128)
    h = (np.pi / 2) / 128
    assert np.isclose(k * h, 2 * np.pi / 10)


def test_both_representations_reach_target() -> None:
    r = compare("opposite_arcs", 128, target=1e-6)
    assert r["lowrank"]["err"] <= 1e-6 and r["butterfly"]["err"] <= 1e-6
    assert r["lowrank"]["rank"] >= r["svd_rank"] - 1     # ID rank is never much below the optimal SVD rank


def test_lowrank_apply_matches_dense() -> None:
    kern, s_t, s_s, _ = far_block("parallel_segments", 64)
    lr = lowrank_id(kern, s_t, s_s, 1e-10)
    bf = butterfly(kern, s_t, s_s, 1e-10)
    g = np.random.default_rng(0).standard_normal(64)
    f = kern(s_t, s_s) @ g
    for rep in (lr, bf):
        assert np.linalg.norm(rep["apply"](g) - f) <= 1e-7 * np.linalg.norm(f)


def test_butterfly_adjoint_is_conjugate_transpose() -> None:
    op = ButterflyOp(128, 1.0, eps=1e-10)
    K = fio_kernel(128, 1.0)(*grids(128))
    f = np.random.default_rng(1).standard_normal(128) + 0j
    assert np.linalg.norm(op.rmatvec(f) - K.conj().T @ f) <= 1e-8 * np.linalg.norm(K.conj().T @ f)


def test_gmres_solves_dense_system() -> None:
    rng = np.random.default_rng(2)
    A = np.eye(40) + 0.1 * rng.standard_normal((40, 40))
    b = rng.standard_normal(40) + 0j
    x, hist = gmres(lambda v: np.asarray(A @ v), b, tol=1e-12, maxiter=40)
    assert hist[-1] <= 1e-12 and np.allclose(A @ x, b)


def test_adjoint_preconditioner_exact_at_t0() -> None:
    """At t = 0 K / sqrt(N) is unitary, so M = K^* / N is the exact inverse: one GMRES step."""
    r = solve_case(ButterflyOp(256, 0.0), "adjoint")
    assert r["iters"] == 1 and r["sol_err"] < 1e-8


def test_jacobian_preconditioner_beats_adjoint_before_caustic() -> None:
    op = ButterflyOp(256, 1.0)
    it = {k: solve_case(op, k, maxiter=100)["iters"] for k in ("none", "adjoint", "jacobian")}
    assert it["jacobian"] < it["adjoint"] < it["none"]
    assert it["jacobian"] <= 4
