"""E9 correctness checks: problem assembly, analytic references, and the harness on LS (small N)."""
from __future__ import annotations

import numpy as np
import pytest

from unified.core import Tree, kernels

from . import ls_problem as L
from . import variants as V


def test_table_assembly_matches_direct_kernel() -> None:
    P = L.build("bump", 7.0, 1.0, 16)
    K = kernels.helmholtz2d(7.0)(P.X, P.X)
    K[np.diag_indices(P.n)] = P.self_term / P.h ** 2
    assert np.abs(np.eye(P.n) + K * P.d[None, :] - P.A).max() < 1e-14


def test_two_sided_kernel_reproduces_offdiagonal_and_is_unscaled_at_proxies() -> None:
    P = L.build("bump", 5.0, 1.0, 16)
    Kt = L.ls_kernel(P, "two_sided")
    off = P.A - np.diag(P.A.diagonal())
    assert np.abs(Kt(P.X, P.X) - off).max() < 1e-14
    Y = np.array([[0.3141, -0.2718], [2.0, 2.0]])            # off-grid points: scale 1
    assert np.allclose(Kt(P.X[:3], Y), kernels.helmholtz2d(5.0)(P.X[:3], Y))


def test_radial_ode_reproduces_mie_for_constant_disk() -> None:
    k, q0, a = 4.0, 2.0, 1.0
    nm = L.nmax_for(k, q0, a)
    bm, _ = L.mie_coeffs(k, q0, a, nm)
    assert L.coeff_error(L.radial_coeffs(k, q0, a, nm, "disk"), bm) < 1e-9
    assert L.unitarity_defect(bm, nm) < 1e-12               # lossless: |s_n| = 1


def test_dense_ls_converges_to_bump_reference() -> None:
    k, q0 = 5.0, 1.0
    nm = L.nmax_for(k, q0, 1.0)
    ref = L.radial_coeffs(k, q0, 1.0, nm, "bump")
    errs = []
    for m in (20, 40):
        P = L.build("bump", k, q0, m)
        errs.append(L.coeff_error(L.discrete_coeffs(P, np.linalg.solve(P.A, P.u_inc()), nm), ref))
    assert errs[1] < 2e-3 and errs[0] / errs[1] > 3.5       # O(h²)


@pytest.mark.parametrize("rule", ["weak", "strong"])
@pytest.mark.parametrize("var", ["exact", "realified", "proxy2"])
def test_rskel_solves_ls_to_eps(rule: V.RuleName, var: V.VariantName) -> None:
    P = L.build("bump", 8.0, 1.0, 40)
    b = P.u_inc()
    x = np.linalg.solve(P.A, b)
    r = V.run(P, Tree(P.X, max_leaf=64), rule, var, 1e-8, b, x)
    assert r.err < 1e-6 and r.relres < 1e-6


def test_realified_keeps_pivots_hermitian_definite() -> None:
    """E5 mechanism carries over: real T => Herm(X_RR) stays positive here (Herm(A) indefinite or not)."""
    P = L.build("bump", 10.0, 1.0, 40)
    b = P.u_inc()
    x = np.linalg.solve(P.A, b)
    tree = Tree(P.X, max_leaf=64)
    assert V.run(P, tree, "weak", "realified", 1e-8, b, x).herm_min > 0


def test_stacked_rank_penalty_only_for_variable_contrast() -> None:
    disk = L.build("disk", 8.0, 1.0, 40)
    bump = L.build("bump", 8.0, 1.0, 40)
    sd = V.stack_ranks(disk, Tree(disk.X, max_leaf=64), "strong", 1e-8, 3)
    sb = V.stack_ranks(bump, Tree(bump.X, max_leaf=64), "strong", 1e-8, 3)
    assert sd.ratio_max < 1.05                              # d constant per box: no penalty
    assert sb.stack > 1.05 * max(sb.col, sb.row)            # d varies: stacked rank exceeds both
