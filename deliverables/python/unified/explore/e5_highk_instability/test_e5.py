"""E5 correctness checks (small N, fast)."""
from __future__ import annotations

import numpy as np

from unified.core import Tree, Weak, Strong, interp_decomp
from unified.explore.e5_highk_instability import _deps as P
from unified.explore.e5_highk_instability.congruent import congruent_factor
from unified.explore.e5_highk_instability.diag import redundant_basis, rhs, run_case
from unified.explore.e5_highk_instability.fixes import RealifiedCompressor, far_block, realify


def test_realified_id_is_real_and_accurate() -> None:
    prob = P.p3_helmholtz(512, 40.0)
    box, far = np.arange(40), np.arange(80, 512)
    sk, rd, T = RealifiedCompressor(prob.A, 1e-10)(box, far)
    assert np.isrealobj(T) or np.abs(np.imag(T)).max() == 0
    M = far_block(prob.A, box, far)
    assert np.linalg.norm(M[:, rd] - M[:, sk] @ T) <= 1e-8 * np.linalg.norm(M)


def test_complex_T_breaks_congruence_even_for_identity() -> None:
    """Pi^T Pi = I + T^T T is singular for T = i: the bilinear pivot of W = I vanishes."""
    Pi = redundant_basis(2, np.array([1]), np.array([0]), np.array([[1j]]))
    assert abs((Pi.T @ Pi)[0, 0]) < 1e-15
    assert (Pi.conj().T @ Pi)[0, 0].real >= 1.0       # the Hermitian (congruent) pivot stays >= 1


def test_real_T_pivot_bound() -> None:
    """Herm(Pi^T W Pi) >= lambda_min(Herm W) for real T and accretive W (claim 3)."""
    rng = np.random.default_rng(0)
    n = 30
    W = np.eye(n) + 0.4 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2 * n)
    M = realify(rng.standard_normal((20, n)) + 1j * rng.standard_normal((20, n)))
    sk, rd, T = interp_decomp(M[:, :] @ np.diag(np.linspace(1, 1e-6, n)), 1e-3)
    Pi = redundant_basis(n, sk, rd, T)
    X = Pi.T @ W @ Pi
    lw = np.linalg.eigvalsh((W + W.conj().T) / 2)[0]
    assert lw > 0
    assert np.linalg.eigvalsh((X + X.conj().T) / 2)[0] >= lw - 1e-10


def test_fixes_restore_accuracy_at_high_k() -> None:
    prob = P.p3_helmholtz(1024, 100.0)
    xd = np.linalg.solve(prob.A, rhs(1024))
    bad = run_case(prob, "weak", "exact", 1e-8, 100.0, xd)
    for v in ("realified", "congruent"):
        good = run_case(prob, "weak", v, 1e-8, 100.0, xd)
        assert good["err"] < 1e-9 and good["cond_rr_max"] < 1e3
        assert good["err"] < 1e-3 * bad["err"]


def test_congruent_matches_dense_low_k() -> None:
    prob = P.p3_helmholtz(512, 10.0)
    b = rhs(512)
    for adm in (Weak(), Strong()):
        F = congruent_factor(prob.A, Tree(prob.X, max_leaf=32), adm, 1e-10)
        x = F.solve(b)
        assert np.linalg.norm(prob.A @ x - b) / np.linalg.norm(b) < 1e-8


def test_fixed_rank_reproduces_reference_ranks() -> None:
    from unified.core import ExactCompressor
    from unified.drivers.rskel import rskel_factor
    from unified.explore.e5_highk_instability.review import FixedRankCompressor, RankRecorder
    prob = P.p3_helmholtz(512, 30.0)
    rec = RankRecorder(ExactCompressor(prob.A, 1e-8))
    F0 = rskel_factor(prob.A, Tree(prob.X, max_leaf=32), Weak(), rec)
    comp = FixedRankCompressor("exact", prob.A, prob.kernel, prob.X, rec.ranks)
    F1 = rskel_factor(prob.A, Tree(prob.X, max_leaf=32), Weak(), comp)
    s0 = sum(v["skel_total"] for v in F0.stats["levels"].values())
    s1 = sum(v["skel_total"] for v in F1.stats["levels"].values())
    assert s0 == s1 and max(comp.far_errs) < 1e-7


def test_proxy_hypothesis_containment() -> None:
    from unified.explore.e5_highk_instability.review import proxy_hypothesis
    prob = P.p3_helmholtz(512, 40.0)
    tree = Tree(prob.X, max_leaf=32)
    h = proxy_hypothesis(prob.kernel, prob.X, prob.A, tree, tree.depth, 40.0)
    assert h["contain"] < 1e-6                  # exact far rows lie in the proxy row space
    assert h["conj_ring"] < h["conj_exact"]     # proxy ring rows are (nearly) closed under conjugation
