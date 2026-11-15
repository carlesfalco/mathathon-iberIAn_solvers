"""Black-box tests of deliverable A1 (recursive skeletonization), written from unified/SPEC.md only.

API under test:  rskel_factor(A, tree, admissibility, compressor) -> F,  F.solve(b),  F.stats.
Acceptance (SPEC): ||A x - b|| / ||b|| <= 100 eps on P1 and P2 for N up to 4096 with both admissibility
rules (Weak = HBS, Strong = FMM-LU) and both compressors; nonsymmetric matrices work; the root size
(unknowns left for the final dense LU) grows with N on P2 for Weak and stays bounded for Strong.
Extra edge cases (SPEC A3): clustered points / empty boxes, N not a power of 2, complex matrices,
eps sweep, trivial trees. Set RUN_SLOW=1 to include N = 4096.
"""
import os
import numpy as np
import pytest

from typing import Any, Callable

from unified.core import Tree, Weak, Strong, ExactCompressor, ProxyCompressor
from unified.core.types import Admissibility, Vector
from unified.bench import problems as P
from unified.bench import adapters as ad

pytest.importorskip("unified.drivers.rskel", reason="A1 driver unified/drivers/rskel.py not written yet")

SLOW = os.environ.get("RUN_SLOW") == "1"
RULES: dict[str, Callable[[], Admissibility]] = {"weak": Weak, "strong": Strong}


def compressor(kind: str, prob: P.Problem, eps: float) -> ExactCompressor | ProxyCompressor:
    """Exact = true entries of A; proxy = proxy circle + explicit far unknowns inside it (from A)."""
    if kind == "exact":
        return ExactCompressor(prob.A, eps)
    return ProxyCompressor(prob.kernel, prob.X, eps, A=prob.A)


def factor_and_check(prob: P.Problem, rule: str, comp: str, eps: float, tree: Tree | None = None, seed: int = 0,
                     factor: float = 100) -> Any:   # the driver's Factorization, reached through the SPEC adapter
    """Factor, solve one random right-hand side and assert the SPEC residual bound. Returns F."""
    tree = tree or Tree(prob.X, max_leaf=32)
    F = ad.rskel().rskel_factor(prob.A, tree, RULES[rule](), compressor(comp, prob, eps))
    rng = np.random.default_rng(seed)
    b: Vector = rng.standard_normal(len(prob.A))
    if np.iscomplexobj(prob.A):
        b = b + 1j * rng.standard_normal(len(prob.A))
    x = F.solve(b)
    res = P.relres(prob.A, x, b)
    assert x.shape == b.shape, f"{prob.name} {rule}/{comp}: solve returned shape {x.shape}, expected {b.shape}"
    assert res <= factor * eps, f"{prob.name} {rule}/{comp} eps={eps}: residual {res:.2e} > {factor}*eps"
    return F


# ---- SPEC acceptance: P1 and P2, both rules, both compressors ------------------------------------
SIZES = [512, 2048] + ([4096] if SLOW else [])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("rule", ["weak", "strong"])
@pytest.mark.parametrize("comp", ["exact", "proxy"])
def test_p1_curve_residual(n: int, rule: str, comp: str) -> None:
    factor_and_check(P.p1_curve(n), rule, comp, 1e-8)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("rule", ["weak", "strong"])
@pytest.mark.parametrize("comp", ["exact", "proxy"])
def test_p2_area_residual(n: int, rule: str, comp: str) -> None:
    factor_and_check(P.p2_area(n), rule, comp, 1e-8)


# ---- stats ----------------------------------------------------------------------------------------
def test_stats_report_every_spec_field() -> None:
    """SPEC: skeleton sizes per level, root size, eliminated unknowns per level, wall time.
    Every unknown is either eliminated somewhere or left for the root LU, so they must add up to N."""
    prob = P.p2_area(1024)
    for rule in RULES:
        F = factor_and_check(prob, rule, "exact", 1e-8)
        s = F.stats
        assert ad.factor_time(s) > 0
        assert ad.max_skeleton(s) >= 1
        assert 0 < ad.root_size(s) <= 1024
        assert ad.total_eliminated(s) + ad.root_size(s) == 1024, \
            f"{rule}: eliminated {ad.total_eliminated(s)} + root {ad.root_size(s)} != N = 1024"


def test_p2_weak_root_grows_strong_root_bounded() -> None:
    """SPEC claim for an area (volume) problem: weak admissibility compresses against neighbours,
    whose interaction rank grows with the box perimeter, so the root grows with N; strong does not."""
    eps = 1e-8
    root: dict[str, list[int]] = {r: [] for r in RULES}
    sizes = [512, 2048] + ([4096] if SLOW else [])
    for n in sizes:
        prob = P.p2_area(n)
        for r in RULES:
            root[r].append(ad.root_size(factor_and_check(prob, r, "proxy", eps).stats))
    assert root["weak"][-1] >= 1.3 * root["weak"][0], f"weak root sizes {root['weak']} for N={sizes}"
    assert root["strong"][-1] <= 1.5 * root["strong"][0] + 32, f"strong root sizes {root['strong']} for N={sizes}"


# ---- nonsymmetric and complex -----------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_nonsymmetric_row_and_column_scaling(rule: str) -> None:
    """D1 A D2 with different smooth diagonal scalings: rows and columns need different interpolation,
    so a driver that reused T^T for the rows would fail here."""
    prob = P.p2_area(1024)
    X = prob.X
    A = (1 + X[:, 0])[:, None] * prob.A * (2 - X[:, 1])[None, :]
    ns = P.Problem("P2-nonsym", X, A, prob.kernel)
    assert np.linalg.norm(A - A.T) > 0.05 * np.linalg.norm(A - np.diag(np.diag(A)))
    factor_and_check(ns, rule, "exact", 1e-8)


@pytest.mark.parametrize("k", [5, 20])
@pytest.mark.parametrize("rule", ["weak", "strong"])
@pytest.mark.parametrize("comp", ["exact", "proxy"])
def test_p3_helmholtz_complex(k: int, rule: str, comp: str) -> None:
    """Complex matrices: P3 Helmholtz curve problem, b complex."""
    factor_and_check(P.p3_helmholtz(1024, k), rule, comp, 1e-8)


# ---- geometry edge cases ------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["weak", "strong"])
@pytest.mark.parametrize("comp", ["exact", "proxy"])
def test_clustered_points_many_empty_boxes(rule: str, comp: str) -> None:
    prob = P.clustered(1200)
    tree = Tree(prob.X, max_leaf=16)
    empty = sum(4 ** l - len(lvl) for l, lvl in enumerate(tree.levels))
    assert empty > 0
    factor_and_check(prob, rule, comp, 1e-8, tree=tree)


@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_deep_tree_on_curve_has_empty_boxes(rule: str) -> None:
    """A curve fills few boxes of a deep quadtree; leaves hold only a handful of points."""
    prob = P.p1_curve(1000)
    tree = Tree(prob.X, depth=6)
    assert len(tree.leaves()) < 4 ** 6 // 4
    factor_and_check(prob, rule, "proxy", 1e-8, tree=tree)


@pytest.mark.parametrize("n", [777, 1001, 1500])
@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_n_not_power_of_two(n: int, rule: str) -> None:
    factor_and_check(P.p1_curve(n), rule, "exact", 1e-8)
    factor_and_check(P.p2_area(n), rule, "proxy", 1e-8)


@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_single_box_tree_is_dense_lu(rule: str) -> None:
    """depth = 0: nothing to compress, the root holds all N unknowns; must be a plain dense solve."""
    prob = P.p2_area(200)
    F = factor_and_check(prob, rule, "exact", 1e-10, tree=Tree(prob.X, depth=0))
    assert ad.root_size(F.stats) == 200


@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_tiny_problem(rule: str) -> None:
    prob = P.p2_area(7)
    factor_and_check(prob, rule, "exact", 1e-10, tree=Tree(prob.X, depth=2))


# ---- eps sweep ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["weak", "strong"])
def test_eps_sweep_residual_tracks_eps_and_ranks_grow(rule: str) -> None:
    """Residual <= 100 eps for every eps, and tighter eps never needs (much) smaller skeletons."""
    prob = P.p1_curve(1024)
    tree = Tree(prob.X, max_leaf=32)
    roots = []
    for eps in [1e-4, 1e-6, 1e-8, 1e-10]:
        F = factor_and_check(prob, rule, "proxy", eps, tree=tree)
        roots.append(ad.root_size(F.stats) + ad.max_skeleton(F.stats))
    assert all(b >= a - 2 for a, b in zip(roots, roots[1:])), f"root+max skeleton vs eps: {roots}"
    assert roots[-1] > roots[0], f"ranks did not grow as eps decreased: {roots}"


def test_solve_is_reusable_and_linear() -> None:
    """A factorization is computed once and applied to many right-hand sides."""
    prob = P.p1_curve(1024)
    F = ad.rskel().rskel_factor(prob.A, Tree(prob.X), Strong(), ExactCompressor(prob.A, 1e-10))
    rng = np.random.default_rng(5)
    b1, b2 = rng.standard_normal(1024), rng.standard_normal(1024)
    x1, x2, x12 = F.solve(b1), F.solve(b2), F.solve(2 * b1 - 3 * b2)
    assert np.linalg.norm(x12 - (2 * x1 - 3 * x2)) <= 1e-10 * np.linalg.norm(x12)
    assert np.allclose(F.solve(b1), x1)  # solve does not mutate the factorization
