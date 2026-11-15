"""Tests for A1 (unified/drivers/rskel.py): one RS factorization, weak and strong admissibility,
exact and proxy compression, checked against dense solves on the spec's P1 and P2 problems."""
import numpy as np
import pytest
from unified.core import Tree, Weak, Strong, ExactCompressor, ProxyCompressor, kernels
from typing import Callable

from unified.core.types import Admissibility, Kernel, Matrix, Points, RealArray, Vector
from unified.drivers.rskel import Factorization, rskel_factor

type TestProblem = tuple[Points, Matrix, Kernel]   # (points X, dense A, kernel reproducing A off the diagonal)

EPS = 1e-8


def starfish_weight(Y: Points, N: int) -> RealArray:
    """Arc-length quadrature weight |gamma'(theta)| 2 pi / N as a function of a point (theta = its angle).
    Used to make the proxy kernel carry the same column weights as A (proxy points just get some weight,
    which only rescales rows of the ID matrix and is harmless)."""
    th = np.arctan2(Y[:, 1], Y[:, 0])
    r, dr = 1 + 0.3 * np.cos(5 * th), -1.5 * np.sin(5 * th)
    w: RealArray = np.sqrt(r ** 2 + dr ** 2) * 2 * np.pi / N
    return w


def p1(N: int) -> TestProblem:
    """P1 curve: starfish, A = I + K * w[None, :]. Nonsymmetric (column weights)."""
    t = 2 * np.pi * np.arange(N) / N
    X = np.c_[(1 + 0.3 * np.cos(5 * t)) * np.cos(t), (1 + 0.3 * np.cos(5 * t)) * np.sin(t)]
    w = starfish_weight(X, N)
    A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
    kernel: Kernel = lambda P, Q: kernels.laplace2d(P, Q) * starfish_weight(np.atleast_2d(Q), N)[None, :]
    return X, A, kernel


def p2(N: int, shift: float = 1.0) -> TestProblem:
    """P2 area: uniform random points in the unit square (seed 0), A = shift * I + K / N."""
    X = np.random.default_rng(0).random((N, 2))
    A = shift * np.eye(N) + kernels.laplace2d(X, X) / N
    return X, A, lambda P, Q: kernels.laplace2d(P, Q) / N


def make_compressor(name: str, X: Points, A: Matrix, kernel: Kernel,
                    eps: float = EPS) -> ExactCompressor | ProxyCompressor:
    return ExactCompressor(A, eps) if name == "exact" else ProxyCompressor(kernel, X, eps, A=A)


def relres(A: Matrix, F: Factorization, seed: int = 1) -> float:
    n: int = A.shape[0]
    b: Vector = np.random.default_rng(seed).standard_normal(n)
    res: float = np.linalg.norm(A @ F.solve(b) - b) / np.linalg.norm(b)
    return res


@pytest.mark.parametrize("N", [512, 4096])
@pytest.mark.parametrize("problem", ["P1", "P2"])
@pytest.mark.parametrize("adm", [Weak(), Strong()], ids=["weak", "strong"])
@pytest.mark.parametrize("comp", ["exact", "proxy"])
def test_residual_against_dense(N: int, problem: str, adm: Admissibility, comp: str) -> None:
    X, A, kernel = (p1 if problem == "P1" else p2)(N)
    F = rskel_factor(A, Tree(X, max_leaf=32), adm, make_compressor(comp, X, A, kernel))
    assert relres(A, F) <= 100 * EPS
    st = F.stats
    assert st["admissibility"] == adm.name and st["root_size"] < N
    assert sum(v["eliminated"] for v in st["levels"].values()) + st["root_size"] == N


def test_solution_matches_dense_solve_and_does_not_touch_A() -> None:
    X, A, kernel = p1(1024)
    A0 = A.copy()
    F = rskel_factor(A, Tree(X, max_leaf=32), Strong(), make_compressor("proxy", X, A, kernel))
    b = np.random.default_rng(3).standard_normal(1024)
    x_ref = np.linalg.solve(A0, b)
    assert np.linalg.norm(F.solve(b) - x_ref) / np.linalg.norm(x_ref) < 100 * EPS
    assert np.array_equal(A, A0)


def test_weak_is_strong_with_empty_near_list() -> None:
    """Weak admissibility = the same code with near_nodes -> []: identical result to a custom rule."""
    class EmptyNear:
        name = "weak"
        def near_nodes(self, tree, node):
            return []
    X, A, kernel = p2(1024)
    tree = Tree(X, max_leaf=32)
    F1 = rskel_factor(A, tree, Weak(), ExactCompressor(A, EPS))
    F2 = rskel_factor(A, tree, EmptyNear(), ExactCompressor(A, EPS))
    b = np.ones(1024)
    assert np.array_equal(F1.solve(b), F2.solve(b))


def test_nonsymmetric_left_and_right_scaling() -> None:
    """A = I + diag(a) K diag(w) with unrelated a, w: row and column far fields differ."""
    N = 1024
    X = np.random.default_rng(5).random((N, 2))
    a: Callable[[Points], RealArray] = lambda Y: 1 + 0.5 * np.sin(7 * Y[:, 0])          # row weight as a function of the point
    w: Callable[[Points], RealArray] = lambda Y: (1 + Y[:, 1] ** 2) / N                  # column weight
    kernel: Kernel = lambda P, Q: a(np.atleast_2d(P))[:, None] * kernels.laplace2d(P, Q) * w(np.atleast_2d(Q))[None, :]
    A = np.eye(N) + kernel(X, X)
    assert np.linalg.norm(A - A.T) > 1e-3 * np.linalg.norm(A)
    for adm in (Weak(), Strong()):
        for comp in (ExactCompressor(A, EPS), ProxyCompressor(kernel, X, EPS, A=A)):
            assert relres(A, rskel_factor(A, Tree(X, max_leaf=32), adm, comp)) <= 100 * EPS


def test_complex_helmholtz_curve() -> None:
    """P3-style complex matrix (k = 5) with both rules."""
    N = 1024
    X, A1, _ = p1(N)
    w = starfish_weight(X, N)
    K = kernels.helmholtz2d(5.0)
    A = np.eye(N) + K(X, X) * w[None, :]
    kernel = lambda P, Q: K(P, Q) * starfish_weight(np.atleast_2d(Q), N)[None, :]
    for adm in (Weak(), Strong()):
        F = rskel_factor(A, Tree(X, max_leaf=32), adm, ProxyCompressor(kernel, X, EPS, A=A))
        assert relres(A, F) <= 100 * EPS


def test_strong_proxy_needs_the_modified_block_set() -> None:
    """The strong-case subtlety: Schur complements modify blocks between two neighbours of a box that are
    far from each other. Proxy compression must include them explicitly (FMM-LU's Q set). On a less
    diagonally dominant P2 (shift 0.01) at eps = 1e-10, dropping them costs orders of magnitude."""
    eps = 1e-10
    X, A, kernel = p2(2048, shift=0.01)
    tree = Tree(X, max_leaf=32)
    with_q = relres(A, rskel_factor(A, tree, Strong(), ProxyCompressor(kernel, X, eps, A=A)))
    without_q = relres(A, rskel_factor(A, tree, Strong(), ProxyCompressor(kernel, X, eps, A=A),
                                       track_modified=False))
    assert with_q <= 100 * eps
    assert without_q > 30 * with_q, (with_q, without_q)


def test_stats_weak_grows_strong_bounded_on_P2() -> None:
    """Area problem: weak skeletons at the top level grow with N, strong ones stay bounded (computed)."""
    tops = {}
    for N in (1024, 4096):
        X, A, kernel = p2(N)
        tree = Tree(X, max_leaf=32)
        for adm in (Weak(), Strong()):
            st = rskel_factor(A, tree, adm, make_compressor("proxy", X, A, kernel)).stats
            # largest skeleton at level 2 (strong compresses there; level 1 has no far field for strong)
            tops[adm.name, N] = st["levels"][2]["skel_max"]
            assert set(st) >= {"levels", "root_size", "factor_time", "n"}
    assert tops["weak", 4096] > 1.5 * tops["weak", 1024]
    assert tops["strong", 4096] < 1.5 * tops["strong", 1024]
    assert tops["strong", 4096] < tops["weak", 4096]


def test_points_not_power_of_two_and_clustered() -> None:
    rng = np.random.default_rng(7)
    X = np.r_[rng.random((700, 2)), 0.5 + 0.01 * rng.standard_normal((300, 2))]  # N = 1000, a cluster
    A = np.eye(1000) + kernels.laplace2d(X, X) / 1000
    for adm in (Weak(), Strong()):
        F = rskel_factor(A, Tree(X, max_leaf=24), adm, ExactCompressor(A, EPS))
        assert relres(A, F) <= 100 * EPS
