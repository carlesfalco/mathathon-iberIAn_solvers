"""Black-box tests of deliverable A2 (ID butterfly), written from unified/SPEC.md only.

SPEC: a 1D butterfly built from the shared Tree / Complementary pairs / interp_decomp, with explicit
scipy.sparse factors and apply(g). Acceptance: B1 relative error <= 10 eps for N = 256 ... 4096,
t in {0, 1, 3}; t = 0 compared with numpy's FFT; storage (nnz) reported and compared with the SVD
reference `unified/reference/butterfly.py`. Set RUN_SLOW=1 to include N = 4096.
"""
import os
import sys
import pathlib
import numpy as np
import pytest

from unified.core import kernels
from unified.core.types import ComplexArray, Spectrum
from unified.bench import problems as P
from unified.bench import adapters as ad

SLOW = os.environ.get("RUN_SLOW") == "1"
SIZES = [256, 1024] + ([4096] if SLOW else [])


def test_b1_t0_is_shifted_inverse_dft() -> None:
    """Sanity check of the problem itself (no driver): K(t=0) @ g == (-1)^j N ifft(g)."""
    n = 256
    g = np.random.default_rng(0).standard_normal(n)
    assert np.allclose(P.b1_matrix(n, 0.0) @ g, (-1.0) ** np.arange(n) * n * np.fft.ifft(g))


bf_mod = pytest.importorskip("unified.drivers.butterfly_id",
                             reason="A2 driver unified/drivers/butterfly_id.py not written yet")


def build(n: int, t: float, eps: float) -> tuple[ComplexArray, ad.ButterflyLike]:
    x, xi = P.b1_grids(n)
    K = P.b1_matrix(n, t)
    return K, ad.butterfly_build(K, x, xi, eps, kernel=kernels.fio1d(t))


def rand_c(n: int, seed: int) -> Spectrum:
    rng = np.random.default_rng(seed)
    g: Spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return g


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("t", [0.0, 1.0, 3.0])
def test_b1_error_within_10_eps(n: int, t: float) -> None:
    eps = 1e-8
    K, bf = build(n, t, eps)
    g = rand_c(n, 1)
    f = ad.butterfly_apply(bf, g)
    err = np.linalg.norm(f - K @ g) / np.linalg.norm(K @ g)
    assert err <= 10 * eps, f"N={n} t={t} eps={eps}: relative error {err:.2e} > 10 eps"


@pytest.mark.parametrize("n", SIZES)
def test_t0_matches_numpy_fft(n: int) -> None:
    """SPEC: t = 0 compared with numpy FFT (independent of the dense matrix)."""
    eps = 1e-10
    _, bf = build(n, 0.0, eps)
    g = rand_c(n, 2)
    ref = (-1.0) ** np.arange(n) * n * np.fft.ifft(g)
    err = np.linalg.norm(ad.butterfly_apply(bf, g) - ref) / np.linalg.norm(ref)
    assert err <= 10 * eps, f"N={n}: error vs FFT {err:.2e}"


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8, 1e-10])
def test_eps_sweep_error_and_storage(eps: float) -> None:
    """Error tracks eps; storage must not exceed the dense N^2 numbers."""
    n = 512
    K, bf = build(n, 1.0, eps)
    g = rand_c(n, 3)
    err = np.linalg.norm(ad.butterfly_apply(bf, g) - K @ g) / np.linalg.norm(K @ g)
    assert err <= 10 * eps, f"eps={eps}: error {err:.2e}"
    assert ad.butterfly_nnz(bf) < n * n


def test_storage_monotone_in_eps() -> None:
    n = 512
    nnz = [ad.butterfly_nnz(build(n, 1.0, e)[1]) for e in (1e-4, 1e-8, 1e-12)]
    assert nnz[0] <= nnz[1] <= nnz[2], f"nnz vs eps (1e-4, 1e-8, 1e-12): {nnz}"


def test_factors_are_scipy_sparse() -> None:
    import scipy.sparse as sp
    _, bf = build(256, 1.0, 1e-8)
    F = ad.butterfly_factors(bf)
    assert len(F) >= int(np.log2(256)), f"expected >= log2 N sparse factors, found {len(F)}"
    assert all(sp.issparse(M) for M in F)


def test_storage_is_n_log_n() -> None:
    """Quasi-linear storage: going from N to 4N should cost about 4 * log(4N)/log(N), well below 16x."""
    eps = 1e-8
    a = ad.butterfly_nnz(build(256, 1.0, eps)[1])
    b = ad.butterfly_nnz(build(1024, 1.0, eps)[1])
    assert b / a < 8, f"nnz(1024) / nnz(256) = {b / a:.2f} (dense would be 16)"


def test_storage_comparable_to_svd_reference() -> None:
    """Compare with code/butterfly.py (SVD bases) at the same eps. An ID keeps skeleton columns
    rather than an optimal basis, so allow a modest constant factor."""
    from unified.reference import butterfly as ref
    n, eps = 1024, 1e-8
    K, bf = build(n, 1.0, eps)
    W, G, Uf, _ = ref.butterfly_factorize(K, eps=eps)
    nnz_svd = W.nnz + sum(g.nnz for g in G) + Uf.nnz
    nnz_id = ad.butterfly_nnz(bf)
    assert nnz_id <= 3 * nnz_svd, f"ID nnz {nnz_id} vs SVD nnz {nnz_svd}"


def test_apply_is_linear_and_handles_real_input() -> None:
    n = 256
    K, bf = build(n, 1.0, 1e-10)
    rng = np.random.default_rng(4)
    g1, g2 = rng.standard_normal(n), rand_c(n, 5)
    f1, f2 = ad.butterfly_apply(bf, g1), ad.butterfly_apply(bf, g2)
    assert np.allclose(ad.butterfly_apply(bf, 2 * g1 - 1j * g2), 2 * f1 - 1j * f2)
    assert np.linalg.norm(f1 - K @ g1) <= 1e-9 * np.linalg.norm(K @ g1)


def test_point_source_columns() -> None:
    """g = e_j picks out one column of K: every column must be reproduced (no lost skeletons)."""
    n = 256
    K, bf = build(n, 3.0, 1e-10)
    for j in (0, 1, n // 2 - 1, n // 2, n - 1):
        e = np.zeros(n, complex); e[j] = 1
        assert np.linalg.norm(ad.butterfly_apply(bf, e) - K[:, j]) <= 1e-8 * np.linalg.norm(K[:, j]), j
