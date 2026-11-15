"""The showcase catalogue: one entry per physics problem, all solved by the SAME unified harness.

Read it top to bottom. Every model has the same parts:

    name            short id used on the command line and in the results table
    physics         one plain sentence
    references      who, when, where (arXiv / DOI), and which known result the check is compared with
    matched         what "matched" means for this model: an exact solution, an invariant, or a published
                    behaviour (never a paper table reproduced number for number)
    kernel          the kernel definition (a short function)
    dials           tree / admissibility / representation / sampling / algebra settings
    params          problem sizes for --quick and for the full run
    build(params)   the matrix (or operator) and everything the check needs
    check(problem)  fast solve or apply, dense reference, and the analytic or invariant comparison
    dial_code       the code that plugs this physics into the harness (kernel, sampler / proxy override,
                    wrappers, forked driver). reproduce.py counts its lines with inspect.getsource.
    physics_code    the problem set-up (formulation, quadrature, geometry, analytic reference). Counted too.

The drivers `unified/drivers/rskel.py` and `unified/drivers/butterfly_id.py` and everything in `unified/core`
are imported unchanged. Physics code comes read-only from the explore/ directories (E2, E5-E10); this file
only adds short kernels for the two baseline curve problems, small wrappers and the checks.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
import scipy.linalg as sla
from scipy.special import hankel1, jv, spherical_jn

from unified.bench import problems as bench
from unified.core import ProxyCompressor, Strong, Tree, Weak, kernels
from unified.core.kernels import ButterflyKernel
from unified.core.types import Admissibility, Compressor, Kernel, KernelMatrix, Matrix, Points, RealArray, Vector
from unified.drivers import butterfly_id as bfly
from unified.drivers.rskel import rskel_factor
from unified.explore.e2_fft_replacements import fftswap
from unified.explore.e5_highk_instability import congruent
from unified.explore.e6_stokes import compressor as e6_comp
from unified.explore.e6_stokes import harness as e6_harness
from unified.explore.e6_stokes import stokes
from unified.explore.e7_yukawa_heat import heat, yukawa
from unified.explore.e8_fractional import fractional, proxies
from unified.explore.e9_lippmann_schwinger import ls_problem as ls
from unified.explore.e9_lippmann_schwinger import variants as ls_variants
from unified.explore.e10_sht_butterfly import sht

type Mode = Literal["quick", "full"]
type Params = Mapping[str, float]
type MatchKind = Literal["exact solution", "invariant", "published behaviour"]
P = TypeVar("P")


# =====================================================================================================
# The record types
# =====================================================================================================
@dataclass(frozen=True)
class Reference:
    """A source in the literature and the specific known result the check is compared with."""
    authors: str
    year: int
    where: str               # arXiv id, DOI, or book
    result: str              # the known result that is matched


@dataclass(frozen=True)
class Dials:
    """The five swappable parts of the harness, as set for one model."""
    tree: str
    admissibility: str
    representation: str
    sampling: str
    algebra: str


@dataclass
class CheckResult:
    """Outcome of one model's check. `passed` needs every comparison inside its stated tolerance.

    ref_fast / ref_dense: the fast and the dense solution measured against the analytic answer or invariant.
    Their difference shows what the fast solver adds on top of the discretization error."""
    metric: str                          # what ref_fast measures
    ref_fast: float
    ref_dense: float | None
    ref_tol: float                       # documented discretization level (or exact invariant tolerance)
    fast_vs_dense: float                 # ||fast - dense|| / ||dense||
    dense_tol: float                     # solver target: 100 * eps (README convention) unless stated
    t_fast: float                        # factor + solve (or build + apply) seconds
    t_dense: float                       # dense LU solve (or dense matvec) seconds
    invariants: dict[str, bool] = field(default_factory=dict)   # extra yes/no conditions that must hold
    extra: dict[str, float] = field(default_factory=dict)       # diagnostics written to results.json
    display: str | None = None           # optional table text for ref_fast (e.g. a rank)

    @property
    def passed(self) -> bool:
        ok = bool(np.isfinite(self.ref_fast) and np.isfinite(self.fast_vs_dense))
        return ok and self.ref_fast <= self.ref_tol and self.fast_vs_dense <= self.dense_tol \
            and all(self.invariants.values())


@dataclass(frozen=True)
class Model(Generic[P]):
    name: str
    physics: str
    references: tuple[Reference, ...]
    matched: MatchKind
    matched_text: str                    # the known result, short (table column)
    kernel: Callable[..., Any]
    dials: Dials
    params: Mapping[Mode, Params]
    build: Callable[[Params], P]
    check: Callable[[P], CheckResult]
    dial_code: tuple[Any, ...]
    physics_code: tuple[Any, ...]
    source: str                          # where the physics lives
    note: str = ""


# =====================================================================================================
# Small shared helpers (measurement only; not counted as dial or physics code)
# =====================================================================================================
def rel(a: Any, b: Any) -> float:
    """Relative 2-norm difference ||a - b|| / ||b||."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def solve_both(A: Matrix, tree: Tree, adm: Admissibility, comp: Compressor, b: Vector
               ) -> tuple[Vector, Vector, float, float]:
    """Fast solve with the unchanged rskel_factor and a dense LU solve: (x_fast, x_dense, t_fast, t_dense)."""
    t0 = time.perf_counter()
    F = rskel_factor(A, tree, adm, comp)
    x: Vector = F.solve(b)
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    xd = dense_solve(A, b)
    return x, xd, t_fast, time.perf_counter() - t0


def dense_solve(A: Matrix, b: Vector) -> Vector:
    """Reference: LU solve of the dense matrix (scipy's stubs are stricter than numpy's dtypes, hence Any)."""
    x: Vector = sla.lu_solve(sla.lu_factor(cast(Any, A)), cast(Any, b))
    return x


def _curve_weights(Y: Points, n: int) -> RealArray:
    """Trapezoid weight |γ'(θ)| 2π/n of starfish points, θ recovered as atan2 (SPEC P1 trick)."""
    w: RealArray = bench._speed(np.arctan2(Y[:, 1], Y[:, 0])) * 2 * np.pi / n
    return w


# Interior targets well inside the starfish (min radius 0.7), so the trapezoid rule is spectrally accurate.
INTERIOR: Points = np.array([[0.3, 0.2], [0.0, 0.5], [-0.4, -0.3], [0.6, 0.0]])
# Point sources outside the starfish (E7's choice): they generate the exact boundary data.
OUTSIDE: Points = np.array([[1.36, 0.0], [0.0, 1.06], [-0.9, -0.9]])


@dataclass
class CurveBVP:
    """A second-kind boundary integral equation A σ = g on the starfish, plus point evaluation."""
    X: Points
    A: Matrix
    kernel: Kernel
    g: Vector
    targets: Points
    u_exact: Vector                       # exact solution at the targets
    evaluate: Callable[[Vector], Vector]  # density -> field at the targets
    eps: float
    h: float = 0.0                        # mean node spacing (arc length / n)
    coarse: CurveBVP | None = None        # the same problem at n/2, for the observed convergence order
    k: float = 0.0                        # wavenumber or screening constant, if any


# =====================================================================================================
# 1. Laplace on a curve (baseline): interior Dirichlet problem by a double layer
# =====================================================================================================
def laplace_dl_kernel(n: int) -> Kernel:
    """Weighted double layer D(x, y) w(y) = (x−y)·n_y / (2π|x−y|²) · w(y) on the starfish.
    Columns off the curve (proxy points used as sources) get the plain monopole −log r / 2π."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        nrm, _, on = yukawa.normals_curvature(Y)
        out: KernelMatrix = stokes.laplace_dl(X, Y, nrm) * _curve_weights(Y, n)[None, :]
        out[:, ~on] = kernels.laplace2d(X, Y[~on])
        return out
    return K


def harmonic_exact(Y: Points) -> RealArray:
    """u = Σ_i −log|y − s_i| / 2π with sources s_i outside the curve: harmonic inside."""
    out: RealArray = kernels.laplace2d(np.atleast_2d(Y), OUTSIDE).sum(axis=1)
    return out


def build_laplace(p: Params) -> CurveBVP:
    """(−½ I + D w) σ = g, diagonal = curvature limit −κ/4π · w (Kress, Linear Integral Equations §6)."""
    n = int(p["n"])
    X, w = bench.starfish(n)
    K = laplace_dl_kernel(n)
    A: Matrix = K(X, X)
    _, curv, _ = yukawa.normals_curvature(X)
    A[np.diag_indices(n)] = -0.5 - curv * w / (4 * np.pi)
    return CurveBVP(X, A, K, harmonic_exact(X), INTERIOR, harmonic_exact(INTERIOR),
                    lambda s: K(INTERIOR, X) @ s, p["eps"])


def observed_order(pb: CurveBVP, err: float) -> float:
    """log2(error at n/2 / error at n) of DENSE solves: the quadrature's convergence order."""
    assert pb.coarse is not None
    c = pb.coarse
    return float(np.log2(rel(c.evaluate(np.linalg.solve(c.A, c.g)), c.u_exact) / err))


def check_curve_bvp(pb: CurveBVP, adm: Admissibility, comp: Compressor, solver_tol: float,
                    ref_tol: float, min_order: float | None = None) -> CheckResult:
    x, xd, tf, td = solve_both(pb.A, Tree(pb.X, max_leaf=32), adm, comp, pb.g)
    res = CheckResult(f"rel. error of u at {len(pb.targets)} interior points", rel(pb.evaluate(x), pb.u_exact),
                      rel(pb.evaluate(xd), pb.u_exact), ref_tol, rel(x, xd), solver_tol, tf, td)
    if min_order is not None:
        q = observed_order(pb, float(res.ref_dense or 0.0))
        res.invariants[f"observed order {q:.2f} >= {min_order} (O(h^3) theory)"] = q >= min_order
        res.extra["observed_order"] = q
    return res


def check_laplace(pb: CurveBVP) -> CheckResult:
    comp = ProxyCompressor(pb.kernel, pb.X, pb.eps, A=pb.A)
    return check_curve_bvp(pb, Weak(), comp, 100 * pb.eps, 1e-9)


LAPLACE = Model(
    name="laplace-curve",
    physics="Steady heat / electrostatic potential inside a starfish-shaped plate with prescribed boundary values.",
    references=(Reference("Kress", 1999, "Linear Integral Equations, 2nd ed., Springer, DOI 10.1007/978-1-4612-0559-3",
                          "double-layer Nyström with the trapezoid rule converges spectrally to the exact harmonic u"),
                Reference("Ho, Greengard", 2012, "arXiv:1110.3105",
                          "recursive skeletonization (weak admissibility) solves it to the ID tolerance")),
    matched="exact solution",
    matched_text="harmonic u from 3 exterior sources (Kress 1999)",
    kernel=laplace_dl_kernel,
    dials=Dials("quadtree, max_leaf 32", "Weak (HBS)", "one ID per box", "core ProxyCompressor (64-point circle)",
                "factor: rskel_factor"),
    params={"quick": {"n": 2048, "eps": 1e-10}, "full": {"n": 8192, "eps": 1e-10}},
    build=build_laplace, check=check_laplace,
    dial_code=(laplace_dl_kernel, stokes.laplace_dl),
    physics_code=(build_laplace, harmonic_exact, _curve_weights, bench.starfish, bench._r, bench._speed,
                  yukawa.normals_curvature, yukawa._gamma_derivs),
    source="unified/bench/problems.py (geometry), E6 laplace_dl, E7 normals",
)


# =====================================================================================================
# 2. Helmholtz on the starfish, with the E5 congruent fix (exterior sound-soft-type Dirichlet problem)
# =====================================================================================================
def helmholtz_dl_kernel(k: float, n: int) -> Kernel:
    """Weighted Helmholtz double layer ∂_{n_y} (i/4) H0(k|x−y|) w(y) = (ik/4) H1(kr) (x−y)·n_y / r · w(y)."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        nrm, _, _ = yukawa.normals_curvature(Y)
        d = X[:, None, :] - Y[None, :, :]
        r = np.linalg.norm(d, axis=2)
        out: KernelMatrix = np.zeros(r.shape, complex)
        nz = r > 0
        out[nz] = 0.25j * k * hankel1(1, k * r[nz]) * np.einsum("ijk,jk->ij", d, nrm)[nz] / r[nz]
        return out * _curve_weights(Y, n)[None, :]
    return K


def radiating_exact(k: float, Y: Points) -> Vector:
    """u = (i/4) H0(k|y − x0|), x0 = (0.1, 0.2) inside the curve: an exact outgoing exterior solution."""
    out: Vector = kernels.helmholtz2d(k)(np.atleast_2d(Y), np.array([[0.1, 0.2]]))[:, 0]
    return out


def build_helmholtz(p: Params) -> CurveBVP:
    """(½ I + D w) σ = g (exterior limit; Colton-Kress DL formulation), targets on the circle |x| = 2.5."""
    n, k = int(p["n"]), p["k"]
    coarse = build_helmholtz({**p, "n": n // 2, "coarse": 0}) if p.get("coarse", 1) else None
    X, w = bench.starfish(n)
    K = helmholtz_dl_kernel(k, n)
    A: Matrix = K(X, X)
    _, curv, _ = yukawa.normals_curvature(X)
    A[np.diag_indices(n)] = 0.5 - curv * w / (4 * np.pi)
    th = 2 * np.pi * np.arange(8) / 8
    T: Points = 2.5 * np.c_[np.cos(th), np.sin(th)]
    return CurveBVP(X, A, K, radiating_exact(k, X), T, radiating_exact(k, T), lambda s: K(T, X) @ s, p["eps"],
                    float(w.sum()) / n, coarse)


def check_helmholtz(pb: CurveBVP) -> CheckResult:
    t0 = time.perf_counter()
    F = congruent.congruent_factor(pb.A, Tree(pb.X, max_leaf=32), Strong(), pb.eps)
    x = F.solve(pb.g)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    xd = dense_solve(pb.A, pb.g)
    td = time.perf_counter() - t0
    err_dense = rel(pb.evaluate(xd), pb.u_exact)
    q = observed_order(pb, err_dense)
    return CheckResult("rel. error of u at 8 exterior points", rel(pb.evaluate(x), pb.u_exact), err_dense, 1e-3,
                       rel(x, xd), 100 * pb.eps, tf, td,
                       invariants={f"observed order {q:.2f} >= 2.5 (O(h^3) theory)": q >= 2.5},
                       extra={"cond_rr_max": F.cond_rr_max(), "observed_order": q})


HELMHOLTZ = Model(
    name="helmholtz-starfish",
    physics="A time-harmonic wave radiating from a source hidden inside a starfish-shaped obstacle.",
    references=(Reference("Colton, Kress", 2013, "Inverse Acoustic and EM Scattering Theory, 3rd ed., DOI 10.1007/978-1-4614-4942-3",
                          "exterior Dirichlet problem by a double layer; exact outgoing field H0(k|x−x0|) as test solution"),
                Reference("Minden, Ho, Damle, Ying", 2017, "arXiv:1609.08130",
                          "strong recursive skeletonization for oscillatory kernels"),
                Reference("E5 (this repo)", 2026, "unified/explore/e5_highk_instability/findings.md",
                          "complex T with a Tᵀ row step loses accuracy at high k; the congruent Tᴴ step fixes it")),
    matched="exact solution",
    matched_text="outgoing H0 field of an interior source (Colton-Kress)",
    kernel=helmholtz_dl_kernel,
    dials=Dials("quadtree, max_leaf 32", "Strong (FMM-LU)", "one ID per box, of [A_FB; A_BFᴴ]",
                "exact far field from the current matrix", "factor: E5 congruent_factor (row step Tᴴ)"),
    params={"quick": {"n": 1024, "k": 40.0, "eps": 1e-8}, "full": {"n": 4096, "k": 160.0, "eps": 1e-8}},
    build=build_helmholtz, check=check_helmholtz,
    dial_code=(helmholtz_dl_kernel, congruent.congruent_factor, congruent.CongruentFactorization, congruent.CStep),
    physics_code=(build_helmholtz, radiating_exact, _curve_weights, bench.starfish, bench._r, bench._speed,
                  yukawa.normals_curvature, yukawa._gamma_derivs),
    source="E5 congruent.py (algebra fix); kernel here",
    note=("The E5 fix is a ~70-line fork of rskel_factor with a Tᴴ row step; it is counted as dial code. "
          "The trapezoid rule on this kernel (≈ r² log r singularity) is O(h³), not spectral. The analytic "
          "tolerance 1e-3 was set AFTER the first run showed 1.4e-4 with dense and fast alike (no prior "
          "fit exists for this kernel); the real test is the observed O(h³) order, which was not tuned."),
)


# =====================================================================================================
# 3. The DFT as a butterfly: bit-reversed ordering turns the ID butterfly into the FFT
# =====================================================================================================
@dataclass
class ButterflyCase:
    N: int
    x: RealArray
    xi: RealArray
    g: Vector
    eps: float
    t: float = 0.0


def dft_kernel() -> ButterflyKernel:
    """K(x, ξ) = exp(2πi x ξ): SPEC B1 at t = 0, a shifted inverse DFT."""
    return kernels.fio1d(0.0)


def build_butterfly_case(p: Params) -> ButterflyCase:
    N = int(p["N"])
    x, xi = bench.b1_grids(N)
    g = np.random.default_rng(0).standard_normal(N) + 1j * np.random.default_rng(1).standard_normal(N)
    return ButterflyCase(N, x, np.asarray(xi, float), g, p["eps"], p.get("t", 0.0))


def check_dft(c: ButterflyCase) -> CheckResult:
    """Butterfly in bit-reversed order (E2): compare with numpy's FFT and count nonzeros."""
    perm = fftswap.bitrev(c.N)
    u = np.arange(c.N, dtype=float)
    t0 = time.perf_counter()
    bf = bfly.butterfly_id(cast(ButterflyKernel, fftswap.permuted_kernel(dft_kernel(), c.x, c.xi, perm, perm)),
                           u, u, c.eps)
    f = np.empty(c.N, complex)
    f[perm] = bf.apply(c.g[perm])
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    fd = dft_kernel()(c.x, c.xi) @ c.g
    td = time.perf_counter() - t0
    exact = (-1.0) ** np.arange(c.N) * c.N * np.fft.ifft(c.g)
    fft_nnz = 2 * c.N * int(np.log2(c.N)) + 2 * c.N
    return CheckResult("rel. error vs (−1)^j N·ifft(g)", rel(f, exact), rel(fd, exact), 1e-12, rel(f, fd),
                       1e-12, tf, td,
                       invariants={"nnz == 2N log2 N + 2N (radix-2 FFT)": bf.nnz == fft_nnz,
                                   "every tile has rank 1": max(bf.ranks) == 1},
                       extra={"nnz": float(bf.nnz), "fft_nnz": float(fft_nnz), "max_rank": float(max(bf.ranks))})


DFT = Model(
    name="dft-as-butterfly",
    physics="The discrete Fourier transform of N samples, computed by the generic ID butterfly.",
    references=(Reference("Cooley, Tukey", 1965, "DOI 10.1090/S0025-5718-1965-0178586-1",
                          "radix-2 FFT: L stages of 2x2 twiddle blocks, 2N log2 N + 2N stored entries"),
                Reference("E2 (this repo)", 2026, "unified/explore/e2_fft_replacements/findings.md",
                          "bit-reversed tiles are rank 1, so the ID butterfly reproduces the FFT exactly")),
    matched="invariant",
    matched_text="FFT identity + Cooley-Tukey entry count",
    kernel=dft_kernel,
    dials=Dials("binary trees over virtual (bit-reversed) indices", "Complementary (butterfly)",
                "one ID per tile", "kernel entries", "apply: butterfly_id"),
    params={"quick": {"N": 1024, "eps": 1e-10}, "full": {"N": 8192, "eps": 1e-10}},
    build=build_butterfly_case, check=check_dft,
    dial_code=(dft_kernel, kernels.fio1d, fftswap.permuted_kernel, fftswap.bitrev),
    physics_code=(build_butterfly_case, bench.b1_grids),
    source="E2 fftswap.py",
)


# =====================================================================================================
# 4. A Fourier integral operator (SPEC B1, t = 1): bounded tile ranks
# =====================================================================================================
def fio_kernel() -> ButterflyKernel:
    """K(x, ξ) = exp(2πi (x ξ + c(x)|ξ|)), c(x) = (2 + sin 2πx)/16: a 1D wave propagator over time t = 1."""
    return kernels.fio1d(1.0)


def check_fio(c: ButterflyCase) -> CheckResult:
    """Butterfly at N and at N/4: the max tile rank must not grow (Candès-Demanet-Ying)."""
    t0 = time.perf_counter()
    bf = bfly.butterfly_id(fio_kernel(), c.x, c.xi, c.eps)
    f = bf.apply(c.g)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    fd = fio_kernel()(c.x, c.xi) @ c.g
    td = time.perf_counter() - t0
    x4, xi4 = bench.b1_grids(c.N // 4)
    r4 = max(bfly.butterfly_id(fio_kernel(), x4, xi4, c.eps).ranks)
    r = max(bf.ranks)
    return CheckResult(f"max tile rank growth, N/4 → N", float(r - r4), None, 2.0, rel(f, fd), 10 * c.eps, tf, td,
                       extra={"rank_N": float(r), "rank_N_over_4": float(r4), "nnz": float(bf.nnz),
                              "dense_entries": float(c.N ** 2)},
                       display=f"rank {r4}→{r}")


FIO = Model(
    name="fio-1d",
    physics="A 1D wave propagator in a variable-speed medium, applied to a random spectrum.",
    references=(Reference("Candès, Demanet, Ying", 2009, "arXiv:0809.0719",
                          "butterfly tiles of a Fourier integral operator have ε-rank bounded independently of N"),
                Reference("O'Neil, Woolfe, Rokhlin", 2010, "Appl. Comput. Harmon. Anal. 28 (2010)",
                          "interpolative (ID) butterfly")),
    matched="published behaviour",
    matched_text="tile rank bounded in N (Candès-Demanet-Ying)",
    kernel=fio_kernel,
    dials=Dials("binary trees over x and ξ", "Complementary (butterfly)", "one ID per tile", "kernel entries",
                "apply: butterfly_id"),
    params={"quick": {"N": 1024, "eps": 1e-8, "t": 1.0}, "full": {"N": 8192, "eps": 1e-8, "t": 1.0}},
    build=build_butterfly_case, check=check_fio,
    dial_code=(fio_kernel, kernels.fio1d, kernels._sine_speed),
    physics_code=(build_butterfly_case, bench.b1_grids),
    source="unified/core/kernels.py (SPEC B1)",
)


# =====================================================================================================
# 5. E6 Stokes: rigid particles in a viscous fluid
# =====================================================================================================
@dataclass
class StokesCase:
    shear: list[tuple[stokes.Geometry, float]]      # free ellipses in shear, with Jeffery's Ω
    circle: tuple[stokes.Geometry, RealArray]        # forced circle, with the exact (U, Ω)
    eps: float


def stokes_kernel(X: Points, Y: Points, NY: Points) -> RealArray:
    """Stresslet double layer (1/π) (r·n_y) r rᵀ/|r|⁴ as interleaved 2x2 blocks (E6)."""
    return stokes.stresslet_dl(X, Y, NY)


def build_stokes(p: Params) -> StokesCase:
    n = int(p["n"])
    shear = [(stokes.ellipse(n, 1.0, b, angle=phi), stokes.jeffery_omega(1.0, b, phi)) for b, phi in ((0.5, 0.7), (0.2, 1.4))]
    F, T = np.array([0.3, -1.0]), 0.7
    return StokesCase(shear, (stokes.ellipse(n, 0.3, 0.3), stokes.circle_mobility(0.3, F, T)), p["eps"])


def check_stokes(c: StokesCase) -> CheckResult:
    """Each case: (geometry, right-hand side, which of (U_x, U_y, Ω) to compare, exact values)."""
    errs: list[float] = []
    out_dense: list[float] = []
    solve: list[float] = []
    tf = td = 0.0
    F, T = np.array([0.3, -1.0]), 0.7
    cases = [(g, stokes.rhs_shear(g), slice(2, 3), np.array([om])) for g, om in c.shear]
    g0, ex0 = c.circle
    cases.append((g0, stokes.rhs_forces(g0, F[None], np.array([T])), slice(0, 3), ex0))
    for g, b, comps, exact in cases:
        A, tree, nc = e6_harness.build(g, "stokes")
        comp = e6_comp.BlockProxyCompressor(g, c.eps, A, tree, ncomp=nc, sources="sl")
        x, xd, a, d = solve_both(A, tree, Weak(), comp, b)
        tf, td = tf + a, td + d
        scale = float(np.abs(exact).max())
        errs.append(float(np.abs(stokes.rigid_motions(g, x)[0][comps] - exact).max()) / scale)
        out_dense.append(float(np.abs(stokes.rigid_motions(g, xd)[0][comps] - exact).max()) / scale)
        solve.append(rel(x, xd))
    return CheckResult("max rel. error of Ω (2 Jeffery ellipses) and (U, Ω) (circle)", max(errs), max(out_dense),
                       1e-9, max(solve), 100 * c.eps, tf, td,
                       extra={"jeffery_b0.5": errs[0], "jeffery_b0.2": errs[1], "circle_mobility": errs[2]})


STOKES = Model(
    name="stokes-particles",
    physics="Rigid particles in a very viscous fluid: how fast a free ellipse spins in shear, how fast a pushed disk moves.",
    references=(Reference("Jeffery", 1922, "DOI 10.1098/rspa.1922.0078",
                          "Ω = −γ (a² sin²φ + b² cos²φ)/(a² + b²) for a free ellipse in simple shear"),
                Reference("Pozrikidis", 1992, "Boundary Integral and Singularity Methods, DOI 10.1017/CBO9780511624124",
                          "translating / rotating circle: U = (½ − log a) F/4π, Ω = T/(4π a²)"),
                Reference("Rachh, Greengard", 2016, "SIAM J. Numer. Anal. 54 (2016)", "completed double-layer formulation in 2D")),
    matched="exact solution",
    matched_text="Jeffery 1922 ellipse rotation + circle mobility",
    kernel=stokes_kernel,
    dials=Dials("quadtree over interleaved dofs (each point twice)", "Weak (HBS)", "one ID per box",
                "E6 BlockProxyCompressor (2x2 kernel, Stokeslet proxy sources, rigid rows)", "factor: rskel_factor"),
    params={"quick": {"n": 512, "eps": 1e-12}, "full": {"n": 2048, "eps": 1e-12}},
    build=build_stokes, check=check_stokes,
    dial_code=(stokes_kernel, stokes.stresslet_dl, stokes.stokeslet, stokes._interleave, e6_comp.BlockProxyCompressor,
               stokes.Geometry.dof_points),
    physics_code=(build_stokes, stokes.ellipse, stokes.Geometry, stokes.rigid_basis, stokes.projector_blocks,
                  stokes.stokes_matrix, stokes.rhs_shear, stokes.rhs_forces, stokes.rigid_motions,
                  stokes.jeffery_omega, stokes.circle_mobility),
    source="unified/explore/e6_stokes",
)


# =====================================================================================================
# 6. E7 Yukawa (screened Poisson): interior Dirichlet problem
# =====================================================================================================
def screened_exact(kappa: float, Y: Points) -> RealArray:
    """u = Σ_i K0(κ|y − s_i|)/2π, sources outside the curve: solves (−Δ + κ²)u = 0 inside."""
    out: RealArray = np.asarray(yukawa.slp_kernel(kappa)(np.atleast_2d(Y), OUTSIDE), float).sum(axis=1)
    return out


def build_yukawa(p: Params) -> CurveBVP:
    """(−½ I + D_κ w) σ = g (E7 dlp_problem), evaluated with the trapezoid rule at interior points."""
    n, kappa = int(p["n"]), p["kappa"]
    P = yukawa.dlp_problem(kappa, n)
    coarse = build_yukawa({**p, "n": n // 2, "coarse": 0}) if p.get("coarse", 1) else None
    _, w = bench.starfish(n)
    return CurveBVP(P.X, P.A, P.kernel, screened_exact(kappa, P.X), INTERIOR, screened_exact(kappa, INTERIOR),
                    lambda s: yukawa.dlp_eval(kappa, n, INTERIOR, s), p["eps"], float(w.sum()) / n, coarse, kappa)


def check_yukawa(pb: CurveBVP) -> CheckResult:
    comp = ProxyCompressor(pb.kernel, pb.X, pb.eps, A=pb.A)
    return check_curve_bvp(pb, Strong(), comp, 100 * pb.eps, 0.02 * pb.k ** 2 * pb.h ** 3, min_order=2.5)


YUKAWA = Model(
    name="yukawa-bvp",
    physics="Screened electrostatics (Debye-Hückel) inside a starfish: −Δu + κ²u = 0 with given boundary values.",
    references=(Reference("Kress", 1999, "Linear Integral Equations, DOI 10.1007/978-1-4612-0559-3",
                          "second-kind double-layer equation; exact u from exterior point sources"),
                Reference("E7 (this repo)", 2026, "unified/explore/e7_yukawa_heat/findings.md",
                          "the kernel swap K0(κr) needs no driver change; proxy columns become monopoles")),
    matched="exact solution",
    matched_text="sum of screened point charges K0(κr)/2π",
    kernel=yukawa.dlp_kernel,
    dials=Dials("quadtree, max_leaf 32", "Strong (FMM-LU)", "one ID per box",
                "core ProxyCompressor; off-curve columns are monopoles (2 lines)", "factor: rskel_factor"),
    params={"quick": {"n": 2048, "kappa": 10.0, "eps": 1e-8}, "full": {"n": 8192, "kappa": 10.0, "eps": 1e-8}},
    build=build_yukawa, check=check_yukawa,
    dial_code=(yukawa.dlp_kernel, yukawa.slp_kernel, yukawa._diff),
    physics_code=(build_yukawa, screened_exact, yukawa.dlp_problem, yukawa.dlp_eval, yukawa.normals_curvature,
                  yukawa._gamma_derivs, bench.starfish, bench._r, bench._speed),
    source="unified/explore/e7_yukawa_heat",
    note=("Analytic tolerance 0.02 κ²h³ is E7's documented fit of the quadrature error (findings.md, P4), fixed "
          "before this run. A first quick run at n = 1024 FAILED the order test (observed 2.3: pre-asymptotic; it is "
          "3.0 from n = 2048 on), so quick mode uses n = 2048; the 2.5 threshold was not changed."),
)


# =====================================================================================================
# 7. E7 heat equation by Rothe's method: every backward-Euler step is a Yukawa solve, factor once
# =====================================================================================================
@dataclass
class HeatCase:
    n: int
    dt: float
    n_steps: int
    eps: float


HEAT_SOURCE = np.array([1.34, 0.0])                      # unit heat source just outside the petal tip (E7)
HEAT_TARGETS: Points = np.array([[1.26, 0.0], [1.2, 0.0], [1.1, 0.0], [0.8, 0.0]])


def build_heat(p: Params) -> HeatCase:
    return HeatCase(int(p["n"]), p["dt"], int(p["steps"]), p["eps"])


def check_heat(c: HeatCase) -> CheckResult:
    """One strong-RS factorization (absolute-tolerance ID + screening truncation) reused for every step."""
    kappa = 1 / np.sqrt(c.dt)
    P = yukawa.dlp_problem(kappa, c.n)
    X, w = bench.starfish(c.n)
    rc = yukawa.cutoff_radius(kappa, c.eps, c.n, float(w.max()), 0.5)
    ex = heat.semidiscrete_exact(kappa, HEAT_SOURCE)
    t0 = time.perf_counter()
    F = rskel_factor(P.A, Tree(X, max_leaf=32), Strong(),
                     yukawa.Truncated(yukawa.AbsTolExact(P.A, 0.5 * c.eps), P.X, rc))
    fast = heat.rothe_heat(kappa, c.n, c.n_steps, ex, HEAT_TARGETS, F.solve)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu = sla.lu_factor(cast(Any, P.A))
    dense = heat.rothe_heat(kappa, c.n, c.n_steps, ex, HEAT_TARGETS, lambda r: sla.lu_solve(lu, r))
    td = time.perf_counter() - t0
    uex = np.array([ex(HEAT_TARGETS, t) for t in fast.t])
    scale = float(np.max(np.abs(uex)))
    return CheckResult(f"max rel. error of u over {c.n_steps} steps × 4 points",
                       float(np.max(np.abs(fast.u - uex))) / scale, float(np.max(np.abs(dense.u - uex))) / scale,
                       1e-5, float(np.max(np.abs(fast.u - dense.u))) / float(np.max(np.abs(dense.u))),
                       100 * c.eps, tf, td, extra={"kappa": float(kappa)})


HEAT = Model(
    name="heat-rothe",
    physics="Heat spreading into a starfish-shaped plate from a source switched on next to its edge.",
    references=(Reference("Chapko, Kress", 1997, "J. Integral Equations Appl., DOI 10.1216/jiea/1181075987",
                          "Rothe's method: backward Euler in time turns each step into a Yukawa boundary integral equation"),
                Reference("E7 (this repo)", 2026, "unified/explore/e7_yukawa_heat/findings.md",
                          "exact backward-Euler solution of a point source: u^n = Σ_p M_p(|x − x0|)")),
    matched="exact solution",
    matched_text="exact backward-Euler point-source solution",
    kernel=yukawa.dlp_kernel,
    dials=Dials("quadtree, max_leaf 32", "Strong (FMM-LU)", "one ID per box, absolute tolerance",
                "exact far field + screening truncation (E7 AbsTolExact, Truncated)",
                "factor once with rskel_factor, solve every time step"),
    params={"quick": {"n": 1024, "dt": 1e-3, "steps": 6, "eps": 1e-8},
            "full": {"n": 1024, "dt": 1e-3, "steps": 16, "eps": 1e-8}},
    build=build_heat, check=check_heat,
    dial_code=(yukawa.dlp_kernel, yukawa._diff, yukawa.slp_kernel, yukawa.AbsTolExact, yukawa.abs_id,
               yukawa.Truncated, yukawa.cutoff_radius),
    physics_code=(build_heat, heat.rothe_heat, heat.history_matrices, heat.semidiscrete_exact, yukawa.monopole_p,
                  yukawa.history_coef, yukawa.dlp_problem, yukawa.normals_curvature, yukawa._gamma_derivs,
                  bench.starfish, bench._r, bench._speed),
    source="unified/explore/e7_yukawa_heat",
)


# =====================================================================================================
# 8. E8 fractional Laplacian: mean exit time of a Lévy flight, lifted proxy
# =====================================================================================================
def build_fractional(p: Params) -> fractional.FracProblem:
    return fractional.frac_poisson(1, p["s"], int(p["n"]))


def check_fractional(pb: fractional.FracProblem) -> CheckResult:
    eps = 1e-8
    comp = proxies.LiftedProxy(pb.kernel, pb.X, eps, A=pb.A)
    x, xd, tf, td = solve_both(pb.A, Tree(pb.X, max_leaf=32), Strong(), comp, pb.b)
    assert pb.u_exact is not None
    return CheckResult("rel. 2-norm error vs Getoor's u", fractional.rel_err(x, pb.u_exact),
                       fractional.rel_err(xd, pb.u_exact), 5e-3, rel(x, xd), 100 * eps, tf, td)


FRACTIONAL = Model(
    name="fractional-exit-time",
    physics="Mean time for a Lévy flight (a random walk with long jumps) to leave the interval (−1, 1).",
    references=(Reference("Getoor", 1961, "Trans. Amer. Math. Soc. 101 (1961) 75-90",
                          "(−Δ)^s u = 1 on the unit ball, u = 0 outside: u = Γ(d/2)/(4^s Γ(1+s) Γ(d/2+s)) (1−|x|²)^s"),
                Reference("Caffarelli, Silvestre", 2007, "arXiv:math/0608640",
                          "extension to one more dimension: why the lifted proxy surface is valid")),
    matched="exact solution",
    matched_text="Getoor 1961 fractional exit time (semicircle at s = ½)",
    kernel=fractional.power_kernel,
    dials=Dials("binary tree, max_leaf 32", "Strong (FMM-LU)", "one ID per box",
                "E8 LiftedProxy: proxy points on a semicircle in R² (one proxy_points override)", "factor: rskel_factor"),
    params={"quick": {"n": 2048, "s": 0.5}, "full": {"n": 8192, "s": 0.5}},
    build=build_fractional, check=check_fractional,
    dial_code=(fractional.power_kernel, proxies.LiftedProxy),
    physics_code=(build_fractional, fractional.frac_poisson, fractional.ball_lattice, fractional.lattice_zeta,
                  fractional.taylor_coef, fractional.neighbour_laplacian, fractional.c_ds, fractional.exact_u),
    source="unified/explore/e8_fractional",
    note="Tolerance 5e-3 is the documented O(h) discretization level of the corrected lattice scheme (E8 R1).",
)


# =====================================================================================================
# 9. E9 Lippmann-Schwinger: a wave scattered by a penetrable disk
# =====================================================================================================
def check_lippmann_schwinger(pb: ls.LSProblem) -> CheckResult:
    eps = 1e-8
    tree = Tree(pb.X, max_leaf=64)
    comp = ProxyCompressor(ls.ls_kernel(pb, "two_sided"), pb.X, eps, A=pb.A,
                           n_proxy=ls_variants.n_proxy_for(pb, tree))
    b = pb.u_inc()
    x, xd, tf, td = solve_both(pb.A, tree, Strong(), comp, b)
    nm = ls.nmax_for(pb.k, pb.q0, pb.a)
    ref = ls.mie_coeffs(pb.k, pb.q0, pb.a, nm)[0]
    bx, bd = ls.discrete_coeffs(pb, x, nm), ls.discrete_coeffs(pb, xd, nm)
    return CheckResult("rel. error of scattering coefficients b_n (far field)", ls.coeff_error(bx, ref),
                       ls.coeff_error(bd, ref), 5e-2, rel(x, xd), 100 * eps, tf, td,
                       extra={"unitarity_defect_fast": ls.unitarity_defect(bx, nm),
                              "unitarity_defect_dense": ls.unitarity_defect(bd, nm), "N": float(pb.n)})


def build_lippmann_schwinger(p: Params) -> ls.LSProblem:
    return ls.build("disk", p["k"], 1.0, int(p["m"]))


LIPPMANN_SCHWINGER = Model(
    name="lippmann-schwinger-disk",
    physics="A plane wave scattered by a glass-like disk whose refractive index squared is 2.",
    references=(Reference("Richmond", 1965, "IEEE Trans. Antennas Propag. 13 (1965)", "pulse-basis LS discretization with the disk self-cell"),
                Reference("Bohren, Huffman", 1983, "Absorption and Scattering of Light by Small Particles, DOI 10.1002/9783527618156",
                          "Bessel-series (Mie for a cylinder) scattering coefficients of a homogeneous disk"),
                Reference("Ho, Ying", 2016, "arXiv:1307.2666", "strong-admissibility skeletonization of volume integral equations")),
    matched="exact solution",
    matched_text="Mie / Bessel series for a disk",
    kernel=ls.ls_kernel,
    dials=Dials("quadtree, max_leaf 64", "Strong (FMM-LU)", "one ID per box",
                "core ProxyCompressor with E9 two-sided kernel (contrast at unknowns only)", "factor: rskel_factor"),
    params={"quick": {"k": 5.0, "m": 36}, "full": {"k": 5.0, "m": 72}},
    build=build_lippmann_schwinger, check=check_lippmann_schwinger,
    dial_code=(ls.ls_kernel, ls_variants.n_proxy_for),
    physics_code=(build_lippmann_schwinger, ls.build, ls.grid_nodes, ls.contrast, ls.self_term, ls.LSProblem,
                  ls.nmax_for, ls.mie_coeffs, ls.discrete_coeffs),
    source="unified/explore/e9_lippmann_schwinger",
    note="Tolerance 5e-2 is the documented O(h) staircase-disk discretization level (E9 r0: 3.6e-2 → 7.7e-3).",
)


# =====================================================================================================
# 10. E10 spherical harmonic transform: the Legendre half as a butterfly
# =====================================================================================================
@dataclass
class SHTCase:
    n_pw: int          # latitudes for the plane-wave check
    n_sphere: int      # latitudes for the full-sphere Parseval check
    eps: float


def sht_kernel(p: sht.SHTProblem) -> ButterflyKernel:
    """K(θ_j, l) = P̃_l^m(cos θ_j), looked up in a table built by the three-term recurrence."""
    return p.kernel()


def plane_wave_exact(k: float, alpha: float, m: int, lmax: int) -> Vector:
    """a_lm = 2 i^l j_l(k) P̃_l^m(cos α): the coefficients of e^{i k k̂·r} on the unit sphere (Rayleigh)."""
    l = np.arange(m, lmax + 1)
    out: Vector = 2 * (1j) ** l * spherical_jn(l, k) * sht.alf_table(np.array([alpha]), m, lmax)[0]
    return out


def build_sht(p: Params) -> SHTCase:
    return SHTCase(int(p["n_pw"]), int(p["n_sphere"]), p["eps"])


def check_sht(c: SHTCase) -> CheckResult:
    """(i) analysis of a plane wave vs Rayleigh's coefficients, 3 orders m; (ii) Parseval on the full sphere."""
    n, k, al = c.n_pw, c.n_pw / 4, 0.7
    th, w = sht.gauss_nodes(n)
    errs, dense_errs, bf_vs_dense, tf, td = [], [], [], 0.0, 0.0
    for m in (0, n // 16, n // 8):
        fm = (1j) ** m * jv(m, k * np.sin(al) * np.sin(th)) * np.exp(1j * k * np.cos(al) * np.cos(th))
        exact = plane_wave_exact(k, al, m, n - 1)
        t0 = time.perf_counter()
        a_bf = sht.build(sht.sht_problem(n, m, clip=1e-16), c.eps).analyze(fm)
        tf += time.perf_counter() - t0
        t0 = time.perf_counter()
        a_dense = sht.alf_table(th, m, n - 1).T @ (w * fm)
        td += time.perf_counter() - t0
        errs.append(rel(a_bf, exact)); dense_errs.append(rel(a_dense, exact)); bf_vs_dense.append(rel(a_bf, a_dense))
    ns, nphi = c.n_sphere, 2 * c.n_sphere
    ths, ws = sht.gauss_nodes(ns)
    t0 = time.perf_counter()
    bfs = {m: sht.build(sht.sht_problem(ns, m, clip=1e-16), c.eps) for m in range(ns)}
    rng = np.random.default_rng(1)
    coef = {m: rng.standard_normal(ns - abs(m)) + 1j * rng.standard_normal(ns - abs(m)) for m in range(-(ns - 1), ns)}
    Fm = np.zeros((ns, nphi), complex)
    for m, a in coef.items():
        Fm[:, m % nphi] = bfs[abs(m)].synth(a)
    f = np.fft.ifft(Fm, axis=1) * nphi
    tf += time.perf_counter() - t0
    energy_field = float(np.sum(ws[:, None] * np.abs(f) ** 2) * 2 * np.pi / nphi)
    energy_coef = 2 * np.pi * sum(float(np.sum(np.abs(a) ** 2)) for a in coef.values())
    parseval = abs(energy_field - energy_coef) / energy_coef
    return CheckResult("max(plane-wave a_lm error, Parseval defect)", max(max(errs), parseval), max(dense_errs),
                       1e-8, max(bf_vs_dense), 100 * c.eps, tf, td,
                       extra={"plane_wave_err": max(errs), "parseval_rel": parseval, "plane_wave_k": float(k)})


SHT = Model(
    name="spherical-harmonics",
    physics="Expanding a field on the sphere (weather, gravity, CMB) in spherical harmonics and back.",
    references=(Reference("Rayleigh / Jacobi-Anger", 1878, "DLMF 10.60.7, https://dlmf.nist.gov/10.60.E7",
                          "plane-wave coefficients a_lm = 2 i^l j_l(k) P̃_l^m(cos α)"),
                Reference("Parseval", 1799, "orthonormality of spherical harmonics",
                          "∫|f|² dΩ = 2π Σ|a_lm|² for the normalisation used here"),
                Reference("Tygert", 2010, "arXiv:0910.5435, J. Comput. Phys. 229 (2010)", "fast SHT via butterfly-compressed Legendre transforms")),
    matched="invariant",
    matched_text="Rayleigh plane-wave expansion + Parseval",
    kernel=sht_kernel,
    dials=Dials("binary trees over θ and l", "Complementary (butterfly)", "one ID per tile",
                "table kernel, entries below 1e-16·max set to 0", "apply: butterfly_id (+ transposed factors)"),
    params={"quick": {"n_pw": 256, "n_sphere": 64, "eps": 1e-10}, "full": {"n_pw": 2048, "n_sphere": 256, "eps": 1e-10}},
    build=build_sht, check=check_sht,
    dial_code=(sht_kernel, sht.SHTProblem, sht.SHTButterfly, sht.build),
    physics_code=(build_sht, plane_wave_exact, sht.alf_table, sht.gauss_nodes, sht._legendre_pn, sht.sht_problem),
    source="unified/explore/e10_sht_butterfly",
)


# =====================================================================================================
# The registry
# =====================================================================================================
MODELS: tuple[Model[Any], ...] = (LAPLACE, HELMHOLTZ, DFT, FIO, STOKES, YUKAWA, HEAT, FRACTIONAL,
                                  LIPPMANN_SCHWINGER, SHT)
BY_NAME: dict[str, Model[Any]] = {m.name: m for m in MODELS}
