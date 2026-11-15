"""Swap (a): compression of ONE far block of the 2D Helmholtz kernel at high frequency.

Same block, same accuracy target, two representations:
  * low rank  : one interpolative decomposition (ID) of the whole block, K ~= K[:, skel] @ P
                (what strong-admissibility recursive skeletonization stores; rank grows ~ k * size).
  * butterfly : the 1D interpolative butterfly of unified/drivers/butterfly_id.py, applied to the block
                K[i, j] = H(gamma_t(s_i), gamma_s(sigma_j)) seen as a kernel of the two 1D ARC PARAMETERS
                (complementary tiles over the parametrisations; Michielssen-Boag 1996 MLMDA,
                Liu et al. 2021 "butterfly in H-matrices").                                   [cited]

Points are placed at `ppw` points per wavelength on each piece (default 10), so N grows with k:
N = ppw * k * len / (2 pi). We choose N = 2^L and set k from it.
"""
from __future__ import annotations
from typing import Any, Callable, NotRequired, TypedDict

import numpy as np

from unified.core import interp_decomp, kernels
from unified.core.types import (ComplexArray, IDResult, InterpMatrix, Kernel, KernelMatrix, Matrix, Points,
                                RealArray, RedundantIdx, SkeletonIdx)
from unified.drivers.butterfly_id import IDButterfly, butterfly_id

# ---- domain aliases --------------------------------------------------------------------------------
type Wavenumber = float
type StorageCount = int                                  # numbers stored (identity parts of [I | T] excluded)
type ArcParam = RealArray                                # (N,) curve parameter s in [0, 1)
type Curve = Callable[[ArcParam], Points]                # s -> (N, 2) points on one piece of the boundary
type ParamKernel = Callable[[ArcParam, ArcParam], KernelMatrix]   # K(s_target, s_source)
type ProbeBlock = Matrix                                 # (N, n_probe) block of probe vectors
type BlockApply = Callable[[ProbeBlock], ProbeBlock]     # G -> K~ G for a compressed representation


class Geometry(TypedDict):
    target: Curve
    source: Curve
    length: float                                        # arc length of each piece


class Representation(TypedDict):
    """A compressed far block: its (max) rank, storage, and how to apply it."""
    rank: int
    storage: StorageCount
    apply: BlockApply
    ranks: NotRequired[list[int]]                        # butterfly: max skeleton size per stage
    nnz_raw: NotRequired[int]                            # butterfly: nnz including identity entries


type MatchedReport = dict[str, Any]                      # Representation without `apply`, plus tol and err
type CompareReport = dict[str, Any]

# typed handle on the (possibly untyped) driver entry point; calling it is then a typed call
_butterfly_id: Callable[..., IDButterfly] = butterfly_id


# ---- geometries: each maps a parameter s in [0, 1) to a point on the (target, source) piece ---------
def _arc(theta0: float, dtheta: float) -> Curve:
    def gamma(s: ArcParam) -> Points:
        P: Points = np.c_[np.cos(theta0 + dtheta * s), np.sin(theta0 + dtheta * s)]
        return P
    return gamma


def _seg(p0_: list[float], p1_: list[float]) -> Curve:
    p0, p1 = np.asarray(p0_, float), np.asarray(p1_, float)

    def gamma(s: ArcParam) -> Points:
        P: Points = p0[None, :] + np.asarray(s, float)[:, None] * (p1 - p0)[None, :]
        return P
    return gamma


GEOMETRIES: dict[str, Geometry] = {
    # quarter arcs of the unit circle in opposite quadrants: gap sqrt(2), each arc has diameter sqrt(2)
    "opposite_arcs": Geometry(target=_arc(np.pi, np.pi / 2), source=_arc(0.0, np.pi / 2), length=np.pi / 2),
    # parallel facing unit segments at distance 1 (a typical strong-admissibility far pair)
    "parallel_segments": Geometry(target=_seg([0, 1], [1, 1]), source=_seg([0, 0], [1, 0]), length=1.0),
    # collinear unit segments with gap 1: phase k|x - y| = k (x - y) is separable -> rank O(1) for all k
    "collinear_segments": Geometry(target=_seg([2, 0], [3, 0]), source=_seg([0, 0], [1, 0]), length=1.0),
}


def far_block(geom: str, N: int, ppw: float = 10.0) -> tuple[ParamKernel, ArcParam, ArcParam, Wavenumber]:
    """Return (kernel_of_parameters, s_target, s_source, k) for the far block of `geom` with N points
    per piece at `ppw` points per wavelength."""
    g = GEOMETRIES[geom]
    k: Wavenumber = 2 * np.pi * N / (ppw * g["length"])
    H: Kernel = kernels.helmholtz2d(k)

    def kern(s: ArcParam, sig: ArcParam) -> KernelMatrix:
        return H(g["target"](np.ravel(s)), g["source"](np.ravel(sig)))
    s: ArcParam = (np.arange(N) + 0.5) / N
    return kern, s, s.copy(), k


def _probe_error(apply_approx: BlockApply, kern: ParamKernel, s_t: ArcParam, s_s: ArcParam,
                 n_probe: int = 8, seed: int = 0, chunk: int = 512) -> float:
    """Relative error ||(K - K~) Om||_F / ||K Om||_F with a Gaussian probe Om (N x n_probe): an unbiased
    estimate of the relative Frobenius error."""
    rng = np.random.default_rng(seed)
    Om: ComplexArray = rng.standard_normal((len(s_s), n_probe)) + 1j * rng.standard_normal((len(s_s), n_probe))
    KO: Matrix = np.vstack([kern(s_t[i:i + chunk], s_s) @ Om for i in range(0, len(s_t), chunk)])
    return float(np.linalg.norm(apply_approx(Om) - KO) / np.linalg.norm(KO))


def lowrank_id(kern: ParamKernel, s_t: ArcParam, s_s: ArcParam, tol: float) -> Representation:
    """One column ID of the whole block. Storage = m*r (skeleton columns) + r*(n - r) (T)."""
    K: KernelMatrix = kern(s_t, s_s)
    idr: IDResult = interp_decomp(K, tol)
    skel: SkeletonIdx = idr.skel
    red: RedundantIdx = idr.red
    T: InterpMatrix = idr.T
    m, n, r = K.shape[0], K.shape[1], len(skel)
    Ks: Matrix = K[:, skel]                              # skeleton columns: the stored "basis"

    def apply(G: ProbeBlock) -> ProbeBlock:
        return Ks @ (G[skel] + T @ G[red])
    return Representation(rank=r, storage=int(m * r + r * (n - r)), apply=apply)


def svd_rank(kern: ParamKernel, s_t: ArcParam, s_s: ArcParam, eps: float) -> int:
    """eps-rank of the block in the 2-norm (sigma_r > eps * sigma_1): the best any low-rank format can do."""
    s: RealArray = np.linalg.svd(kern(s_t, s_s), compute_uv=False)
    return int(np.sum(s > eps * s[0]))


def butterfly(kern: ParamKernel, s_t: ArcParam, s_s: ArcParam, tol: float) -> Representation:
    """ID butterfly of the block. Storage counts the nnz of all factors MINUS the identity entries of the
    interpolation matrices P = [I | T] (one per skeleton column in W and every G_l), so that it is
    counted exactly like the low-rank ID (whose identity part is not stored either)."""
    bf = _butterfly_id(kern, s_t, s_s, eps=tol)
    identity: int = bf.W.shape[0] + sum(G.shape[0] for G in bf.G)
    return Representation(rank=max(bf.ranks), ranks=list(bf.ranks), storage=int(bf.nnz - identity),
                          nnz_raw=int(bf.nnz), apply=bf.apply)


TOLS: list[float] = [1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10]

type Builder = Callable[[ParamKernel, ArcParam, ArcParam, float], Representation]


def matched(builder: Builder, kern: ParamKernel, s_t: ArcParam, s_s: ArcParam,
            target: float = 1e-6) -> MatchedReport:
    """Largest ID tolerance in TOLS whose ACHIEVED relative error is <= target (fair comparison:
    the butterfly's per-tile errors accumulate over log2 N stages, the single ID's do not)."""
    for tol in TOLS:
        rep = builder(kern, s_t, s_s, tol)
        err = _probe_error(rep["apply"], kern, s_t, s_s)
        if err <= target:
            break
    out: MatchedReport = {k: v for k, v in rep.items() if k != "apply"}
    out.update(tol=tol, err=err)
    return out


def compare(geom: str, N: int, target: float = 1e-6, ppw: float = 10.0) -> CompareReport:
    """Storage of the far block as matched-accuracy low-rank ID vs butterfly. [computed]"""
    kern, s_t, s_s, k = far_block(geom, N, ppw)
    lr = matched(lowrank_id, kern, s_t, s_s, target)
    bf = matched(butterfly, kern, s_t, s_s, target)
    return dict(geom=geom, N=N, k=float(k), dense=N * N, svd_rank=svd_rank(kern, s_t, s_s, target),
                lowrank=lr, butterfly=bf, ratio_bf_over_lr=bf["storage"] / lr["storage"])
