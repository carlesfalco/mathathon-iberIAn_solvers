"""E3 compression swaps: reproduce every number and figure in findings.md.

    .venv/bin/python -m unified.explore.e3_compression_swaps.run [--quick]

Experiments (all eps = 1e-8, max_leaf = 32, N = 2048 unless stated; one random RHS, error vs dense solve):
  E1 baseline     10 problems x {weak, strong} x {exact, proxy, randomized}
  E2 resonance    area Helmholtz with kernels H0 / Y0 / J0 at k = z / R_l (z a zero of Y_0 or J_0, R_l the
                  level-l proxy radius) times (1 + delta), delta in {0, 1e-8, ..., 1e-2}; proxy vs 2-ring proxy
  E3 k sweep      P3 (starfish, H0) k = 5..100 plus k = j01 / R_l: all compressors, both rules, + worst cond(X_RR)
  E4 non-PDE      Gaussian RBF sigma sweep and Cauchy c sweep: exact / proxy / randomized
  E5 cost vs N    P2 and RBF(0.2), N = 1024, 2048, 4096: flops (matvec-equivalents) and skeleton sizes
  E6 extras       P3 k = 80 eps sweep (exact vs proxy); Y0 kernel on the starfish at y01 / R_l
Outputs: results/e*.json and results/e*.png
"""
from __future__ import annotations
import argparse
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.special import jn_zeros, yn_zeros  # noqa: E402

from . import _deps as P  # noqa: E402
from unified.core import Tree  # noqa: E402
from . import e3_problems as E  # noqa: E402
from .e3_problems import HelmholtzFamily  # noqa: E402
from .harness import case, dense_ref, CaseResult, RuleName, CompKind  # noqa: E402

type Wavenumber = float
type Row = dict[str, Any]

OUT = os.path.join(HERE, "results")
EPS = 1e-8
COMPS: tuple[CompKind, ...] = ("exact", "proxy", "randomized")
RULES: tuple[RuleName, ...] = ("weak", "strong")


def _save(name: str, obj: object) -> None:
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _log(r: CaseResult) -> None:
    print(f"  {r['problem']:34s} {r['rule']:6s} {r['comp']:10s} err={r['err']:.1e} res={r['residual']:.1e} "
          f"skmax={r['skel_max']} mv={r['matvec_equiv']:.1f} condRR={r['cond_rr_max']:.1e}", flush=True)


def run_set(prob: P.Problem, rules: Sequence[RuleName], comps: Sequence[CompKind],
            k_wave: Wavenumber = 0.0, **meta: Any) -> list[Row]:
    """Factor one problem with every (rule, compressor) pair against one dense reference solve."""
    xd = dense_ref(prob)
    rows: list[Row] = []
    for rule in rules:
        for comp in comps:
            r = case(prob, rule, comp, eps=EPS, x_dense=xd, k_wave=k_wave)
            _log(r)
            rows.append({**r, **meta})
    return rows


def proxy_radii(X: np.ndarray) -> dict[int, float]:
    """R_l = radius_factor * half-width of level-l boxes (ProxyCompressor default radius_factor = 2.5)."""
    t = Tree(X, max_leaf=32)
    return {lvl: 2.5 * t.levels[lvl][0].half_width for lvl in range(1, t.depth + 1)}


# ---- E1 -------------------------------------------------------------------------------------------------
def e1(n: int) -> list[Row]:
    print("E1 baseline")
    makers: list[Callable[[], P.Problem]] = [
        lambda: P.p1_curve(n), lambda: P.p2_area(n), lambda: P.p3_helmholtz(n, 20.0),
        lambda: E.double_layer(n), lambda: E.advdiff(n, 20.0),
        lambda: E.helm_area(n, 10.0, "Y0"),
        lambda: E.rbf(n, 0.2), lambda: E.cauchy(n, 0.2), lambda: E.skew_cauchy(n, 0.2)]
    rows = [row for mk in makers for row in run_set(mk(), RULES, COMPS)]
    _save("e1_baseline", rows)
    return rows


# ---- E2 -------------------------------------------------------------------------------------------------
def e2(n: int, deltas: Sequence[float]) -> list[Row]:
    print("E2 proxy-kernel zeros (area Helmholtz)")
    X = np.random.default_rng(0).random((n, 2))
    R = proxy_radii(X)
    targets: list[tuple[HelmholtzFamily, str, int, Wavenumber]] = []
    for lvl in (2, 3):
        targets.append(("Y0", "y01", lvl, float(yn_zeros(0, 1)[0]) / R[lvl]))
        targets.append(("J0", "j01", lvl, float(jn_zeros(0, 1)[0]) / R[lvl]))
        targets.append(("H0", "j01", lvl, float(jn_zeros(0, 1)[0]) / R[lvl]))
    rows: list[Row] = []
    for which, zero, lvl, k0 in targets:
        for d in deltas:
            comps: tuple[CompKind, ...] = ("proxy", "proxy2", "randomized") if d == 0 else ("proxy", "proxy2")
            rows += run_set(E.helm_area(n, k0 * (1 + d), which), RULES, comps, k_wave=k0 * (1 + d),
                            kernel=which, zero=zero, level=lvl, k0=k0, delta=d)
    _save("e2_resonance", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rule in zip(axes, RULES):
        for ti, (which, zero, lvl, k0) in enumerate(targets):
            for comp, ls in (("proxy", "-"), ("proxy2", ":")):
                sel = [r for r in rows if r["kernel"] == which and r["level"] == lvl and r["rule"] == rule
                       and r["comp"] == comp]
                ax.loglog([max(r["delta"], 1e-10) for r in sel], [r["err"] for r in sel], ls, marker="o", ms=3, color=f"C{ti}",
                          label=f"{which} at {zero}/R{lvl} ({comp})" if comp == "proxy" else None)
        ax.axhline(EPS, color="gray", lw=0.8)
        ax.set_title(f"{rule}: k = k*(1+delta); solid 1 ring, dotted 2 rings", fontsize=9)
        ax.set_xlabel("relative detuning delta (0 plotted at 1e-10)")
    axes[0].set_ylabel("||x - x_dense|| / ||x_dense||")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e2_resonance.png"), dpi=130); plt.close(fig)
    return rows


# ---- E3 -------------------------------------------------------------------------------------------------
def e3(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("E3 P3 k sweep")
    X, _ = P.starfish(n)
    R = proxy_radii(X)
    k_res = [float(jn_zeros(0, 1)[0]) / R[lvl] for lvl in (2, 3, 4)]   # interior Dirichlet eigenvalues
    allk = sorted(set(list(ks) + k_res))
    rows: list[Row] = []
    for k in allk:
        rows += run_set(P.p3_helmholtz(n, float(k)), RULES, COMPS + ("proxy2",), k_wave=k, k=k,
                        at_dirichlet_eig=k in k_res)
    _save("e3_ksweep", rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for j, rule in enumerate(RULES):
        for comp in COMPS + ("proxy2",):
            sel = [r for r in rows if r["rule"] == rule and r["comp"] == comp]
            axes[0, j].semilogy([r["k"] for r in sel], [r["err"] for r in sel], marker="o", ms=3, label=comp)
            axes[1, j].semilogy([r["k"] for r in sel], [r["cond_rr_max"] for r in sel], marker="o", ms=3,
                                label=comp)
        for kr in k_res:
            axes[0, j].axvline(kr, color="gray", ls=":", lw=0.8)
        axes[0, j].axhline(EPS, color="gray", lw=0.8)
        axes[0, j].set_title(f"P3 starfish, {rule}: solve error (dotted: J0(kR_l)=0)")
        axes[1, j].set_title("worst cond of eliminated block X_RR")
        axes[1, j].set_xlabel("k")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e3_ksweep.png"), dpi=130); plt.close(fig)
    return rows


# ---- E4 -------------------------------------------------------------------------------------------------
def e4(n: int, sigmas: Sequence[float], cs: Sequence[float]) -> list[Row]:
    print("E4 non-PDE kernels")
    rows: list[Row] = []
    for s in sigmas:
        rows += run_set(E.rbf(n, s), RULES, COMPS, family="rbf", width=s)
    for c in cs:
        rows += run_set(E.cauchy(n, c), RULES, COMPS, family="cauchy", width=c)
    _save("e4_nonpde", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rule in zip(axes, RULES):
        for fam, mk in (("rbf", "o"), ("cauchy", "s")):
            for comp in COMPS:
                sel = [r for r in rows if r["family"] == fam and r["rule"] == rule and r["comp"] == comp]
                ax.loglog([r["width"] for r in sel], [r["err"] for r in sel], marker=mk, ms=4, label=f"{fam} {comp}")
        ax.axhline(EPS, color="gray", lw=0.8)
        ax.set_title(f"{rule}: non-PDE kernels on 2048 pts in unit square")
        ax.set_xlabel("kernel width (sigma or c)")
    axes[0].set_ylabel("||x - x_dense|| / ||x_dense||")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e4_nonpde.png"), dpi=130); plt.close(fig)
    return rows


# ---- E5 -------------------------------------------------------------------------------------------------
def e5(sizes: Sequence[int]) -> list[Row]:
    print("E5 cost vs N")
    rows: list[Row] = []
    for n in sizes:
        rows += run_set(P.p2_area(n), RULES, COMPS, family="P2")
        rows += run_set(E.rbf(n, 0.2), RULES, COMPS, family="rbf0.2")
    _save("e5_cost", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rule in zip(axes, RULES):
        for fam, mk in (("P2", "o"), ("rbf0.2", "s")):
            for comp in COMPS:
                sel = [r for r in rows if r["family"] == fam and r["rule"] == rule and r["comp"] == comp]
                ax.loglog([r["N"] for r in sel], [r["matvec_equiv"] for r in sel], marker=mk, label=f"{fam} {comp}")
        ax.set_title(f"{rule}: compression flops / (2 N^2)")
        ax.set_xlabel("N")
    axes[0].set_ylabel("dense-matvec equivalents")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e5_cost.png"), dpi=130); plt.close(fig)
    return rows


# ---- E6 -------------------------------------------------------------------------------------------------
def e6(n: int) -> list[Row]:
    """(a) P3 at k = 80: does tightening eps help Exact? (b) Y0 kernel on the CURVE at y01 / R_l."""
    print("E6 eps sweep at k = 80; Y0 on the curve")
    rows: list[Row] = []
    prob = P.p3_helmholtz(n, 80.0)
    xd = dense_ref(prob)
    for eps in (1e-6, 1e-8, 1e-10, 1e-12):
        for comp in ("exact", "proxy"):
            r = case(prob, "weak", comp, eps=eps, x_dense=xd)
            _log(r)
            rows.append({**r, "part": "eps_sweep"})
    X, _ = P.starfish(n)
    R = proxy_radii(X)
    for lvl in (2, 3, 4):
        k = float(yn_zeros(0, 1)[0]) / R[lvl]
        rows += run_set(E.helm_curve(n, k, "Y0"), RULES, ("proxy", "proxy2", "exact"), k_wave=k,
                        part="curve_Y0", level=lvl, k=k)
    _save("e6_extra", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N = 1024 and fewer sweep points")
    ap.add_argument("--only", default="", help="comma-separated subset of e1..e6 (default: all)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n = 1024 if a.quick else 2048
    only = set(a.only.split(",")) if a.only else {"e1", "e2", "e3", "e4", "e5", "e6"}
    if "e6" in only:
        e6(n)
    if "e1" in only:
        e1(n)
    if "e2" in only:
        e2(n, [0.0, 1e-6, 1e-3] if a.quick else [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    if "e3" in only:
        e3(n, [10.0, 40.0, 80.0] if a.quick else [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    if "e4" in only:
        e4(n, [0.1, 0.5] if a.quick else [0.05, 0.1, 0.2, 0.5, 1.0], [0.2] if a.quick else [0.05, 0.2, 1.0])
    if "e5" in only:
        e5([512, 1024] if a.quick else [1024, 2048, 4096])


if __name__ == "__main__":
    main()
