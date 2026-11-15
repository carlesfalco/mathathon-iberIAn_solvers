"""E8 fractional Laplacian / Riesz kernels: reproduce every number and figure in findings.md.

    .venv/bin/python -m unified.explore.e8_fractional.run [--quick] [--only r1,r2,...]

  R1 physics     fractional Poisson (-Lap)^s u = 1 on the unit ball (d = 1, 2): dense solve vs the exact
                 (1-|x|^2)^s solution (raw and Taylor-corrected quadrature), and the fast solver
                 (strong RS + lifted proxy) vs dense, N up to 8192
  R2 proxy sweep N = 2048, both kernels (hypersingular p = d+2s, Riesz p = d-2s), d = 1, 2, s sweep,
                 weak/strong x {exact, plane, lifted, annulus}: error vs dense, ranks, skeletons, ID cost
  R3 resolution  lifted proxy: number of latitudes (2D) / semicircle points (1D), and radius factor
  R4 rank vs N   exact and lifted, weak and strong, N = 1024 .. 8192
  R6 equal rank  N = 2048, both kernels, d = 1, 2, s sweep, weak/strong: exact vs plane vs lifted (4 latitudes)
                 vs annulus vs Xing-Chow-style selected volume proxy (xc), at equal tolerance eps = 1e-8 AND at
                 equal rank (per-box ranks copied from the exact run); true far-field ID error, solve error, cost
  R5 breaks      eps sweep (weak error ~ cond(A) eps), power exponent p sweep incl. growing kernels (p <= 0),
                 s near 0 and 1
Outputs: results/r*.json, results/r*.png. All eps = 1e-8, max_leaf = 32 unless stated.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fractional import Dim, Family, FracProblem, frac_poisson, make, power_second_kind, rel_err  # noqa: E402
from .harness import COMPS, CaseResult, RuleName, case, dense_solve  # noqa: E402
from .equalrank import ECOMPS, ecase  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
RULES: tuple[RuleName, ...] = ("weak", "strong")
DIMS: tuple[Dim, ...] = (1, 2)
type Row = dict[str, Any]


def _save(name: str, obj: object) -> None:
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _log(r: CaseResult) -> None:
    print(f"  {r['problem']:32s} {r['rule']:6s} {r['comp']:8s} err={r['err']:.1e} skel={r['skel_total']:5d} "
          f"rank={r['rank_max']} rows={r['id_rows']}", flush=True)


# ---- R1 ---------------------------------------------------------------------------------------------------
def r1(quick: bool) -> list[Row]:
    print("R1 physics: fractional Poisson vs exact solution")
    rows: list[Row] = []
    sizes: dict[int, list[int]] = {1: [256, 1024, 4096] if quick else [256, 512, 1024, 2048, 4096, 8192],
                                   2: [256, 1024] if quick else [256, 512, 1024, 2048, 4096]}
    for d in DIMS:
        for s in (0.25, 0.5, 0.75):
            for n in sizes[d] + ([8192] if (d == 2 and s == 0.5 and not quick) else []):
                raw = frac_poisson(d, s, n, corrected=False)
                P = frac_poisson(d, s, n)
                assert P.u_exact is not None
                xd = dense_solve(P)
                inner = np.sum(P.X ** 2, axis=1) < 0.25
                r = case(P, "strong", "lifted", x_dense=xd)
                row = dict(d=d, s=s, N=len(P.X), h=P.h, raw_err=rel_err(dense_solve(raw), P.u_exact),
                           dense_err=rel_err(xd, P.u_exact), dense_err_inner=rel_err(xd[inner], P.u_exact[inner]),
                           fast_vs_dense=r["err"], fast_vs_exact=r["err_vs_exact"], skel=r["skel_total"],
                           factor_s=r["factor_s"], cond=float(np.linalg.cond(P.A)) if len(P.X) <= 4096 else None)
                print(f"  d={d} s={s} N={len(P.X):5d} raw={row['raw_err']:.2e} corr={row['dense_err']:.2e} "
                      f"(|x|<.5: {row['dense_err_inner']:.2e}) fast-dense={r['err']:.1e} "
                      f"fast-exact={r['err_vs_exact']:.2e} t={r['factor_s']:.1f}s", flush=True)
                rows.append(row)
    _save("r1_physics", rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, d in enumerate(DIMS):
        for s, c in zip((0.25, 0.5, 0.75), ("C0", "C1", "C2")):
            sub = [r for r in rows if r["d"] == d and r["s"] == s]
            N = [r["N"] for r in sub]
            ax[i].loglog(N, [r["raw_err"] for r in sub], "--", color=c, label=f"s={s} raw")
            ax[i].loglog(N, [r["dense_err"] for r in sub], "o-", color=c, label=f"s={s} corrected")
            ax[i].loglog(N, [r["fast_vs_dense"] for r in sub], "x:", color=c, label=f"s={s} fast vs dense")
        ax[i].set(title=f"d = {d}: rel. error vs exact (1-|x|^2)^s", xlabel="N")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r1_physics.png"), dpi=120)
    return rows


# ---- R2 ---------------------------------------------------------------------------------------------------
def r2(quick: bool) -> list[Row]:
    print("R2 proxy sweep")
    n = 1024 if quick else 2048
    grid: list[tuple[Family, Dim, float]] = (
        [("hyper", 1, s) for s in (0.1, 0.25, 0.5, 0.75, 0.9)] + [("hyper", 2, s) for s in (0.1, 0.25, 0.5, 0.75, 0.9)]
        + [("riesz", 1, s) for s in (0.1, 0.25, 0.4)] + [("riesz", 2, s) for s in (0.1, 0.25, 0.5, 0.75, 0.9)])
    if quick:
        grid = [g for g in grid if g[2] in (0.25, 0.75)]
    rows: list[Row] = []
    for fam, d, s in grid:
        P = make(fam, d, s, n)
        xd = dense_solve(P)
        for rule in RULES:
            for comp in COMPS:
                r = case(P, rule, comp, x_dense=xd); _log(r); rows.append(r)
    _save("r2_sweep", rows)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
    panels: list[tuple[str, int]] = [("hyper", 1), ("hyper", 2), ("riesz", 1), ("riesz", 2)]
    for j, (famp, dp) in enumerate(panels):
        for comp, mk in zip(COMPS, "osD^"):
            for rule, ls in zip(RULES, (":", "-")):
                sub = [r for r in rows if r["family"] == famp and r["d"] == dp and r["comp"] == comp and r["rule"] == rule]
                if sub:
                    ax[0, j].semilogy([r["s"] for r in sub], [r["err"] for r in sub], mk + ls, label=f"{comp} {rule}")
                    ax[1, j].plot([r["s"] for r in sub], [r["skel_total"] / r["N"] for r in sub], mk + ls)
        ax[0, j].axhline(1e-8, color="k", lw=0.5)
        ax[0, j].set(title=f"{famp} d={dp} (p = d {'+' if famp == 'hyper' else '-'} 2s)")
        ax[1, j].set(xlabel="s")
    ax[0, 0].set_ylabel("solve error vs dense"); ax[1, 0].set_ylabel("skeletons / N")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r2_sweep.png"), dpi=120)
    return rows


# ---- R3 ---------------------------------------------------------------------------------------------------
def r3(quick: bool) -> list[Row]:
    print("R3 lifted proxy resolution")
    n = 1024 if quick else 2048
    rows: list[Row] = []
    for fam in ("hyper", "riesz"):
        P2 = make(fam, 2, 0.5 if fam == "hyper" else 0.25, n)
        xd = dense_solve(P2)
        for n_lat in (0, 1, 2, 3, 4, 6, 8, 12):
            r = case(P2, "strong", "lifted", x_dense=xd, n_lat=n_lat); _log(r); rows.append({**r, "sweep": "n_lat"})
        for rf in (1.75, 2.0, 2.5, 3.0):
            r = case(P2, "strong", "lifted", x_dense=xd, radius_factor=rf); _log(r); rows.append({**r, "sweep": "radius"})
        P1 = make(fam, 1, 0.5 if fam == "hyper" else 0.25, n)
        xd1 = dense_solve(P1)
        for npx in (2, 4, 8, 16, 32, 64):
            r = case(P1, "strong", "lifted", x_dense=xd1, n_proxy=npx); _log(r); rows.append({**r, "sweep": "n_proxy"})
    _save("r3_resolution", rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for fam, c in (("hyper", "C0"), ("riesz", "C1")):
        sub = [r for r in rows if r["family"] == fam and r["sweep"] == "n_lat"]
        ax[0].semilogy([r["n_lat"] for r in sub], [r["err"] for r in sub], "o-", color=c, label=f"{fam} d=2")
        sub = [r for r in rows if r["family"] == fam and r["sweep"] == "n_proxy"]
        ax[1].semilogy([r["n_proxy"] for r in sub], [r["err"] for r in sub], "o-", color=c, label=f"{fam} d=1")
    ax[0].set(xlabel="latitudes above the equator (0 = plane circle)", ylabel="error vs dense", title="2D, strong")
    ax[1].set(xlabel="semicircle intervals", title="1D, strong"); ax[1].set_xscale("log", base=2)
    for a in ax:
        a.axhline(1e-8, color="k", lw=0.5); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r3_resolution.png"), dpi=120)
    return rows


# ---- R4 ---------------------------------------------------------------------------------------------------
def r4(quick: bool) -> list[Row]:
    print("R4 rank vs N")
    rows: list[Row] = []
    sizes = [512, 1024, 2048] if quick else [1024, 2048, 4096, 8192]
    for d in DIMS:
        for n in sizes:
            P = frac_poisson(d, 0.5, n)
            xd = dense_solve(P)
            for rule in RULES:
                for comp in ("exact", "lifted"):
                    t0 = time.perf_counter()
                    r = case(P, rule, comp, x_dense=xd)
                    r["wall_s"] = time.perf_counter() - t0
                    _log(r); rows.append(r)
            del P, xd
    _save("r4_rank_vs_n", rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, d in enumerate(DIMS):
        for rule, ls in zip(RULES, (":", "-")):
            for comp, mk in (("exact", "o"), ("lifted", "D")):
                sub = [r for r in rows if r["d"] == d and r["rule"] == rule and r["comp"] == comp]
                ax[i].loglog([r["N"] for r in sub], [max(r["rank_max"].values()) for r in sub], mk + ls,
                             label=f"{comp} {rule}")
        ax[i].set(title=f"fractional Poisson d={d}, s=0.5", xlabel="N", ylabel="max rank over boxes")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r4_rank_vs_n.png"), dpi=120)
    return rows


# ---- R5 ---------------------------------------------------------------------------------------------------
def r5(quick: bool) -> dict[str, list[Row]]:
    print("R5 where it breaks")
    n = 1024 if quick else 2048
    out: dict[str, list[Row]] = {"eps": [], "power": [], "extreme_s": []}
    for d in DIMS:
        for s in (0.25, 0.5, 0.75):
            P = frac_poisson(d, s, n)
            xd = dense_solve(P); cond = float(np.linalg.cond(P.A))
            for eps in (1e-6, 1e-8, 1e-10, 1e-12):
                for rule in RULES:
                    r = case(P, rule, "lifted", eps=eps, x_dense=xd)
                    r["cond"] = cond; _log(r); out["eps"].append(r)
    for d in DIMS:
        for p in (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5):
            P = power_second_kind(d, p, n)
            xd = dense_solve(P)
            for rule in RULES:
                for comp in COMPS:
                    r = case(P, rule, comp, x_dense=xd); _log(r); out["power"].append(r)
    for d in DIMS:
        for s in (0.02, 0.05, 0.95, 0.98):
            P = frac_poisson(d, s, n)
            xd = dense_solve(P)
            for comp in ("exact", "lifted"):
                r = case(P, "strong", comp, x_dense=xd); _log(r); out["extreme_s"].append(r)
    _save("r5_breaks", out)
    return out


# ---- R6 ---------------------------------------------------------------------------------------------------
def r6(quick: bool) -> list[Row]:
    print("R6 equal tolerance vs equal rank")
    n = 1024 if quick else 2048
    grid: list[tuple[Family, Dim, float]] = (
        [("hyper", 1, s) for s in (0.25, 0.5, 0.75)] + [("hyper", 2, s) for s in (0.25, 0.5, 0.75)]
        + [("riesz", 1, s) for s in (0.1, 0.25, 0.4)] + [("riesz", 2, s) for s in (0.25, 0.5, 0.75)])
    if quick:
        grid = grid[1::3]
    rows: list[Row] = []
    for fam, d, s in grid:
        P = make(fam, d, s, n)
        xd = dense_solve(P)
        for rule in RULES:
            ref, sched = ecase(P, rule, "exact", xd)
            rows.append(ref)
            for comp in ECOMPS[1:]:
                for schedule in (None, sched):
                    r, _ = ecase(P, rule, comp, xd, schedule=schedule)
                    rows.append(r)
            for r in rows[-9:]:
                print(f"  {fam} d={d} s={s} {rule:6s} {r['comp']:7s} {r['mode']:10s} solve={r['err']:.1e} "
                      f"far_max={r['far_err_max']:.1e} far_abs={r['far_abs_max']:.1e} skel={r['skel_compressed']} "
                      f"proxies={r['n_proxy_max']} flops={r['id_flops']:.2e} sel={r['select_flops']:.1e}", flush=True)
    _save("r6_equal_rank", rows)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    panels: list[tuple[str, int]] = [("hyper", 1), ("hyper", 2), ("riesz", 1), ("riesz", 2)]
    for i, rule in enumerate(RULES):
        for j, (famp, dp) in enumerate(panels):
            for comp, c in zip(ECOMPS, ("k", "C3", "C5", "C7", "C2")):
                for mode, ls in (("equal_tol", "-"), ("equal_rank", ":")):
                    sub = [r for r in rows if r["family"] == famp and r["d"] == dp and r["rule"] == rule
                           and r["comp"] == comp and r["mode"] == mode]
                    if sub:
                        ax[i, j].semilogy([r["s"] for r in sub], [r["err"] for r in sub], "o" + ls, color=c,
                                          label=f"{comp} {mode.replace('equal_', '=')}")
            ax[i, j].axhline(1e-8, color="k", lw=0.5)
            ax[i, j].set(title=f"{famp} d={dp}, {rule}", xlabel="s")
    ax[0, 0].set_ylabel("solve error vs dense"); ax[1, 0].set_ylabel("solve error vs dense")
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r6_equal_rank.png"), dpi=120)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="r1,r2,r3,r4,r5,r6")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    for name in a.only.split(","):
        {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6}[name.strip()](a.quick)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
