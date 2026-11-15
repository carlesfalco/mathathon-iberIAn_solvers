"""Reproduce every number and figure of findings.md (≈ 3-4 min on a laptop).

    .venv/bin/python -m unified.explore.e4_butterfly_in_solver.run [--quick]      (from the repo root)

Writes results/compression.json, results/gmres.json, results/compression.png, results/gmres.png.
"""
from __future__ import annotations
import json
import pathlib
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from unified.explore.e4_butterfly_in_solver.arcs import CompareReport, Wavenumber, compare  # noqa: E402
from unified.explore.e4_butterfly_in_solver.fio_solve import (ButterflyOp, PrecondKind, T_CAUSTIC,  # noqa: E402
                                                              solve_case, spectrum_stats)

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
EPS = 1e-6
type Results = dict[str, Any]

# plot style of unified/bench/common.py (copied: that module is untyped, so calling it fails mypy --strict)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARKERS = ["o", "s", "^", "D"]
KINDS3: tuple[PrecondKind, ...] = ("none", "adjoint", "jacobian")
KINDS4: tuple[PrecondKind, ...] = ("none", "adjoint", "jacobian", "parametrix")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    })


def crossover_k(rows: list[CompareReport]) -> Wavenumber | None:
    """k where storage(butterfly) / storage(low rank) crosses 1 (log-log interpolation)."""
    for a, b in zip(rows, rows[1:]):
        ra, rb = a["ratio_bf_over_lr"], b["ratio_bf_over_lr"]
        if ra > 1 >= rb:
            s = np.log(ra) / (np.log(ra) - np.log(rb))
            return float(np.exp(np.log(a["k"]) + s * (np.log(b["k"]) - np.log(a["k"]))))
    return None


def part_a(Ns: list[int]) -> Results:
    out: Results = {"eps_target": EPS, "geometries": {}}
    for geom in ("opposite_arcs", "parallel_segments", "collinear_segments"):
        rows = []
        for N in Ns:
            t0 = time.perf_counter()
            r = compare(geom, N, EPS)
            r["time"] = time.perf_counter() - t0
            rows.append(r)
            print(f"[a] {geom:18s} N={N:5d} k={r['k']:7.1f}  LR rank {r['lowrank']['rank']:4d} "
                  f"store {r['lowrank']['storage']:>9,d} err {r['lowrank']['err']:.1e} | BF max rank "
                  f"{r['butterfly']['rank']:3d} store {r['butterfly']['storage']:>9,d} err {r['butterfly']['err']:.1e}"
                  f" | BF/LR {r['ratio_bf_over_lr']:.2f}  ({r['time']:.1f}s)", flush=True)
        ks, ranks = np.array([r["k"] for r in rows]), np.array([r["lowrank"]["rank"] for r in rows])
        big = ks >= 100
        slope = float(np.polyfit(np.log(ks[big]), np.log(ranks[big]), 1)[0]) if big.sum() >= 2 else None
        out["geometries"][geom] = dict(rows=rows, crossover_k=crossover_k(rows), lowrank_rank_loglog_slope=slope)
    return out


def part_b(quick: bool) -> Results:
    out: Results = {"t_caustic": T_CAUSTIC, "gmres_tol": 1e-8, "butterfly_eps": 1e-10}
    # B1: sweep t and damping at fixed N
    N = 512 if quick else 1024
    sweep: list[dict[str, Any]] = []
    for sigma in (0.0, 1.0, 3.0, 6.0):
        for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            op = ButterflyOp(N, t, sigma)
            sK = np.linalg.svd(op.kernel(op.x, op.xi), compute_uv=False)
            row: dict[str, Any] = dict(N=N, t=t, sigma=sigma, cond_K=float(sK[0] / sK[-1]), bf_max_rank=max(op.bf.ranks))
            for kind in KINDS4:
                r = solve_case(op, kind, maxiter=200)
                row[kind] = {k: r[k] for k in ("iters", "converged", "final_res", "sol_err", "true_res")}
            sweep.append(row)
            print(f"[b] N={N} t={t:3.1f} sigma={sigma:3.1f} cond(K)={row['cond_K']:8.1e} iters "
                  + " ".join(f"{k}={row[k]['iters']}{'' if row[k]['converged'] else '!'}(x err {row[k]['sol_err']:.0e})"
                             for k in ("none", "adjoint", "jacobian", "parametrix")), flush=True)
    out["sweep"] = sweep
    # B2: iterations vs N (sigma = 0), with residual histories for the figure
    scaling: list[dict[str, Any]] = []
    for N in ((256, 1024) if quick else (256, 1024, 4096)):
        for t in (0.5, 1.0, 3.0):
            op = ButterflyOp(N, t)
            for kind in KINDS3:
                r = solve_case(op, kind, maxiter=300, check_dense=False)
                scaling.append({k: r[k] for k in ("N", "t", "precond", "iters", "converged", "final_res", "sol_err", "hist")})
                print(f"[b] scaling N={N:5d} t={t} {kind:9s} iters={r['iters']:3d} conv={r['converged']} "
                      f"x err {r['sol_err']:.1e}", flush=True)
    out["scaling"] = scaling
    # B3: dense spectra (why): eigenvalues of K M far from 1, at the caustic onset
    spec: list[dict[str, Any]] = []
    for t in (1.0, 2.0, 2.5, 2.6, 3.0):
        for skind in KINDS3[1:]:
            s = spectrum_stats(512, t, 0.0, skind)
            spec.append(s)
            print(f"[b] spectrum N=512 t={t} {skind:9s} cond(K)={s['cond_K']:.1e} cond(KM)={s['cond']:.1e} "
                  f"#|lam-1|>1e-3: {s['n_outliers']}", flush=True)
    out["spectrum"] = spec
    return out


def figures(a: Results, b: Results) -> None:
    style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (geom, g) in enumerate(a["geometries"].items()):
        ks = [r["k"] for r in g["rows"]]
        axs[0].plot(ks, [r["lowrank"]["storage"] for r in g["rows"]], color=SERIES[i], marker=MARKERS[i],
                    ls="-", label=f"{geom}: low-rank ID")
        axs[0].plot(ks, [r["butterfly"]["storage"] for r in g["rows"]], color=SERIES[i], marker=MARKERS[i],
                    ls="--", mfc="none", label=f"{geom}: butterfly")
        axs[1].plot(ks, [r["ratio_bf_over_lr"] for r in g["rows"]], color=SERIES[i], marker=MARKERS[i], label=geom)
    ks = [r["k"] for r in a["geometries"]["opposite_arcs"]["rows"]]
    axs[0].plot(ks, [r["dense"] for r in a["geometries"]["opposite_arcs"]["rows"]], color=INK2, ls=":", label="dense N²")
    axs[0].set(xscale="log", yscale="log", xlabel="wavenumber k (10 points / wavelength)",
               ylabel="numbers stored", title="Far block storage at achieved error ≤ 1e-6")
    axs[0].legend(fontsize=7)
    axs[1].axhline(1, color=INK2, lw=1)
    axs[1].set(xscale="log", yscale="log", xlabel="wavenumber k", ylabel="storage butterfly / low rank",
               title="Below 1: butterfly wins")
    axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "compression.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    Nmax = max(r["N"] for r in b["scaling"])
    for i, kind in enumerate(KINDS3):
        for t, ls in ((1.0, "-"), (3.0, ":")):
            h = next(r["hist"] for r in b["scaling"] if r["N"] == Nmax and r["t"] == t and r["precond"] == kind)
            axs[0].semilogy(h, color=SERIES[i], ls=ls, label=f"{kind}, t={t:g}")
    axs[0].set(xlabel="GMRES iteration", ylabel="relative residual", title=f"FIO solve, N={Nmax} (t=3: caustic)")
    axs[0].legend(fontsize=7)
    sw = b["sweep"]
    for i, kind in enumerate(KINDS4[1:]):
        for sigma, ls in ((0.0, "-"), (3.0, "--")):
            rows = [r for r in sw if r["sigma"] == sigma]
            axs[1].plot([r["t"] for r in rows], [r[kind]["iters"] for r in rows], color=SERIES[i],
                        marker=MARKERS[i], ls=ls, label=f"{kind}, σ={sigma:g}")
    axs[1].axvline(T_CAUSTIC, color=INK2, lw=1); axs[1].text(T_CAUSTIC, 205, " caustic", color=INK2, fontsize=8)
    axs[1].set(xlabel="propagation time t", ylabel="GMRES iterations (cap 200)", yscale="log",
               title=f"Iterations to 1e-8, N={sw[0]['N']}")
    axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RES / "gmres.png", dpi=130); plt.close(fig)


def main() -> None:
    quick = "--quick" in sys.argv
    RES.mkdir(exist_ok=True)
    Ns = [32, 64, 128, 256, 512, 1024] if quick else [32, 64, 128, 256, 512, 1024, 2048, 4096]
    a = part_a(Ns)
    (RES / "compression.json").write_text(json.dumps({"label": "computed", **a}, indent=1))
    b = part_b(quick)
    (RES / "gmres.json").write_text(json.dumps({"label": "computed", **b}, indent=1))
    figures(a, b)
    for geom, g in a["geometries"].items():
        print(f"crossover k ({geom}): {g['crossover_k']}, low-rank rank ~ k^{g['lowrank_rank_loglog_slope']}")


if __name__ == "__main__":
    main()
