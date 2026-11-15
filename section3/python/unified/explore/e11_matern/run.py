"""E11 driver: reproduces every number and figure in findings.md.

    PYTHONPATH=. .venv/bin/python -m unified.explore.e11_matern.run            # full (≈ 6-8 min)
    PYTHONPATH=. .venv/bin/python -m unified.explore.e11_matern.run --quick    # N = 1024, fewer cases
    ... --only r1,r3                                                           # a subset

R1  compressor x nu table (N = 2048, ell = 0.1, sigma2 = 1e-2), strong and weak; K0 and radiative-transfer controls
R2  lifted-proxy resolution: latitudes 0..8, one kernel vs beta kernels
R3  equal rank: every compressor truncated at the exact ID's per-box ranks (strong)
R4  GP sweep: nu x ell x sigma2 (N = 2048): compression residual vs conditioning, log det, posterior mean
R5  N scaling to 8192 (strong; exact, liftM, xc)
R6  statistics task: log marginal likelihood vs ell, fast (liftM, xc) vs dense
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .harness import COMPS, CaseResult, CompName, RuleName, case, dense_ref, schedule_of, BoxRecord
from .matern import gp_problem, lift_order, log_marginal_likelihood

OUT = Path(__file__).parent / "results"
NUS = [0.5, 0.7, 1.0, 1.5, 2.5]
EPS = 1e-8


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(OUT / "run_log.txt", "a") as f:
        f.write(msg + "\n")


def fmt(r: CaseResult) -> str:
    return (f"  {r['rule']:6s} {r['comp']:8s} bwd {r['backward']:.1e} res {r['residual']:.1e} err {r['err']:.1e} far(max/med) "
            f"{r['far_err_max']:.1e}/{r['far_err_median']:.1e} logdet {r['logdet_abs_err']:.1e} skel {r['skel_total']} "
            f"rmax {r['rank_max']} rows {r['id_rows']} npx {r['n_proxy_max']} t {r['factor_s']:.1f}s")


def save(name: str, data: Any) -> None:
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(data, f, indent=1, default=float)


def r1(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    probs = [("matern", nu) for nu in NUS] + [("k0", 0.0), ("rte", 0.5)]
    for kind, nu in probs:
        p = gp_problem(N, nu, 0.1, 1e-2, kind=kind)
        ref = dense_ref(p)
        log(f"R1 {p.name} cond {ref['cond']:.2e}  (beta, D) = {lift_order(nu) if kind == 'matern' else '-'}")
        for rule in ("strong", "weak"):
            for c in COMPS:
                r = case(p, rule, c, ref, EPS)
                log(fmt(r)); out.append(r)
    save("r1_table", out)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, rule in zip(axs, ("strong", "weak")):
        for c in COMPS:
            xs = [r["nu"] for r in out if r["rule"] == rule and r["comp"] == c and r["kind"] == "matern"]
            ys = [r["backward"] for r in out if r["rule"] == rule and r["comp"] == c and r["kind"] == "matern"]
            ax.semilogy(xs, ys, "o-", label=c)
        ax.axhline(EPS, color="k", ls=":", lw=1)
        ax.set_xlabel("Matérn smoothness ν"); ax.set_ylabel("normwise backward error")
        ax.set_title(f"R1: {rule} RS, N = {N}, ℓ = 0.1, σ² = 1e-2, ε = 1e-8")
    axs[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "r1_table.png", dpi=120); plt.close(fig)
    return out


def r2(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    for nu in (0.5, 0.7, 1.5, 2.5):
        p = gp_problem(N, nu, 0.1, 1e-2)
        ref = dense_ref(p, with_cond=False)
        for c in ("lift1", "liftM"):
            for nl in (0, 1, 2, 4, 8):
                r = case(p, "strong", c, ref, EPS, n_lat=nl)
                log(f"R2 nu {nu} n_lat {nl}" + fmt(r)); out.append(r)
    save("r2_latitudes", out)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for nu in (0.5, 0.7, 1.5, 2.5):
        for c, ls in (("lift1", "--"), ("liftM", "-")):
            rr = [r for r in out if r["nu"] == nu and r["comp"] == c]
            ax.semilogy([r["n_lat"] for r in rr], [r["far_err_max"] for r in rr], ls, marker="o", label=f"{c} ν={nu}")
    ax.axhline(EPS, color="k", ls=":", lw=1)
    ax.set_xlabel("latitudes above the equator (0 = circle)"); ax.set_ylabel("worst per-box far-field error")
    ax.set_title(f"R2: lifted resolution, strong, N = {N}"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "r2_latitudes.png", dpi=120); plt.close(fig)
    return out


def r3(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    for nu in (0.5, 0.7, 1.5, 2.5):
        p = gp_problem(N, nu, 0.1, 1e-2)
        ref = dense_ref(p, with_cond=False)
        recs: list[BoxRecord] = []
        base = case(p, "strong", "exact", ref, EPS, records_out=recs)
        sched = schedule_of(recs)
        log(f"R3 nu {nu} (reference)" + fmt(base)); out.append(base)
        for c in ("rand", "plane1", "planeM", "lift1", "liftM", "annulus", "xc"):
            r = case(p, "strong", c, ref, EPS, schedule=sched)
            log(f"R3 nu {nu} equal-rank" + fmt(r)); out.append(r)
    save("r3_equal_rank", out)
    return out


def r4(N: int, quick: bool) -> list[CaseResult]:
    out: list[CaseResult] = []
    ells = (0.03, 0.1, 0.3)
    for nu in (0.5, 0.7, 1.5, 2.5):
        for ell in ells:
            for s2 in (1e-1, 1e-4):
                p = gp_problem(N, nu, ell, s2)
                ref = dense_ref(p)
                pairs: tuple[tuple[RuleName, CompName], ...] = (
                    ("strong", "exact"), ("strong", "liftM"), ("strong", "xc"), ("weak", "exact"), ("weak", "liftM"))
                for rule, c in pairs:
                    if quick and rule == "weak":
                        continue
                    r = case(p, rule, c, ref, EPS, far_check=False)
                    log(f"R4 nu {nu} ell {ell} s2 {s2:g} cond {ref['cond']:.1e}" + fmt(r)
                        + f" mean {r['mean_err']:.1e}")
                    out.append(r)
    save("r4_gp_sweep", out)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, mk in (("exact", "o"), ("liftM", "s"), ("xc", "^")):
        rr = [r for r in out if r["comp"] == c and r["rule"] == "strong"]
        ax.loglog([r["cond"] for r in rr], [r["err"] for r in rr], mk, label=f"strong {c}: err vs dense", alpha=.7)
        ax.loglog([r["cond"] for r in rr], [r["backward"] for r in rr], mk, mfc="none", label=f"strong {c}: backward err")
    cs = np.geomspace(1, 1e8, 10)
    ax.loglog(cs, cs * EPS, "k:", label="cond·ε")
    ax.set_xlabel("cond(C + σ² I)"); ax.set_ylabel("relative error / residual")
    ax.set_title(f"R4: GP sweep, N = {N}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "r4_conditioning.png", dpi=120); plt.close(fig)
    return out


def r5(Ns: list[int]) -> list[CaseResult]:
    out: list[CaseResult] = []
    for nu in (0.7, 1.5):
        for N in Ns:
            p = gp_problem(N, nu, 0.1, 1e-2)
            t0 = time.perf_counter(); ref = dense_ref(p, with_cond=False); td = time.perf_counter() - t0
            for c in ("exact", "liftM", "xc"):
                r = case(p, "strong", c, ref, EPS, far_check=False)
                r["dense_s"] = td
                log(f"R5 nu {nu} N {N} dense {td:.1f}s" + fmt(r) + f" mean {r['mean_err']:.1e}")
                out.append(r)
            del p
    save("r5_scaling", out)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for nu, ls in ((0.7, "--"), (1.5, "-")):
        for c in ("exact", "liftM", "xc"):
            rr = [r for r in out if r["nu"] == nu and r["comp"] == c]
            axs[0].plot([r["N"] for r in rr], [r["rank_max_level"][str(2)] if str(2) in r["rank_max_level"]
                                               else r["rank_max_level"].get(2, 0) for r in rr], ls, marker="o",
                        label=f"{c} ν={nu}")
            axs[1].semilogy([r["N"] for r in rr], [max(r["backward"], 1e-17) for r in rr], ls, marker="o",
                            label=f"{c} ν={nu}")
    axs[0].set_xscale("log", base=2); axs[1].set_xscale("log", base=2)
    axs[0].set_xlabel("N"); axs[0].set_ylabel("max skeleton per box, level 2")
    axs[1].set_xlabel("N"); axs[1].set_ylabel("normwise backward error"); axs[1].axhline(EPS, color="k", ls=":")
    axs[0].legend(fontsize=7); fig.suptitle("R5: strong RS on Matérn GP matrices, ℓ = 0.1, σ² = 1e-2")
    fig.tight_layout(); fig.savefig(OUT / "r5_scaling.png", dpi=120); plt.close(fig)
    return out


def r6(N: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ells = [0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9]
    for ell in ells:
        p = gp_problem(N, 1.5, ell, 1e-2)
        ref = dense_ref(p, with_cond=False)
        row: dict[str, Any] = {"ell": ell, "dense": log_marginal_likelihood(p.y, ref["x"], ref["logdet"])}
        for c in ("liftM", "xc"):
            r = case(p, "strong", c, ref, EPS, far_check=False)
            row[c] = -0.5 * r["yalpha"] - 0.5 * r["logdet_fast"] - 0.5 * N * np.log(2 * np.pi)
            row[c + "_logdet_err"] = r["logdet_abs_err"]
            row[c + "_err"] = r["err"]
        log(f"R6 ell {ell}: loglik dense {row['dense']:.6f} liftM {row['liftM']:.6f} xc {row['xc']:.6f}")
        out.append(row)
    save("r6_likelihood", out)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(ells, [r["dense"] for r in out], "k-o", label="dense")
    ax.semilogx(ells, [r["liftM"] for r in out], "s--", mfc="none", label="strong RS, liftM")
    ax.set_xlabel("length scale ℓ"); ax.set_ylabel("log marginal likelihood")
    ax.set_title(f"R6: ν = 1.5, σ² = 1e-2, N = {N}"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "r6_likelihood.png", dpi=120); plt.close(fig)
    return out


def lines() -> dict[str, Any]:
    """Solver-dial lines vs physics/statistics set-up lines (the showcase's inspect.getsource counter)."""
    from unified.showcase.reproduce import code_lines
    from . import harness, matern, proxies
    dial = {"MaternProxy (dial 4: geometry)": code_lines(proxies.MaternProxy),
            "StackedKernel.__call__ (dial 4: kernel orders)": code_lines(matern.StackedKernel.__call__),
            "multi_orders + lift_order (dial 4)": code_lines(matern.multi_orders) + code_lines(matern.lift_order),
            "_pairdist (lifted points)": code_lines(matern._pairdist),
            "logdet (algebra read-out)": code_lines(harness.logdet)}
    physics = {f: code_lines(getattr(matern, f)) for f in
               ("radial_rK", "matern_cov", "kappa_of", "test_function", "GPProblem", "gp_problem",
                "posterior_mean", "log_marginal_likelihood")}
    physics["StackedKernel.main/block (kernel formulas)"] = (code_lines(matern.StackedKernel.main)
                                                             + code_lines(matern.StackedKernel.block))
    out = dict(dial=dial, dial_total=sum(dial.values()), physics=physics, physics_total=sum(physics.values()),
               note="MaternProxy includes the annulus and xc comparison geometries; the lifted-multi path alone is "
                    "_hemisphere + StackedKernel.__call__ + multi_orders + lift_order + _pairdist")
    out["lifted_multi_only"] = (code_lines(proxies.MaternProxy._hemisphere) + code_lines(proxies.MaternProxy._circle)
                                + dial["StackedKernel.__call__ (dial 4: kernel orders)"]
                                + dial["multi_orders + lift_order (dial 4)"] + dial["_pairdist (lifted points)"])
    log(f"L lines: dial {out['dial_total']} (lifted-multi path {out['lifted_multi_only']}) {dial}; "
        f"physics {out['physics_total']} {physics}")
    save("lines", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="r1,r2,r3,r4,r5,r6,l")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # Accelerate's slogdet emits spurious warnings
    only = set(a.only.split(","))
    N = 1024 if a.quick else 2048
    t0 = time.perf_counter()
    log(f"=== E11 run quick={a.quick} only={sorted(only)} pid={os.getpid()}")
    if "r1" in only: r1(N)
    if "r2" in only: r2(N)
    if "r3" in only: r3(N)
    if "r4" in only: r4(N, a.quick)
    if "r5" in only: r5([1024, 2048] if a.quick else [2048, 4096, 8192])
    if "r6" in only: r6(N)
    if "l" in only: lines()
    log(f"=== total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
