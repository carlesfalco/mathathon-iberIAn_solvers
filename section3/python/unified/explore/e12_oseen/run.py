"""E12 driver: reproduces every number and figure in findings.md.

    PYTHONPATH=. .venv/bin/python -m unified.explore.e12_oseen.run            # full (≈ 5 min)
    PYTHONPATH=. .venv/bin/python -m unified.explore.e12_oseen.run --quick    # smaller N
    ... --only r1,r6

R1  far-block ranks vs direction theta and Oseen parameter lam: per-block relative / absolute eps-rank and the
    driver's stacked-ID rank (rows + columns)
R2  area matrix A = I + G_lam/N (N = 2048): weak / strong / directional rules x exact / circle proxy / ellipse proxy
R3  N scaling (2048 .. 8192) at lam = 0 and 80: strong vs dir(2,2,1)
R4  proxy-circle resolution: n_proxy vs lam (does the stock proxy still work? when must it grow?)
R5  the diagonal-similarity route (Oseen = D Yukawa D^-1): error vs lam
R6  tensor Oseen drag on a cylinder vs Lamb (1911), dense convergence, fast vs dense, pivot health (E5 check)
L   lines of code: solver dials vs physics set-up (inspect.getsource, as the showcase does)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .directional import Directional, block_rank, stacked_id_rank
from .harness import (CaseResult, EllipseProxy, TensorProxy, area_case, cylinder_case, dense_solve,
                      similarity_route)
from .oseen import (ScalarOseen, area_problem, cylinder, interleave, lamb_drag, near_constant,
                    oseen_tensor, oseen_tensor_blocks)

OUT = Path(__file__).parent / "results"
EPS = 1e-8


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(OUT / "run_log.txt", "a") as f:
        f.write(msg + "\n")


def save(name: str, data: Any) -> None:
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(data, f, indent=1, default=float)


def short(r: CaseResult) -> str:
    keys = ("err", "err_mod_null", "backward", "skel_total", "root", "stored", "flops", "pivot_cond",
            "pivot_herm_min", "drag", "factor_s")
    return " ".join(f"{k} {r[k]:.3g}" if isinstance(r[k], float) else f"{k} {r[k]}" for k in keys if k in r)


def r1() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    w = 1 / 8
    thetas = np.linspace(0, np.pi, 13)
    for lam in (0.0, 1.0, 5.0, 20.0, 80.0, 320.0):
        K = ScalarOseen(lam)
        for D in (2 * w, 4 * w):
            down = block_rank(K, w, D, 0.0, EPS)[2]
            for th in thetas:
                rel, ab, nrm = block_rank(K, w, D, float(th), EPS, abs_scale=down)
                st = stacked_id_rank(K, w, D, float(th), EPS)
                out.append(dict(lam=lam, D=D, theta=float(th), rel_rank=rel, abs_rank=ab, norm=nrm,
                                norm_ratio=nrm / down, stacked_rank=st))
            rr = [o for o in out if o["lam"] == lam and o["D"] == D]
            log(f"R1 lam {lam:5g} D/w {D / w:.0f}: rel {[o['rel_rank'] for o in rr]}")
            log(f"                      abs {[o['abs_rank'] for o in rr]}")
            log(f"                  stacked {[o['stacked_rank'] for o in rr]}")
    save("r1_block_ranks", out)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for lam in (0.0, 5.0, 80.0, 320.0):
        rr = [o for o in out if o["lam"] == lam and o["D"] == 2 * w]
        degs = [np.degrees(o["theta"]) for o in rr]
        axs[0].plot(degs, [o["rel_rank"] for o in rr], "o-", label=f"λ = {lam:g}")
        axs[1].plot(degs, [o["abs_rank"] for o in rr], "o-", label=f"λ = {lam:g}")
        axs[2].plot(degs, [o["stacked_rank"] for o in rr], "o-", label=f"λ = {lam:g}")
    for ax, t in zip(axs, ("relative ε-rank of K(F,B)", "absolute ε-rank (ε × downstream norm)",
                           "driver's stacked ID rank [K(F,B); K(B,F)ᵀ]")):
        ax.set_xlabel("direction of F from B (deg; 0 = downstream)"); ax.set_title(t, fontsize=10)
    axs[0].set_ylabel("rank at ε = 1e-8, w = 1/8, D = 2w"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "r1_block_ranks.png", dpi=120); plt.close(fig)
    return out


LAMS2 = (0.0, 1.0, 5.0, 20.0, 80.0, 320.0)
RULES2 = ("weak", "strong", "dir:2,2,1", "dir:3,3,1", "dir:3,1,1", "dir:1,1,2")


def r2(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    for lam in LAMS2:
        p = area_problem(N, lam)
        xd = dense_solve(p.A, p.b)
        npx = 128 if lam >= 320 else 64        # R4: the circle needs 128 points at lam = 320
        for rule in RULES2:
            for comp in ("exact", "proxy", "ellipse"):
                if comp == "ellipse" and not rule.startswith("dir"):
                    continue
                r = area_case(p, rule, comp, xd, health=rule in ("strong", "weak"), n_proxy=npx)
                log(f"R2 lam {lam:g} {rule:10s} {comp:8s} " + short(r)); out.append(r)
    save("r2_admissibility", out)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    lams = LAMS2
    for rule in RULES2:
        for ax, key in zip(axs, ("skel_total", "stored", "flops")):
            vals = [next(r[key] for r in out if r["lam"] == l and r["rule"] == rule and r["comp"] == "exact")
                    for l in lams]
            base = [next(r[key] for r in out if r["lam"] == l and r["rule"] == "strong" and r["comp"] == "exact")
                    for l in lams]
            ax.plot(range(len(lams)), np.array(vals) / np.array(base), "o-", label=rule)
    for ax, t in zip(axs, ("total skeletons", "numbers stored", "elimination flops")):
        ax.set_xticks(range(len(lams)), [f"{l:g}" for l in lams]); ax.set_xlabel("λ")
        ax.set_title(f"{t} ÷ strong (exact ID)", fontsize=10); ax.axhline(1, color="k", lw=.6)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"R2: A = I + G_λ/N, N = {N}, ε = 1e-8"); fig.tight_layout()
    fig.savefig(OUT / "r2_admissibility.png", dpi=120); plt.close(fig)
    return out


def r3(Ns: list[int]) -> list[CaseResult]:
    out: list[CaseResult] = []
    for lam in (0.0, 80.0):
        for N in Ns:
            p = area_problem(N, lam)
            t0 = time.perf_counter(); xd = dense_solve(p.A, p.b); td = time.perf_counter() - t0
            for rule, comp in (("strong", "exact"), ("strong", "proxy"), ("dir:2,2,1", "exact"),
                               ("dir:2,2,1", "ellipse")):
                r = area_case(p, rule, comp, xd)
                r["dense_s"] = td
                log(f"R3 lam {lam:g} N {N} {rule:10s} {comp:8s} " + short(r)); out.append(r)
            del p
    save("r3_scaling", out)
    return out


def r4(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    for lam in (20.0, 80.0, 320.0, 1280.0):
        p = area_problem(N, lam)
        xd = dense_solve(p.A, p.b)
        for npx in (16, 32, 64, 128, 256):
            r = area_case(p, "strong", "proxy", xd, n_proxy=npx)
            log(f"R4 lam {lam:g} n_proxy {npx} " + short(r)); out.append(r)
    save("r4_proxy_resolution", out)
    return out


def r5(N: int) -> list[CaseResult]:
    out: list[CaseResult] = []
    for lam in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 80.0):
        p = area_problem(N, lam)
        xd = dense_solve(p.A, p.b)
        r = similarity_route(p, EPS, xd)
        log(f"R5 lam {lam:g} similarity route: err {r['err']:.2e} residual {r['residual']:.2e} "
            f"cond(D) {r['cond_D']:.2e} skel {r['skel_total']}"); out.append(r)
    save("r5_similarity", out)
    return out


def r6(quick: bool) -> dict[str, Any]:
    conv: list[dict[str, Any]] = []
    ks = (0.01, 0.03, 0.1, 0.3, 1.0)
    for k in ks:
        for n in (64, 128, 256, 512):
            c = cylinder(k, n)
            f = dense_solve(c.A, c.rhs)
            conv.append(dict(k=k, n=n, drag=c.drag(f), lamb=lamb_drag(k), rel_dev=c.drag(f) / lamb_drag(k) - 1,
                             cond=float(np.linalg.cond(c.A))))
            log(f"R6 dense k {k} n {n}: drag {c.drag(f):.10f} Lamb {lamb_drag(k):.10f} "
                f"rel dev {c.drag(f) / lamb_drag(k) - 1:+.3e} cond {conv[-1]['cond']:.2e}")
    fast: list[CaseResult] = []
    n_fast = 256 if quick else 1024
    for k in (0.1, 1.0, 10.0):
        c = cylinder(k, n_fast)
        fd = dense_solve(c.A, c.rhs)
        for rule in ("weak", "strong", "dir:2,2,1"):
            for comp in ("exact", "proxy"):
                r = cylinder_case(c, rule, comp, fd, EPS)
                r["drag_dense"] = c.drag(fd)
                r["drag_rel_err"] = abs(r["drag"] / c.drag(fd) - 1)
                log(f"R6 fast k {k} n {n_fast} {rule:10s} {comp:6s} drag rel err {r['drag_rel_err']:.2e} " + short(r))
                fast.append(r)
    res = dict(convergence=conv, fast=fast)
    save("r6_cylinder", res)
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in (64, 512):
        rr = [o for o in conv if o["n"] == n]
        ax.loglog([o["k"] for o in rr], [abs(o["rel_dev"]) for o in rr], "o-", label=f"|dense drag / Lamb − 1|, n = {n}")
    kk = np.array(ks)
    ax.loglog(kk, 2 * kk ** 2 * np.log(1 / kk + 1), "k:", label="∝ (ka)² log(1/ka) (guide)")
    ax.set_xlabel("ka (Oseen parameter × radius)"); ax.set_ylabel("relative deviation from Lamb (1911)")
    ax.set_title("R6: Oseen drag on a cylinder"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "r6_drag.png", dpi=120); plt.close(fig)
    return res


def lines() -> dict[str, Any]:
    from unified.showcase.reproduce import code_lines
    from . import directional, harness, oseen
    dial = {"Directional (dial 2)": code_lines(Directional), "EllipseProxy (dial 4)": code_lines(EllipseProxy),
            "TensorProxy (dial 4, tensor)": code_lines(TensorProxy)}
    physics = {f: code_lines(getattr(oseen, f)) for f in
               ("ScalarOseen", "oseen_tensor_blocks", "interleave", "oseen_tensor", "near_constant", "Cylinder",
                "cylinder", "lamb_drag", "area_problem")}
    out = dict(dial=dial, dial_total=sum(dial.values()), physics=physics, physics_total=sum(physics.values()),
               note="stock ProxyCompressor + scalar kernel needs 0 dial lines; experiment code (probes, harness, "
                    "run) is not counted", modules=[directional.__name__, harness.__name__])
    log(f"L lines: dial {out['dial_total']} {dial}; physics {out['physics_total']}")
    save("lines", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="r1,r2,r3,r4,r5,r6,l")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    only = set(a.only.split(","))
    N = 1024 if a.quick else 2048
    t0 = time.perf_counter()
    log(f"=== E12 run quick={a.quick} only={sorted(only)}")
    _ = (oseen_tensor, oseen_tensor_blocks, interleave, near_constant)   # imported for readers of this file
    if "r1" in only: r1()
    if "r2" in only: r2(N)
    if "r3" in only: r3([1024, 2048] if a.quick else [2048, 4096, 8192])
    if "r4" in only: r4(N)
    if "r5" in only: r5(N)
    if "r6" in only: r6(a.quick)
    if "l" in only: lines()
    log(f"=== total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
