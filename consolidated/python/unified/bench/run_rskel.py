"""Benchmarks for recursive skeletonization (unified/drivers/rskel.py): weak (HBS) vs strong (FMM-LU).

1. Scaling. P1 curve and P2 area, N = 1024 ... 16384, proxy compressor, eps = 1e-8.
   The headline quantity is the MAX SKELETON SIZE PER BOX at each tree level (stats['levels'][l]['skel_max']).
   It is the rank of a box's interaction with its far field, and it is what the admissibility rule controls.
   - Level 2 is the coarsest level where strong admissibility has a far field in 2D.
   - The finest two levels are where most unknowns are eliminated.
   Factor time, solve time and residual are also recorded, plus fitted log-log slopes of factor time vs N.
   Root size is kept only as a secondary number. Under Strong in 2D the level-1 boxes have no far field,
   so the root holds every level-2 skeleton by construction, and comparing root sizes across rules misleads.
   -> results/rskel_scaling.{png,json}
2. Helmholtz. P3, N = 2048, k in {5, 20, 50}: max skeleton per level and root size.
   -> results/rskel_helmholtz.{png,json}
Run:  .venv/bin/python -m unified.bench.run_rskel [--quick] [--max-n 16384]

Memory: the driver keeps a dense N x N working matrix (2 GB real at N = 16384). Cases run one at a time
with overwrite_a=True, so A itself is the working matrix. The residual is then recomputed row block by row
block from the kernel. A case that runs out of memory is skipped and recorded as skipped in the JSON.
"""
from __future__ import annotations
import argparse
import gc
import pathlib
import time
from typing import Callable, NotRequired, Sequence, TypedDict

import numpy as np
import matplotlib.pyplot as plt

from unified.core import Tree, Weak, Strong, ProxyCompressor
from unified.core.types import Admissibility, Vector
from unified.bench import problems as P, adapters as ad, common as C

type RuleName = str                                  # "weak" | "strong": a key of RULES
type ProblemBuilder = Callable[[int], P.Problem]     # N -> Problem (P1, P2, or P3 at fixed k)

RULES: dict[RuleName, Callable[[], Admissibility]] = {"weak": Weak, "strong": Strong}
PROBLEMS: dict[str, ProblemBuilder] = {"P1": P.p1_curve, "P2": P.p2_area}


class RSkelRow(TypedDict):
    """One factorization of results/rskel_*.json. [computed]"""
    problem: str
    rule: RuleName
    N: int
    eps: float
    depth: int
    factor_s: float
    solve_s: float
    root: int
    skel_max_per_level: dict[int, int]
    skel_mean_per_level: dict[int, float]
    residual: np.floating             # relres_kernel: ||A x - b|| / ||b||
    k: NotRequired[float]             # Helmholtz runs only: wavenumber


class SkippedCase(TypedDict):
    problem: str
    N: int
    rule: RuleName
    reason: str


class SlopeFit(TypedDict, total=False):
    """Least-squares slope of log(factor time) vs log N."""
    all_N: float
    N_range: list[int]
    largest_3_N: float


def factor_once(make: ProblemBuilder, n: int, rule: RuleName, eps: float, max_leaf: int = 32) -> RSkelRow:
    """Build the problem, build the tree, factor with the proxy compressor (in place), and solve one
    random right-hand side. The residual is computed from the kernel because A was overwritten."""
    prob = make(n)
    tree = Tree(prob.X, max_leaf=max_leaf)
    comp = ProxyCompressor(prob.kernel, prob.X, eps, A=prob.A)
    t0 = time.perf_counter()
    F = ad.rskel().rskel_factor(prob.A, tree, RULES[rule](), comp, overwrite_a=True)
    t_factor = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    b: Vector = rng.standard_normal(n) + (1j * rng.standard_normal(n) if np.iscomplexobj(prob.A) else 0)
    t0 = time.perf_counter(); x: Vector = F.solve(b); t_solve = time.perf_counter() - t0
    lv = F.stats["levels"]
    out = RSkelRow(problem=prob.name, rule=rule, N=n, eps=eps, depth=tree.depth, factor_s=t_factor,
               solve_s=t_solve, root=ad.root_size(F.stats),
               skel_max_per_level={int(l): int(v["skel_max"]) for l, v in lv.items()},
               skel_mean_per_level={int(l): float(v["skel_mean"]) for l, v in lv.items()},
               residual=P.relres_kernel(prob, x, b))
    del F, prob, comp
    gc.collect()
    return out


def scaling(sizes: Sequence[int], eps: float) -> tuple[list[RSkelRow], list[SkippedCase]]:
    rows: list[RSkelRow] = []
    skipped: list[SkippedCase] = []
    for tag, make in PROBLEMS.items():
        for n in sizes:
            for rule in RULES:
                try:
                    r = factor_once(make, n, rule, eps)
                except MemoryError as e:
                    skipped.append(SkippedCase(problem=tag, N=n, rule=rule, reason=f"MemoryError: {e}"))
                    print(f"{tag} N={n} {rule}: SKIPPED (out of memory)", flush=True)
                    gc.collect()
                    continue
                rows.append(r)
                sk = dict(sorted(r["skel_max_per_level"].items()))
                print(f"{r['problem']:14s} {rule:6s} depth {r['depth']} factor {r['factor_s']:7.2f}s "
                      f"solve {r['solve_s']*1e3:7.1f}ms root {r['root']:5d} skel_max/level {sk} "
                      f"residual {r['residual']:.1e}", flush=True)
    return rows, skipped


def slopes(rows: list[RSkelRow]) -> dict[str, SlopeFit]:
    """Least-squares slope of log(factor time) vs log N, per problem and rule: over all N and over the
    largest three N (closest to the asymptotic regime)."""
    out: dict[str, SlopeFit] = {}
    for tag in PROBLEMS:
        for rule in RULES:
            rs = sorted((r for r in rows if r["problem"].startswith(tag) and r["rule"] == rule), key=lambda r: r["N"])
            if len(rs) < 2:
                continue
            N, t = np.log([r["N"] for r in rs]), np.log([r["factor_s"] for r in rs])
            e = SlopeFit(all_N=float(np.polyfit(N, t, 1)[0]), N_range=[rs[0]["N"], rs[-1]["N"]])
            if len(rs) >= 3:
                e["largest_3_N"] = float(np.polyfit(N[-3:], t[-3:], 1)[0])
            out[f"{tag}/{rule}"] = e
    return out


def plot_scaling(rows: list[RSkelRow], slope: dict[str, SlopeFit]) -> pathlib.Path:
    """Rows: P1, P2. Columns: factor time, solve time, max skeleton at level 2, max skeleton at the
    finest two levels. Colour = rule (weak blue, strong orange); solid = finest level, dashed = next."""
    C.style()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    notes: list[str] = []
    for i, (tag, title) in enumerate([("P1", "P1 curve (Laplace, starfish)"), ("P2", "P2 area (Laplace, unit square)")]):
        for s, rule in enumerate(RULES):
            rs = sorted((r for r in rows if r["problem"].startswith(tag) and r["rule"] == rule), key=lambda r: r["N"])
            if not rs:
                continue
            N = [r["N"] for r in rs]
            sl = slope.get(f"{tag}/{rule}", {}).get("all_N", float("nan"))
            C.line(axes[i, 0], N, [r["factor_s"] for r in rs], s, f"{rule} (slope {sl:.2f})", direct=False)
            C.line(axes[i, 1], N, [r["solve_s"] for r in rs], s, rule, direct=False)
            C.line(axes[i, 2], N, [r["skel_max_per_level"].get(2, np.nan) for r in rs], s, rule, direct=False)
            ax = axes[i, 3]
            ax.plot(N, [r["skel_max_per_level"][r["depth"]] for r in rs], color=C.SERIES[s],
                    marker=C.MARKERS[s], label=f"{rule}, finest level")
            ax.plot(N, [r["skel_max_per_level"].get(r["depth"] - 1, np.nan) for r in rs], color=C.SERIES[s],
                    marker=C.MARKERS[s], ls="--", mfc="none", label=f"{rule}, finest − 1")
            notes.append(f"{tag} {rule}: root {rs[0]['root']}→{rs[-1]['root']}")
        heads = ["factor time [s]", "solve time [s]", "max skeleton per box, level 2",
                 "max skeleton per box, finest two levels"]
        for j, h in enumerate(heads):
            ax = axes[i, j]
            ax.set_xscale("log", base=2)
            if j < 2:
                ax.set_yscale("log")
            ax.set(xlabel="N", ylabel=h.split(",")[0] if j >= 2 else h, title=f"{title}\n{h}" if j == 0 else h)
            ax.legend(loc="best", fontsize=8)
    Ns = sorted({r["N"] for r in rows})
    fig.text(0.01, 0.005, f"Root size (secondary; N = {Ns[0]} → {Ns[-1]}): " + ";  ".join(notes) +
             ".  Under Strong in 2D the root holds all level-2 skeletons by construction.",
             fontsize=8, color=C.INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return C.save(fig, "rskel_scaling")


def helmholtz(n: int, ks: Sequence[int], eps: float) -> list[RSkelRow]:
    rows: list[RSkelRow] = []
    for k in ks:
        for rule in RULES:
            r = factor_once(lambda m: P.p3_helmholtz(m, k), n, rule, eps)
            r["k"] = k
            rows.append(r)
            print(f"P3 k={k:3d} {rule:6s} root {r['root']:5d} skel_max/level "
                  f"{dict(sorted(r['skel_max_per_level'].items()))} residual {r['residual']:.1e}", flush=True)
    return rows


def plot_helmholtz(rows: list[RSkelRow], ks: Sequence[int]) -> pathlib.Path:
    C.style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for s, k in enumerate(ks):
        for ax, rule in zip(axes[:2], RULES):
            r = next(r for r in rows if r["k"] == k and r["rule"] == rule)
            lv = sorted(r["skel_max_per_level"])
            C.line(ax, lv, [r["skel_max_per_level"][l] for l in lv], s, f"k={k}", direct=False)
    for ax, rule in zip(axes[:2], RULES):
        ax.set(xlabel="tree level (larger = smaller boxes)", ylabel="max skeleton per box",
               title=f"P3 Helmholtz, {rule}: rank per level")
        ax.set_xticks(sorted(rows[0]["skel_max_per_level"]))
        ax.legend(loc="upper right")
    for s, rule in enumerate(RULES):
        rs = [r for r in rows if r["rule"] == rule]
        C.line(axes[2], [r["k"] for r in rs], [r["skel_max_per_level"][2] for r in rs], s, rule, direct=False)
    axes[2].set(xlabel="wavenumber k", ylabel="max skeleton per box, level 2",
                title=f"P3 level-2 rank vs k (N = {rows[0]['N']})")
    axes[2].legend(loc="upper left")
    return C.save(fig, "rskel_helmholtz")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N up to 2048 only")
    ap.add_argument("--max-n", type=int, default=16384)
    a = ap.parse_args(argv)
    eps = 1e-8
    sizes = [n for n in ([512, 1024, 2048] if a.quick else [1024, 2048, 4096, 8192, 16384]) if n <= a.max_n]
    rows, skipped = scaling(sizes, eps)
    slope = slopes(rows)
    for key, v in slope.items():
        print(f"factor-time slope {key}: {v}")
    print(plot_scaling(rows, slope),
          C.dump("rskel_scaling", {"eps": eps, "compressor": "proxy", "max_leaf": 32, "overwrite_a": True,
                                   "factor_time_loglog_slope": slope, "skipped": skipped, "rows": rows}))
    ks, n = [5, 20, 50], (1024 if a.quick else 2048)
    hrows = helmholtz(n, ks, eps)
    print(plot_helmholtz(hrows, ks), C.dump("rskel_helmholtz", {"eps": eps, "compressor": "proxy", "rows": hrows}))


if __name__ == "__main__":
    main()
