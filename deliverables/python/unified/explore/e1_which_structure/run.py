"""E1 'which structure for which kernel': storage of weak (HODLR) / strong (H) / complementary (butterfly)
compression of 12 kernels as N grows, plus the natural-vs-bit-reversed ordering test for DFT / FIO.

    .venv/bin/python -m unified.explore.e1_which_structure.run            # full (N = 256..4096), ~5 min
    .venv/bin/python -m unified.explore.e1_which_structure.run --quick    # N <= 1024

Writes results/sweep.json, results/ordering.json, results/*.png.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np

from unified.core import Tree, Weak, Strong, kernels
from unified.core.types import RealArray, FrequencyGrid, Kernel, KernelMatrix, Points, PositionGrid
from .structures import (Ordering, Report, StorageCount, hmatrix_storage, butterfly_storage, order_tree,
                         tile_ranks, bf_storage)

# ---- domain aliases ---------------------------------------------------------------------------------------
type Wavenumber = float
type Bandwidth = float                                       # Gaussian RBF width h
type Exponent = float                                        # alpha in storage ~ N^alpha
type Structure = str                                         # "weak" | "strong" | "complementary"
type RowPoints = Points | PositionGrid
type ColPoints = Points | FrequencyGrid
type Instance = tuple[KernelMatrix, RowPoints, ColPoints]    # dense K plus the row / column points
type Builder = Callable[[int], Instance]                     # N -> instance
type Problem = tuple[int, str, Builder]                      # (dimension, family, builder)
type Summary = dict[str, Any]

EPS = 1e-6
HERE = pathlib.Path(__file__).parent
OUT = HERE / "results"
NS: list[int] = [256, 512, 1024, 2048, 4096]
STRUCTS: list[Structure] = ["weak", "strong", "complementary"]
LEAF_H: tuple[int, ...] = (16, 64)           # leaf sizes tried for weak / strong (best storage kept)
LEAF_BF: dict[int, tuple[int, ...]] = {1: (1, 8), 2: (4, 16)}   # leaf sizes tried for the butterfly, per dimension


def starfish(n: int) -> tuple[Points, RealArray]:
    """Points and trapezoid weights on the starfish r(θ) = 1 + 0.3 cos 5θ. Same as
    unified.bench.problems.starfish (re-stated here so this directory type-checks on its own)."""
    th: RealArray = 2 * np.pi * np.arange(n) / n
    r: RealArray = 1 + 0.3 * np.cos(5 * th)
    X: Points = np.c_[r * np.cos(th), r * np.sin(th)]
    w: RealArray = np.sqrt(r ** 2 + (1.5 * np.sin(5 * th)) ** 2) * 2 * np.pi / n
    return X, w


def starfish_perimeter(n: int = 1 << 14) -> float:
    """Length of the starfish curve (trapezoid rule)."""
    X, w = starfish(n)
    return float(np.sum(w))


PERIM = starfish_perimeter()


def n_min_ppw(k: Wavenumber, ppw: float = 10) -> int:
    """Points needed on the starfish for ppw points per wavelength at wavenumber k."""
    return int(np.ceil(ppw * k * PERIM / (2 * np.pi)))


# ---- problems: name -> (dim, family, builder(N) -> (K, row points, col points)) -------------------------
def curve(N: int) -> Points:
    """N points on the starfish."""
    X: Points
    X, _ = starfish(N)
    return X


def area(N: int) -> Points:
    """N uniform random points in the unit square (seed 0, as in P2)."""
    X: Points = np.random.default_rng(0).random((N, 2))
    return X


def _self(kern: Callable[[int], Kernel], pts: Callable[[int], Points]) -> Builder:
    """Self-interaction K(X, X) on a 2D point set."""
    def build(N: int) -> Instance:
        X = pts(N)
        return kern(N)(X, X), X, X
    return build


def _fio(t: float) -> Builder:
    """1D FIO fio1d(t) on the B1 grids (t = 0: shifted inverse DFT)."""
    def build(N: int) -> Instance:
        x: PositionGrid = np.arange(N) / N
        xi: FrequencyGrid = np.arange(N, dtype=float) - N // 2
        return kernels.fio1d(t)(x, xi), x, xi
    return build


def gauss(h: Bandwidth) -> Kernel:
    """Gaussian RBF exp(-|x - y|^2 / h^2)."""
    def K(X: Points, Y: Points) -> KernelMatrix:
        out: KernelMatrix = np.exp(-((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / h ** 2)
        return out
    return K


PROBLEMS: dict[str, Problem] = {
    "Laplace curve":        (2, "smooth", _self(lambda N: kernels.laplace2d, curve)),
    "Laplace area":         (2, "smooth", _self(lambda N: kernels.laplace2d, area)),
    "Helmholtz k=1":        (2, "smooth", _self(lambda N: kernels.helmholtz2d(1.0), curve)),
    "Helmholtz k=10":       (2, "osc-fixed-k", _self(lambda N: kernels.helmholtz2d(10.0), curve)),
    "Helmholtz k=50":       (2, "osc-fixed-k", _self(lambda N: kernels.helmholtz2d(50.0), curve)),
    "Helmholtz k=150":      (2, "osc-fixed-k", _self(lambda N: kernels.helmholtz2d(150.0), curve)),
    "Helmholtz 10 ppw":     (2, "osc-scaling", _self(lambda N: kernels.helmholtz2d(2 * np.pi * N / (10 * PERIM)), curve)),
    "DFT (FIO t=0)":        (1, "osc-scaling", _fio(0.0)),
    "FIO t=1":              (1, "osc-scaling", _fio(1.0)),
    "FIO t=3":              (1, "osc-scaling", _fio(3.0)),
    "Gauss h=0.5 (area)":   (2, "smooth", _self(lambda N: gauss(0.5), area)),
    "Gauss h=0.05 (area)":  (2, "smooth", _self(lambda N: gauss(0.05), area)),
}
FIXED_K: dict[str, Wavenumber] = {"Helmholtz k=50": 50.0, "Helmholtz k=150": 150.0}


def sizes(name: str, Ns: list[int]) -> list[int]:
    """N values for a problem: fixed-k Helmholtz needs >= 10 points per wavelength (at least the
    smallest N >= that bound, so k=150 still gets measured)."""
    if name in FIXED_K:
        nmin = n_min_ppw(FIXED_K[name])
        ok = [N for N in Ns if N >= nmin]
        return ok or [Ns[-1]]
    return Ns


def trees(dim: int, rpts: RowPoints, cpts: ColPoints, leaf: int) -> tuple[Tree, Tree]:
    """Row and column trees with ~leaf points per leaf. 1D: binary trees on the natural order (depth
    log2(N/leaf)); 2D: one quadtree shared by rows and columns."""
    if dim == 1:
        N = len(rpts)
        L = int(round(np.log2(N // leaf)))
        return order_tree(N, L), order_tree(len(cpts), L)
    T = Tree(rpts, max_leaf=leaf)
    return T, T


def job(name: str, N: int) -> Report:
    """All three structures for one (kernel, N); for each, the best over the leaf sizes tried."""
    dim, fam, build = PROBLEMS[name]
    t0 = time.perf_counter()
    K, rp, cp = build(N)
    out: dict[Structure, Report] = {}
    for st in STRUCTS:
        best: Report | None = None
        for leaf in (LEAF_BF[dim] if st == "complementary" else LEAF_H):
            rt, ct = trees(dim, rp, cp, leaf)
            if st == "complementary":
                r = butterfly_storage(K, rt, ct, EPS)
            else:
                r = hmatrix_storage(K, rt, ct, Weak() if st == "weak" else Strong(), EPS)
            r.update(leaf=leaf, depth=rt.depth)
            if best is None or r["storage"] < best["storage"]:
                best = r
        assert best is not None
        out[st] = best
    return dict(kernel=name, N=N, family=fam, dim=dim, results=out, seconds=time.perf_counter() - t0)


def ordering_job(t: float, N: int, ro: Ordering, co: Ordering) -> Report:
    """Complementary tiles of fio1d(t) at leaf size 1 with rows / cols in natural or bit-reversed order."""
    x: PositionGrid = np.arange(N) / N
    xi: FrequencyGrid = np.arange(N, dtype=float) - N // 2
    K: KernelMatrix = kernels.fio1d(t)(x, xi)
    L = int(round(np.log2(N)))
    rt, ct = order_tree(N, L, ro), order_tree(N, L, co)
    R = tile_ranks(K, rt, ct, EPS)
    res: Report = dict(t=t, N=N, rows=ro, cols=co, max_rank_per_stage=[max(Rl.values()) for Rl in R],
               max_rank=max(max(Rl.values()) for Rl in R), bf_storage=bf_storage(R, rt, ct))
    rule: Weak | Strong
    for name, rule in (("weak", Weak()), ("strong", Strong())):   # same orders, H block tree, leaf 16
        r16, c16 = order_tree(N, L - 4, ro), order_tree(N, L - 4, co)
        h = hmatrix_storage(K, r16, c16, rule, EPS)
        res[name] = dict(storage=h["storage"], max_rank=h["max_rank"])
    return res


def fit_exponent(Ns: list[int], vals: list[StorageCount]) -> Exponent | None:
    """Least-squares slope of log(storage) vs log(N)."""
    if len(Ns) < 2:
        return None
    return float(np.polyfit(np.log(Ns), np.log(vals), 1)[0])


def summarize(rows: list[Report]) -> dict[str, Summary]:
    """Per kernel: storage/N at the largest N, growth exponent per structure, best structure."""
    summ: dict[str, Summary] = {}
    for name in PROBLEMS:
        rr = sorted([r for r in rows if r["kernel"] == name], key=lambda r: r["N"])
        if not rr:
            continue
        Ns = [r["N"] for r in rr]
        last = rr[-1]
        s: Summary = dict(N_max=last["N"], Ns=Ns, family=last["family"])
        for st in STRUCTS:
            s[st] = dict(storage_per_N=last["results"][st]["storage"] / last["N"],
                         max_rank=last["results"][st]["max_rank"],
                         err=last["results"][st]["err"],
                         exponent=fit_exponent(Ns, [r["results"][st]["storage"] for r in rr]),
                         max_rank_series=[r["results"][st]["max_rank"] for r in rr],
                         storage_series=[r["results"][st]["storage"] for r in rr])
        s["best"] = min(STRUCTS, key=lambda st: s[st]["storage_per_N"])
        summ[name] = s
    return summ


# ---- figures -----------------------------------------------------------------------------------------------
def plot(summ: dict[str, Summary], ordering: list[Report]) -> None:
    """heatmap.png, storage_vs_N.png, ordering.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summ)
    # 1. heatmap: log10(storage / best storage) at N_max, annotated with storage/N and exponent
    M = np.array([[np.log10(summ[n][st]["storage_per_N"] / summ[n][summ[n]["best"]]["storage_per_N"])
                   for st in STRUCTS] for n in names])
    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(names) + 1.6))
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=max(1.5, M.max()), aspect="auto")
    for i, n in enumerate(names):
        for j, st in enumerate(STRUCTS):
            d = summ[n][st]
            e = d["exponent"]
            txt = f"{d['storage_per_N']:.0f}·N  r≤{d['max_rank']}\n" + (f"N^{e:.2f}" if e is not None else "(one N)")
            star = summ[n]["best"] == st
            ax.text(j, i, ("★ " if star else "") + txt, ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.75 * max(1.5, M.max()) else "black",
                    fontweight="bold" if star else "normal")
    ax.set_xticks(range(3), ["weak (HODLR)", "strong (H)", "complementary (butterfly)"])
    ax.set_yticks(range(len(names)), [f"{n}  (N={summ[n]['N_max']})" for n in names], fontsize=8)
    ax.set_title(f"Storage at ε={EPS:g}: shade = log10(storage / best); ★ = best; exponent fit over N", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10(storage / best)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=150)
    plt.close(fig)

    # 2. storage/N vs N small multiples
    multi = [n for n in names if len(summ[n]["Ns"]) >= 2]
    cols = 4
    rows_ = int(np.ceil(len(multi) / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(3.1 * cols, 2.5 * rows_), squeeze=False)
    sty: dict[Structure, tuple[str, str]] = {"weak": ("tab:blue", "o-"), "strong": ("tab:orange", "s-"), "complementary": ("tab:green", "^-")}
    for ax, n in zip(axs.flat, multi):
        for st in STRUCTS:
            c, m = sty[st]
            ax.loglog(summ[n]["Ns"], np.array(summ[n][st]["storage_series"]) / np.array(summ[n]["Ns"]), m,
                      color=c, ms=3, label=f"{st} N^{summ[n][st]['exponent']:.2f}")
        ax.set_title(n, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(multi):]:
        ax.axis("off")
    fig.supxlabel("N")
    fig.supylabel("storage / N  (numbers per unknown)")
    fig.tight_layout()
    fig.savefig(OUT / "storage_vs_N.png", dpi=150)
    plt.close(fig)

    # 3. ordering: max tile rank per stage, DFT and FIO, natural vs bit-reversed
    Nmax = max(o["N"] for o in ordering)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, t in zip(axs, sorted({o["t"] for o in ordering})):
        for o in ordering:
            if o["t"] == t and o["N"] == Nmax:
                ax.semilogy(o["max_rank_per_stage"], "o-", ms=3, label=f"rows {o['rows']}, cols {o['cols']}")
        ax.set_title(f"fio1d t={t:g}, N={Nmax}" + ("  (DFT)" if t == 0 else ""), fontsize=9)
        ax.set_xlabel("butterfly stage l")
    axs[0].set_ylabel("max tile ε-rank")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ordering.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    Ns = [N for N in NS if N <= (1024 if a.quick else 4096)]
    OUT.mkdir(exist_ok=True)
    jobs = [(n, N) for n in PROBLEMS for N in sizes(n, Ns)]
    jobs.sort(key=lambda j: -j[1])                          # big jobs first for load balance
    ojobs = [(t, N, ro, co) for t in (0.0, 1.0, 3.0) for N in (256, Ns[-1])
             for ro in ("natural", "bitrev") for co in ("natural", "bitrev")]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        ofuts = [ex.submit(ordering_job, *j) for j in ojobs]
        rows: list[Report] = []
        for f in futs:
            r = f.result()
            rows.append(r)
            print(f"{r['kernel']:22s} N={r['N']:5d} " + "  ".join(
                f"{st[:4]}: {r['results'][st]['storage'] / r['N']:7.0f}N r{r['results'][st]['max_rank']:4d}"
                f" e{r['results'][st]['err']:.0e}" for st in STRUCTS) + f"  ({r['seconds']:.0f}s)", flush=True)
        ordering = [f.result() for f in ofuts]
    summ = summarize(rows)
    meta = dict(eps=EPS, leaf_H=LEAF_H, leaf_BF=LEAF_BF, starfish_perimeter=PERIM, Ns=Ns,
                seconds=time.perf_counter() - t0)
    (OUT / "sweep.json").write_text(json.dumps(dict(meta=meta, rows=rows, summary=summ), indent=1))
    (OUT / "ordering.json").write_text(json.dumps(ordering, indent=1))
    print("\nbest structure / exponents (weak, strong, complementary):")
    for n, s in summ.items():
        print(f"  {n:22s} best={s['best']:13s} " + "  ".join(
            f"{st[:4]} {s[st]['exponent']:.2f}" if s[st]["exponent"] is not None else f"{st[:4]}  -  "
            for st in STRUCTS))
    print("\nordering (max tile rank, butterfly storage/N, weak/strong storage/N):")
    for o in ordering:
        print(f"  t={o['t']:g} N={o['N']:5d} rows={o['rows']:7s} cols={o['cols']:7s} r={o['max_rank']:4d} "
              f"BF={o['bf_storage'] / o['N']:7.0f}N  weak={o['weak']['storage'] / o['N']:6.0f}N "
              f"strong={o['strong']['storage'] / o['N']:6.0f}N")
    plot(summ, ordering)
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
