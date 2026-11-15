"""E2 runner: reproduces every number and figure in findings.md (results/*.json, results/*.png).

    .venv/bin/python unified/explore/e2_fft_replacements/run.py            # full (~5 min), N up to 4096
    .venv/bin/python unified/explore/e2_fft_replacements/run.py --quick    # N <= 1024
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = pathlib.Path(__file__).resolve().parent
# (this directory is imported as the package unified.explore.e2_fft_replacements)
sys.path.insert(0, str(HERE.parents[2]))    # repository root, for `unified`
from unified.explore.e2_fft_replacements.fftswap import (MEDIA, medium_range, bitrev, permuted_kernel,  # noqa: E402
                                         LowRankPhaseFFT, fft_cost, lowrank_cost, b1_kernel, exact_apply,
                                         time_it, Permutation)
from unified.explore.e2_fft_replacements._deps import (butterfly_id, load_svd_reference, ButterflyFactorization,  # noqa: E402
                                       SparseFactor)
from unified.core.kernels import ButterflyKernel  # noqa: E402
from unified.core.types import ComplexArray, Field, FrequencyGrid, PositionGrid, Spectrum  # noqa: E402

type Row = dict[str, Any]          # one JSON record of results
type Ordering = str                # "natural", "bitrev both", ...

RES = HERE / "results"
EPS = 1e-6
COLORS: dict[str, str] = {"constant": "#2a78d6", "weak": "#eb6834", "sine": "#1baf7a", "lens": "#eda100"}  # fixed order
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "lines.linewidth": 2})


def rel(a: Field, b: Field) -> float:
    """Relative 2-norm error ||a - b|| / ||b||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def grids(N: int) -> tuple[PositionGrid, FrequencyGrid]:
    """B1 grids: x = j/N, xi = k - N/2."""
    return np.arange(N) / N, (np.arange(N) - N // 2).astype(float)


def rand_g(N: int, seed: int = 0) -> Spectrum:
    """Random complex Gaussian input g(xi)."""
    r = np.random.default_rng(seed)
    g: ComplexArray = r.standard_normal(N) + 1j * r.standard_normal(N)
    return g


def id_bf_ordered(kernel: ButterflyKernel, N: int, prow: Permutation, pcol: Permutation, eps: float = EPS
                  ) -> tuple[ButterflyFactorization, Callable[[Spectrum], Field]]:
    """ID butterfly of K[prow][:, pcol] via the virtual-coordinate wrapper; returns (bf, apply(g))."""
    x, xi = grids(N)
    u = np.arange(N, dtype=float)
    bf = butterfly_id(permuted_kernel(kernel, x, xi, prow, pcol), u, u, eps)

    def apply(g: Spectrum) -> Field:
        f: ComplexArray = np.empty(N, complex)
        f[prow] = bf.apply(g[pcol])
        return f
    return bf, apply


def svd_nnz(W: SparseFactor, G: list[SparseFactor], U: SparseFactor) -> int:
    """Stored nonzeros of the SVD reference butterfly's factors."""
    return int(W.nnz + sum(g.nnz for g in G) + U.nnz)


# ---- part (a) ---------------------------------------------------------------------------------------
def part_a(Ns: list[int]) -> list[Row]:
    """(a) Natural vs bit-reversed orderings: ranks, nnz and error of the ID (and SVD) butterfly."""
    ref = load_svd_reference()
    rows: list[Row] = []
    for N in Ns:
        x, xi = grids(N)
        I, br = np.arange(N), bitrev(N)
        g = rand_g(N)
        orders: dict[Ordering, tuple[Permutation, Permutation]] = {"natural": (I, I), "bitrev both": (br, br)}
        if N == 1024:
            orders.update({"bitrev rows only": (br, I), "bitrev cols only": (I, br)})
        for t in (0.0, 1.0):
            if t > 0 and N > 1024:
                continue
            k = b1_kernel(t, "sine")
            f = exact_apply(k, x, xi, g)
            for name, (pr, pc) in orders.items():
                bf, ap = id_bf_ordered(k, N, pr, pc)
                twiddle: float | None = None
                if name == "bitrev both" and t == 0:   # all G_l entries should be unimodular twiddles
                    vals = np.concatenate([Gl.data for Gl in bf.G])
                    twiddle = float(np.max(np.abs(np.abs(vals) - 1)))
                row: Row = {"N": N, "t": t, "ordering": name, "id_ranks": bf.ranks, "id_nnz": bf.nnz,
                       "id_err": rel(ap(g), f), "fft_nnz": fft_cost(N), "max_abs_twiddle_dev": twiddle}
                if N <= 1024:
                    Kd = k(x, xi)[np.ix_(pr, pc)]
                    W, G, U, r = ref.butterfly_factorize(Kd, eps=EPS)
                    fs: ComplexArray = np.empty(N, complex); fs[pr] = ref.butterfly_apply(W, G, U, g[pc])
                    row.update(svd_ranks=r, svd_nnz=svd_nnz(W, G, U), svd_err=rel(fs, f))
                rows.append(row)
                print(f"(a) N={N:5d} t={t} {name:17s} ID ranks max {max(bf.ranks):3d} nnz {bf.nnz:9,d} "
                      f"(FFT {fft_cost(N):,d}) err {row['id_err']:.1e}"
                      + (f" | SVD nnz {row['svd_nnz']:,d} err {row['svd_err']:.1e}" if "svd_nnz" in row else ""))
    return rows


# ---- part (b) ---------------------------------------------------------------------------------------
TS: list[float] = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def part_b(Ns: list[int], svd_maxN: int) -> list[Row]:
    """(b) R FFTs vs butterfly over (N, medium, t): rank, error, cost, storage, wall time."""
    ref = load_svd_reference()
    rows: list[Row] = []
    for N in Ns:
        x, xi = grids(N)
        g = rand_g(N)
        for med in MEDIA:
            dc = medium_range(med)
            for t in TS:
                k = b1_kernel(t, med)
                f = exact_apply(k, x, xi, g)
                lr = LowRankPhaseFFT(N, t, MEDIA[med], EPS)
                bf = butterfly_id(k, x, xi, EPS)
                row: Row = {"N": N, "medium": med, "t": t, "dc": dc, "B": t * dc * N / 2,
                       "R": lr.R, "lr_err": rel(lr.apply(g), f), "lr_storage": lr.storage,
                       "lr_cost": lowrank_cost(N, lr.R), "lr_time": time_it(lambda: lr.apply(g)),
                       "lr_build": lr.build_time,
                       "id_ranks": bf.ranks, "id_nnz": bf.nnz, "id_err": rel(bf.apply(g), f),
                       "id_time": time_it(lambda: bf.apply(g)), "id_build": bf.build_time}
                if N <= svd_maxN or (med == "sine" and t == 1.0):
                    W, G, U, r = ref.butterfly_factorize(k(x, xi), eps=EPS)
                    row.update(svd_ranks=r, svd_nnz=svd_nnz(W, G, U),
                               svd_err=rel(ref.butterfly_apply(W, G, U, g), f))
                row["winner_cost"] = "R-FFT" if row["lr_cost"] < row["id_nnz"] else "butterfly"
                rows.append(row)
                print(f"(b) N={N:5d} {med:8s} t={t:<5} B={row['B']:7.2f} R={lr.R:4d} lr_err {row['lr_err']:.1e} "
                      f"cost R-FFT {row['lr_cost']:11,d} vs ID-BF {bf.nnz:10,d} (err {row['id_err']:.1e}) "
                      f"-> {row['winner_cost']}")
    return rows


METRICS: dict[str, tuple[str, str]] = {      # metric -> (R-FFT field, butterfly field)
    "cost": ("lr_cost", "id_nnz"),            # complex multiply-adds per apply (model)
    "storage": ("lr_storage", "id_nnz"),      # numbers stored
    "time": ("lr_time", "id_time"),           # measured apply wall time (numpy FFT vs scipy.sparse)
}


def _first_crossing(ts: list[float], ratio: list[float]) -> float | None:
    """Smallest t where ratio (R-FFT / butterfly) crosses 1 upward, log-log interpolated; None if never."""
    for ta, tb, ra, rb in zip(ts, ts[1:], ratio, ratio[1:]):
        if ra < 1 <= rb:
            la, lb = np.log(max(ta, 1e-4)), np.log(tb)
            return float(np.exp(la + (lb - la) * (0 - np.log(ra)) / (np.log(rb) - np.log(ra))))
    return None


def crossovers(rows: list[Row]) -> list[Row]:
    """Per (medium, N) and metric: the t* where R-FFT stops beating the ID butterfly, B* = t* dc N / 2,
    and (cost) the break-even number of FFTs R* = butterfly nnz / (FFT cost + 2N)."""
    out: list[Row] = []
    for med in MEDIA:
        for N in sorted({r["N"] for r in rows}):
            rs = sorted([r for r in rows if r["medium"] == med and r["N"] == N], key=lambda r: r["t"])
            ts = [float(r["t"]) for r in rs]
            row: Row = {"medium": med, "N": N,
                        "R_star_breakeven": float(np.mean([r["id_nnz"] for r in rs]) / (fft_cost(N) + 2 * N))}
            for m, (lf, bf) in METRICS.items():
                ratio = [r[lf] / r[bf] for r in rs]
                tstar = _first_crossing(ts, ratio)
                row[f"{m}_t_star"] = tstar
                row[f"{m}_B_star"] = None if tstar is None else tstar * rs[0]["dc"] * N / 2
                row[f"{m}_R_FFT_wins_all_t"] = all(x < 1 for x in ratio)
                row[f"{m}_ratio_at_t"] = dict(zip(ts, ratio))
            out.append(row)
    return out


def fmt(v: float | None) -> str:
    return "never" if v is None else f"{v:.3g}"


# ---- figures ----------------------------------------------------------------------------------------
def _n_axis(ax: Any, Ns: list[int]) -> None:
    """Log N axis labelled only at the N values we ran (no overlapping minor labels)."""
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_a(rows: list[Row]) -> None:
    """nnz vs N: natural vs bit-reversed ID butterfly at t = 0, against the radix-2 FFT count."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, col, mk in [("natural", "#2a78d6", "o"), ("bitrev both", "#eb6834", "s")]:
        rs = [r for r in rows if r["ordering"] == name and r["t"] == 0]
        N = [r["N"] for r in rs]
        ax.loglog(N, [r["id_nnz"] for r in rs], marker=mk, ms=8, color=col, label=f"ID butterfly, {name}")
    Ns = sorted({r["N"] for r in rows})
    ax.loglog(Ns, [fft_cost(n) for n in Ns], ls="--", color=MUTED, label=r"$2N\log_2N+2N$ (radix-2 FFT)")
    _n_axis(ax, Ns)
    ax.set_xlabel("N"); ax.set_ylabel("stored nonzeros (nnz)")
    ax.set_title("(a) DFT (t = 0): bit-reversed tiles are rank 1", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RES / "fig_a_nnz.png", dpi=150); plt.close(fig)


def fig_R(rows: list[Row]) -> None:
    """R vs B for every (N, medium, t > 0): the rank depends on B = t dc N / 2 only."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for med in ["weak", "sine", "lens"]:
        rs = [r for r in rows if r["medium"] == med and r["t"] > 0]
        ax.loglog([r["B"] for r in rs], [r["R"] for r in rs], "o", ms=8, color=COLORS[med], label=med,
                  markeredgecolor="white", markeredgewidth=1)
    B = np.logspace(0.5, 3, 50)
    ax.loglog(B, B, ls="--", color=MUTED, lw=1, label="R = B")
    ax.set_xlabel(r"$B = t\,\Delta c\,N/2$  (phase-perturbation bandwidth)")
    ax.set_ylabel(r"$\varepsilon$-rank R of $e^{2\pi i t c(x)|\xi|}$")
    ax.set_title(r"(b) $\varepsilon$-rank vs B ($\varepsilon=10^{-6}$; constant c: R = 1)", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RES / "fig_b_rank.png", dpi=150); plt.close(fig)


def fig_cost(rows: list[Row]) -> None:
    """Apply-cost ratio R-FFT / butterfly vs t (largest N) and vs N (t = 0.01, 1)."""
    Ns = sorted({r["N"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    N = Ns[-1]
    ax = axs[0]
    for med in MEDIA:
        rs = sorted([r for r in rows if r["medium"] == med and r["N"] == N and r["t"] > 0], key=lambda r: r["t"])
        ax.loglog([r["t"] for r in rs], [r["lr_cost"] / r["id_nnz"] for r in rs], marker="o", ms=7,
                  color=COLORS[med], label=med)
    ax.axhline(1, color=INK, lw=1)
    ax.text(0.0032, 1.15, "R-FFT costs more above this line", color=MUTED, fontsize=8)
    ax.set_xlabel("t"); ax.set_ylabel("apply cost: R-FFT / ID butterfly nnz")
    ax.set_title(f"cost ratio vs t, N = {N}", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax = axs[1]
    for med in ["weak", "sine", "lens"]:
        rs = sorted([r for r in rows if r["medium"] == med and r["t"] == 1.0], key=lambda r: r["N"])
        ax.loglog([r["N"] for r in rs], [r["lr_cost"] / r["id_nnz"] for r in rs], marker="o", ms=7,
                  color=COLORS[med], label=f"{med}, t = 1")
        rs = sorted([r for r in rows if r["medium"] == med and r["t"] == 0.01], key=lambda r: r["N"])
        ax.loglog([r["N"] for r in rs], [r["lr_cost"] / r["id_nnz"] for r in rs], marker="s", ms=7, ls=":",
                  color=COLORS[med], label=f"{med}, t = 0.01")
    ax.axhline(1, color=INK, lw=1)
    _n_axis(ax, Ns)
    ax.set_xlabel("N"); ax.set_title("cost ratio vs N (sine and lens overlap)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(RES / "fig_b_cost.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N <= 1024")
    ap.add_argument("--reuse", action="store_true",
                    help="reload results/partA.json, partB.json and redo only crossovers + figures")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    tag = "_quick" if a.quick else ""
    if a.reuse:
        ra = json.loads((RES / f"partA{tag}.json").read_text())
        rb = json.loads((RES / f"partB{tag}.json").read_text())
    else:
        Ns_a = [256, 1024] if a.quick else [256, 1024, 4096]
        Ns_b = [256, 512, 1024] if a.quick else [256, 512, 1024, 2048, 4096]
        ra = part_a(Ns_a)
        rb = part_b(Ns_b, svd_maxN=1024)
    cx = crossovers(rb)
    for c in cx:
        print(f"crossover {c['medium']:8s} N={c['N']:5d}: break-even R = {c['R_star_breakeven']:5.1f} | "
              + " | ".join(f"{m}: t* {fmt(c[m + '_t_star'])}, B* {fmt(c[m + '_B_star'])}" for m in METRICS))
    for name, obj in [("partA", ra), ("partB", rb), ("crossover", cx)]:
        (RES / f"{name}{tag}.json").write_text(json.dumps(obj, indent=1))
    fig_a(ra); fig_R(rb); fig_cost(rb)


if __name__ == "__main__":
    main()
