"""Reproduce the error/complexity comparison and results/comparison.png."""
import csv
import math
import timeit
import os
from pathlib import Path

# Avoid charging OpenBLAS thread startup and scheduling noise to tiny benchmarks.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

from butterfly import butterfly_dft, butterfly_fio, dense_dft, dense_fio, fft_dft

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def relative_errors(approximation, exact):
    residual = approximation - exact
    return (
        float(np.linalg.norm(residual) / np.linalg.norm(exact)),
        float(np.max(np.abs(residual)) / np.max(np.abs(exact))),
    )


def time_function(function, number, repeat=5):
    return min(timeit.repeat(function, number=number, repeat=repeat)) / number


def main():
    q_default = 8
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    # Accuracy against interpolation order, for one representative input.
    n_accuracy = 128
    rng = np.random.default_rng(19)
    f_accuracy = rng.normal(size=n_accuracy) + 1j * rng.normal(size=n_accuracy)
    exact_accuracy = fft_dft(f_accuracy)
    accuracy_rows = []
    for q in (3, 4, 5, 6, 8, 10, 12, 14):
        approximate = butterfly_dft(f_accuracy, q)
        l2, linf = relative_errors(approximate, exact_accuracy)
        accuracy_rows.append({"N": n_accuracy, "q": q, "relative_l2": l2, "relative_inf": linf})

    with (RESULTS / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=accuracy_rows[0].keys())
        writer.writeheader()
        writer.writerows(accuracy_rows)

    # Runtime and N-scaling.  Dense timing is limited to avoid large matrices.
    scaling_rows = []
    for n in sizes:
        rng = np.random.default_rng(100 + n)
        f = rng.normal(size=n) + 1j * rng.normal(size=n)
        exact = fft_dft(f)
        approximate = butterfly_dft(f, q_default)
        l2, linf = relative_errors(approximate, exact)

        butterfly_seconds = time_function(lambda: butterfly_dft(f, q_default), number=1)
        fft_seconds = time_function(lambda: fft_dft(f), number=max(20, 4096 // n))
        dense_seconds = ""
        if n <= 256:
            dense_dft(f)  # warm up the dense library path before timing it
            dense_seconds = time_function(lambda: dense_dft(f), number=1)
            np.testing.assert_allclose(exact, dense_dft(f), rtol=1e-11, atol=1e-11)

        levels = int(math.log2(n))
        scaling_rows.append({
            "N": n,
            "q": q_default,
            "relative_l2": l2,
            "relative_inf": linf,
            "butterfly_seconds": butterfly_seconds,
            "fft_seconds": fft_seconds,
            "dense_seconds": dense_seconds,
            "pair_count": n * (levels + 1),
            "transfer_scalar_products": n * q_default**2 * (2 * levels + 1),
        })
        print(
            f"N={n:4d}  error={l2:.3e}  "
            f"butterfly={butterfly_seconds:.4f}s  FFT={fft_seconds:.2e}s"
        )

    with (RESULTS / "scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scaling_rows[0].keys())
        writer.writeheader()
        writer.writerows(scaling_rows)

    # A genuinely non-DFT oscillatory kernel, compared only with its dense sum.
    non_dft_phase = lambda x, p, n: n * (x * p + 0.1 * x**2 * p**2)
    non_dft_f = (
        np.sin(2 * np.pi * np.arange(n_accuracy) / n_accuracy)
        + 0.3j * np.cos(4 * np.pi * np.arange(n_accuracy) / n_accuracy)
    )
    non_dft_exact = dense_fio(non_dft_f, non_dft_phase)
    non_dft_rows = []
    for q in (3, 4, 5, 6, 8, 10, 12, 14):
        approximate = butterfly_fio(non_dft_f, q, non_dft_phase)
        l2, linf = relative_errors(approximate, non_dft_exact)
        non_dft_rows.append({"N": n_accuracy, "q": q, "relative_l2": l2, "relative_inf": linf})
    with (RESULTS / "non_dft_accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=non_dft_rows[0].keys())
        writer.writeheader()
        writer.writerows(non_dft_rows)

    make_figure(accuracy_rows, scaling_rows)
    make_non_dft_figure(non_dft_f, non_dft_exact, non_dft_phase, non_dft_rows)
    make_summary(accuracy_rows, scaling_rows, non_dft_rows)


def make_figure(accuracy_rows, scaling_rows):
    n = 128
    p = np.arange(n) / n
    signal = (
        np.sin(2 * np.pi * 3 * p)
        + 0.35 * np.sin(2 * np.pi * 11 * p)
        + 0.6 * np.exp(-((p - 0.68) / 0.09) ** 2)
        + 0.25j * np.cos(2 * np.pi * 5 * p)
    )
    exact = fft_dft(signal)
    approximate = butterfly_dft(signal, q=10)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    colours = {
        "reference": "#355C7D",   # desaturated dark blue, for legibility
        "mint": "#82C9BB",
        "sky": "#77B5D9",
        "lavender": "#A8A1D9",
        "guide": "#8E9AAF",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(np.abs(exact), color=colours["reference"], lw=1.8,
            label=r"$N\,\mathrm{IFFT}(f)$ (FFT reference)")
    ax.plot(np.abs(approximate), "--", color=colours["mint"], lw=1.8,
            label=r"butterfly, $q=10$")
    ax.set(title=r"Magnitude of the transformed signal ($N=128$)",
           xlabel=r"frequency index $j$", ylabel=r"$|u_j|$")
    ax.legend()

    ax = axes[0, 1]
    size_errors = [row["relative_l2"] for row in scaling_rows]
    ax.semilogx([row["N"] for row in scaling_rows], size_errors,
                "o-", color=colours["sky"], lw=1.4)
    ax.set(title=r"Butterfly error versus problem size ($q=8$)",
           xlabel=r"$N$", ylabel=r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    ns = np.array([row["N"] for row in scaling_rows])
    bt = np.array([row["butterfly_seconds"] for row in scaling_rows])
    ft = np.array([row["fft_seconds"] for row in scaling_rows])
    ax.loglog(ns, bt, "o-", color=colours["sky"], label=r"butterfly, $q=8$")
    ax.loglog(ns, ft, "o-", color=colours["lavender"], label=r"$N\,\mathrm{IFFT}(f)$")
    dense_ns = [row["N"] for row in scaling_rows if row["dense_seconds"] != ""]
    dense_t = [row["dense_seconds"] for row in scaling_rows if row["dense_seconds"] != ""]
    ax.loglog(dense_ns, dense_t, "o-", color=colours["mint"],
              label=r"dense matvec ($N^2$ entries)")
    guide_n2 = bt[-1] * ns**2 / ns[-1] ** 2
    guide_nlogn = bt[-1] * ns * np.log2(ns) / (ns[-1] * np.log2(ns[-1]))
    ax.loglog(ns, guide_n2, ":", color=colours["guide"], label=r"guide $\propto N^2$")
    ax.loglog(ns, guide_nlogn, "-.", color=colours["reference"],
              label=r"guide $\propto N\log_2N$")
    ax.set(title=r"Measured wall time ($q=8$)", xlabel=r"$N$",
           ylabel=r"time $t$ (seconds)", xscale="log", yscale="log")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    qs = [row["q"] for row in accuracy_rows]
    errors = [row["relative_l2"] for row in accuracy_rows]
    ax.semilogy(qs, errors, "o-", color=colours["lavender"])
    ax.set(title=r"Butterfly error at $N=128$",
           xlabel=r"interpolation order $q$",
           ylabel=r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(r"Butterfly factorisation for $\Psi(x,p)=xp$: exact FFT reference",
                 fontsize=14)
    fig.savefig(RESULTS / "comparison.png", dpi=150)


def make_non_dft_figure(f, exact, phase, rows):
    colours = {"reference": "#355C7D", "mint": "#82C9BB", "lavender": "#A8A1D9"}
    approximate = butterfly_fio(f, 10, phase)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(np.abs(exact), color=colours["reference"], lw=1.8,
            label=r"dense kernel evaluation")
    ax.plot(np.abs(approximate), "--", color=colours["mint"], lw=1.8,
            label=r"butterfly, $q=10$")
    ax.set(title=r"Non-DFT output ($N=128$)", xlabel=r"output index $i$",
           ylabel=r"$|u_i|$")
    ax.legend()
    ax = axes[1]
    ax.semilogy([row["q"] for row in rows], [row["relative_l2"] for row in rows],
                "o-", color=colours["lavender"])
    ax.set(title=r"Error against the dense evaluation ($N=128$)",
           xlabel=r"interpolation order $q$", ylabel=r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(r"Oscillatory test kernel $\Psi(x,p)=xp+0.1x^2p^2$", fontsize=14)
    fig.savefig(RESULTS / "non_dft_comparison.png", dpi=150)


def make_summary(accuracy_rows, scaling_rows, non_dft_rows):
    lines = [
        "# Measured results",
        "",
        "All errors compare the butterfly output with the positive-sign FFT.",
        "Absolute timing values depend on the machine; rerun `python compare.py` to reproduce them.",
        "Timings use one BLAS thread and an explicit warm-up for the dense path.",
        "",
        "## Scaling at q=8",
        "",
        "| N | relative L2 error | butterfly (s) | FFT (s) | dense matvec (s) | pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scaling_rows:
        dense = f"{row['dense_seconds']:.3e}" if row["dense_seconds"] != "" else "not run"
        lines.append(
            f"| {row['N']} | {row['relative_l2']:.3e} | "
            f"{row['butterfly_seconds']:.3e} | {row['fft_seconds']:.3e} | "
            f"{dense} | {row['pair_count']} |"
        )
    lines += [
        "",
        "The FFT is much faster in wall time because it is an optimised specialised",
        "implementation of this exact transform. The useful check here is scaling:",
        "the butterfly has O(N log N) paired coefficient updates, whereas a dense",
        "matrix has N^2 entries. Python loop overhead is not representative of an",
        "optimised implementation.",
        "",
        "## Accuracy at N=128",
        "",
        "| q | relative L2 error |",
        "|---:|---:|",
    ]
    lines += [f"| {row['q']} | {row['relative_l2']:.3e} |" for row in accuracy_rows]
    lines += [
        "",
        "Here relative $L^2$ error means $||u_q-u_{reference}||_2/||u_{reference}||_2$.",
        "",
        "![Four-panel DFT comparison](comparison.png)",
        "",
        "## Non-DFT oscillatory kernel",
        "",
        "The phase is $Psi(x,p)=xp+0.1x^2p^2$ at $N=128$. The reference is the",
        "explicit dense evaluation of the same kernel, not an FFT.",
        "",
        "| q | relative L2 error |",
        "|---:|---:|",
    ]
    lines += [f"| {row['q']} | {row['relative_l2']:.3e} |" for row in non_dft_rows]
    lines += [
        "",
        "![Non-DFT comparison](non_dft_comparison.png)",
        "",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
