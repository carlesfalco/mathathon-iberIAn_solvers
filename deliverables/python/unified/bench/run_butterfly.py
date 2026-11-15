"""Benchmark B1: ID butterfly (unified/drivers/butterfly_id.py) vs the SVD reference (code/butterfly.py).

For N = 256 ... 4096 and t in {0, 1, 3}: relative error of apply(g) against the dense K @ g, storage
(nnz over all sparse factors) of both versions, build time. Output: results/butterfly.{png,json}.
Run:  .venv/bin/python -m unified.bench.run_butterfly [--quick]
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from typing import NotRequired, Sequence, TypedDict

import numpy as np
import matplotlib.pyplot as plt

from unified.core import kernels
from unified.core.types import ComplexArray, Field, Spectrum
from unified.bench import problems as P, adapters as ad, common as C

from unified.reference import butterfly as ref  # noqa: E402  (packaged SVD comparison baseline)


class ButterflyRow(TypedDict):
    """One (N, t) case of results/butterfly.json. [computed]"""
    N: int
    t: float
    eps: float
    err_id: float              # ||apply(g) - K g|| / ||K g||, ID butterfly
    err_svd: float             # same, SVD reference
    nnz_id: int
    nnz_svd: int
    build_id_s: float
    build_svd_s: float
    err_id_vs_fft: NotRequired[float]   # t = 0 only: against numpy's inverse FFT


def run(sizes: Sequence[int], ts: Sequence[float], eps: float) -> list[ButterflyRow]:
    rows: list[ButterflyRow] = []
    rng = np.random.default_rng(0)
    for n in sizes:
        x, xi = P.b1_grids(n)
        for t in ts:
            K: ComplexArray = P.b1_matrix(n, t)   # dense B1 matrix K[x, xi]
            g: Spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            Kg: Field = K @ g
            t0 = time.perf_counter(); bf = ad.butterfly_build(K, x, xi, eps, kernel=kernels.fio1d(t))
            t_id = time.perf_counter() - t0
            err_id = np.linalg.norm(ad.butterfly_apply(bf, g) - Kg) / np.linalg.norm(Kg)
            t0 = time.perf_counter(); W, G, Uf, _ = ref.butterfly_factorize(K, eps=eps)
            t_svd = time.perf_counter() - t0
            err_svd = np.linalg.norm(ref.butterfly_apply(W, G, Uf, g) - Kg) / np.linalg.norm(Kg)
            r = ButterflyRow(N=n, t=t, eps=eps, err_id=err_id, err_svd=err_svd, nnz_id=ad.butterfly_nnz(bf),
                     nnz_svd=W.nnz + sum(m.nnz for m in G) + Uf.nnz, build_id_s=t_id, build_svd_s=t_svd)
            if t == 0:  # SPEC: compare t = 0 with numpy's FFT
                fft: Field = (-1.0) ** np.arange(n) * n * np.fft.ifft(g)
                r["err_id_vs_fft"] = np.linalg.norm(ad.butterfly_apply(bf, g) - fft) / np.linalg.norm(fft)
            rows.append(r)
            print(f"N={n:5d} t={t}: err ID {err_id:.1e} SVD {err_svd:.1e} | nnz ID {r['nnz_id']:,} "
                  f"SVD {r['nnz_svd']:,} dense {n*n:,} | build ID {t_id:.2f}s SVD {t_svd:.2f}s")
    return rows


def plot(rows: list[ButterflyRow], ts: Sequence[float], eps: float) -> pathlib.Path:
    C.style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for s, t in enumerate(ts):
        rs = [r for r in rows if r["t"] == t]
        C.line(a1, [r["N"] for r in rs], [r["err_id"] for r in rs], s, f"t={t:g}", direct=False)
    a1.axhline(10 * eps, color=C.INK2, ls="--", lw=1)
    a1.annotate("SPEC bound 10·eps", (rows[0]["N"], 10 * eps), xytext=(0, 4), textcoords="offset points",
                color=C.INK2, fontsize=8)
    a1.set(xscale="log", yscale="log", xlabel="N", ylabel="relative error ‖Kg − apply(g)‖/‖Kg‖",
           title=f"ID butterfly error (eps = {eps:g})")
    a1.set_xscale("log", base=2)
    a1.legend(loc="lower right")
    t = max(ts)
    rs = [r for r in rows if r["t"] == t]
    N = np.array([r["N"] for r in rs])
    C.line(a2, N, [r["nnz_id"] for r in rs], 0, "ID (this repo)")
    C.line(a2, N, [r["nnz_svd"] for r in rs], 1, "SVD (code/butterfly.py)")
    C.line(a2, N, N ** 2, 2, "dense N²")
    a2.set(xscale="log", yscale="log", xlabel="N", ylabel="stored numbers (nnz)", title=f"storage, t = {t:g}")
    a2.set_xscale("log", base=2)
    a2.legend(loc="upper left")
    return C.save(fig, "butterfly")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N up to 1024 only")
    a = ap.parse_args(argv)
    sizes = [256, 512, 1024] if a.quick else [256, 512, 1024, 2048, 4096]
    ts, eps = [0.0, 1.0, 3.0], 1e-8
    rows = run(sizes, ts, eps)
    print(plot(rows, ts, eps), C.dump("butterfly", {"eps": eps, "rows": rows}))


if __name__ == "__main__":
    main()
