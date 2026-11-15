"""E10 runner: reproduces every number and figure in findings.md (results/*.json, results/*.png).

    PYTHONPATH=. .venv/bin/python -m unified.explore.e10_sht_butterfly.run            # full (~10 min)
    PYTHONPATH=. .venv/bin/python -m unified.explore.e10_sht_butterfly.run --quick    # n <= 1024 (~1 min)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial, jv, lpmv, spherical_jn

from unified.explore.e10_sht_butterfly.sht import (Coord, absolute_tolerance, SHTButterfly, SHTProblem, alf_table, build,
                                                   cheb_problem, gauss_nodes, sht_problem, tile_ranks)

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
type Row = dict[str, Any]


def rel(a: NDArray[Any], b: NDArray[Any]) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def save(name: str, obj: Any) -> None:
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def median_time(fn: Callable[[], Any], reps: int = 7) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(p: SHTProblem, eps: float, seed: int = 0, timing: bool = False) -> Row:
    """Build the butterfly of A^(m); report ranks, storage, synthesis error and round-trip error."""
    B = build(p, eps)
    rng = np.random.default_rng(seed)
    nc = p.n - p.m
    a = rng.standard_normal(nc) + 1j * rng.standard_normal(nc)        # random band-limited field (coefficients)
    f = p.A @ a
    fb = B.synth(a)
    a2 = B.analyze(fb)                                                  # butterfly synthesis then butterfly analysis
    row: Row = dict(n=p.n, m=p.m, eps=eps, ranks=B.bf.ranks, max_rank=max(B.bf.ranks), nnz=B.nnz,
                    dense=p.n * nc, nnz_ratio=B.nnz / (p.n * nc), synth_err=rel(fb, f), roundtrip_err=rel(a2, a),
                    adjoint_err=rel(B.adjoint(f), p.A.T @ f), build_s=B.bf.build_time,
                    finite=bool(np.all(np.isfinite(fb))))
    if timing:
        Wr, Ur = B.bf.W.real.tocsr(), B.bf.U.real.tocsr()               # A is real: so are the factors
        Gr = [G.real.tocsr() for G in B.bf.G]
        v = np.zeros((p.n, 2)); v[:nc] = np.c_[a.real, a.imag]
        def bf_apply() -> Any:
            c = Wr @ v
            for G in Gr:
                c = G @ c
            return Ur @ c
        assert rel(bf_apply()[:, 0] + 1j * bf_apply()[:, 1], f) < 10 * eps + 1e-12
        vd = v[:nc].copy()
        row["t_bf"] = median_time(bf_apply)
        row["t_dense"] = median_time(lambda: p.A @ vd)
    return row


# ---- R0: the kernel itself ----------------------------------------------------------------------------
def r0_validation(ns: list[int]) -> Row:
    th, _ = gauss_nodes(64)
    err_lpmv = 0.0
    for m in (0, 3, 20):
        A = alf_table(th, m, 60)
        ref = np.stack([lpmv(m, l, np.cos(th)) * np.sqrt((2 * l + 1) / 2 * factorial(l - m) / factorial(l + m))
                        * (-1) ** m for l in range(m, 61)], 1)
        err_lpmv = max(err_lpmv, float(np.abs(A - ref).max()))
    orth = []
    for n in ns:
        for m in (0, n // 4, n // 2):
            p = sht_problem(n, m)
            G = p.A.T @ (p.w[:, None] * p.A)
            orth.append(dict(n=n, m=m, orth_err=float(np.abs(G - np.eye(n - m)).max()),
                             max_abs=float(np.abs(p.A).max()), frac_subnormal=float(np.mean((p.A != 0) & (np.abs(p.A) < 2.3e-308)))))
    return dict(lpmv_err=err_lpmv, orth=orth)


# ---- R1: m = 0 (Legendre) and Chebyshev, scaling in n ------------------------------------------------
def r1_scaling(ns: list[int], epss: list[float]) -> list[Row]:
    rows = []
    for kind in ("legendre", "chebyshev"):
        for eps in epss:
            for n in ns:
                if kind == "chebyshev" and n > 4096:
                    continue
                p = sht_problem(n, 0) if kind == "legendre" else cheb_problem(n)
                r = measure(p, eps, timing=True)
                r["kind"] = kind
                rows.append(r)
                print(f"  R1 {kind:9s} eps={eps:.0e} n={n:5d} ranks={r['ranks']} nnz/dense={r['nnz_ratio']:.3f} "
                      f"synth={r['synth_err']:.1e} rt={r['roundtrip_err']:.1e} t_bf/t_dense={r['t_bf'] / r['t_dense']:.2f}"
                      f" build={r['build_s']:.1f}s", flush=True)
    return rows


# ---- R2: sweep the order m; raw kernel vs zeroing the evanescent entries -----------------------------
def r2_msweep(ns: list[int], epss: list[float]) -> list[Row]:
    rows = []
    for n in ns:
        for eps in epss:
            for frac in (0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8):
                m = int(frac * n)
                for variant, clip in (("raw", 0.0), ("clip1e-16", 1e-16), ("clip_eps", eps)):
                    if n > 2048 and variant != "clip_eps":
                        continue
                    r = measure(sht_problem(n, m, clip=clip), eps)
                    r["variant"] = variant
                    if variant != "raw":                       # error of the clipped butterfly vs the TRUE A
                        p0 = sht_problem(n, m)
                        p1 = sht_problem(n, m, clip=clip)
                        r["clip_kernel_err"] = float(np.linalg.norm(p1.A - p0.A, 2) / np.linalg.norm(p0.A, 2))
                    rows.append(r)
                    print(f"  R2 n={n} eps={eps:.0e} m={m:5d} {variant:9s} max_r={r['max_rank']:3d} "
                          f"nnz/dense={r['nnz_ratio']:.3f} synth={r['synth_err']:.1e} rt={r['roundtrip_err']:.1e}", flush=True)
    return rows


# ---- R3: the ordering / clustering dial ---------------------------------------------------------------
def r3_ordering(n: int, eps: float) -> list[Row]:
    rows = []
    cases: list[tuple[str, int, Coord]] = [("legendre", 0, c) for c in ("theta", "x", "bitrev", "random")]
    cases += [("legendre", n // 4, c) for c in ("theta", "x", "random")]
    cases += [("chebyshev", 0, c) for c in ("theta", "x", "bitrev", "random")]
    for kind, m, coord in cases:
        p = sht_problem(n, m, coord, clip=1e-16) if kind == "legendre" else cheb_problem(n, coord)
        r = measure(p, eps)
        r.update(kind=kind, coord=coord)
        rows.append(r)
        print(f"  R3 {kind:9s} m={m:4d} {coord:7s} ranks={r['ranks']} nnz/dense={r['nnz_ratio']:.3f} "
              f"synth={r['synth_err']:.1e}", flush=True)
    return rows


# ---- R4: ranks around the turning curve ----------------------------------------------------------------
def r4_turning(n: int, epss: list[float]) -> tuple[list[Row], dict[str, Any]]:
    rows = []
    L = int(np.log2(n))
    maps: dict[str, Any] = {}
    for eps in epss:
        for m in (0, n // 8, n // 4, n // 2):
            p = sht_problem(n, m, clip=1e-16)
            for lev in (L // 2 - 1, L // 2, L // 2 + 1):
                for absolute in (False, True):
                    tr = tile_ranks(p, lev, eps, absolute)
                    r = np.array([t["rank"] for t in tr]); c = np.array([t["cls"] for t in tr])
                    row: Row = dict(n=n, m=m, eps=eps, level=lev, tol="abs" if absolute else "rel")
                    for k, name in ((2.0, "osc"), (1.0, "straddle"), (0.0, "evan")):
                        sel = c == k
                        row[f"{name}_count"] = int(sel.sum())
                        row[f"{name}_max"] = int(r[sel].max()) if sel.any() else None
                        row[f"{name}_mean"] = float(r[sel].mean()) if sel.any() else None
                    rows.append(row)
                    if m == n // 4 and lev == L // 2 and eps == epss[-1]:
                        maps[row["tol"]] = tr
            print(f"  R4 eps={eps:.0e} m={m} done", flush=True)
    return rows, maps


# ---- R5: physics checks: plane wave (analytic), full sphere round trip + Parseval ---------------------
def r5_physics(n_pw: int, n_sphere: int, eps: float) -> Row:
    out: Row = {}
    # (a) Plane wave e^{i k k̂·r}, k̂ at polar angle α: its m-th Fourier mode in φ is
    #     f_m(θ) = i^m J_m(k sin α sin θ) e^{i k cos α cos θ}      (Jacobi-Anger)
    # and its coefficients are a_lm = 2 i^l j_l(k) P̃_l^m(cos α)   (Rayleigh / addition theorem). [cited]
    k, al = n_pw / 4, 0.7
    th, w = gauss_nodes(n_pw)
    pw = []
    for m in (0, n_pw // 16, n_pw // 8):
        B = build(sht_problem(n_pw, m, clip=1e-16), eps)
        fm = (1j) ** m * jv(m, k * np.sin(al) * np.sin(th)) * np.exp(1j * k * np.cos(al) * np.cos(th))
        l = np.arange(m, n_pw)
        exact = 2 * (1j) ** l * spherical_jn(l, k) * alf_table(np.array([al]), m, n_pw - 1)[0]
        a_bf = B.analyze(fm)
        a_dense = alf_table(th, m, n_pw - 1).T @ (w * fm)
        pw.append(dict(m=m, k=k, err_bf_vs_exact=rel(a_bf, exact), err_dense_vs_exact=rel(a_dense, exact),
                       resynth_err=rel(B.synth(a_bf), fm)))
        print(f"  R5 plane wave n={n_pw} m={m}: |a_bf - exact|={pw[-1]['err_bf_vs_exact']:.1e} "
              f"(dense {pw[-1]['err_dense_vs_exact']:.1e})", flush=True)
    out["plane_wave"] = pw
    # (b) Full sphere: random band-limited complex field, lmax = n - 1, n Gauss latitudes x 2n longitudes.
    n = n_sphere
    nphi = 2 * n
    rng = np.random.default_rng(1)
    th, w = gauss_nodes(n)
    bfs: dict[int, SHTButterfly] = {}
    t0 = time.perf_counter()
    for m in range(n):
        bfs[m] = build(sht_problem(n, m, clip=1e-16), eps)
    t_build = time.perf_counter() - t0
    coef = {m: rng.standard_normal(n - abs(m)) + 1j * rng.standard_normal(n - abs(m)) for m in range(-(n - 1), n)}
    F = np.zeros((n, nphi), complex)                         # F[:, m mod nphi] = f_m(θ)
    for m, a in coef.items():
        F[:, m % nphi] = bfs[abs(m)].synth(a)
    f = np.fft.ifft(F, axis=1) * nphi                        # f(θ_j, φ_k) = Σ_m f_m(θ_j) e^{i m φ_k}
    Fb = np.fft.fft(f, axis=1) / nphi
    err2 = 0.0; nrm2 = 0.0
    for m, a in coef.items():
        a2 = bfs[abs(m)].analyze(Fb[:, m % nphi])
        err2 += float(np.sum(np.abs(a2 - a) ** 2)); nrm2 += float(np.sum(np.abs(a) ** 2))
    energy_field = float(np.sum(w[:, None] * np.abs(f) ** 2) * 2 * np.pi / nphi)   # ∫|f|² dΩ by quadrature
    energy_coef = 2 * np.pi * nrm2                                                  # Parseval: 2π Σ|a_lm|²
    out["sphere"] = dict(n=n, nphi=nphi, lmax=n - 1, eps=eps, roundtrip_err=float(np.sqrt(err2 / nrm2)),
                         parseval_rel=abs(energy_field - energy_coef) / energy_coef, build_all_m_s=t_build,
                         total_nnz=int(sum(b.nnz for b in bfs.values())),
                         total_dense=int(sum(n * (n - m) for m in range(n))))
    print(f"  R5 sphere n={n}: roundtrip={out['sphere']['roundtrip_err']:.1e} parseval={out['sphere']['parseval_rel']:.1e}",
          flush=True)
    return out


# ---- R6: dial 4 variant: absolute instead of per-tile relative tolerance in the ID ---------------------
def r6_abs_tol(n: int, epss: list[float]) -> list[Row]:
    rows = []
    for eps in epss:
        for frac in (0, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8):
            m = int(frac * n)
            p = sht_problem(n, m)                                   # raw kernel (subnormals included)
            with absolute_tolerance(float(np.abs(p.A).max())):
                r = measure(p, eps)
            r["variant"] = "abs_tol"
            rows.append(r)
            print(f"  R6 n={n} eps={eps:.0e} m={m:5d} abs_tol   max_r={r['max_rank']:3d} "
                  f"nnz/dense={r['nnz_ratio']:.3f} synth={r['synth_err']:.1e} rt={r['roundtrip_err']:.1e}", flush=True)
    return rows


# ---- figures -------------------------------------------------------------------------------------------
def figures(r1: list[Row], r2: list[Row], maps: dict[str, Any], n4: int) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, ls in (("legendre", "-"), ("chebyshev", "--")):
        for eps, col in ((1e-6, "C0"), (1e-10, "C3")):
            rr = [r for r in r1 if r["kind"] == kind and r["eps"] == eps]
            ax[0].plot([r["n"] for r in rr], [r["nnz_ratio"] for r in rr], ls, color=col, marker="o",
                       label=f"{kind} ε={eps:.0e}")
            ax[1].plot([r["n"] for r in rr], [r["t_bf"] / r["t_dense"] for r in rr], ls, color=col, marker="o",
                       label=f"{kind} ε={eps:.0e}")
    for a, t in zip(ax, ("storage: butterfly nnz / dense n²", "apply time: butterfly / dense BLAS")):
        a.set_xscale("log", base=2); a.set_yscale("log"); a.axhline(1, color="k", lw=0.8); a.set_title(t)
        a.set_xlabel("n (latitudes = degrees)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RES / "r1_crossover.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.8))
    n2 = max(r["n"] for r in r2 if r["variant"] == "raw")
    for variant, mk in (("raw", "x"), ("clip1e-16", "o"), ("clip_eps", "s"), ("abs_tol", "^")):
        for eps, col in ((1e-6, "C0"), (1e-10, "C3")):
            rr = [r for r in r2 if r["variant"] == variant and r["eps"] == eps and r["n"] == n2]
            if rr:
                ax.plot([r["m"] / r["n"] for r in rr], [r["nnz_ratio"] for r in rr], marker=mk, color=col,
                        label=f"{variant} ε={eps:.0e}")
    ax.set_xlabel("m / n"); ax.set_ylabel("nnz / dense"); ax.set_title(f"n = {n2}: storage vs order m")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(RES / "r2_msweep.png", dpi=120); plt.close(fig)

    if maps:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, tol in zip(ax, ("rel", "abs")):
            tr = maps[tol]
            sc = a.scatter([t["theta"] for t in tr], [t["l"] for t in tr], c=[t["rank"] for t in tr], s=14,
                           cmap="viridis", marker="s")
            tt = np.linspace(0.05, np.pi - 0.05, 200)
            a.plot(tt, (n4 // 4) / np.sin(tt) - 0.5, "r-", lw=1, label="turning curve (l+½) sin θ = m")
            a.set_ylim(n4 // 4, n4); a.set_xlabel("θ (tile centre)"); a.set_ylabel("l (tile centre)")
            a.set_title(f"tile ε-rank, tolerance {'relative to tile' if tol == 'rel' else 'relative to ‖A‖'}")
            fig.colorbar(sc, ax=a); a.legend(fontsize=7, loc="lower center")
        fig.tight_layout(); fig.savefig(RES / "r4_turning_map.png", dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["r6"], default=None, help="run one experiment only")
    args = ap.parse_args()
    q = args.quick
    t0 = time.perf_counter()
    if args.only == "r6":
        save("r6_abs_tol", r6_abs_tol(512 if q else 2048, [1e-6, 1e-10]))
        return
    ns = [256, 512, 1024] if q else [256, 512, 1024, 2048, 4096, 8192]
    print("R0 validation"); save("r0_validation", r0 := r0_validation([256, 1024] if q else [256, 1024, 4096]))
    print(f"  lpmv err {r0['lpmv_err']:.1e}; max orth err {max(o['orth_err'] for o in r0['orth']):.1e}")
    print("R1 scaling"); save("r1_scaling", r1 := r1_scaling(ns, [1e-6, 1e-10]))
    print("R2 m sweep"); save("r2_msweep", r2 := r2_msweep([512] if q else [2048, 4096], [1e-6, 1e-10]))
    print("R3 ordering"); save("r3_ordering", r3_ordering(512 if q else 1024, 1e-10))
    n4 = 512 if q else 1024
    print("R4 turning"); r4, maps = r4_turning(n4, [1e-6, 1e-10]); save("r4_turning", r4)
    print("R5 physics"); save("r5_physics", r5_physics(512 if q else 2048, 64 if q else 256, 1e-10))
    print("R6 absolute tolerance"); save("r6_abs_tol", r6 := r6_abs_tol(512 if q else 2048, [1e-6, 1e-10]))
    figures(r1, r2 + r6, maps, n4)
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
