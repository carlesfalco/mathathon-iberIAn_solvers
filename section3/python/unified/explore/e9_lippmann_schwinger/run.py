"""E9: Lippmann-Schwinger (2D volume scattering) through the unchanged harness.

Reproduce:  PYTHONPATH=. .venv/bin/python -m unified.explore.e9_lippmann_schwinger.run [--quick]
Writes results/*.json and results/*.png (about 12 min; --quick about 2 min, N <= 2048).

    r0  verification: dense LS vs Mie (disk) / radial ODE (bump); unitarity; fast vs dense   -> r0_verify
    r1  ranks vs N at k = 2 and k = 10, weak vs strong (proxy2)                               -> r1_ranks_n
    r1b ranks vs k at N ~ 4096                                                                -> r1b_ranks_k
    r2  one-sided contrast: stacked vs one-sided ranks; naive vs two-sided proxy              -> r2_contrast
    r3  E5 question: k sweep, exact (Tᵀ, complex T) vs realified (real T) vs proxy2           -> r3_highk
    r4  contrast strength sweep q0 (constant disk, Mie reference), cond(A), Herm(A)          -> r4_q0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from unified.core import Tree, interp_decomp
from unified.core.types import Matrix, Vector

from . import ls_problem as L
from . import variants as V
from .ls_problem import ContrastKind, LSProblem

OUT = os.path.join(os.path.dirname(__file__), "results")
EPS = 1e-8
MAX_LEAF = 64
type Row = dict[str, Any]
# Okabe-Ito subset, validated (dataviz validate_palette.js: all checks pass; CVD 7.6 -> marker+linestyle used too)
COL = {"exact": "#0072B2", "realified": "#D55E00", "proxy2": "#009E73", "proxyN": "#CC79A7",
       "weak": "#0072B2", "strong": "#D55E00"}
MK = {"exact": "o", "realified": "s", "proxy2": "^", "proxyN": "v", "weak": "o", "strong": "s"}


def _save(name: str, rows: Any) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)


def _log(*a: Any) -> None:
    print(*a, flush=True)


def dense(prob: LSProblem, b: Vector) -> tuple[Vector, float]:
    """Dense reference solve by LU, and cond_2(A) estimated by 40 power iterations on (A^H A)^{-1}."""
    lu = sla.lu_factor(cast(Any, prob.A))
    x: Vector = sla.lu_solve(lu, b)
    v: Vector = np.random.default_rng(1).standard_normal(prob.n) + 0j
    for _ in range(40):
        v = sla.lu_solve(lu, sla.lu_solve(lu, v, trans=2), trans=0)
        v /= np.linalg.norm(v)
    inv_norm = float(np.linalg.norm(sla.lu_solve(lu, v)))   # ~ 1 / sigma_min after the power iterations
    u: Vector = np.random.default_rng(2).standard_normal(prob.n) + 0j
    for _ in range(40):                                          # ~ sigma_max by power iteration on A^H A
        u = prob.A.conj().T @ (prob.A @ u)
        u /= np.linalg.norm(u)
    return x, float(np.linalg.norm(prob.A @ u)) * inv_norm


def herm_min(A: Matrix) -> float:
    return float(sla.eigh((A + A.conj().T) / 2, eigvals_only=True, subset_by_index=[0, 0])[0])


def ppw(prob: LSProblem) -> float:
    """Grid points per interior wavelength 2π / (k √(1 + max q) h)."""
    return float(2 * np.pi / (prob.k * np.sqrt(1 + max(prob.q0, 0)) * prob.h))


def tree_of(prob: LSProblem) -> Tree:
    return Tree(prob.X, max_leaf=MAX_LEAF)


# ---- r0 ------------------------------------------------------------------------------------------
def r0(quick: bool) -> list[Row]:
    rows: list[Row] = []
    ms = [24, 36, 48] if quick else [24, 36, 48, 72, 102]
    for kind in ("disk", "bump"):
        for k in (5.0, 10.0):
            q0, a = 1.0, 1.0
            nm = L.nmax_for(k, q0, a)
            ref = L.mie_coeffs(k, q0, a, nm)[0] if kind == "disk" else L.radial_coeffs(k, q0, a, nm, "bump")
            for m in ms:
                if m == 102 and k != 10.0:
                    continue
                P = L.build(kind, k, q0, m, a)
                b = P.u_inc()
                x, cond = dense(P, b)
                bx = L.discrete_coeffs(P, x, nm)
                row: Row = dict(kind=kind, k=k, m=m, n=P.n, h=P.h, ppw=ppw(P), cond=cond,
                                coef_err=L.coeff_error(bx, ref), unit_dense=L.unitarity_defect(bx, nm),
                                unit_ref=L.unitarity_defect(ref, nm))
                if kind == "disk":
                    uf = L.mie_field(k, q0, a, P.X, nm)
                    row["field_err"] = float(np.linalg.norm(x - uf) / np.linalg.norm(uf))
                if m == (48 if quick else 72):      # fast solvers vs dense AND vs the analytic reference
                    tree = tree_of(P)
                    for rule in ("weak", "strong"):
                        for var in ("exact", "realified", "proxy2"):
                            r = V.run(P, tree, rule, var, EPS, b, x, diagnostics=False)
                            row[f"{rule}_{var}_err"] = r.err
                            row[f"{rule}_{var}_coef_err"] = L.coeff_error(L.discrete_coeffs(P, r.x, nm), ref)
                rows.append(row)
                _log("r0", {kk: (f"{vv:.2e}" if isinstance(vv, float) else vv) for kk, vv in row.items()})
    _save("r0_verify", rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, kind in enumerate(("disk", "bump")):
        for k, ls in ((5.0, "-"), (10.0, "--")):
            rr = [r for r in rows if r["kind"] == kind and r["k"] == k]
            h = np.array([r["h"] for r in rr])
            ax[i].loglog(h, [r["coef_err"] for r in rr], ls, marker="o", color=COL["weak"], label=f"k={k:g}: far-field coeff. error")
            ax[i].loglog(h, [r["unit_dense"] for r in rr], ls, marker="s", color=COL["strong"], label=f"k={k:g}: unitarity defect")
        h0 = np.array([0.02, 0.09])
        ax[i].loglog(h0, (0.3 if kind == "disk" else 4.0) * h0 ** (1 if kind == "disk" else 2), ":", color="0.5",
                     label="slope 1" if kind == "disk" else "slope 2")
        ax[i].set_title(("constant disk vs Mie series" if kind == "disk" else "C² bump vs radial ODE") + " (dense LS)")
        ax[i].set_xlabel("grid spacing h"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
    ax[0].set_ylabel("relative error")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r0_verify.png"), dpi=130); plt.close(fig)
    return rows


# ---- r1 ------------------------------------------------------------------------------------------
def r1(quick: bool) -> list[Row]:
    rows: list[Row] = []
    ns = [512, 1024, 2048] if quick else [512, 1024, 2048, 4096, 8192]
    for k in (2.0, 10.0):
        for n in ns:
            P = L.build("bump", k, 1.0, L.m_for_n("bump", n))
            b = P.u_inc()
            x, _ = dense(P, b)
            tree = tree_of(P)
            for rule in ("weak", "strong"):
                r = V.run(P, tree, rule, "proxy2", EPS, b, x, diagnostics=False)
                row: Row = dict(k=k, n=P.n, depth=tree.depth, rule=rule, err=r.err, skel_total=r.skel_total,
                                root=r.root, storage=r.storage, t=r.t_factor,
                                skel_l1=r.skel_max_by_level.get(1, 0), skel_l2=r.skel_max_by_level.get(2, 0),
                                skel_l3=r.skel_max_by_level.get(3, 0), levels=r.skel_max_by_level)
                rows.append(row); _log("r1", row)
            del P
    _save("r1_ranks_n", rows)
    return rows


def r1b(quick: bool) -> list[Row]:
    rows: list[Row] = []
    n = 2048 if quick else 4096
    for k in ([2.0, 10.0, 20.0] if quick else [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]):
        P = L.build("bump", k, 1.0, L.m_for_n("bump", n))
        b = P.u_inc()
        x, _ = dense(P, b)
        tree = tree_of(P)
        for rule in ("weak", "strong"):
            r = V.run(P, tree, rule, "proxy2", EPS, b, x, diagnostics=False)
            row: Row = dict(k=k, n=P.n, ppw=ppw(P), rule=rule, err=r.err, skel_total=r.skel_total, root=r.root,
                            storage=r.storage, skel_l1=r.skel_max_by_level.get(1, 0),
                            skel_l2=r.skel_max_by_level.get(2, 0), skel_l3=r.skel_max_by_level.get(3, 0))
            rows.append(row); _log("r1b", row)
    _save("r1b_ranks_k", rows)
    return rows


def plot_r1(rn: list[Row], rk: list[Row]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, ls in ((2.0, "-"), (10.0, "--")):
        for rule in ("weak", "strong"):
            rr = [r for r in rn if r["k"] == k and r["rule"] == rule]
            n = np.array([r["n"] for r in rr])
            ax[0].loglog(n, [r["skel_l2"] for r in rr], ls, marker=MK[rule], color=COL[rule], label=f"{rule}, k={k:g}")
            ax[1].loglog(n, [r["storage"] for r in rr], ls, marker=MK[rule], color=COL[rule], label=f"{rule}, k={k:g}")
    n0 = np.array([512, 8192])
    ax[0].loglog(n0, 2.2 * n0 ** 0.5, ":", color="0.5", label="∝ N^½")
    ax[1].loglog(n0, 300 * n0, ":", color="0.5", label="∝ N")
    ax[1].loglog(n0, 25 * n0 ** 1.5, "-.", color="0.5", label="∝ N^1.5")
    ax[0].set_title("max skeleton, level-2 boxes (fixed physical size)"); ax[0].set_xlabel("N")
    ax[1].set_title("factor storage (complex numbers)"); ax[1].set_xlabel("N")
    for rule in ("weak", "strong"):
        rr = [r for r in rk if r["rule"] == rule]
        ax[2].plot([r["k"] for r in rr], [r["skel_l2"] for r in rr], marker=MK[rule], color=COL[rule], label=f"{rule} level 2")
        ax[2].plot([r["k"] for r in rr], [r["skel_l3"] for r in rr], ":", marker=MK[rule], color=COL[rule], label=f"{rule} level 3")
    ax[2].set_title(f"max skeleton vs k (N≈{rk[0]['n']})"); ax[2].set_xlabel("k")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r1_ranks.png"), dpi=130); plt.close(fig)


# ---- r2 ------------------------------------------------------------------------------------------
def far_fit_error(P: LSProblem, tree: Tree, rule: V.RuleName, var: V.VariantName, level: int) -> float:
    """Max over boxes at `level` of ||M[:, red] - M[:, skel] T|| / ||M|| with M = [A_FB; A_BFᵀ] the TRUE
    stacked far field (original A) and (skel, T) from the compressor under test. Tests compression only."""
    comp = V.make_compressor(var, P, tree, EPS)
    worst = 0.0
    for B in tree.levels[level]:
        near = np.concatenate([nb.idx for nb in V.RULES[rule].near_nodes(tree, B)] + [np.zeros(0, np.intp)])
        mask = np.ones(P.n, bool); mask[B.idx] = False; mask[near] = False
        far = np.flatnonzero(mask)
        if far.size == 0:
            continue
        sk, rd, T = comp(B.idx, far, B)
        M = np.vstack([P.A[np.ix_(far, B.idx)], P.A[np.ix_(B.idx, far)].T])
        if rd.size:
            worst = max(worst, float(np.linalg.norm(M[:, rd] - M[:, sk] @ T) / np.linalg.norm(M)))
    return worst


def r2(quick: bool) -> Row:
    out: Row = {"stack": [], "proxy": [], "fit": []}
    n = 2048
    for kind in ("disk", "bump", "square"):
        P = L.build(kind, 10.0, 1.0, L.m_for_n(kind, n))
        tree = tree_of(P)
        for rule in ("weak", "strong"):
            for lvl in range(tree.depth, 1, -1):
                s = asdict(V.stack_ranks(P, tree, rule, EPS, lvl)); s.update(kind=kind, rule=rule)
                out["stack"].append(s); _log("r2 stack", s)
                for var in ("exact", "proxy2", "proxyN"):
                    fe = dict(kind=kind, rule=rule, level=lvl, variant=var, fit=far_fit_error(P, tree, rule, var, lvl))
                    out["fit"].append(fe); _log("r2 fit", fe)
        if kind == "square":
            continue
        b = P.u_inc()
        x, _ = dense(P, b)
        rules: tuple[V.RuleName, ...] = ("weak", "strong")
        pvars: tuple[V.VariantName, ...] = ("exact", "proxy2", "proxyN")
        for eps in ([1e-8] if quick else [1e-6, 1e-8, 1e-10]):
            for rule in rules:
                for var in pvars:
                    r = V.run(P, tree, rule, var, eps, b, x, diagnostics=False)
                    row: Row = dict(kind=kind, eps=eps, rule=rule, variant=var, err=r.err, skel_total=r.skel_total)
                    out["proxy"].append(row); _log("r2 proxy", row)
            for rule in rules:                   # symmetrized Â (HIF-IE): exact ID, plain and realified
                for re_ in (False, True):
                    r = V.run_symmetrized(P, tree, rule, eps, b, x, realify=re_)
                    row = dict(kind=kind, eps=eps, rule=rule, variant=r.variant, err=r.err, skel_total=r.skel_total,
                               cond_rr=r.cond_rr_max, herm_rr=r.herm_min)
                    out["proxy"].append(row); _log("r2 proxy", row)
    _save("r2_contrast", out)
    return out


# ---- r3 ------------------------------------------------------------------------------------------
def _sweep_row(P: LSProblem, extra: Row, variants: tuple[V.VariantName, ...]) -> list[Row]:
    b = P.u_inc()
    x, cond = dense(P, b)
    hm = herm_min(P.A)
    tree = tree_of(P)
    rows: list[Row] = []
    for rule in ("weak", "strong"):
        for var in variants:
            r = V.run(P, tree, rule, var, EPS, b, x)
            row: Row = dict(extra, n=P.n, ppw=ppw(P), condA=cond, hermA=hm, rule=rule, variant=var, err=r.err,
                            relres=r.relres, cond_rr=r.cond_rr_max, herm_rr=r.herm_min, skel_total=r.skel_total)
            rows.append(row); _log("sweep", {kk: (f"{vv:.3g}" if isinstance(vv, float) else vv) for kk, vv in row.items()})
    return rows


def r3(quick: bool) -> list[Row]:
    n = 2048 if quick else 4096
    rows: list[Row] = []
    for k in ([10.0, 40.0] if quick else [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0]):
        P = L.build("bump", k, 1.0, L.m_for_n("bump", n))
        rows += _sweep_row(P, dict(k=k, q0=1.0), ("exact", "realified", "proxy2"))
    k = 20.0 if quick else 80.0                  # eps sweep at the largest k (E5: tighter eps hurt exact)
    P = L.build("bump", k, 1.0, L.m_for_n("bump", n))
    b = P.u_inc()
    x, _ = dense(P, b)
    tree = tree_of(P)
    evars: tuple[V.VariantName, ...] = ("exact", "realified")
    for eps in (1e-6, 1e-10, 1e-12):
        for var in evars:
            r = V.run(P, tree, "weak", var, eps, b, x)
            row: Row = dict(k=k, eps=eps, rule="weak", variant=var, err=r.err, cond_rr=r.cond_rr_max,
                            herm_rr=r.herm_min, skel_total=r.skel_total, eps_sweep=True)
            rows.append(row); _log("r3 eps", row)
    _save("r3_highk", rows)
    return rows


def r4(quick: bool) -> list[Row]:
    n = 2048 if quick else 4096
    rows: list[Row] = []
    k = 10.0
    for q0 in ([0.5, 4.0] if quick else [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]):
        P = L.build("disk", k, q0, L.m_for_n("disk", n))
        nm = L.nmax_for(k, q0, 1.0)
        x = np.linalg.solve(P.A, P.u_inc())
        mie = L.coeff_error(L.discrete_coeffs(P, x, nm), L.mie_coeffs(k, q0, 1.0, nm)[0])
        rows += _sweep_row(P, dict(k=k, q0=q0, mie_err=mie), ("exact", "realified", "proxy2"))
    _save("r4_q0", rows)
    return rows


def plot_sweeps(r3rows: list[Row], r4rows: list[Row]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for j, (rows, key, lab) in enumerate(((r3rows, "k", "k (bump, q0 = 1)"), (r4rows, "q0", "q0 (constant disk, k = 10)"))):
        for rule, ls in (("weak", "-"), ("strong", "--")):
            for var in ("exact", "realified", "proxy2"):
                rr = [r for r in rows if r["rule"] == rule and r["variant"] == var and not r.get("eps_sweep")]
                xs = [r[key] for r in rr]
                ax[0, j].semilogy(xs, [r["err"] for r in rr], ls, marker=MK[var], color=COL[var], label=f"{rule} {var}")
                ax[1, j].semilogy(xs, [r["cond_rr"] for r in rr], ls, marker=MK[var], color=COL[var], label=f"{rule} {var}")
        rr = [r for r in rows if r["rule"] == "weak" and r["variant"] == "exact" and not r.get("eps_sweep")]
        ax[0, j].semilogy([r[key] for r in rr], [EPS * r["condA"] for r in rr], ":", color="0.4", label="ε·cond(A)")
        ax[0, j].set_title(f"error vs dense solve — {lab}"); ax[1, j].set_title(f"worst cond(X_RR) — {lab}")
        for a in ax[:, j]:
            a.set_xlabel(key); a.grid(alpha=0.3); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r3_r4_sweeps.png"), dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="r0,r1,r2,r3,r4")
    a = ap.parse_args()
    t0 = time.time()
    only = a.only.split(",")
    if "r0" in only:
        r0(a.quick)
    if "r1" in only:
        plot_r1(r1(a.quick), r1b(a.quick))
    if "r2" in only:
        r2(a.quick)
    if "r3" in only or "r4" in only:
        plot_sweeps(r3(a.quick), r4(a.quick))
    _log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
