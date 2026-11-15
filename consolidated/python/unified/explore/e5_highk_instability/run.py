"""E5: why exact-ID recursive skeletonization breaks down on P3 (2D Helmholtz, starfish) at high k, and a fix.

Reproduce everything:  PYTHONPATH=. .venv/bin/python -m unified.explore.e5_highk_instability.run  [--quick]
Writes results/*.json and results/*.png. About 4 min at N = 2048 (--quick: N = 1024, fewer k).

    r1  k sweep 10..120 (weak, strong) for exact / proxy / realified / congruent   -> r1_ksweep
    r2  fine k sweep 40..100 step 1 (exact, weak): smooth growth or resonances?     -> r2_fine
    r3  eps sweep and oversampling at k = 80                                        -> r3_eps
    r4  localisation + mechanism per box (levels, k*hw, Hermitian parts, Im T)      -> r4_boxes
    r5  leaf level, W = A: exact vs proxy redundant subspaces (angles, off-curve field, Im T) -> r5_leaf
    r6  fixes table at k = 80, 100 (weak, strong), all variants                     -> r6_fixes
    r7  reviewer: eps -> rank -> cond(T), max|T_ij|, cond(X_RR) per worst box         -> r7_eps_T
    r8  reviewer: exact / proxy / randomized / realified at EQUAL rank per box       -> r8_equal_rank
    r9  reviewer: proxy hypothesis (row-space containment, conjugation closure)      -> r9_proxy_hypothesis
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.linalg as sla

from unified.core import ExactCompressor, ProxyCompressor, Tree
from unified.core.types import Compressor, Matrix, Points

from . import _deps as P
from .diag import instrument
from .diag import RULES, VARIANTS, BoxRecord, Variant, Wavenumber, redundant_basis, rhs, run_case
from .fixes import RealifiedCompressor
from .review import FixedKind, FixedRankCompressor, RankMap, RankRecorder, proxy_hypothesis
from unified.drivers.rskel import rskel_factor
from unified.core import Weak

OUT = os.path.join(os.path.dirname(__file__), "results")
EPS = 1e-8
type Row = dict[str, Any]


def _save(name: str, rows: Any) -> None:
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)


def _log(r: Row) -> None:
    print(f"  k={r['k']:6.1f} {r['rule']:6s} {r['comp']:11s} eps={r['eps']:.0e} err={r['err']:.2e} "
          f"cond={r['cond_rr_max']:.2e} skel={r['skel_total']}")


def _dense(prob: P.Problem) -> Any:
    return np.linalg.solve(prob.A, rhs(len(prob.X)))


# ---- r1 -----------------------------------------------------------------------------------------------
def r1(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("r1 k sweep")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        xd = _dense(prob)
        H = (prob.A + prob.A.conj().T) / 2
        herm_min_a = float(np.linalg.eigvalsh(H)[0])
        for rule in RULES:
            for v in ("exact", "proxy", "realified", "congruent"):
                r = run_case(prob, rule, v, EPS, k, xd)
                r["herm_min_A"] = herm_min_a
                _log(r); rows.append(r)
    _save("r1_ksweep", rows)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for j, rule in enumerate(RULES):
        for v in ("exact", "proxy", "realified", "congruent"):
            sel = [r for r in rows if r["rule"] == rule and r["comp"] == v]
            ax[0, j].semilogy([r["k"] for r in sel], [r["err"] for r in sel], marker="o", ms=3, label=v)
            ax[1, j].semilogy([r["k"] for r in sel], [r["cond_rr_max"] for r in sel], marker="o", ms=3, label=v)
        ax[0, j].axhline(EPS, color="gray", lw=0.8)
        ax[0, j].set_title(f"P3 N={n}, {rule}: solve error")
        ax[1, j].set_title("worst cond(X_RR)")
        ax[1, j].set_xlabel("k")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r1_ksweep.png"), dpi=130); plt.close(fig)
    return rows


# ---- r2 -----------------------------------------------------------------------------------------------
def r2(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("r2 fine k sweep (exact, weak), worst cond per level")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        recs: list[BoxRecord] = []
        r = run_case(prob, "weak", "exact", EPS, k, _dense(prob), records=recs)
        levels = sorted({b["level"] for b in recs})
        r["cond_by_level"] = {lv: max(b["cond_x"] for b in recs if b["level"] == lv) for lv in levels}
        _log(r); rows.append(r)
    _save("r2_fine", rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    kk = [r["k"] for r in rows]
    ax[0].semilogy(kk, [r["err"] for r in rows], marker=".", label="error")
    ax[0].semilogy(kk, [r["cond_rr_max"] * 1e-16 for r in rows], ls=":", label="1e-16 cond(X_RR)")
    ax[0].axhline(EPS, color="gray", lw=0.8); ax[0].legend(fontsize=8); ax[0].set_xlabel("k")
    ax[0].set_title("exact, weak: error vs k (step 1)")
    for lv in sorted({lv for r in rows for lv in r["cond_by_level"]}):
        ax[1].semilogy(kk, [r["cond_by_level"].get(lv, np.nan) for r in rows], marker=".", label=f"level {lv}")
    ax[1].set_title("worst cond(X_RR) per tree level"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r2_fine.png"), dpi=130); plt.close(fig)
    return rows


# ---- r3 -----------------------------------------------------------------------------------------------
def r3(n: int) -> list[Row]:
    print("r3 eps sweep at k = 80 (weak)")
    prob = P.p3_helmholtz(n, 80.0)
    xd = _dense(prob)
    rows: list[Row] = []
    for eps in (1e-6, 1e-8, 1e-10, 1e-12):
        for v in ("exact", "proxy", "realified", "congruent"):
            r = run_case(prob, "weak", v, eps, 80.0, xd)
            _log(r); rows.append(r)
    _save("r3_eps", rows)
    return rows


# ---- r4 -----------------------------------------------------------------------------------------------
def _level_table(recs: list[BoxRecord]) -> list[Row]:
    out: list[Row] = []
    for lv in sorted({b["level"] for b in recs}, reverse=True):
        s = [b for b in recs if b["level"] == lv]
        out.append(dict(level=lv, boxes=len(s), k_hw=s[0]["k_hw"], cond_max=max(b["cond_x"] for b in s),
                        herm_min_x=min(b["herm_min_x"] for b in s), herm_min_wbb=min(b["herm_min_wbb"] for b in s),
                        dev_wbb_max=max(b["dev_wbb"] for b in s), t_imag_median=float(np.median([b["t_imag"] for b in s])),
                        t_norm_max=max(b["t_norm"] for b in s),
                        skel_frac_median=float(np.median([b["ns"] / b["nb"] for b in s]))))
    return out


def r4(n: int, ks: Sequence[Wavenumber]) -> Row:
    print("r4 per-box localisation + Hermitian-part mechanism")
    data: Row = {}
    allrecs: dict[str, list[BoxRecord]] = {}
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        xd = _dense(prob)
        for rule in RULES:
            for v in ("exact", "proxy", "realified", "hybrid", "oversample8", "srrqr", "abstol"):
                recs: list[BoxRecord] = []
                run_case(prob, rule, v, EPS, k, xd, records=recs)
                key = f"k={k:g}/{rule}/{v}"
                data[key] = _level_table(recs)
                allrecs[key] = recs
                if v in ("exact", "realified", "proxy"):
                    for t in data[key]:
                        print(f"  {key:26s} L{t['level']} k*hw={t['k_hw']:5.1f} cond={t['cond_max']:.1e} "
                              f"minHerm(X)={t['herm_min_x']:+.3f} minHerm(W_BB)={t['herm_min_wbb']:+.3f} "
                              f"||W_BB-I||={t['dev_wbb_max']:.1e} ImT={t['t_imag_median']:.2f}")
    # theorem check: with real T, Herm(X_RR) >= lambda_min(Herm W_BB) for every box (up to O(eps))
    margin = min(b["herm_min_x"] - b["herm_min_wbb"] for key, rs in allrecs.items() if "/realified" in key for b in rs)
    data["realified_min_margin_hermX_minus_hermWBB"] = margin
    print(f"  realified: min over boxes of lmin Herm(X_RR) - lmin Herm(W_BB) = {margin:.2e} (theory: >= 0)")
    data["box_records"] = {k: v for k, v in allrecs.items() if "/weak/" in k and ("exact" in k or "realified" in k)}
    _save("r4_boxes", data)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for key, rs in data["box_records"].items():
        mk = "o" if "exact" in key else "x"
        ax[0].loglog([b["k_hw"] for b in rs], [b["cond_x"] for b in rs], mk, ms=3, alpha=0.5, label=key)
        ax[1].semilogy([b["ns"] / b["nb"] for b in rs], [b["cond_x"] for b in rs], mk, ms=3, alpha=0.5)
        ax[2].plot([b["herm_min_wbb"] for b in rs], [b["herm_min_x"] for b in rs], mk, ms=3, alpha=0.5)
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("k * box half-width"); ax[0].set_ylabel("cond(X_RR)"); ax[0].legend(fontsize=6)
    ax[0].set_title("weak RS, every box (o exact, x realified)")
    ax[1].set_xlabel("skeleton fraction |S|/|B|"); ax[1].set_title("cond(X_RR) vs skeleton fraction")
    ax[2].set_xscale("symlog", linthresh=0.1); ax[2].set_yscale("symlog", linthresh=0.1)
    lim = [-1e3, 1.5]
    ax[2].plot(lim, lim, "k:", lw=0.8)
    ax[2].set_xlabel("lambda_min Herm(W_BB)"); ax[2].set_ylabel("lambda_min Herm(X_RR)")
    ax[2].set_title("accretivity: realified stays above the diagonal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r4_boxes.png"), dpi=130); plt.close(fig)
    return data


# ---- r5 -----------------------------------------------------------------------------------------------
def _max_sin(U: Matrix, V: Matrix) -> float:
    """sin of the largest principal angle between span(U) and span(V) (0 if either is empty)."""
    if U.shape[1] == 0 or V.shape[1] == 0:
        return 0.0
    return float(np.sin(sla.subspace_angles(U, V).max()))


def _orth(M: Matrix) -> Matrix:
    Q: Matrix = np.linalg.qr(M)[0]      # Pi = [I; -T] has full column rank
    return Q


def r5(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    """Leaf boxes with W = A (no Schur fill yet), weak far field. For exact / proxy / realified IDs:
    - t_imag: how complex T is;
    - conj_gap: sin of the largest principal angle between span(Pi) and span(conj Pi) (0 iff closed under conj);
    - angle_vs_proxy: sin of the largest principal angle between exact's span(Pi) and proxy's span(Pi);
    - offcurve: sigma_max(K(C, B) Pi) / sigma_max(K(C, B)) on a circle C of radius 3 hw: do the redundant
      combinations radiate OFF the curve (exact: yes; proxy: no)?
    - pivot_herm_min: lambda_min Herm(Pi^T A_BB Pi), and lambda_min Herm(Pi^T Pi)."""
    print("r5 leaf-level subspaces")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        A, X = prob.A, prob.X
        tree = Tree(X, max_leaf=32)
        comps: dict[str, Compressor] = {"exact": ExactCompressor(A, EPS), "proxy": ProxyCompressor(prob.kernel, X, EPS, A=A),
                 "realified": RealifiedCompressor(A, EPS)}
        for leaf in tree.leaves():
            box = leaf.idx
            far = np.setdiff1d(np.arange(len(X)), box)
            th = 2 * np.pi * np.arange(256) / 256
            C: Points = leaf.center[None, :] + 3 * leaf.half_width * np.c_[np.cos(th), np.sin(th)]
            KC: Matrix = prob.kernel(C, X[box])
            spans: dict[str, Matrix] = {}
            for name, comp in comps.items():
                sk, rd, T = comp(box, far, leaf)
                if len(rd) == 0:
                    continue
                Pi = redundant_basis(len(box), sk, rd, T)
                spans[name] = _orth(Pi)
                X_rr: Matrix = Pi.T @ A[np.ix_(box, box)] @ Pi
                G: Matrix = Pi.T @ Pi
                tn = float(np.linalg.norm(T))
                rows.append(dict(k=k, box=str(leaf.coord), comp=name, ns=len(sk), nb=len(box),
                                 t_imag=float(np.linalg.norm(T.imag) / tn) if tn else 0.0,
                                 conj_gap=_max_sin(Pi, Pi.conj()),
                                 offcurve=float(np.linalg.norm(KC @ _orth(Pi), 2) / np.linalg.norm(KC, 2)),
                                 cond_x=float(np.linalg.cond(X_rr)) if X_rr.size else 1.0,
                                 pivot_herm_min=float(np.linalg.eigvalsh((X_rr + X_rr.conj().T) / 2)[0]),
                                 gram_herm_min=float(np.linalg.eigvalsh((G + G.conj().T) / 2)[0])))
            if "exact" not in spans or "proxy" not in spans:
                continue
            ang = _max_sin(spans["exact"], spans["proxy"])
            for r in rows:
                r.setdefault("angle_exact_vs_proxy", ang)
    _save("r5_leaf", rows)
    print(f"  {'k':>5s} {'comp':10s} {'ImT':>5s} {'conjgap':>7s} {'offcurve':>9s} {'minHerm(PiTAPi)':>15s} "
          f"{'minHerm(PiTPi)':>14s} {'sin<exact,proxy>':>16s} (medians over leaves; Herm mins are minima)")
    for k in ks:
        for c in ("exact", "proxy", "realified"):
            s = [r for r in rows if r["k"] == k and r["comp"] == c]
            print(f"  {k:5.0f} {c:10s} {np.median([r['t_imag'] for r in s]):5.2f} "
                  f"{np.median([r['conj_gap'] for r in s]):7.2f} {np.median([r['offcurve'] for r in s]):9.1e} "
                  f"{min(r['pivot_herm_min'] for r in s):+15.3f} {min(r['gram_herm_min'] for r in s):+14.3f} "
                  f"{np.median([r['angle_exact_vs_proxy'] for r in s]):16.2f}")
    return rows


# ---- r6 -----------------------------------------------------------------------------------------------
def r6(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("r6 fixes")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        xd = _dense(prob)
        for rule in RULES:
            for v in VARIANTS:
                r = run_case(prob, rule, v, EPS, k, xd)
                _log(r); rows.append(r)
    _save("r6_fixes", rows)
    return rows


# ---- r7: reviewer (1) eps -> rank -> conditioning of T and X_RR ----------------------------------------
def r7(n: int, k: Wavenumber = 80.0) -> list[Row]:
    print(f"r7 eps sweep at k = {k:g} (weak): worst box per run, and level-1 medians")
    prob = P.p3_helmholtz(n, k)
    xd = _dense(prob)
    rows: list[Row] = []
    for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        for v in ("exact", "realified", "proxy"):
            recs: list[BoxRecord] = []
            r = run_case(prob, "weak", v, eps, k, xd, records=recs)
            w = max(recs, key=lambda b: b["cond_x"])
            top = [b for b in recs if b["level"] == min(b2["level"] for b2 in recs)]
            r.update(worst=dict(w), top_level_median=dict(
                rank=float(np.median([b["ns"] for b in top])), t_cond=float(np.median([b["t_cond"] for b in top])),
                t_max=float(np.median([b["t_max"] for b in top])), cond_x=float(np.median([b["cond_x"] for b in top]))),
                t_max_all=max(b["t_max"] for b in recs), t_cond_all=max(b["t_cond"] for b in recs))
            print(f"  eps={eps:.0e} {v:9s} err={r['err']:.1e} skel={r['skel_total']} | worst box L{w['level']} "
                  f"nb={w['nb']} rank={w['ns']} cond(X_RR)={w['cond_x']:.1e} cond(T)={w['t_cond']:.1e} "
                  f"max|T|={w['t_max']:.2f} ImT={w['t_imag']:.2f} | all boxes: max|T|={r['t_max_all']:.2f} "
                  f"max cond(T)={r['t_cond_all']:.1e}")
            rows.append(r)
    _save("r7_eps_T", rows)
    return rows


# ---- r8: reviewer (2) equal-rank comparison -------------------------------------------------------------
def r8(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("r8 equal rank (weak): ranks copied per box from a reference run")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        xd = _dense(prob)
        b = rhs(n)
        refs: dict[str, RankMap] = {}
        for ref in ("exact", "proxy", "realified"):
            base: Compressor = (ExactCompressor(prob.A, EPS) if ref == "exact" else RealifiedCompressor(prob.A, EPS)
                                if ref == "realified" else ProxyCompressor(prob.kernel, prob.X, EPS, A=prob.A))
            rec = RankRecorder(base)
            rskel_factor(prob.A, Tree(prob.X, max_leaf=32), Weak(), rec)
            refs[ref] = rec.ranks
        for ref, ranks in refs.items():
            for kind in ("exact", "proxy", "randomized", "realified"):
                fk: FixedKind = kind
                comp = FixedRankCompressor(fk, prob.A, prob.kernel, prob.X, ranks)
                recs: list[BoxRecord] = []
                with instrument(recs, k):
                    F = rskel_factor(prob.A, Tree(prob.X, max_leaf=32), Weak(), comp)
                x = F.solve(b)
                r: Row = dict(k=k, ranks_from=ref, comp=kind, skel_total=int(sum(ranks.values())),
                              err=float(np.linalg.norm(x - xd) / np.linalg.norm(xd)),
                              cond_rr_max=max(float(np.linalg.cond(np.triu(s.lu[0]))) for s in F.steps),
                              far_err_max=max(comp.far_errs), far_err_median=float(np.median(comp.far_errs)),
                              t_imag_median=float(np.median([q["t_imag"] for q in recs])),
                              herm_min_x=min(q["herm_min_x"] for q in recs), levels=_level_table(recs))
                print(f"  k={k:5.0f} ranks of {ref:5s} ({r['skel_total']}) {kind:10s} err={r['err']:.1e} "
                      f"cond={r['cond_rr_max']:.1e} far-field ID err max={r['far_err_max']:.1e} "
                      f"median={r['far_err_median']:.1e} ImT={r['t_imag_median']:.2f} minHerm(X)={r['herm_min_x']:+.2f}")
                rows.append(r)
    _save("r8_equal_rank", rows)
    return rows


# ---- r9: reviewer (3) the proxy hypothesis on W = A -----------------------------------------------------
def r9(n: int, ks: Sequence[Wavenumber]) -> list[Row]:
    print("r9 proxy hypothesis: containment of exact far rows in proxy rows; conjugation closure")
    rows: list[Row] = []
    for k in ks:
        prob = P.p3_helmholtz(n, k)
        tree = Tree(prob.X, max_leaf=32)
        for lvl in range(tree.depth, 1, -1):
            h = proxy_hypothesis(prob.kernel, prob.X, prob.A, tree, lvl, k)
            print(f"  k={k:5.0f} L{lvl} contain={h['contain']:.1e} conj-gap exact={h['conj_exact']:.2f} "
                  f"proxy(all rows)={h['conj_proxy']:.2f} proxy ring only={h['conj_ring']:.3f} "
                  f"rank exact={h['rank_exact']:.1f} proxy={h['rank_proxy']:.1f}")
            rows.append(dict(k=k, level=lvl, **h))
    _save("r9_proxy_hypothesis", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N = 1024 and fewer k")
    ap.add_argument("--only", default="", help="comma-separated subset of r1..r6")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n = 1024 if a.quick else 2048
    only = set(a.only.split(",")) if a.only else {"r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"}
    if "r1" in only:
        r1(n, [10.0, 60.0, 100.0] if a.quick else [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                                                    110.0, 120.0])
    if "r2" in only:
        r2(n, list(np.arange(40.0, 101.0, 10.0 if a.quick else 1.0)))
    if "r3" in only:
        r3(n)
    if "r4" in only:
        r4(n, [80.0] if a.quick else [20.0, 80.0, 100.0])
    if "r5" in only:
        r5(n, [20.0, 80.0] if a.quick else [10.0, 20.0, 40.0, 80.0, 100.0])
    if "r6" in only:
        r6(n, [80.0] if a.quick else [80.0, 100.0, 120.0])
    if "r7" in only:
        r7(n)
    if "r8" in only:
        r8(n, [80.0] if a.quick else [80.0, 100.0, 120.0])
    if "r9" in only:
        r9(n, [20.0, 80.0])


if __name__ == "__main__":
    main()
