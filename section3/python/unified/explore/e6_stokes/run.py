"""E6: rigid particles in 2D Stokes flow through the unchanged harness (kernel swap + vector proxy).

Reproduce everything:  PYTHONPATH=. .venv/bin/python -m unified.explore.e6_stokes.run  [--quick]
Writes results/*.json and results/*.png.

    r0  core ProxyCompressor with the Stokes kernel, unchanged: does it run?
    r1  analytic checks: translating/rotating circle (mobility), Jeffery rate of a free ellipse in shear
    r2  Stokes vs Laplace skeleton ranks per level, N sweep (starfish), weak/strong, exact/proxy; eps sweep
    r3  suspensions of K ellipses (N_dof up to 8192): accuracy, ranks, factor time vs dense LU
    r4  proxy sources (Stokeslet / stresslet / both) and the degenerate scale R = sqrt(e) (Laplace: R = 1)
    r5  is the rigid-body projector seen by the proxy? (smooth and rough projector weights; containment)
    r6  where it breaks: gap between two ellipses -> 0, thin ellipses (b/a -> 0)
    r7  physics demo: full 3K x 3K mobility matrix of a suspension from one factorization; reciprocity
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unified.core import ProxyCompressor, Tree, Weak
from unified.core.types import Matrix, Points, Vector
from unified.drivers.rskel import rskel_factor

from . import stokes as S
from .harness import Physics, RuleName, CompKind, build, case, make_comp

RULES: tuple[RuleName, ...] = ("weak", "strong")
PHYS: tuple[Physics, ...] = ("laplace", "stokes")
EXP: tuple[CompKind, ...] = ("exact", "proxy-sl")
PB: tuple[CompKind, ...] = ("proxy-sl", "proxy-both")
PN: tuple[CompKind, ...] = ("proxy-sl", "proxy-norigid")
DEG: tuple[tuple[Physics, float], ...] = (("stokes", float(np.sqrt(np.e))), ("laplace", 1.0))
SRC: dict[str, CompKind] = {"sl": "proxy-sl", "dl": "proxy-dl", "both": "proxy-both"}

OUT = os.path.join(os.path.dirname(__file__), "results")
type Record = dict[str, Any]


def _save(name: str, data: Any) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def _strip(r: Record) -> Record:
    return {k: v for k, v in r.items() if not k.startswith("_")}


# ---- r0 ---------------------------------------------------------------------------------------------------
def r0() -> Record:
    """Plug the Stokes kernel into core.ProxyCompressor exactly as the scalar problems do."""
    g = S.starfish(256)
    A, tree, _ = build(g, "stokes")
    def kern(X: Points, Y: Points) -> Matrix:   # normals are not even available from coordinates alone
        return S.stresslet_dl(X, Y, np.zeros_like(Y))
    comp = ProxyCompressor(kern, g.dof_points(), 1e-10, A=A)
    try:
        rskel_factor(A, tree, Weak(), comp)
        out: Record = {"runs": True}
    except Exception as e:   # the expected outcome: block shapes do not match
        out = {"runs": False, "error": f"{type(e).__name__}: {e}"}
    print("r0", out)
    _save("r0_core_proxy", out)
    return out


# ---- r1 ---------------------------------------------------------------------------------------------------
def r1(n: int) -> Record:
    rows: list[Record] = []
    F, T = np.array([0.3, -1.0]), 0.7
    for a in [0.3, 1.0, float(np.sqrt(np.e)), 3.0]:
        g = S.ellipse(n, a, a)
        b = S.rhs_forces(g, F[None], np.array([T]))
        r = case(g, "stokes", "weak", "proxy-sl", eps=1e-12, b=b, keep=True)
        ex = S.circle_mobility(a, F, T)
        got = S.rigid_motions(g, r["_x"])[0]
        rows.append(dict(test="circle", a=a, exact=ex, fast=got, abs_err=float(np.abs(got - ex).max()),
                         rel_err=float(np.abs(got - ex).max() / np.abs(ex).max()), solve_err=r["err"]))
    for bb in [1.0, 0.5, 0.2, 0.1]:
        for phi in [0.0, 0.7, 1.4]:
            g = S.ellipse(n, 1.0, bb, angle=phi)
            r = case(g, "stokes", "weak", "proxy-sl", eps=1e-12, b=S.rhs_shear(g), keep=True)
            got = S.rigid_motions(g, r["_x"])[0]
            om = S.jeffery_omega(1.0, bb, phi)
            rows.append(dict(test="jeffery", b=bb, phi=phi, exact=om, fast=got[2], U=got[:2],
                             abs_err=float(abs(got[2] - om)), rel_err=float(abs(got[2] - om) / abs(om)),
                             solve_err=r["err"]))
    for x in rows:
        print("r1", x["test"], {k: v for k, v in x.items() if k in ("a", "b", "phi", "abs_err", "solve_err")})
    _save("r1_analytic", rows)
    return {"rows": rows}


# ---- r2 ---------------------------------------------------------------------------------------------------
def r2(ns: list[int], eps_list: list[float], n_eps: int) -> Record:
    rows: list[Record] = []
    for n in ns:
        g = S.starfish(n)
        for phys in PHYS:
            for rule in RULES:
                for kind in EXP:
                    r = _strip(case(g, phys, rule, kind, eps=1e-10))
                    rows.append(r)
                    print("r2", n, phys, rule, kind, f"err {r['err']:.1e}", "skel", r["skel_mean"], "root", r["root"])
    erows: list[Record] = []
    g = S.starfish(n_eps)
    for eps in eps_list:
        for phys in PHYS:
            for kind in EXP:
                r = _strip(case(g, phys, "weak", kind, eps=eps))
                erows.append(r)
                print("r2-eps", eps, phys, kind, f"err {r['err']:.1e}", r["skel_mean"])
    _save("r2_ranks", {"N_sweep": rows, "eps_sweep": erows})
    _plot_r2(rows, erows, ns)
    return {"N_sweep": rows, "eps_sweep": erows}


def _ratio(rows: list[Record], **sel: Any) -> dict[int, float]:
    def pick(phys: str) -> Record:
        return next(r for r in rows if r["physics"] == phys and all(r[k] == v for k, v in sel.items()))
    s, lp = pick("stokes"), pick("laplace")
    return {int(l): s["skel_mean"][l] / lp["skel_mean"][l] for l in s["skel_mean"] if lp["skel_mean"][l] > 0}


def _plot_r2(rows: list[Record], erows: list[Record], ns: list[int]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    nmax = max(ns)
    for phys, ls in (("laplace", "--"), ("stokes", "-")):
        for rule, c in (("weak", "C0"), ("strong", "C3")):
            r = next(x for x in rows if x["N_pts"] == nmax and x["physics"] == phys and x["rule"] == rule
                     and x["comp"] == "exact")
            lv = sorted(r["skel_mean"])
            ax[0].plot(lv, [r["skel_mean"][l] for l in lv], ls, color=c, marker="o", label=f"{phys} {rule}")
    ax[0].set(xlabel="tree level (leaves = right)", ylabel="mean skeleton size per box",
              title=f"exact ID, starfish N_pts={nmax}, eps=1e-10", yscale="log")
    ax[0].legend(fontsize=8)
    for rule, c in (("weak", "C0"), ("strong", "C3")):
        for kind, mk in (("exact", "o"), ("proxy-sl", "s")):
            rat = [np.mean(list(_ratio([x for x in rows if x["N_pts"] == n], rule=rule, comp=kind).values()))
                   for n in ns]
            ax[1].plot(ns, rat, marker=mk, color=c, label=f"{rule} {kind}")
    ax[1].axhline(2, color="k", lw=0.8, ls=":")
    ax[1].set(xscale="log", xlabel="N points (Stokes dofs = 2N)", ylabel="Stokes / Laplace skeleton (mean over levels)",
              title="rank ratio", ylim=(1, 3))
    ax[1].legend(fontsize=8)
    eps_vals = sorted({r["eps"] for r in erows})
    for kind, mk in (("exact", "o"), ("proxy-sl", "s")):
        rat = [np.mean(list(_ratio([x for x in erows if x["eps"] == e], comp=kind).values())) for e in eps_vals]
        ax[2].plot(eps_vals, rat, marker=mk, label=kind)
    ax[2].axhline(2, color="k", lw=0.8, ls=":")
    ax[2].set(xscale="log", xlabel="eps", ylabel="Stokes / Laplace skeleton", title="rank ratio vs eps (weak)",
              ylim=(1, 3))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r2_ranks.png"), dpi=120)
    plt.close(fig)


# ---- r3 ---------------------------------------------------------------------------------------------------
def r3(Ks: list[int], n_per: int) -> Record:
    rows: list[Record] = []
    for K in Ks:
        g = S.random_suspension(K, n_per, box=1.6 * np.sqrt(K), gap=0.05, seed=K)
        for phys in PHYS:
            A, _, _ = build(g, phys)
            b: Vector = np.random.default_rng(1).standard_normal(size=(int(A.shape[0]),))
            t0 = time.perf_counter()
            xd = np.linalg.solve(A, b)
            td = time.perf_counter() - t0
            for rule in RULES:
                for kind in EXP:
                    r = _strip(case(g, phys, rule, kind, eps=1e-10, A=A, b=b, x_dense=xd))
                    r.update(K=K, dense_s=td, dense_storage=int(A.size))
                    rows.append(r)
                    print("r3", K, phys, rule, kind, f"N_dof {r['N_dof']} err {r['err']:.1e} root {r['root']}"
                          f" factor {r['factor_s']:.1f}s dense {td:.1f}s storage/N^2 {r['storage'] / A.size:.3f}")
            del A
    _save("r3_suspension", rows)
    return {"rows": rows}


# ---- r4 ---------------------------------------------------------------------------------------------------
def _rowside_residual(g: S.Geometry, phys: Physics, sources: str) -> tuple[float, float, float]:
    """Largest leaf box. Row side (box velocity due to far sources) vs the span of the proxy sources, truncated
    at 1e-10 relative: residual of the true far rows, of the constant fields on the box's curve points, and of
    the constant fields on 64 random points filling the box (an AREA sample, as in a volume problem)."""
    A, tree, nc = build(g, phys)
    B = max(tree.levels[-1], key=lambda nd: nd.idx.size)
    R = 2.5 * B.half_width
    th = 2 * np.pi * np.arange(64) / 64
    nrm: Points = np.c_[np.cos(th), np.sin(th)]
    Pp = B.center + R * nrm
    pts = np.unique(B.idx // nc)
    Xb = g.X[pts]
    if phys == "stokes":
        sl, dl = S.stokeslet(Xb, Pp).T, S.stresslet_dl(Xb, Pp, nrm).T
    else:
        r = np.linalg.norm(Xb[:, None] - Pp[None], axis=2)
        sl, dl = (-np.log(r) / (2 * np.pi)).T, S.laplace_dl(Xb, Pp, nrm).T
    basis: Matrix = {"sl": sl, "dl": dl, "both": np.vstack([sl, dl])}[sources]
    dofxy = np.repeat(g.X, nc, axis=0)
    far = np.setdiff1d(np.arange(A.shape[0]), B.idx)
    far = far[np.linalg.norm(dofxy[far] - B.center, axis=1) > 1.2 * R]
    Y = A[np.ix_(B.idx, far)].T
    U, sv, _ = np.linalg.svd(basis.T, full_matrices=False)
    Q = U[:, sv > 1e-10 * sv[0]]
    const = np.tile(np.eye(nc), (pts.size, 1)).T

    def res(X: Matrix) -> float:
        return float(np.linalg.norm(X - (X @ Q) @ Q.T) / np.linalg.norm(X))

    Xa = B.center + B.half_width * np.random.default_rng(0).uniform(-1, 1, (64, 2))
    if phys == "stokes":
        bas_a: Matrix = {"sl": S.stokeslet(Xa, Pp).T, "dl": S.stresslet_dl(Xa, Pp, nrm).T,
                         "both": np.vstack([S.stokeslet(Xa, Pp).T, S.stresslet_dl(Xa, Pp, nrm).T])}[sources]
    else:
        la = (-np.log(np.linalg.norm(Xa[:, None] - Pp[None], axis=2)) / (2 * np.pi)).T
        bas_a = {"sl": la, "dl": S.laplace_dl(Xa, Pp, nrm).T, "both": np.vstack([la, S.laplace_dl(Xa, Pp, nrm).T])}[sources]
    Ua, sa, _ = np.linalg.svd(bas_a.T, full_matrices=False)
    Qa = Ua[:, sa > 1e-10 * sa[0]]
    ca = np.tile(np.eye(nc), (64, 1)).T
    area = float(np.linalg.norm(ca - (ca @ Qa) @ Qa.T) / np.linalg.norm(ca))
    return res(Y), res(const), area


def r4(n: int, fracs: list[float]) -> Record:
    g0 = S.starfish(n)
    hw = Tree(g0.X, max_leaf=32).levels[-1][0].half_width
    rows: list[Record] = []
    for phys, Rstar in DEG:
        for f in fracs:
            g = S.starfish(n, scale=Rstar / (2.5 * hw) * f)     # leaf proxy radius = f * R*
            rec: Record = dict(physics=phys, R_over_Rstar=f, R=f * Rstar)
            for src in ("sl", "dl", "both"):
                rec[f"err_{src}"] = case(g, phys, "weak", SRC[src], eps=1e-10)["err"]
                rec[f"rowres_{src}"], rec[f"constres_{src}"], rec[f"arearesconst_{src}"] = _rowside_residual(g, phys, src)
            rows.append(rec)
            print("r4", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in rec.items()})
    # proxy-ring size: n_proxy and radius factor (Stokes, sl and both)
    prow: list[Record] = []
    for npx in [8, 16, 24, 32, 64]:
        for rf in [1.5, 2.5, 4.0]:
            for kind in PB:
                r = case(g0, "stokes", "weak", kind, eps=1e-10, n_proxy=npx, radius_factor=rf)
                prow.append(dict(n_proxy=npx, radius_factor=rf, comp=kind, err=r["err"], skel_total=r["skel_total"]))
                print("r4-ring", prow[-1])
    _save("r4_proxy", {"degenerate": rows, "ring": prow})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for phys, c in (("stokes", "C0"), ("laplace", "C1")):
        rr = [r for r in rows if r["physics"] == phys]
        x = [r["R_over_Rstar"] for r in rr]
        ax[0].semilogy(x, [r["constres_sl"] for r in rr], "o-", color=c, label=f"{phys}: constant field, SL proxies")
        ax[0].semilogy(x, [r["rowres_sl"] for r in rr], "s--", color=c, label=f"{phys}: true far rows, SL proxies")
        ax[0].semilogy(x, [r["rowres_dl"] for r in rr], "^:", color=c, label=f"{phys}: true far rows, DL proxies")
        ax[0].semilogy(x, [r["arearesconst_sl"] for r in rr], "x-", color=c, lw=2,
                       label=f"{phys}: constant field on AREA points, SL")
        for src, mk in (("sl", "o"), ("dl", "^"), ("both", "s")):
            ax[1].semilogy(x, [r[f"err_{src}"] for r in rr], mk + "-", color=c, label=f"{phys} {src}")
    ax[0].set(xlabel="leaf proxy radius / degenerate radius", ylabel="residual outside proxy span",
              title="row-side containment (largest leaf box)")
    ax[1].set(xlabel="leaf proxy radius / degenerate radius", ylabel="solve error vs dense",
              title="weak RS, eps = 1e-10")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r4_degenerate.png"), dpi=120)
    plt.close(fig)
    return {"degenerate": rows, "ring": prow}


def r4b(n: int, fracs: list[float]) -> Record:
    """Area (volume-sum) analogue of P2 with the Stokeslet: proxy span loses constants at R = sqrt(e) on AREA points;
    does the solve notice? Proxy radius at level depth-1 swept through the degenerate radius."""
    from .compressor import BlockProxyCompressor
    from unified.core import Strong
    g0, _ = S.area_problem(n)
    t = Tree(g0.X, max_leaf=32)
    lev = t.depth - 1
    hwl = t.levels[lev][0].half_width
    rows: list[Record] = []
    for phys, Rstar, nc in (("stokes", float(np.sqrt(np.e)), 2), ("laplace", 1.0, 1)):
        for f in fracs:
            g, A = S.area_problem(n, Rstar / (2.5 * hwl) * f, phys)
            tree = Tree(np.repeat(g.X, nc, axis=0), depth=t.depth)
            b: Vector = np.random.default_rng(1).standard_normal(size=(int(A.shape[0]),))
            xd = np.linalg.solve(A, b)
            rec: Record = dict(physics=phys, level=lev, R_over_Rstar=f)
            for src in ("sl", "dl", "both"):
                for rule in RULES:
                    comp = BlockProxyCompressor(g, 1e-10, A, tree, ncomp=nc, sources=src, rigid_rows=False,
                                                strong=rule == "strong", matrix_kernel="sl")
                    x = rskel_factor(A, tree, Strong() if rule == "strong" else Weak(), comp).solve(b)
                    rec[f"err_{src}_{rule}"] = float(np.linalg.norm(x - xd) / np.linalg.norm(xd))
            rows.append(rec)
            print("r4b", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in rec.items()})
    _save("r4b_area_degenerate", rows)
    return {"rows": rows}


# ---- r5 ---------------------------------------------------------------------------------------------------
def r5(n: int) -> Record:
    rows: list[Record] = []
    rng = np.random.default_rng(0)
    geoms = {"starfish": S.starfish(n), "suspension K=8": S.random_suspension(8, n // 8, 4.0, 0.05, seed=1)}
    for gname, g in geoms.items():
        for wname, gg in (("smooth", g), ("rough", dataclasses.replace(g, pw=g.w * rng.uniform(0.2, 1.8, g.n)))):
            for rule in RULES:
                for kind in PN:
                    r = case(gg, "stokes", rule, kind, eps=1e-10)
                    rows.append(dict(geometry=gname, weights=wname, rule=rule, comp=kind, err=r["err"],
                                     skel_total=r["skel_total"]))
                    print("r5", rows[-1])
    _save("r5_projector", rows)
    return {"rows": rows}


# ---- r6 ---------------------------------------------------------------------------------------------------
def r6(n_per: int, gaps: list[float], aspects: list[float], n_ellipse: int) -> Record:
    grow: list[Record] = []
    F = np.array([[0.0, -1.0], [0.0, -1.0]])
    T = np.zeros(2)
    for d in gaps:
        def pair(m: int) -> S.Geometry:
            return S.union([S.ellipse(m, 0.5, 0.5, (-0.5 - d / 2, 0.0)), S.ellipse(m, 0.5, 0.5, (0.5 + d / 2, 0.0))])
        ref_g = pair(4 * n_per)                         # reference: 4x finer, dense
        ref = S.rigid_motions(ref_g, np.linalg.solve(S.stokes_matrix(ref_g), S.rhs_forces(ref_g, F, T)))
        g = pair(n_per)
        b = S.rhs_forces(g, F, T)
        A = S.stokes_matrix(g)
        xd = np.linalg.solve(A, b)
        dense = S.rigid_motions(g, xd)
        rec: Record = dict(gap=d, h=float(g.w.max()), ref_U=ref, disc_err=float(np.abs(dense - ref).max() / np.abs(ref).max()))
        for rule in RULES:
            r = case(g, "stokes", rule, "proxy-sl", eps=1e-10, A=A, b=b, x_dense=xd, keep=True)
            rec[f"{rule}_err"] = r["err"]
            rec[f"{rule}_mob_err"] = float(np.abs(S.rigid_motions(g, r["_x"]) - dense).max() / np.abs(dense).max())
            rec[f"{rule}_skel"] = r["skel_mean"]
            rec[f"{rule}_root"] = r["root"]
        rec["cond_A"] = float(np.linalg.cond(A))
        grow.append(rec)
        print("r6-gap", {k: v for k, v in rec.items() if k not in ("ref_U", "weak_skel", "strong_skel")})
    arow: list[Record] = []
    for asp in aspects:
        g = S.ellipse(n_ellipse, 1.0, asp, angle=0.4)
        b = S.rhs_shear(g)
        A = S.stokes_matrix(g)
        xd = np.linalg.solve(A, b)
        ex = S.jeffery_omega(1.0, asp, 0.4)
        rec = dict(aspect=asp, dense_jeffery_err=float(abs(S.rigid_motions(g, xd)[0, 2] - ex) / abs(ex)))
        for rule in RULES:
            r = case(g, "stokes", rule, "proxy-sl", eps=1e-10, A=A, b=b, x_dense=xd, keep=True)
            rec[f"{rule}_err"] = r["err"]
            rec[f"{rule}_jeffery_err"] = float(abs(S.rigid_motions(g, r["_x"])[0, 2] - ex) / abs(ex))
            rec[f"{rule}_skel_total"] = r["skel_total"]
            rec[f"{rule}_root"] = r["root"]
        arow.append(rec)
        print("r6-aspect", rec)
    _save("r6_breaks", {"gap": grow, "aspect": arow})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    gx = [r["gap"] for r in grow]
    ax[0].loglog(gx, [r["disc_err"] for r in grow], "o-", label=f"discretisation (n={n_per}/body) vs 4x finer")
    ax[0].loglog(gx, [r["weak_mob_err"] for r in grow], "s-", label="fast (weak) vs dense, same n")
    ax[0].loglog(gx, [r["strong_mob_err"] for r in grow], "^-", label="fast (strong) vs dense, same n")
    ax[0].set(xlabel="gap between two unit circles", ylabel="relative error in (U, Omega)",
              title="two sedimenting circles: what breaks first")
    ax[0].legend(fontsize=8)
    ax2 = ax[0].twinx()
    ax2.semilogx(gx, [r["weak_root"] for r in grow], "k:", label="weak root size")
    ax2.set_ylabel("weak root size (dotted)")
    axx = [r["aspect"] for r in arow]
    ax[1].loglog(axx, [r["dense_jeffery_err"] for r in arow], "o-", label=f"dense vs Jeffery (n={n_ellipse})")
    ax[1].loglog(axx, [r["weak_jeffery_err"] for r in arow], "s--", label="fast weak vs Jeffery")
    ax[1].loglog(axx, [r["weak_err"] for r in arow], "^:", label="fast weak vs dense (density)")
    ax[1].set(xlabel="aspect ratio b/a", ylabel="relative error", title="thin ellipse in shear")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r6_breaks.png"), dpi=120)
    plt.close(fig)
    return {"gap": grow, "aspect": arow}


# ---- r7 ---------------------------------------------------------------------------------------------------
def r7(K: int, n_per: int) -> Record:
    g = S.random_suspension(K, n_per, box=1.6 * np.sqrt(K), gap=0.05, seed=7)
    A, tree, _ = build(g, "stokes")
    t0 = time.perf_counter()
    Fz = rskel_factor(A, tree, Weak(), make_comp("proxy-sl", g, A, tree, 2, "weak", 1e-10))
    tf = time.perf_counter() - t0
    M = np.zeros((3 * K, 3 * K))
    Md = np.zeros((3 * K, 3 * K))
    t0 = time.perf_counter()
    rhs = []
    for j in range(3 * K):
        Fv, Tv = np.zeros((K, 2)), np.zeros(K)
        if j % 3 < 2:
            Fv[j // 3, j % 3] = 1.0
        else:
            Tv[j // 3] = 1.0
        rhs.append(S.rhs_forces(g, Fv, Tv))
    Bm = np.stack(rhs, 1)
    for j in range(3 * K):
        M[:, j] = S.rigid_motions(g, Fz.solve(Bm[:, j])).ravel()
    ts = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xd = np.linalg.solve(A, Bm)
    td = time.perf_counter() - t0
    for j in range(3 * K):
        Md[:, j] = S.rigid_motions(g, Xd[:, j]).ravel()
    sym = float(np.linalg.norm(M - M.T) / np.linalg.norm(M))
    sym_d = float(np.linalg.norm(Md - Md.T) / np.linalg.norm(Md))
    out: Record = dict(K=K, N_dof=A.shape[0], fast_vs_dense=float(np.linalg.norm(M - Md) / np.linalg.norm(Md)),
                       asym_fast=sym, asym_dense=sym_d, eig_sym_min=float(np.linalg.eigvalsh((M + M.T) / 2).min()),
                       eig_sym_max=float(np.linalg.eigvalsh((M + M.T) / 2).max()),
                       factor_s=tf, solves_s=ts, dense_multi_rhs_s=td)
    # dissipation F.U >= 0 is only finite for zero net force in 2D (Stokes paradox): restrict to sum_k F_k = 0
    C = np.zeros((2, 3 * K))
    for k in range(K):
        C[0, 3 * k], C[1, 3 * k + 1] = 1.0, 1.0
    Nsp = np.linalg.svd(C)[2][2:].T                   # basis of {sum F = 0} (torques free)
    Ms = Nsp.T @ ((M + M.T) / 2) @ Nsp
    out["eig_forcefree_min"] = float(np.linalg.eigvalsh(Ms).min())
    print("r7", out)
    # sedimentation (all F = (0, -1)) velocity field picture
    Fv = np.tile([0.0, -1.0], (K, 1))
    U = (M @ np.c_[Fv, np.zeros(K)].ravel()).reshape(K, 3)
    out["sediment_U"] = U
    fig, ax = plt.subplots(figsize=(6, 6))
    for k in range(K):
        P = g.X[g.body == k]
        ax.fill(P[:, 0], P[:, 1], color="0.8", ec="k", lw=0.6)
        ax.arrow(*g.centers[k], *(U[k, :2] * 3), width=0.01, color="C3")
    ax.set_aspect("equal")
    ax.set_title(f"{K} ellipses sedimenting (F = (0,-1) each): U x 3 (red)\nmobility matrix asym "
                 f"{sym:.1e}, fast vs dense {out['fast_vs_dense']:.1e}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r7_suspension.png"), dpi=120)
    plt.close(fig)
    _save("r7_mobility", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    q = a.quick
    todo = a.only.split(",") if a.only else ["r0", "r1", "r2", "r3", "r4", "r4b", "r5", "r6", "r7"]
    t0 = time.perf_counter()
    if "r0" in todo:
        r0()
    if "r1" in todo:
        r1(256 if q else 1024)
    if "r2" in todo:
        r2([512, 1024] if q else [512, 1024, 2048, 4096], [1e-6, 1e-10] if q else [1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
           1024 if q else 2048)
    if "r3" in todo:
        r3([4, 8] if q else [4, 8, 16, 32], 128)
    if "r4" in todo:
        r4(512 if q else 1024, [0.9, 1.0, 1.1] if q else [0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.25])
    if "r4b" in todo:
        r4b(1024, [0.97, 1.0, 1.03] if q else [0.9, 0.97, 0.99, 1.0, 1.01, 1.03, 1.1])
    if "r5" in todo:
        r5(512 if q else 1024)
    if "r6" in todo:
        r6(128, [0.5, 0.1] if q else [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
           [1.0, 0.1] if q else [1.0, 0.3, 0.1, 0.03, 0.01], 512 if q else 1024)
    if "r7" in todo:
        r7(8 if q else 24, 128)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
