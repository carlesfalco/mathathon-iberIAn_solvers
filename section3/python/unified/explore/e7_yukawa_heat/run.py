"""E7: Yukawa / modified Helmholtz kernel K0(κr) in the unified harness; heat equation by Rothe's method.

Reproduce:  PYTHONPATH=. .venv/bin/python -m unified.explore.e7_yukawa_heat.run [--quick]
Writes results/*.json and results/*.png (≈ 5 min; --quick ≈ 1 min).

    r1  rank model: per-level ranks of true weak / strong far blocks and of proxy-ring rows vs κ·hw,
        relative (harness) vs absolute tolerance                                  -> r1_ranks
    r2  κ sweep of the full harness on Y1 (SLP, kernel-only swap) and Y2 (DLP Dirichlet BVP):
        err vs dense, err vs analytic, skeletons, factor flops/storage, breakdowns -> r2_ksweep
    r3  where the ANSWER breaks: analytic error vs κh for several N                   -> r3_quadrature
    r4  factor cost vs N at fixed κ (does truncation make it O(N)?)                 -> r4_scaling
    r5  heat equation: Rothe + one fast factorization, vs semi-discrete exact and continuous (O(dt)) -> r5_heat
    r6  κ → 0 (Laplace limit): monopole-proxy precision loss and the centred-proxy fix; Y1 conditioning -> r6_small_kappa
    r7  near-field truncation (NearTruncated: norm / rank criterion) vs weak / strong: Y2 curve, Y3 volume,
        and the heat equation with the truncated-strong factor                                  -> r7_near
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from unified.bench.problems import Problem, dense_identity_plus, starfish
from unified.core import ExactCompressor, ProxyCompressor, Strong, Tree, Weak
from unified.core.types import Admissibility, Compressor, Points, RealArray, Vector
from unified.drivers.rskel import Factorization, rskel_factor

from .heat import continuous_exact, rothe_heat, semidiscrete_exact
from .yukawa import (AbsTolExact, CenteredProxy, NearTruncated, Truncated, cutoff_radius, dlp_eval, dlp_problem, slp_kernel,
                     slp_problem)

OUT = os.path.join(os.path.dirname(__file__), "results")
EPS = 1e-8
SCALE = 0.5                  # matrix scale for the absolute tolerance: A = −½I + (small), |A_ii| ≈ ½
type Row = dict[str, Any]

# BVP analytic check: sum of three screened point charges OUTSIDE the starfish; targets inside, ≥ 0.05 from Γ
# (the closest source-target pair is 0.10 apart, so K0 stays above underflow up to κ ≈ 3000)
SOURCES: Points = np.array([[1.36, 0.0], [0.0, 1.06], [-0.9, -0.9]])
TARGETS: Points = np.array([[1.26, 0.0], [1.2, 0.0], [1.1, 0.05], [0.0, 0.94], [0.3, 0.2]])
X0_HEAT = np.array([1.34, 0.0])           # heat source 0.04 outside the petal tip (1.3, 0)
HEAT_TARGETS: Points = np.array([[1.26, 0.0], [1.2, 0.0], [1.1, 0.0], [0.8, 0.0]])


def _save(name: str, obj: Any) -> None:
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def factor_cost(F: Factorization) -> tuple[float, int]:
    """(flop estimate, numbers stored) of a factorization: per step LU of X_RR (r³·2/3), the two
    off-diagonal solves and the Schur update (2 r² s + 2 r s²), plus the root LU."""
    flops, store = 0.0, 0
    for s in F.steps:
        r, k = s.red.size, s.sn.size
        flops += 2 / 3 * r ** 3 + 2 * r * r * k + 2 * r * k * k
        store += s.T.size + s.lu[0].size + s.X_snr.size + s.X_rsn.size
    m = F.root.size
    return flops + 2 / 3 * m ** 3, store + m * m


def configs(P: Problem, kappa: float, n: int, rules: tuple[str, ...]) -> list[tuple[str, Admissibility, Compressor]]:
    """The compressor variants compared in r2/r4 (same ε everywhere)."""
    X, w = starfish(n)
    rc = cutoff_radius(kappa, EPS, n, float(w.max()), SCALE)
    out: list[tuple[str, Admissibility, Compressor]] = []
    for rule in rules:
        adm: Admissibility = Weak() if rule == "weak" else Strong()
        out += [(f"{rule}/exact-rel", adm, ExactCompressor(P.A, EPS)),
                (f"{rule}/exact-abs", adm, AbsTolExact(P.A, EPS * SCALE)),
                (f"{rule}/proxy-rel", adm, ProxyCompressor(P.kernel, P.X, EPS)),
                (f"{rule}/abs+trunc", adm, Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))]
        if rule == "weak":
            out.append((f"{rule}/proxy+trunc", adm, Truncated(ProxyCompressor(P.kernel, P.X, EPS), P.X, rc)))
    return out


def run_one(P: Problem, tree: Tree, name: str, adm: Admissibility, comp: Compressor, b: Vector,
            x_ref: Vector) -> Row:
    row: Row = {"config": name}
    t0 = time.perf_counter()
    try:
        with np.errstate(all="ignore"):
            F = rskel_factor(P.A, tree, adm, comp)
            x = F.solve(b)
    except (ValueError, np.linalg.LinAlgError) as e:
        row.update(breakdown=type(e).__name__ + ": " + str(e)[:60])
        return row
    row["factor_time"] = time.perf_counter() - t0
    err = float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    row["err_dense"] = err if np.isfinite(err) else None
    if not np.isfinite(err):
        row["breakdown"] = "non-finite solution"
    fl, st = factor_cost(F)
    lv = F.stats["levels"]
    row.update(flops=fl, storage=st, root=F.stats["root_size"], x=x,
               skel_total=int(sum(v["skel_total"] for v in lv.values())),
               skel_mean={int(k): v["skel_mean"] for k, v in lv.items()})
    return row


# ---- r1 ----------------------------------------------------------------------------------------------------
def r1_ranks(n: int, kappas: RealArray) -> list[Row]:
    """Per level: mean numerical rank of [A_FB; A_BFᵀ] (weak far / strong far) and of the proxy-ring rows."""
    rows: list[Row] = []
    X, _ = starfish(n)
    tree = Tree(X, max_leaf=32)
    for kappa in kappas:
        P = dlp_problem(float(kappa), n)
        for l in range(2, tree.depth + 1):
            acc: dict[str, list[int]] = {}
            boxes = tree.levels[l][:: max(1, len(tree.levels[l]) // 6)]
            for B in boxes:
                near = np.concatenate([nb.idx for nb in tree.neighbors(B)])
                mask = np.ones(n, bool); mask[B.idx] = False
                far_w = np.flatnonzero(mask); mask[near] = False
                far_s = np.flatnonzero(mask)
                prox = ProxyCompressor(P.kernel, X, EPS, n_proxy=128)
                Pp = prox.proxy_points(B)
                blocks = {"weak": np.vstack([P.A[np.ix_(far_w, B.idx)], P.A[np.ix_(B.idx, far_w)].T]),
                          "strong": np.vstack([P.A[np.ix_(far_s, B.idx)], P.A[np.ix_(B.idx, far_s)].T]),
                          "ring": np.vstack([P.kernel(Pp, X[B.idx]), P.kernel(X[B.idx], Pp).T])}
                for key, M in blocks.items():
                    if M.shape[0] == 0:
                        continue
                    s = np.linalg.svd(M, compute_uv=False)
                    acc.setdefault(key + "_rel", []).append(int(np.sum(s > EPS * s[0])) if s[0] > 0 else 0)
                    if key != "ring":     # ring rows have no absolute meaning (unknown source strengths)
                        acc.setdefault(key + "_abs", []).append(int(np.sum(s > EPS * SCALE)))
            hw = tree.levels[l][0].half_width
            rows.append({"kappa": float(kappa), "level": l, "hw": hw, "box_size": float(np.mean([len(B.idx) for B in boxes])),
                         **{k: float(np.mean(v)) for k, v in acc.items()}})
        print(f"  r1 kappa={kappa:8.3g} done", flush=True)
    return rows


# ---- r2 ----------------------------------------------------------------------------------------------------
def r2_ksweep(n: int, kappas: list[float]) -> list[Row]:
    rows: list[Row] = []
    X, _ = starfish(n)
    tree = Tree(X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    for kappa in kappas:
        for prob in ("Y2-DLP", "Y1-SLP"):
            P = dlp_problem(kappa, n) if prob == "Y2-DLP" else slp_problem(kappa, n)
            t0 = time.perf_counter()
            lu = sla.lu_factor(cast(Any, P.A))
            t_dense = time.perf_counter() - t0
            x_ref = sla.lu_solve(lu, b)
            cond = float(np.linalg.cond(P.A)) if n <= 4096 else None
            g = slp_kernel(kappa)(X, SOURCES).sum(1)
            u_ex = slp_kernel(kappa)(TARGETS, SOURCES).sum(1)
            cfgs = configs(P, kappa, n, ("weak", "strong"))
            if prob == "Y1-SLP":              # the kernel-only swap: harness defaults only
                cfgs = [c for c in cfgs if c[0].endswith(("exact-rel", "proxy-rel"))]
            for name, adm, comp in cfgs:
                row = run_one(P, tree, name, adm, comp, b, x_ref)
                row.update(problem=prob, kappa=kappa, n=n, cond=cond, dense_time=t_dense,
                           dense_flops=2 / 3 * n ** 3)
                row.pop("x", None)
                if prob == "Y2-DLP" and "breakdown" not in row:   # analytic check through the FAST solver
                    with np.errstate(all="ignore"):
                        F = rskel_factor(P.A, tree, adm, comp)
                        u = dlp_eval(kappa, n, TARGETS, F.solve(g))
                    ud = dlp_eval(kappa, n, TARGETS, sla.lu_solve(lu, g))
                    scale = float(np.max(np.abs(u_ex)))
                    ok = scale > 1e-280             # otherwise the analytic solution underflows at every target
                    row["err_analytic"] = float(np.max(np.abs(u - u_ex)) / scale) if ok else None
                    row["err_analytic_dense"] = float(np.max(np.abs(ud - u_ex)) / scale) if ok else None
                rows.append(row)
                print(f"  {prob} k={kappa:8.3g} {name:18s} " + (f"BREAKDOWN {row['breakdown']}" if "breakdown" in row else
                      f"err={row['err_dense']:.1e} skel={row['skel_total']:5d} flops={row['flops']:.2e} "
                      f"an={row.get('err_analytic') or float('nan'):.1e}"), flush=True)
    return rows


# ---- r3 ----------------------------------------------------------------------------------------------------
def r3_quadrature(ns: list[int], kappas: list[float]) -> list[Row]:
    rows: list[Row] = []
    for n in ns:
        X, w = starfish(n)
        tree = Tree(X, max_leaf=32)
        for kappa in kappas:
            P = dlp_problem(kappa, n)
            g = slp_kernel(kappa)(X, SOURCES).sum(1)
            u_ex = slp_kernel(kappa)(TARGETS, SOURCES).sum(1)
            rc = cutoff_radius(kappa, EPS, n, float(w.max()), SCALE)
            F = rskel_factor(P.A, tree, Strong(), Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))
            sf = F.solve(g)
            sd = np.linalg.solve(P.A, g)
            e_f = float(np.max(np.abs(dlp_eval(kappa, n, TARGETS, sf) - u_ex)) / np.max(np.abs(u_ex)))
            e_d = float(np.max(np.abs(dlp_eval(kappa, n, TARGETS, sd) - u_ex)) / np.max(np.abs(u_ex)))
            rows.append({"n": n, "kappa": kappa, "kappa_h": kappa * 9.0172 / n, "err_fast": e_f, "err_dense": e_d,
                         "fast_vs_dense": float(np.linalg.norm(sf - sd) / np.linalg.norm(sd))})
            print(f"  r3 N={n} k={kappa:7.3g} kh={kappa*9.0172/n:6.3f} analytic err fast={e_f:.1e} dense={e_d:.1e}", flush=True)
    return rows


# ---- r4 ----------------------------------------------------------------------------------------------------
def r4_scaling(ns: list[int], kappas: list[float]) -> list[Row]:
    rows: list[Row] = []
    for n in ns:
        X, _ = starfish(n)
        tree = Tree(X, max_leaf=32)
        b = np.random.default_rng(1).standard_normal(n)
        for kappa in kappas:
            P = dlp_problem(kappa, n)
            x_ref = np.linalg.solve(P.A, b)
            for name, adm, comp in configs(P, kappa, n, ("weak", "strong")):
                if not name.endswith(("exact-rel", "abs+trunc", "proxy-rel")):
                    continue
                row = run_one(P, tree, name, adm, comp, b, x_ref)
                row.pop("x", None)
                row.update(n=n, kappa=kappa)
                rows.append(row)
                print(f"  r4 N={n} k={kappa:6.3g} {name:16s} " + (f"BREAKDOWN" if "breakdown" in row else
                      f"err={row['err_dense']:.1e} flops={row['flops']:.2e} store={row['storage']:.2e} t={row['factor_time']:.2f}"), flush=True)
            del P
    return rows


# ---- r5 ----------------------------------------------------------------------------------------------------
def r5_heat(n: int, dts: list[float], n_steps: int, T: float, conv_steps: list[int]) -> Row:
    out: Row = {"semidiscrete": [], "convergence": []}
    X, w = starfish(n)
    tree = Tree(X, max_leaf=32)
    for dt in dts:
        kappa = 1 / np.sqrt(dt)
        P = dlp_problem(kappa, n)
        rc = cutoff_radius(kappa, EPS, n, float(w.max()), SCALE)
        t0 = time.perf_counter()
        F = rskel_factor(P.A, tree, Strong(), Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))
        t_fac = time.perf_counter() - t0
        lu = sla.lu_factor(cast(Any, P.A))
        ex = semidiscrete_exact(kappa, X0_HEAT)
        t0 = time.perf_counter()
        fast = rothe_heat(kappa, n, n_steps, ex, HEAT_TARGETS, F.solve)
        t_march = time.perf_counter() - t0
        dense = rothe_heat(kappa, n, n_steps, ex, HEAT_TARGETS, lambda r: sla.lu_solve(lu, r))
        uex = np.array([ex(HEAT_TARGETS, t) for t in fast.t])
        gmax = max(float(np.max(ex(X, t))) for t in fast.t)
        fl, st = factor_cost(F)
        row = {"dt": dt, "kappa": kappa, "n_steps": n_steps, "rcut": rc, "factor_time": t_fac, "march_time": t_march,
               "flops": fl, "storage": st,
               "err_fast_vs_exact": float(np.max(np.abs(fast.u - uex)) / np.max(np.abs(uex))),
               "err_dense_vs_exact": float(np.max(np.abs(dense.u - uex)) / np.max(np.abs(uex))),
               "fast_vs_dense": float(np.max(np.abs(fast.u - dense.u)) / np.max(np.abs(dense.u))),
               "max_principle": bool(np.all(fast.u >= -1e-10 * gmax) and np.all(fast.u <= gmax * (1 + 1e-10))),
               "u_min": float(fast.u.min()), "u_max": float(fast.u.max()), "g_max": gmax,
               "t": fast.t.tolist(), "u_target0": fast.u[:, 1].tolist(), "uex_target0": uex[:, 1].tolist()}
        out["semidiscrete"].append(row)
        print(f"  r5 dt={dt:.0e} kappa={kappa:6.1f} fast-vs-exact={row['err_fast_vs_exact']:.1e} "
              f"dense-vs-exact={row['err_dense_vs_exact']:.1e} maxprinciple={row['max_principle']} "
              f"factor={t_fac:.2f}s march={t_march:.2f}s", flush=True)
    cont = continuous_exact(X0_HEAT)
    for ns in conv_steps:                     # O(dt) convergence to the continuous solution at time T
        kappa = float(np.sqrt(ns / T))
        P = dlp_problem(kappa, n)
        rc = cutoff_radius(kappa, EPS, n, float(w.max()), SCALE)
        F = rskel_factor(P.A, tree, Strong(), Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))
        run = rothe_heat(kappa, n, ns, cont, HEAT_TARGETS, F.solve)
        u_T = cont(HEAT_TARGETS, T)
        row = {"n_steps": ns, "dt": T / ns, "kappa": kappa,
               "err_vs_continuous": float(np.max(np.abs(run.u[-1] - u_T)) / np.max(np.abs(u_T))),
               "err_semidiscrete_formula": float(np.max(np.abs(semidiscrete_exact(kappa, X0_HEAT)(HEAT_TARGETS, T) - u_T))
                                                 / np.max(np.abs(u_T)))}
        out["convergence"].append(row)
        print(f"  r5 conv steps={ns} dt={T/ns:.1e} err vs continuous={row['err_vs_continuous']:.2e} "
              f"(point-source formula alone {row['err_semidiscrete_formula']:.2e})", flush=True)
    return out


# ---- r6 ----------------------------------------------------------------------------------------------------
def r6_small_kappa(n: int, kappas: list[float]) -> list[Row]:
    rows: list[Row] = []
    X, _ = starfish(n)
    tree = Tree(X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    for kappa in kappas:
        P = dlp_problem(kappa, n)
        x_ref = np.linalg.solve(P.A, b)
        for rule, adm in (("weak", Weak()), ("strong", Strong())):
            comps: list[tuple[str, Compressor]] = [("exact-rel", ExactCompressor(P.A, EPS)),
                                                   ("proxy-rel", ProxyCompressor(P.kernel, P.X, EPS)),
                                                   ("centred-proxy", CenteredProxy(P.kernel, P.X, EPS))]
            for cname, comp in comps:
                row = run_one(P, tree, f"{rule}/{cname}", adm, comp, b, x_ref)
                row.pop("x", None)
                row.update(kappa=kappa, n=n, problem="Y2-DLP")
                rows.append(row)
                print(f"  r6 k={kappa:7.1e} {rule}/{cname:14s} err={row.get('err_dense') or float('nan'):.1e} "
                      f"skel={row.get('skel_total')}", flush=True)
        Y1 = slp_problem(kappa, n)
        rows.append({"problem": "Y1-SLP", "kappa": kappa, "n": n, "cond": float(np.linalg.cond(Y1.A)),
                     "c_kappa": float((np.log(2 / kappa) - 0.5772156649) / (2 * np.pi))})
        print(f"  r6 k={kappa:7.1e} Y1 cond={rows[-1]['cond']:.2f}", flush=True)
    return rows


# ---- r7 ----------------------------------------------------------------------------------------------------
NEAR_RULES = ("weak", "strong", "nt-norm", "nt-rank")


def near_rule(name: str, A: Any) -> Admissibility:
    """weak / strong / strong with near-field truncation (norm ≤ τ, or abs-rank ≤ ¼ of the smaller box)."""
    if name == "weak":
        return Weak()
    if name == "strong":
        return Strong()
    return NearTruncated(Strong(), A, EPS * SCALE, 0.0 if name == "nt-norm" else 0.25, name=name)


def _near_row(A: Any, tree: Tree, rule: str, comp: Compressor, b: Vector, x_ref: Vector) -> Row:
    adm = near_rule(rule, A)
    t0 = time.perf_counter()
    F = rskel_factor(A, tree, adm, comp)
    t_fac = time.perf_counter() - t0
    x = F.solve(b)
    fl, st = factor_cost(F)
    return {"rule": rule, "flops": fl, "storage": st, "factor_time": t_fac, "root": F.stats["root_size"],
            "err_dense": float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)),
            "near_dropped": getattr(adm, "dropped", None), "near_kept": getattr(adm, "kept", None), "F": F}


def r7_near(n: int, kappas_curve: list[float], kappas_vol: list[float], n_heat: int, dts: list[float],
            n_steps: int, T: float, conv_steps: list[int]) -> Row:
    out: Row = {"curve": [], "volume": [], "heat": [], "heat_conv": []}
    X, w = starfish(n)
    tree = Tree(X, max_leaf=32)
    b = np.random.default_rng(0).standard_normal(n)
    for kappa in kappas_curve:                       # Y2: double-layer BVP on the starfish
        P = dlp_problem(kappa, n)
        x_ref = np.linalg.solve(P.A, b)
        g = slp_kernel(kappa)(X, SOURCES).sum(1)
        u_ex = slp_kernel(kappa)(TARGETS, SOURCES).sum(1)
        scale = float(np.max(np.abs(u_ex)))
        rc = cutoff_radius(kappa, EPS, n, float(w.max()), SCALE)
        for rule in NEAR_RULES:
            row = _near_row(P.A, tree, rule, Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc), b, x_ref)
            F = row.pop("F")
            u = dlp_eval(kappa, n, TARGETS, F.solve(g))
            row.update(kappa=kappa, dt=1 / kappa ** 2, n=n, dense_storage=n * n, kappa_h=kappa * 9.0172 / n,
                       err_analytic=float(np.max(np.abs(u - u_ex)) / scale) if scale > 1e-280 else None)
            out["curve"].append(row)
            print(f"  r7 curve k={kappa:7.3g} {rule:8s} err={row['err_dense']:.1e} flops={row['flops']:.2e} "
                  f"store={row['storage']:.2e} near kept/dropped={row['near_kept']}/{row['near_dropped']}", flush=True)
        del P
    Xv = np.random.default_rng(0).random((n, 2))     # Y3: P2 with the screened kernel (volume points)
    tv = Tree(Xv, max_leaf=32)
    for kappa in kappas_vol:
        G = slp_kernel(kappa)
        def kv(P_: Points, Q_: Points) -> Any:
            return G(P_, Q_) / n
        A = dense_identity_plus(kv, Xv)
        x_ref = np.linalg.solve(A, b)
        for rule in NEAR_RULES:
            row = _near_row(A, tv, rule, AbsTolExact(A, EPS * SCALE), b, x_ref)
            row.pop("F")
            row.update(kappa=kappa, n=n, dense_storage=n * n, kappa_h=kappa / np.sqrt(n))
            out["volume"].append(row)
            print(f"  r7 volume k={kappa:7.3g} {rule:8s} err={row['err_dense']:.1e} flops={row['flops']:.2e} "
                  f"store={row['storage']:.2e} near kept/dropped={row['near_kept']}/{row['near_dropped']}", flush=True)
        del A
    Xh, wh = starfish(n_heat)                        # heat equation through the truncated-strong factor
    th = Tree(Xh, max_leaf=32)
    for dt in dts:
        kappa = 1 / np.sqrt(dt)
        P = dlp_problem(kappa, n_heat)
        rc = cutoff_radius(kappa, EPS, n_heat, float(wh.max()), SCALE)
        lu = sla.lu_factor(cast(Any, P.A))
        ex = semidiscrete_exact(kappa, X0_HEAT)
        dense = rothe_heat(kappa, n_heat, n_steps, ex, HEAT_TARGETS, lambda r: sla.lu_solve(lu, r))
        uex = np.array([ex(HEAT_TARGETS, t) for t in dense.t])
        for rule in NEAR_RULES:
            adm = near_rule(rule, P.A)
            F = rskel_factor(P.A, th, adm, Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))
            run = rothe_heat(kappa, n_heat, n_steps, ex, HEAT_TARGETS, F.solve)
            fl, st = factor_cost(F)
            row = {"dt": dt, "kappa": kappa, "rule": rule, "flops": fl, "storage": st,
                   "err_vs_exact": float(np.max(np.abs(run.u - uex)) / np.max(np.abs(uex))),
                   "err_dense_vs_exact": float(np.max(np.abs(dense.u - uex)) / np.max(np.abs(uex))),
                   "fast_vs_dense": float(np.max(np.abs(run.u - dense.u)) / np.max(np.abs(dense.u)))}
            out["heat"].append(row)
            print(f"  r7 heat dt={dt:.0e} {rule:8s} vs exact={row['err_vs_exact']:.1e} (dense {row['err_dense_vs_exact']:.1e}) "
                  f"fast-vs-dense={row['fast_vs_dense']:.1e} flops={fl:.2e}", flush=True)
    cont = continuous_exact(X0_HEAT)
    for ns in conv_steps:
        kappa = float(np.sqrt(ns / T))
        P = dlp_problem(kappa, n_heat)
        rc = cutoff_radius(kappa, EPS, n_heat, float(wh.max()), SCALE)
        F = rskel_factor(P.A, th, near_rule("nt-rank", P.A), Truncated(AbsTolExact(P.A, EPS * SCALE), P.X, rc))
        run = rothe_heat(kappa, n_heat, ns, cont, HEAT_TARGETS, F.solve)
        u_T = cont(HEAT_TARGETS, T)
        row = {"n_steps": ns, "dt": T / ns, "err_vs_continuous": float(np.max(np.abs(run.u[-1] - u_T)) / np.max(np.abs(u_T)))}
        out["heat_conv"].append(row)
        print(f"  r7 heat conv (nt-rank) steps={ns} err vs continuous={row['err_vs_continuous']:.2e}", flush=True)
    return out


def plot_r7(res: Row) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, key, title in ((axs[0], "curve", "Y2 curve (DLP), N=%d"), (axs[1], "volume", "Y3 volume (I + K0/N), N=%d")):
        rows = res[key]
        for rule in NEAR_RULES:
            rr = [r for r in rows if r["rule"] == rule]
            ax.loglog([r["kappa"] for r in rr], [r["flops"] for r in rr], "-o", ms=3, label=rule)
        ax.set_title(title % rows[0]["n"]); ax.set_xlabel("κ (= 1/√dt)"); ax.set_ylabel("factor flops (estimate)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    for rule in NEAR_RULES:
        rr = [r for r in res["heat"] if r["rule"] == rule]
        axs[2].loglog([r["dt"] for r in rr], [r["err_vs_exact"] for r in rr], "-o", ms=3, label=rule)
    rr = [r for r in res["heat"] if r["rule"] == "weak"]
    axs[2].loglog([r["dt"] for r in rr], [r["err_dense_vs_exact"] for r in rr], "k:", label="dense LU")
    axs[2].set_xlabel("dt"); axs[2].set_ylabel("heat: error vs semi-discrete exact"); axs[2].grid(alpha=.3)
    axs[2].legend(fontsize=8); axs[2].set_title("heat: all rules give the dense answer")
    fig.suptitle("r7: near-field truncation (strong minus negligible / low-rank neighbours)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r7_near.png"), dpi=120); plt.close(fig)


# ---- plots -------------------------------------------------------------------------------------------------
def plot_r1(rows: list[Row]) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    levels = sorted({r["level"] for r in rows})
    for ax, key, title in zip(axs, ["strong", "weak", "ring"], ["strong far field", "weak far field", "proxy ring rows (128 pts)"]):
        for l in levels:
            rr = [r for r in rows if r["level"] == l and key + "_rel" in r]
            kh = [r["kappa"] * r["hw"] for r in rr]
            ln, = ax.semilogx(kh, [r[key + "_rel"] for r in rr], "-o", ms=3, label=f"level {l} rel")
            if key != "ring":
                ax.semilogx(kh, [r[key + "_abs"] for r in rr], "--", color=ln.get_color(), label=f"level {l} abs")
        ax.set_xlabel("κ · box half-width"); ax.set_ylabel("rank at ε = 1e-8"); ax.set_title(title)
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("r1: ranks vs screening; solid = relative tolerance (harness), dashed = absolute (ε·½)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r1_ranks.png"), dpi=120); plt.close(fig)


def plot_r2(rows: list[Row]) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    y2 = [r for r in rows if r["problem"] == "Y2-DLP"]
    for cfg in sorted({r["config"] for r in y2}):
        rr = [r for r in y2 if r["config"] == cfg and "breakdown" not in r]
        bad = [r for r in y2 if r["config"] == cfg and "breakdown" in r]
        ls = "-" if cfg.startswith("strong") else "--"
        ln, = axs[0].loglog([r["kappa"] for r in rr], [max(r["flops"], 1) for r in rr], ls + "o", ms=3, label=cfg)
        axs[1].semilogx([r["kappa"] for r in rr], [r["skel_total"] for r in rr], ls + "o", ms=3, color=ln.get_color())
        axs[2].loglog([r["kappa"] for r in rr], [max(r["err_dense"], 1e-17) for r in rr], ls + "o", ms=3, color=ln.get_color())
        for r in bad:
            axs[2].plot(r["kappa"], 1, "x", ms=10, color=ln.get_color())
    d = [r for r in y2 if r["config"] == y2[0]["config"]]
    axs[0].loglog([r["kappa"] for r in d], [r["dense_flops"] for r in d], "k:", label="dense LU")
    axs[0].set_title("factor flops (estimate)"); axs[1].set_title("total skeletons"); axs[2].set_title("‖x − x_dense‖/‖x_dense‖ (x = breakdown)")
    axs[2].axhline(EPS, color="gray", lw=.5)
    for ax in axs:
        ax.set_xlabel("κ (= 1/√dt)"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("r2: Y2 double-layer Dirichlet BVP on the starfish, N = %d" % y2[0]["n"])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r2_ksweep.png"), dpi=120); plt.close(fig)


def plot_r3(rows: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for n in sorted({r["n"] for r in rows}):
        rr = [r for r in rows if r["n"] == n]
        ln, = ax.loglog([r["kappa_h"] for r in rr], [r["err_dense"] for r in rr], "-o", ms=3, label=f"N={n} dense")
        ax.loglog([r["kappa_h"] for r in rr], [r["err_fast"] for r in rr], "x", color=ln.get_color())
    ax.set_xlabel("κ h  (h = L/N)"); ax.set_ylabel("max rel. error vs analytic"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("r3: discretization limit (x = fast solver, strong abs+trunc)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r3_quadrature.png"), dpi=120); plt.close(fig)


def plot_r4(rows: list[Row]) -> None:
    kappas = sorted({r["kappa"] for r in rows})
    fig, axs = plt.subplots(1, len(kappas), figsize=(5 * len(kappas), 4))
    for ax, k in zip(np.atleast_1d(axs), kappas):
        for cfg in sorted({r["config"] for r in rows}):
            rr = [r for r in rows if r["kappa"] == k and r["config"] == cfg and "breakdown" not in r]
            ax.loglog([r["n"] for r in rr], [r["flops"] for r in rr], ("-" if cfg.startswith("strong") else "--") + "o", ms=3, label=cfg)
        ns = np.array(sorted({r["n"] for r in rows}), float)
        ax.loglog(ns, 2 / 3 * ns ** 3, "k:", label="dense LU")
        ax.set_title(f"κ = {k:g}"); ax.set_xlabel("N"); ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].set_ylabel("factor flops (estimate)"); np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r4_scaling.png"), dpi=120); plt.close(fig)


def plot_r5(res: Row) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for r in res["semidiscrete"]:
        ln, = axs[0].plot(r["t"], r["u_target0"], "o", ms=3, label=f"Rothe+fast, dt={r['dt']:.0e}")
        axs[0].plot(r["t"], r["uex_target0"], "-", color=ln.get_color(), lw=.8)
    axs[0].set_xscale("log"); axs[0].set_xlabel("t"); axs[0].set_ylabel("u at (1.2, 0)"); axs[0].legend(fontsize=8)
    axs[0].set_title("semi-discrete exact (lines) vs computed (dots)")
    c = res["convergence"]
    axs[1].loglog([r["dt"] for r in c], [r["err_vs_continuous"] for r in c], "o-", label="Rothe + BIE + fast solver")
    dts = np.array([r["dt"] for r in c])
    axs[1].loglog(dts, c[0]["err_vs_continuous"] * dts / dts[0], "k:", label="O(dt)")
    axs[1].set_xlabel("dt"); axs[1].set_ylabel("error vs continuous at T"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "r5_heat.png"), dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of parts to run, e.g. r6 (default: all)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    q = a.quick
    want = set(a.only.split(",")) if a.only else {"r1", "r2", "r3", "r4", "r5", "r6", "r7"}
    if "r1" in want:
        print("r1 rank model", flush=True)
        r1 = r1_ranks(1024 if q else 4096, np.logspace(-3, 4, 8 if q else 22)); _save("r1_ranks", r1); plot_r1(r1)
    if "r2" in want:
        print("r2 kappa sweep", flush=True)
        kap = [1e-3, 1.0, 30.0, 300.0, 3000.0] if q else [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4]
        r2 = r2_ksweep(1024 if q else 4096, kap); _save("r2_ksweep", r2); plot_r2(r2)
    if "r3" in want:
        print("r3 quadrature limit", flush=True)
        r3 = r3_quadrature([512, 1024] if q else [512, 1024, 2048, 4096],
                           [1.0, 10.0, 100.0, 1000.0] if q else [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
        _save("r3_quadrature", r3); plot_r3(r3)
    if "r4" in want:
        print("r4 scaling", flush=True)
        r4 = r4_scaling([512, 1024, 2048] if q else [1024, 2048, 4096, 8192], [1.0, 100.0] if q else [1.0, 30.0, 300.0])
        _save("r4_scaling", r4); plot_r4(r4)
    if "r5" in want:
        print("r5 heat", flush=True)
        r5 = r5_heat(512 if q else 1024, [1e-2, 1e-3, 1e-4] if q else [1e-2, 1e-3, 1e-4, 1e-5], 8 if q else 16, 0.04,
                     [4, 8, 16] if q else [4, 8, 16, 32])
        _save("r5_heat", r5); plot_r5(r5)
    if "r6" in want:
        print("r6 small kappa", flush=True)
        r6 = r6_small_kappa(1024 if q else 2048, [1e-12, 1e-3, 1.0] if q else [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0, 30.0])
        _save("r6_small_kappa", r6)
    if "r7" in want:
        print("r7 near-field truncation", flush=True)
        r7 = r7_near(1024 if q else 4096,
                     [1.0, 300.0, 3000.0] if q else [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4, 3e4],
                     [1.0, 300.0] if q else [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
                     512 if q else 1024, [1e-2, 1e-4] if q else [1e-2, 1e-3, 1e-4, 1e-5], 8 if q else 16, 0.04,
                     [4, 8] if q else [4, 8, 16, 32])
        _save("r7_near", r7); plot_r7(r7)

if __name__ == "__main__":
    main()
