"""Golden values for a Rust port: exact numbers the Python core/drivers produce on small cases.

All inputs are deterministic and RNG-free on purpose: numpy's PCG64 stream is not reproducible from a
dependency-free Rust program, so a cross-language equivalence test must share *inputs*, not RNG state.
Writes `golden.txt` (key=value lines).

Run:  PYTHONPATH=<repo>/updated python3 tools/golden.py > golden.txt
"""
from __future__ import annotations

import numpy as np

from unified.core import Tree, Weak, Strong, interp_decomp, ExactCompressor, ProxyCompressor, kernels
from unified.drivers import rskel, butterfly_id as bid

OUT: list[tuple[str, str]] = []


def emit_s(k: str, v: str) -> None:
    OUT.append((k, v))


def emit_coords(k: str, coords) -> None:
    emit_s(k, " ".join("_".join(str(int(c)) for c in coord) for coord in coords))


def emit(k: str, v) -> None:
    if isinstance(v, (list, tuple, np.ndarray)):
        emit_s(k, " ".join(str(int(x)) if float(x).is_integer() else f"{float(x):.17g}" for x in v))
    elif isinstance(v, (int, np.integer)):
        emit_s(k, str(int(v)))
    else:
        emit_s(k, f"{float(v):.17g}")


# ---- deterministic inputs (bit-identical in Python and Rust: integer arithmetic + libm) -------------
def weyl2d(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return np.c_[((i * 7919) % 5003) / 5003.0, ((i * 104729) % 4999) / 4999.0]


def rank7_matrix(m: int, n: int, k: int = 7) -> np.ndarray:
    return (np.sin(0.3 * np.outer(np.arange(m), np.arange(k)) + 0.1)
            @ np.cos(0.5 * np.outer(np.arange(k), np.arange(n)) + 0.2))


def lap1d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    r = np.abs(np.subtract.outer(np.asarray(X, float).ravel(), np.asarray(Y, float).ravel()))
    K = -0.5 * r
    K[r == 0] = 0.0
    return K


def starfish(N: int) -> tuple[np.ndarray, np.ndarray]:
    th = 2 * np.pi * np.arange(N) / N
    r = 1 + 0.3 * np.cos(5 * th)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    dr = -1.5 * np.sin(5 * th)
    w = np.sqrt(r ** 2 + dr ** 2) * 2 * np.pi / N
    return X, w


def p1_kernel(N: int):
    """Column weight |gamma'| 2 pi / N evaluated at the angles of the *column* argument (P1 problem)."""
    def K(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, float))
        th = np.arctan2(Q[:, 1], Q[:, 0])
        r, dr = 1 + 0.3 * np.cos(5 * th), -1.5 * np.sin(5 * th)
        w = np.sqrt(r ** 2 + dr ** 2) * 2 * np.pi / N
        return kernels.laplace2d(P, Q) * w[None, :]
    return K


def rhs(N: int) -> np.ndarray:
    return np.sin(3.0 * np.arange(N)) + 0.5


# ---- cases ----------------------------------------------------------------------------------------
def case_tree() -> None:
    T = Tree(np.arange(64) / 64, depth=6)
    emit("tree1d_levels", [len(l) for l in T.levels])
    emit("tree1d_leaf_sizes", sorted({len(n.idx) for n in T.leaves()}))
    X = weyl2d(500)
    T2 = Tree(X, max_leaf=16)
    emit("tree2d_depth", T2.depth)
    emit("tree2d_occupied", [len(l) for l in T2.levels])
    mid = T2.levels[2][len(T2.levels[2]) // 2]
    emit_coords("tree2d_mid_coord", [mid.coord])
    emit_coords("tree2d_neighbors_mid", sorted(tuple(n.coord) for n in T2.neighbors(mid)))
    emit_coords("tree2d_ilist_mid", sorted(tuple(n.coord) for n in T2.interaction_list(mid)))


def case_id() -> None:
    M = rank7_matrix(80, 50)
    for eps in (1e-10, 1e-3):
        sk, rd, Tm = interp_decomp(M, eps)
        rel = float(np.linalg.norm(M[:, rd] - M[:, sk] @ Tm) / np.linalg.norm(M))
        emit(f"id_eps{eps:g}_k", len(sk))
        emit(f"id_eps{eps:g}_skel", sk)
        emit(f"id_eps{eps:g}_relres", rel)
    sk, rd, Tm = interp_decomp(np.zeros((5, 0)), 1e-8)
    emit("id_n0_k", len(sk))
    sk, rd, Tm = interp_decomp(np.zeros((0, 5)), 1e-8)
    emit("id_m0_k", len(sk))
    emit("id_m0_red", rd)
    emit("id_30x40_k", len(interp_decomp(rank7_matrix(30, 40), 1e-12)[0]))


def stats(name: str, tree: Tree, F, A: np.ndarray, b: np.ndarray) -> None:
    st = F.stats
    emit(f"{name}_depth", tree.depth)
    emit(f"{name}_levels", [len(tree.levels[l]) for l in range(tree.depth + 1)])
    order = sorted(st["levels"])
    emit(f"{name}_skelmax", [st["levels"][l]["skel_max"] for l in order])
    emit(f"{name}_skeltotal", [st["levels"][l]["skel_total"] for l in order])
    emit(f"{name}_eliminated", [st["levels"][l]["eliminated"] for l in order])
    emit(f"{name}_boxes", [st["levels"][l]["boxes"] for l in order])
    emit(f"{name}_root", st["root_size"])
    emit(f"{name}_nsteps", st["n_steps"])
    x = F.solve(b)
    emit(f"{name}_relres", float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)))


def case_rs_1d(adm_name: str, adm, N: int = 256, max_leaf: int = 8, eps: float = 1e-10) -> None:
    X = np.arange(N) / N
    w = 1.0 / N
    A = np.eye(N) + lap1d(X, X) * w
    tree = Tree(X, max_leaf=max_leaf)
    stats(f"rs1d_{adm_name}", tree, rskel.rskel_factor(A, tree, adm, ExactCompressor(A, eps)), A, rhs(N))


def case_p1(adm_name: str, adm, comp_kind: str, N: int, max_leaf: int, eps: float) -> None:
    X, w = starfish(N)
    A = np.eye(N) + kernels.laplace2d(X, X) * w[None, :]
    tree = Tree(X, max_leaf=max_leaf)
    comp = (ExactCompressor(A, eps) if comp_kind == "exact"
            else ProxyCompressor(p1_kernel(N), X, eps, A=A))
    stats(f"p1_{adm_name}_{comp_kind}", tree, rskel.rskel_factor(A, tree, adm, comp), A, rhs(N))


def case_butterfly(N: int, t: float, eps: float) -> None:
    tag = f"bf_N{N}_t{t:g}_e{eps:g}"
    kernel, x, xi = bid.b1_problem(N, t)
    bf = bid.butterfly_id(kernel, x, xi, eps)
    emit(f"{tag}_L", bf.stats["L"])
    emit(f"{tag}_ranks", bf.ranks)
    emit(f"{tag}_nnz", bf.nnz)
    emit(f"{tag}_nnz_W_U", [bf.W.nnz, bf.U.nnz])
    emit(f"{tag}_nnz_G", [g.nnz for g in bf.G])
    g = np.sin(1.7 * np.arange(N)) + 0.3 + 1j * (np.cos(0.9 * np.arange(N)) - 0.2)
    ref = bid.exact_apply(kernel, x, xi, g)
    emit(f"{tag}_relerr", float(np.linalg.norm(bf.apply(g) - ref) / np.linalg.norm(ref)))


def main() -> None:
    case_tree()
    case_id()
    case_rs_1d("weak", Weak())
    case_rs_1d("strong", Strong())
    for adm_name, adm in (("weak", Weak()), ("strong", Strong())):
        for ck in ("exact", "proxy"):
            case_p1(adm_name, adm, ck, 256, 8, 1e-8)
    for N, t, eps in ((64, 0.0, 1e-10), (64, 3.0, 1e-8), (128, 1.0, 1e-8)):
        case_butterfly(N, t, eps)
    for k, v in OUT:
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
