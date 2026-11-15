import numpy as np
from unified.core.types import Points
from unified.core import Tree, Weak, Strong, Complementary, interp_decomp, ExactCompressor, ProxyCompressor, kernels


def starfish(n: int) -> Points:
    t = 2 * np.pi * np.arange(n) / n
    r = 1 + 0.3 * np.cos(5 * t)
    X: Points = np.c_[r * np.cos(t), r * np.sin(t)]
    return X


def test_tree_partitions_points_at_every_level() -> None:
    X = np.random.default_rng(0).random((500, 2))
    T = Tree(X, max_leaf=16)
    for lvl in T.levels:
        assert sorted(np.concatenate([n.idx for n in lvl])) == list(range(500))
    for n in T.nodes_postorder():
        if n.children:
            assert sorted(np.concatenate([c.idx for c in n.children])) == sorted(n.idx)


def test_neighbors_and_interaction_list_brute_force() -> None:
    X = np.random.default_rng(1).random((800, 2))
    T = Tree(X, depth=4)
    for l in (2, 3, 4):
        for n in T.levels[l]:
            nb = {m.coord for m in T.neighbors(n)}
            brute = {m.coord for m in T.levels[l] if m is not n and max(abs(m.coord[k] - n.coord[k]) for k in range(2)) <= 1}
            assert nb == brute
            il = {m.coord for m in T.interaction_list(n)}
            brute_il = {m.coord for m in T.levels[l] if max(abs(m.coord[k] - n.coord[k]) for k in range(2)) > 1
                        and max(abs(m.coord[k] // 2 - n.coord[k] // 2) for k in range(2)) <= 1}
            assert il == brute_il


def test_1d_tree() -> None:
    T = Tree(np.arange(64) / 64, depth=6)
    assert [len(l) for l in T.levels] == [1, 2, 4, 8, 16, 32, 64]
    assert all(len(n.idx) == 1 for n in T.leaves())


def test_interp_decomp_accuracy_and_rank() -> None:
    rng = np.random.default_rng(2)
    M = rng.standard_normal((80, 7)) @ rng.standard_normal((7, 50))
    skel, red, T = interp_decomp(M, 1e-10)
    assert len(skel) == 7
    assert np.linalg.norm(M[:, red] - M[:, skel] @ T) < 1e-8 * np.linalg.norm(M)


def test_proxy_rank_matches_exact_rank() -> None:
    X = starfish(1024)
    A = kernels.laplace2d(X, X)
    T = Tree(X, depth=4)
    exact, proxy = ExactCompressor(A, 1e-8), ProxyCompressor(kernels.laplace2d, X, 1e-8, A=A)
    for n in T.leaves():
        far = np.setdiff1d(np.arange(len(X)), n.idx)
        k_exact = len(exact(n.idx, far, n)[0])
        k_proxy = len(proxy(n.idx, far, n)[0])
        assert abs(k_exact - k_proxy) <= 4, (n, k_exact, k_proxy)
    # and the proxy ID is accurate against the TRUE far field
    n = max(T.leaves(), key=lambda m: len(m.idx))
    far = np.setdiff1d(np.arange(len(X)), n.idx)
    skel, red, Tm = proxy(n.idx, far, n)
    M = A[np.ix_(far, n.idx)]
    assert np.linalg.norm(M[:, red] - M[:, skel] @ Tm) < 1e-6 * np.linalg.norm(M)


def test_admissibility_rules() -> None:
    X = np.random.default_rng(3).random((400, 2))
    T = Tree(X, depth=3)
    n = T.levels[3][10]
    assert Weak().near_nodes(T, n) == []
    assert set(map(id, Strong().near_nodes(T, n))) == set(map(id, T.neighbors(n)))
    R, Cc = Tree(np.arange(16) / 16, depth=4), Tree(np.arange(16) - 8, depth=4)
    for l in range(5):
        pairs = Complementary().pairs(R, Cc, l)
        assert len(pairs) == 16 and all(len(a.idx) * len(b.idx) == 16 for a, b in pairs)


def test_bbox_points_on_dyadic_boundaries() -> None:
    T = Tree(np.arange(64) / 64, depth=6, bbox=([0.0], [1.0]))
    assert len(T.leaves()) == 64 and all(len(n.idx) == 1 for n in T.leaves())
