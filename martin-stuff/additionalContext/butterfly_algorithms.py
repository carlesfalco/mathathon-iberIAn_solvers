"""Controlled implementations of two middle-seeded butterfly algorithms.

The mixed application follows Li et al.'s source factors, middle coupling, and
target factors.  The target-only application replaces its source-and-middle
prefix by a diagonal and coarse target transfers from a small-QR descent.

Both constructions start from the same local middle SVDs.  No matrix entries
are read during application or during either unshared construction branch.
Slot arrays are packed once, before timing, so both online implementations use
exactly the same transfer loop.  This is clear comparison code, not a production
implementation or an optimized sparse-matrix benchmark.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr


def size(N):
    L = int(np.log2(N))
    if N < 4 or 2**L != N:
        raise ValueError("N must be a power of two, at least four")
    if L % 2:
        raise ValueError("log2(N) must be even for the balanced middle level")
    return L, L // 2


def level_ranks(N, r):
    L, _ = size(N)
    return [min(r, 2**ell, N // 2**ell) for ell in range(L + 1)]


def slot_indices(N, ell):
    """The N slots at actual complementary-block level ell."""
    L, _ = size(N)
    for i in range(2**ell):
        for j in range(2**(L - ell)):
            yield i, j


@dataclass
class MiddleSeed:
    """Identical SVD data supplied to both construction branches."""

    P: dict
    Q: dict
    sigma: dict
    W: dict
    represented: np.ndarray
    residual_frobenius: float
    build_seconds: float = 0.0


def make_test_matrix(N, r, random_seed=1701):
    """Dense rank-r matrix: all complementary blocks have rank at most r."""
    rng = np.random.default_rng(random_seed + N)
    left = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
    right = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
    return (left @ right.conj().T) / np.sqrt(N * r)


def make_middle_seed(K, r):
    """Compute the shared seed K^h_ij approximately P Sigma Q^*."""
    N = K.shape[0]
    L, h = size(N)
    rho = level_ranks(N, r)
    rows, cols = N // 2**h, 2**h
    P, Q, sigma, W = {}, {}, {}, {}
    represented = np.zeros_like(K)
    residual_sq = 0.0
    for i, j in slot_indices(N, h):
        block = K[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols]
        left, singular, right_h = np.linalg.svd(block, full_matrices=False)
        k = rho[h]
        p, q, s = left[:, :k], right_h.conj().T[:, :k], singular[:k]
        P[h, i, j], Q[h, i, j], sigma[i, j] = p, q, s
        W[h, i, j] = s[:, None] * q.conj().T
        represented[i * rows:(i + 1) * rows,
                    j * cols:(j + 1) * cols] = P[h, i, j] @ W[h, i, j]
        residual_sq += float(np.sum(singular[k:] ** 2))
    return MiddleSeed(P, Q, sigma, W, represented, residual_sq**0.5)


def refine_target_level(P_coarse, N, ell, rank_cap):
    """One upward basis step, by a truncated SVD of a tall stacked basis."""
    m = N // 2**(ell + 1)
    knew = min(rank_cap, 2**(ell + 1), m)
    P_fine = {}
    for iprime in range(2**(ell + 1)):
        i, child = iprime // 2, iprime % 2
        for jp in range(N // 2**(ell + 1)):
            stack = np.hstack((
                P_coarse[ell, i, 2 * jp][child * m:(child + 1) * m],
                P_coarse[ell, i, 2 * jp + 1][child * m:(child + 1) * m],
            ))
            left, _, _ = np.linalg.svd(stack, full_matrices=False)
            P_fine[ell + 1, iprime, jp] = left[:, :knew]
    return P_fine


def pack_transfers(transfers, ell, N, fine_level=None):
    """Pack local matrices in destination-slot order once, before a matvec."""
    dest = ell + 1 if fine_level is None else fine_level
    rows, cols = 2**dest, N // 2**dest
    sample = transfers[ell, 0, 0]
    packed = np.empty((rows, cols, *sample.shape), dtype=sample.dtype)
    for i, j in slot_indices(N, dest):
        packed[i, j] = transfers[ell, i, j]
    return packed


def build_common_target_upper(seed, N, r):
    """Target factors at levels ell >= h, identical in both algorithms."""
    L, h = size(N)
    P = dict(seed.P)
    for ell in range(h, L):
        P.update(refine_target_level(P, N, ell, r))
    G = {}
    for ell in range(h, L):
        m = N // 2**(ell + 1)
        for iprime in range(2**(ell + 1)):
            i, child = iprime // 2, iprime % 2
            for jp in range(N // 2**(ell + 1)):
                stack = np.hstack((
                    P[ell, i, 2 * jp][child * m:(child + 1) * m],
                    P[ell, i, 2 * jp + 1][child * m:(child + 1) * m],
                ))
                G[ell, iprime, jp] = P[ell + 1, iprime, jp].conj().T @ stack
    terminal = np.array([P[L, i, 0][0, 0] for i in range(N)])
    return terminal, {ell: pack_transfers(G, ell, N) for ell in range(h, L)}


def build_source_branch(seed, N, r):
    """The paper/source branch: Q recursion, F chain, and middle C^h."""
    L, h = size(N)
    Q = dict(seed.Q)
    # Apply the target upward recursion to K^*, transposing slot indices.
    for s in range(h, 0, -1):
        dual_coarse = {(L - s, j, i): Q[s, i, j]
                       for i, j in slot_indices(N, s)}
        dual_fine = refine_target_level(dual_coarse, N, L - s, r)
        for t, j, i in dual_fine:
            Q[s - 1, i, j] = dual_fine[t, j, i]

    F = {}
    for s in range(1, h + 1):
        half = 2**(s - 1)
        for iprime, jp in slot_indices(N, s):
            i = iprime // 2
            qfine = Q[s, iprime, jp]
            left = qfine[:half].conj().T @ Q[s - 1, i, 2 * jp]
            right = qfine[half:].conj().T @ Q[s - 1, i, 2 * jp + 1]
            F[s, iprime, jp] = np.hstack((left, right))
    q0 = np.array([np.conj(Q[0, 0, j][0, 0]) for j in range(N)])
    coupling = np.empty((2**h, 2**h, level_ranks(N, r)[h]), dtype=complex)
    for i, j in slot_indices(N, h):
        coupling[i, j] = seed.sigma[i, j]
    return q0, {s: pack_transfers(F, s, N, fine_level=s) for s in range(1, h + 1)}, coupling


def build_target_coarse_branch(seed, N, r):
    """Downward small-pivoted-QR recursion for target-only G^ell, ell<h."""
    L, h = size(N)
    rho = level_ranks(N, r)
    W = dict(seed.W)
    G = {}
    for ell in range(h - 1, -1, -1):
        width = 2**ell
        small = {}
        for i, j in slot_indices(N, ell):
            jp = j // 2
            cols = slice(0, width) if j % 2 == 0 else slice(width, 2 * width)
            stack = np.vstack((
                W[ell + 1, 2 * i, jp][:, cols],
                W[ell + 1, 2 * i + 1, jp][:, cols],
            ))
            qsmall, _, _ = qr(stack, mode="economic", pivoting=True,
                              check_finite=False)
            qsmall = qsmall[:, :rho[ell]]
            W[ell, i, j] = qsmall.conj().T @ stack
            small[i, j] = qsmall
        for iprime, jp in slot_indices(N, ell + 1):
            i, child = iprime // 2, iprime % 2
            qa, qb = small[i, 2 * jp], small[i, 2 * jp + 1]
            half = qa.shape[0] // 2
            rows = slice(child * half, (child + 1) * half)
            G[ell, iprime, jp] = np.hstack((qa[rows], qb[rows]))
    diagonal = np.array([W[0, 0, j][0, 0] for j in range(N)])
    return diagonal, {ell: pack_transfers(G, ell, N) for ell in range(h)}


def advance_slots(state, transfers):
    """The same two-input, one-output local update for both sweeps."""
    nparent = state.shape[0]
    old_pairs = state.reshape(nparent, state.shape[1] // 2, -1)
    destinations = np.repeat(old_pairs, 2, axis=0)
    return np.einsum("ijkp,ijp->ijk", transfers, destinations, optimize=False)


def apply_mixed(g, q0, F, coupling, G_upper, terminal, N):
    """Paper-style application: V^L, source F's, C^h, then target G's."""
    L, h = size(N)
    state = (q0 * g)[None, :, None]
    for s in range(1, h + 1):
        state = advance_slots(state, F[s])
    state = state * coupling
    for ell in range(h, L):
        state = advance_slots(state, G_upper[ell])
    return terminal * state[:, 0, 0]


def apply_target_only(g, diagonal, G_lower, G_upper, terminal, N):
    """Target-only application: D, then target G's at every level."""
    L, _ = size(N)
    state = (diagonal * g)[None, :, None]
    for ell in range(L):
        transfers = G_lower[ell] if ell in G_lower else G_upper[ell]
        state = advance_slots(state, transfers)
    return terminal * state[:, 0, 0]
