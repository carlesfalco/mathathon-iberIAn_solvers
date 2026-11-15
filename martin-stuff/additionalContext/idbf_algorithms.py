"""Controlled IDBF applications with a shared two-sided interpolative seed.

The purpose is the same as the companion SVD experiment: compare online
organizations, not replace the paper's optimized sampled-entry constructor.

At a middle block, deterministic full-row/column rank-revealing QR IDs give
    K ~= U R,   R = K[selected_rows,:],
    R ~= S V,   S = R[:,selected_columns],
so K ~= (U S) S^{-1} R.  We store P = K[:,selected_columns] (= U S up to the
ID residual), C = S^{-1}, and the target coefficients W = V.  The original
branch takes R g to C R g.  The target-only branch takes W g instead.

No INVERSE IS APPLIED ONLINE.  The current dense, exact-rank test forms small
local interpolative decompositions from visible entries.  It is deliberately a
mathematical oracle: these routines do NOT claim the O(N log N) construction
complexity of the paper's carefully organized sampled LRCS/MSCS code.  Online
storage and work use only the small local matrices together with terminal
constant-height output blocks (selected-column lists are used for construction).
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr, solve_triangular


def dimensions(N):
    L = int(np.log2(N))
    if N < 16 or 2**L != N or L % 2:
        raise ValueError("N must have even log2(N), with N >= 16")
    return L, L // 2


def level_ranks(N, rank):
    L, _ = dimensions(N)
    return [min(rank, 2**ell, N//2**ell) for ell in range(L+1)]


def slot_indices(N, ell):
    L, _ = dimensions(N)
    for i in range(2**ell):
        for j in range(2**(L-ell)):
            yield i, j


def make_test_matrix(N, rank, random_seed=1701):
    rng = np.random.default_rng(random_seed + N)
    a = rng.standard_normal((N, rank)) + 1j*rng.standard_normal((N, rank))
    b = rng.standard_normal((N, rank)) + 1j*rng.standard_normal((N, rank))
    return (a @ b.conj().T)/np.sqrt(N*rank)


def column_id(A, k):
    """A[:,p[:k]] V ~= A, using pivoted QR of the supplied sample A."""
    _, R, p = qr(A, mode="economic", pivoting=True, check_finite=False)
    if k > min(A.shape):
        raise ValueError("sample cannot reveal the requested rank")
    V = np.zeros((k, A.shape[1]), dtype=complex)
    V[:, p[:k]] = np.eye(k)
    if k < A.shape[1]:
        V[:, p[k:]] = solve_triangular(R[:k, :k], R[:k, k:],
                                       lower=False, check_finite=False)
    return p[:k], V


def sampled_rid(K, k):
    """Row ID K ~= U K[rows,:], with all global columns used in this oracle."""
    p, V = column_id(K.conj().T, k)
    return p, V.conj().T, K[p, :]


@dataclass
class SharedMiddle:
    P: dict             # Selected original columns, explicit m_h by rho_h.
    q: dict             # Their global column indices.
    R: dict             # K[selected original rows, merged band].
    C: dict             # Small S^{-1}: K ~= P C R.
    W: dict             # Target coefficients V: K ~= P W.
    represented: np.ndarray
    max_residual: float


def make_shared_middle(K, rank):
    """Two-sided local ID seed K_ij^h ~= P_ij C_ij R_ij = P_ij W_ij."""
    N = K.shape[0]
    L, h = dimensions(N)
    m, n = N//2**h, 2**h
    k = min(rank, m, n)
    P, q, R, C, W = {}, {}, {}, {}, {}
    represented = np.zeros_like(K)
    max_residual = 0.0
    for i, j in slot_indices(N, h):
        rows = slice(i*m, (i+1)*m)
        cols = slice(j*n, (j+1)*n)
        block = K[rows, cols]
        rows_id, U, rmat = sampled_rid(block, k)
        selected, v = column_id(rmat, k)
        columns_id = j*n + selected
        pmat = K[rows, columns_id]
        S = rmat[:, selected]
        # P is the product U S from LRCS up to roundoff.  Keep a small solve
        # for the full-rank cross skeleton, rather than any dense online map.
        coupling = np.linalg.solve(S, np.eye(k))
        P[i, j], q[i, j], R[i, j] = pmat, columns_id, rmat
        C[i, j], W[i, j] = coupling, v
        rec = pmat @ v
        represented[rows, cols] = rec
        max_residual = max(max_residual,
                           np.linalg.norm(block-rec)/max(np.linalg.norm(block), 1.0))
        assert np.linalg.norm(U @ S - pmat) < 1e-9*max(np.linalg.norm(pmat), 1.0)
    return SharedMiddle(P, q, R, C, W, represented, max_residual)


def build_common_upper(K, seed, rank):
    """Target cIDs B=[R P_a,R P_b] ~= P_new G, shared by both applications."""
    N = K.shape[0]
    L, h = dimensions(N)
    P = {(h, i, j): value for (i, j), value in seed.P.items()}
    q = {(h, i, j): value for (i, j), value in seed.q.items()}
    upper = {}
    for ell in range(h, L):
        child_height = N//2**(ell+1)
        transfers = {}
        for ip, jp in slot_indices(N, ell+1):
            i = ip//2
            qa, qb = q[ell, i, 2*jp], q[ell, i, 2*jp+1]
            candidates = np.concatenate((qa, qb))
            row_start = ip*child_height
            B = K[row_start:row_start+child_height, candidates]
            knew = min(rank, child_height, 2**(ell+1))
            p, G = column_id(B, knew)
            P[ell+1, ip, jp] = B[:, p]
            q[ell+1, ip, jp] = candidates[p]
            transfers[ip, jp] = G
        upper[ell] = transfers
    terminal = {i: P[L, i, 0] for i in range(N)}
    return upper, terminal, P, q


def build_original_lower(K, seed, rank):
    """Original-style selected-row states R^ell g and small ID transfers.

    The boundary at level zero keeps each scalar input entry exactly.  A new
    source state is related to its two parent states by the small interpolation
    coefficient matrix R_new [R_a;R_b]^+.  This is an explicit dense-oracle
    realization; an optimized constructor would obtain these from sampled IDs.
    """
    N = K.shape[0]
    L, h = dimensions(N)
    R = {(0, 0, j): np.ones((1, 1), dtype=complex) for j in range(N)}
    transfers = {}
    for ell in range(1, h+1):
        band = 2**ell
        parent_band = band//2
        lower = {}
        for ip, jp in slot_indices(N, ell):
            i = ip//2
            height = N//2**ell
            rows = slice(ip*height, (ip+1)*height)
            cols = slice(jp*band, (jp+1)*band)
            if ell == h:
                rnew = seed.R[ip, jp]
            else:
                _, _, rnew = sampled_rid(K[rows, cols], min(rank, height, band))
            R[ell, ip, jp] = rnew
            ra = R[ell-1, i, 2*jp]
            rb = R[ell-1, i, 2*jp+1]
            old = np.zeros((ra.shape[0]+rb.shape[0], band), dtype=complex)
            old[:ra.shape[0], :parent_band] = ra
            old[ra.shape[0]:, parent_band:] = rb
            lower[ip, jp] = rnew @ np.linalg.pinv(old)
        transfers[ell] = lower
    return transfers, R


def build_target_lower(K, seed, rank, shared_upper):
    """One-sided original-column cIDs, aligned to the common middle seed.

    Before the middle, q lists are subsets of the two parent q lists.  At the
    middle, use P^h from the shared seed and interpolate it from the two parent
    candidate bases.  This records the common basis for both online paths while
    preserving the target-only partial-product invariant.
    """
    N = K.shape[0]
    L, h = dimensions(N)
    # Caller supplies upper's P,q metadata to avoid reconstructing anything.
    _, _, P_upper, q_upper = shared_upper
    q = {(0, 0, j): np.array([j], dtype=int) for j in range(N)}
    P = {(0, 0, j): K[:, [j]] for j in range(N)}
    transfers = {}
    for ell in range(h):
        transfers[ell] = {}
        band = 2**(ell+1)
        height = N//2**(ell+1)
        knew = min(rank, height, band)
        for ip, jp in slot_indices(N, ell+1):
            i = ip//2
            qa, qb = q[ell, i, 2*jp], q[ell, i, 2*jp+1]
            candidates = np.concatenate((qa, qb))
            rows = slice(ip*height, (ip+1)*height)
            B = K[rows, candidates]
            if ell+1 == h:
                # Interpolate the fixed shared seed from the candidate span.
                dest = P_upper[h, ip, jp]
                G = np.linalg.lstsq(dest, B, rcond=None)[0]
                qnew, pnew = q_upper[h, ip, jp], dest
            else:
                selected, G = column_id(B, knew)
                qnew, pnew = candidates[selected], B[:, selected]
            q[ell+1, ip, jp], P[ell+1, ip, jp] = qnew, pnew
            transfers[ell][ip, jp] = G
    return transfers, q, P


def advance(state, level_transfers, N, ell):
    """The same destination-slot update for both online organizations."""
    output = {}
    for ip, jp in slot_indices(N, ell+1):
        i = ip//2
        output[ip, jp] = level_transfers[ip, jp] @ np.concatenate((
            state[i, 2*jp], state[i, 2*jp+1]))
    return output


def apply_idbf(g, lower_source, seed, upper, terminal, N):
    """Selected-row/source states -> small middle coupling -> target states."""
    L, h = dimensions(N)
    state = {(0, j): g[j:j+1].copy() for j in range(N)}
    for ell in range(1, h+1):
        state = advance(state, lower_source[ell], N, ell-1)
    state = {(i, j): seed.C[i, j] @ x for (i, j), x in state.items()}
    for ell in range(h, L):
        state = advance(state, upper[ell], N, ell)
    return np.concatenate([terminal[i] @ state[i, 0] for i in range(N)])


def apply_target_only(g, lower_target, upper, terminal, N):
    """The same target partial-product invariant at every online level."""
    L, _ = dimensions(N)
    state = {(0, j): g[j:j+1].copy() for j in range(N)}
    transfers = dict(lower_target)
    transfers.update(upper)
    for ell in range(L):
        state = advance(state, transfers[ell], N, ell)
    return np.concatenate([terminal[i] @ state[i, 0] for i in range(N)])

def check_target_invariant(K, lower, upper, terminal, P_lower, P_upper, rng):
    """Independently check P^ell w^ell = K^ell g^ell for every slot and level."""
    N=K.shape[0]
    L,_=dimensions(N)
    basis=dict(P_lower)
    basis.update(P_upper)
    g=rng.standard_normal(N)+1j*rng.standard_normal(N)
    state={(0,j):g[j:j+1].copy() for j in range(N)}
    transfers=dict(lower)
    transfers.update(upper)
    worst=0.0
    for ell in range(L):
        state=advance(state,transfers[ell],N,ell)
        height=N//2**(ell+1)
        band=2**(ell+1)
        for ip,jp in slot_indices(N,ell+1):
            rows=slice(ip*height,(ip+1)*height)
            cols=slice(jp*band,(jp+1)*band)
            expected=K[rows,cols]@g[cols]
            Pnew=terminal[ip] if ell+1==L else basis[ell+1,ip,jp]
            actual=Pnew@state[ip,jp]
            error = np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1.0)
            worst=max(worst, error)
    return worst


def entry_count(lower, seed, upper, terminal, target=False):
    count = sum(x.size for level in upper.values() for x in level.values())
    count += sum(x.size for x in terminal.values())
    count += sum(x.size for level in lower.values() for x in level.values())
    if not target:
        count += sum(x.size for x in seed.C.values())
    return count
