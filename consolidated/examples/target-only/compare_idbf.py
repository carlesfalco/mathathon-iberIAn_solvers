"""Correctness and controlled timings for idbf_algorithms.py.

Run from the repository root with one BLAS/OpenMP thread:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 results/compare_idbf.py

IMPORTANT SCOPE LIMIT.  This is an exact-rank, dense-oracle comparison of two
IDBF online organizations.  It does not implement the paper's carefully
organized linear-entry sampled LRCS/MSCS constructor.  Unshared construction
timings are for these visible-entry prototype branches only; they are not
claims about the paper's O(N log N) construction implementation.  The common
middle seed is the sequential row-ID/column-ID form U S V := P C R, where
P = U S contains original selected columns, C = S^{-1}, and V = C R.
"""

from dataclasses import dataclass
import json
import os
import platform
from statistics import median
import sys
from time import perf_counter

import numpy as np
import scipy

from idbf_algorithms import (
    apply_idbf, apply_target_only, build_common_upper, build_original_lower,
    build_target_lower, check_target_invariant, dimensions, entry_count,
    level_ranks, make_shared_middle, make_test_matrix, advance, slot_indices,
)

RANK = 4
CHECK_SIZES = (16, 64, 256)
TIMING_SIZES = (16, 64, 256)


@dataclass
class Case:
    K: np.ndarray
    seed: object
    upper: dict
    terminal: dict
    P_upper: dict
    q_upper: dict
    lower_source: dict
    source_rows: dict
    lower_target: dict
    q_target: dict
    P_target: dict
    upper_seconds: float
    source_seconds: float
    target_seconds: float


def build_pair(N):
    K = make_test_matrix(N, RANK)
    start = perf_counter()
    seed = make_shared_middle(K, RANK)
    seed_seconds = perf_counter()-start
    start = perf_counter()
    upper, terminal, P_upper, q_upper = build_common_upper(K, seed, RANK)
    upper_seconds = perf_counter()-start
    shared_upper = (upper, terminal, P_upper, q_upper)
    start = perf_counter()
    lower_source, source_rows = build_original_lower(K, seed, RANK)
    source_seconds = perf_counter()-start
    start = perf_counter()
    lower_target, q_target, P_target = build_target_lower(
        K, seed, RANK, shared_upper)
    target_seconds = perf_counter()-start
    return (Case(K, seed, upper, terminal, P_upper, q_upper,
                 lower_source, source_rows, lower_target, q_target, P_target,
                 upper_seconds, source_seconds, target_seconds), seed_seconds)


def original(g, case, N):
    return apply_idbf(g, case.lower_source, case.seed, case.upper,
                      case.terminal, N)


def target(g, case, N):
    return apply_target_only(g, case.lower_target, case.upper,
                             case.terminal, N)


def timed_median(function, batch_seconds=0.25, repetitions=7):
    function()
    start = perf_counter()
    function()
    estimate = perf_counter()-start
    batch = max(1, min(10000, int(batch_seconds/max(estimate, 2e-7))))
    times = []
    for _ in range(repetitions):
        start = perf_counter()
        for _ in range(batch):
            function()
        times.append((perf_counter()-start)/batch)
    return median(times), batch, min(times), max(times)


def check_case(N):
    case, _ = build_pair(N)
    rng = np.random.default_rng(20261115+N)
    worst = np.zeros(3)
    for _ in range(4):
        g = rng.standard_normal(N)+1j*rng.standard_normal(N)
        expected = case.K @ g
        scale = max(np.linalg.norm(expected), 1.0)
        a, b = original(g, case, N), target(g, case, N)
        worst = np.maximum(worst, [
            np.linalg.norm(a-expected)/scale,
            np.linalg.norm(b-expected)/scale,
            np.linalg.norm(a-b)/scale,
        ])
    invariant = check_target_invariant(case.K, case.lower_target, case.upper,
        case.terminal, case.P_target, case.P_upper,
        np.random.default_rng(9000+N))
    source_error, middle_error = 0.0, 0.0
    L, h = dimensions(N)
    g = rng.standard_normal(N)+1j*rng.standard_normal(N)
    state = {(0, j): g[j:j+1].copy() for j in range(N)}
    for ell in range(1, h+1):
        state = advance(state, case.lower_source[ell], N, ell-1)
        band = 2**ell
        rows_count = N//2**ell
        for ip, jp in slot_indices(N, ell):
            cols = slice(jp*band, (jp+1)*band)
            expected_source = case.source_rows[ell, ip, jp] @ g[cols]
            scale = max(np.linalg.norm(expected_source), 1.0)
            source_error = max(source_error,
                np.linalg.norm(state[ip, jp]-expected_source)/scale)
            if ell == h:
                expected_output = case.K[ip*rows_count:(ip+1)*rows_count, cols] @ g[cols]
                actual_output = case.seed.P[ip, jp] @ (case.seed.C[ip, jp] @ state[ip, jp])
                middle_error = max(middle_error,
                    np.linalg.norm(actual_output-expected_output) /
                    max(np.linalg.norm(expected_output), 1.0))
    assert case.seed.max_residual < 2e-10
    assert worst.max() < 2e-10, (N, "application error", worst)
    assert invariant < 2e-10, (N, "intermediate invariant error", invariant)
    assert max(source_error, middle_error) < 2e-10, (N, "source invariant error")
    mutual = np.linalg.norm(dense_original_error(case))/np.linalg.norm(case.K)
    assert mutual < 2e-10, (N, "dense operators differ", mutual)
    print(f"N={N:4d}: original error {worst[0]:.3e}; target error {worst[1]:.3e}; "
          f"mutual {worst[2]:.3e}; block invariants {invariant:.3e}; "
          f"source invariants {source_error:.3e}; middle products {middle_error:.3e}; "
          f"dense original {mutual:.3e}")
    return dict(N=N, original_error=float(worst[0]), target_error=float(worst[1]),
                mutual_error=float(worst[2]), invariant_error=float(invariant),
                source_invariant_error=source_error, middle_product_error=middle_error,
                dense_original_error=float(mutual), seed_residual=case.seed.max_residual)


def dense_original_error(case):
    """Independent full operator check, not just common-direction tests."""
    N = case.K.shape[0]
    difference = np.zeros_like(case.K)
    for j in range(N):
        g = np.zeros(N, dtype=complex)
        g[j] = 1.0
        a, b = original(g, case, N), target(g, case, N)
        difference[:, j] = a-case.K[:, j]
        assert np.linalg.norm(b-case.K[:, j]) < 2e-10*max(np.linalg.norm(case.K[:, j]), 1.0)
    return difference


def timing_case(N):
    case, seed_seconds = build_pair(N)
    rng = np.random.default_rng(601+N)
    g = rng.standard_normal(N)+1j*rng.standard_normal(N)
    t_seed, _, _, _ = timed_median(lambda: make_shared_middle(case.K, RANK),
                                    0.08, repetitions=5)
    t_upper, _, _, _ = timed_median(lambda: build_common_upper(case.K, case.seed, RANK),
                                     0.08, repetitions=5)
    t_source, b_source, _, _ = timed_median(
        lambda: build_original_lower(case.K, case.seed, RANK), 0.08, repetitions=5)
    lower = case.lower_target
    t_target, b_target, _, _ = timed_median(
        lambda: build_target_lower(case.K, case.seed, RANK,
            (case.upper, case.terminal, case.P_upper, case.q_upper)), 0.08, repetitions=5)
    t_original, b_original, spread_original_low, spread_original_high = timed_median(
        lambda: original(g, case, N))
    t_target_apply, b_target_apply, spread_target_low, spread_target_high = timed_median(
        lambda: target(g, case, N))
    a, b = original(g, case, N), target(g, case, N)
    expected = case.K @ g
    scale = max(np.linalg.norm(expected), 1.0)
    mutual = np.linalg.norm(a-b)/scale
    error_original = np.linalg.norm(a-expected)/scale
    error_target = np.linalg.norm(b-expected)/scale
    assert mutual < 2e-10
    entries_original = entry_count(case.lower_source, case.seed, case.upper, case.terminal)
    entries_target = entry_count(case.lower_target, case.seed, case.upper, case.terminal,
                                 target=True)
    record = dict(
        N=N, rank=RANK, ranks=level_ranks(N, RANK),
        seed_residual=case.seed.max_residual,
        original_error=float(error_original), target_error=float(error_target),
        mutual_error=float(mutual),
        shared_seed_ms=1e3*t_seed, shared_upper_ms=1e3*t_upper,
        original_build_ms=1e3*t_source, target_build_ms=1e3*t_target,
        original_full_setup_ms=1e3*(t_seed+t_upper+t_source),
        target_full_setup_ms=1e3*(t_seed+t_upper+t_target),
        original_apply_us=1e6*t_original, target_apply_us=1e6*t_target_apply,
        original_target_apply_ratio=float(t_original/t_target_apply),
        original_target_build_ratio=float(t_source/t_target),
        original_target_full_setup_ratio=float((t_seed+t_upper+t_source)/(t_seed+t_upper+t_target)),
        entries_original=entries_original, entries_target=entries_target,
        batches=[b_source, b_target, b_original, b_target_apply],
        original_apply_min_us=1e6*spread_original_low,
        original_apply_max_us=1e6*spread_original_high,
        target_apply_min_us=1e6*spread_target_low,
        target_apply_max_us=1e6*spread_target_high,
    )
    print(f"\nN={N:4d}, r={RANK}; cap-derived rank budget: {level_ranks(N, RANK)}")
    print(f"shared setup: middle two-sided ID {1e3*t_seed:.3f} ms; "
          f"common upper cIDs {1e3*t_upper:.3f} ms")
    print(f"prototype branches: source interpolation {1e3*t_source:.3f} ms; "
          f"target cID descent {1e3*t_target:.3f} ms; ratio {t_source/t_target:.3f}x")
    print(f"full visible-entry setup: original {1e3*(t_seed+t_upper+t_source):.3f} ms; "
          f"target-only {1e3*(t_seed+t_upper+t_target):.3f} ms")
    print(f"application: source+C {1e6*t_original:.3f} us; "
          f"target-only {1e6*t_target_apply:.3f} us; "
          f"ratio {t_original/t_target_apply:.3f}x")
    print(f"structural complex entries: original {entries_original}; "
          f"target {entries_target}; mutual relative error {mutual:.3e}")
    print(f"batch sizes: source/target builds {b_source}/{b_target}; "
          f"original/target applications {b_original}/{b_target_apply}")
    return record


def main():
    print("Correctness of two IDBF online organizations on an exact-rank complex matrix")
    checks = [check_case(n) for n in CHECK_SIZES]
    print("\nTiming with one BLAS/OpenMP thread, medians over repeated batches")
    timings = [timing_case(n) for n in TIMING_SIZES]
    metadata = dict(python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    logical_cpus=os.cpu_count(),
                    tested_matrix="K=A B^*/sqrt(N*r), complex standard normal, exact rank r",
                    protocol="one BLAS thread; 5 setup batches (0.08s); 7 apply batches (0.25s); common seed and upper cIDs",
                    limitation="dense visible-entry mathematical oracle, not the paper's optimized sampled-entry construction",
                    checks=checks, timings=timings)
    with open("results/idbf_benchmark_results.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
