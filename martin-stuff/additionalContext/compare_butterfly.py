"""Correctness checks and controlled timings for results/butterfly_algorithms.py.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 results/compare_butterfly.py

The test matrix has exact global rank RANK, hence exact complementary rank at
most RANK at every level.  This deliberately removes a confounding variable:
both algorithms must approximate the same operator to roundoff before timing.

Application timings include every operation for one matrix--vector product.
Construction timings compare only the unshared branches from the common seed:
  source-basis recursion + F factors + packing + M data
versus
  target coefficient QR descent + G factors + packing + D.
The common seed uses literal LAPACK SVDs of every middle block, for a clean
mathematical oracle rather than an implementation of the paper's randomized
seed.  Seed and common-upper time is therefore printed separately, and the full
setup estimates below are explicitly literal-SVD setup, not the paper's
asymptotically cheaper randomized construction.
"""

from dataclasses import dataclass
from statistics import median
from time import perf_counter
import numpy as np

from butterfly_algorithms import (
    apply_mixed,
    apply_target_only,
    build_common_target_upper,
    build_source_branch,
    build_target_coarse_branch,
    level_ranks,
    make_middle_seed,
    make_test_matrix,
)


RANK = 4
CHECK_SIZES = (16, 64)
TIMING_SIZES = (16, 64, 256)


@dataclass
class Case:
    K: np.ndarray
    seed: object
    terminal: np.ndarray
    G_upper: dict
    q0: np.ndarray
    F: dict
    coupling: np.ndarray
    diagonal: np.ndarray
    G_lower: dict


def build_pair(N):
    K = make_test_matrix(N, RANK)
    start = perf_counter()
    seed = make_middle_seed(K, RANK)
    seed.build_seconds = perf_counter() - start
    start = perf_counter()
    terminal, G_upper = build_common_target_upper(seed, N, RANK)
    shared_upper_seconds = perf_counter() - start
    q0, F, coupling = build_source_branch(seed, N, RANK)
    diagonal, G_lower = build_target_coarse_branch(seed, N, RANK)
    return Case(K, seed, terminal, G_upper, q0, F, coupling,
                diagonal, G_lower), shared_upper_seconds


def mixed(g, case, N):
    return apply_mixed(g, case.q0, case.F, case.coupling,
                       case.G_upper, case.terminal, N)


def target(g, case, N):
    return apply_target_only(g, case.diagonal, case.G_lower,
                             case.G_upper, case.terminal, N)


def timed_median(function, batch_seconds, repetitions=5):
    function()  # warm all code paths before timing
    start = perf_counter()
    function()
    estimate = perf_counter() - start
    batch = max(1, min(10000, int(batch_seconds / max(estimate, 2e-7))))
    times = []
    for _ in range(repetitions):
        start = perf_counter()
        for _ in range(batch):
            function()
        times.append((perf_counter() - start) / batch)
    return median(times), batch


def check_case(N):
    case, _ = build_pair(N)
    rng = np.random.default_rng(20261114 + N)
    worst = np.zeros(3)
    for _ in range(4):
        g = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        expected = case.K @ g
        scale = max(np.linalg.norm(expected), 1.0)
        out_mixed, out_target = mixed(g, case, N), target(g, case, N)
        worst = np.maximum(worst, [
            np.linalg.norm(out_mixed - out_target) / scale,
            np.linalg.norm(out_mixed - expected) / scale,
            np.linalg.norm(out_target - expected) / scale,
        ])
    print(f"N={N:4d}, r={RANK}: mutual error {worst[0]:.3e}; "
          f"source+M versus exact K {worst[1]:.3e}; "
          f"target-only versus exact K {worst[2]:.3e}")
    assert worst.max() < 2e-10, f"Factorization mismatch at N={N}"


def timing_case(N):
    case, shared_upper_seconds = build_pair(N)
    rng = np.random.default_rng(601 + N)
    g = rng.standard_normal(N) + 1j * rng.standard_normal(N)

    source_build, source_batch = timed_median(
        lambda: build_source_branch(case.seed, N, RANK), 0.08)
    target_build, target_batch = timed_median(
        lambda: build_target_coarse_branch(case.seed, N, RANK), 0.08)
    mixed_apply, mixed_batch = timed_median(
        lambda: mixed(g, case, N), 0.25, repetitions=7)
    target_apply, target_batch = timed_median(
        lambda: target(g, case, N), 0.25, repetitions=7)

    reference = case.K @ g
    scale = max(np.linalg.norm(reference), 1.0)
    mutual = np.linalg.norm(mixed(g, case, N) - target(g, case, N)) / scale
    assert mutual < 2e-10, f"Timed approximations differ at N={N}"

    rho = level_ranks(N, RANK)
    L = int(np.log2(N))
    transfer_nnz = 2 * N * sum(rho[ell] * rho[ell + 1] for ell in range(L))
    paper_nnz = transfer_nnz + 2 * N + N * rho[L // 2]
    target_nnz = transfer_nnz + 2 * N

    print(f"\nN={N:4d}, r={RANK}; rank budget: {rho}")
    common_setup = case.seed.build_seconds + shared_upper_seconds
    total_source = common_setup + source_build
    total_target = common_setup + target_build
    print(f"shared setup (excluded from branch timings): "
          f"middle seed {1e3 * case.seed.build_seconds:.3f} ms; "
          f"upper target factors {1e3 * shared_upper_seconds:.3f} ms")
    print(f"unshared construction, median: source+M "
          f"{1e3 * source_build:.3f} ms; target QR descent "
          f"{1e3 * target_build:.3f} ms; ratio "
          f"{source_build / target_build:.3f}x")
    print(f"full setup with the literal-SVD seed: source+M "
          f"{1e3 * total_source:.3f} ms; target-only "
          f"{1e3 * total_target:.3f} ms; ratio "
          f"{total_source / total_target:.3f}x")
    print(f"application, median: source+M "
          f"{1e6 * mixed_apply:.2f} us; target-only "
          f"{1e6 * target_apply:.2f} us; ratio "
          f"{mixed_apply / target_apply:.3f}x")
    print(f"structural complex entries: source+M {paper_nnz}; "
          f"target-only {target_nnz}; mutual relative error {mutual:.3e}")
    print(f"batch sizes (source build, target build, source apply, target apply): "
          f"{source_batch}, {target_batch}, {mixed_batch}, {target_batch}")


if __name__ == "__main__":
    print("Correctness against one exact complementary-low-rank matrix")
    for n in CHECK_SIZES:
        check_case(n)
    print("\nTiming with one BLAS thread, median of repeated batches")
    for n in TIMING_SIZES:
        timing_case(n)
