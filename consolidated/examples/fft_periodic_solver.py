#!/usr/bin/env python3
"""Reproduce the periodic FFT-solver checks in the consolidated report."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local"
OUT.mkdir(parents=True, exist_ok=True)


def unitary_dft(n: int) -> np.ndarray:
    j = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)


def symbol(n: int) -> np.ndarray:
    k = np.arange(n)
    return 1.0 + 4.0 * n * n * np.sin(np.pi * k / n) ** 2


def periodic_matrix(n: int) -> np.ndarray:
    j = np.arange(n)
    a = (1.0 + 2.0 * n * n) * np.eye(n)
    a[j, (j + 1) % n] -= n * n
    a[j, (j - 1) % n] -= n * n
    return a


def recursive_fft(x: np.ndarray) -> np.ndarray:
    """Unnormalised negative-sign radix-2 DFT, for verification only."""
    n = len(x)
    if n == 1:
        return x.copy()
    if n % 2:
        raise ValueError("radix-2 input length must be a power of two")
    even = recursive_fft(x[::2])
    odd = recursive_fft(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate((even + twiddle, even - twiddle))


def bit_reversed(n: int) -> np.ndarray:
    levels = n.bit_length() - 1
    if 1 << levels != n:
        raise ValueError("n must be a power of two")
    return np.array([int(f"{j:0{levels}b}"[::-1], 2) for j in range(n)])


def sparse_stage(n: int, block: int) -> np.ndarray:
    """One normalised Cooley--Tukey combination stage."""
    half = block // 2
    d = np.exp(-2j * np.pi * np.arange(half) / block)
    eye = np.eye(half)
    b = np.block([[eye, np.diag(d)], [eye, -np.diag(d)]]) / np.sqrt(2.0)
    s = np.zeros((n, n), dtype=complex)
    for start in range(0, n, block):
        s[start:start + block, start:start + block] = b
    return s


def sparse_stage_factorization(n: int) -> np.ndarray:
    product = np.eye(n)[bit_reversed(n), :].astype(complex)
    block = 2
    while block <= n:
        product = sparse_stage(n, block) @ product
        block *= 2
    return product


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    factor_rows = []
    max_unitary = 0.0
    max_stage = 0.0
    max_recursive = 0.0
    rng = np.random.default_rng(20261115)
    for n in (8, 16, 32, 64):
        j = np.arange(n)
        x = j / n
        a = periodic_matrix(n)
        fmat = unitary_dft(n)
        lam = symbol(n)
        exact = np.cos(2 * np.pi * x) + 0.2 * np.sin(4 * np.pi * x)
        forcing = a @ exact
        fft_solution = np.fft.ifft(np.fft.fft(forcing) / lam)
        dense_solution = np.linalg.solve(a, forcing)
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        unitary_error = relative(fmat.conj().T @ fmat, np.eye(n))
        stage_error = relative(sparse_stage_factorization(n), fmat)
        recursive_error = relative(recursive_fft(z), n ** 0.5 * fmat @ z)
        max_unitary = max(max_unitary, unitary_error)
        max_stage = max(max_stage, stage_error)
        max_recursive = max(max_recursive, recursive_error)
        row = {
            "N": n,
            "factorisation_error": relative(a, fmat.conj().T @ np.diag(lam) @ fmat),
            "solve_error_vs_dense": relative(fft_solution, dense_solution),
            "relative_residual": relative(a @ fft_solution, forcing),
            "manufactured_solution_error": relative(fft_solution, exact),
            "unitary_error": unitary_error,
            "sparse_stage_error": stage_error,
            "recursive_fft_error": recursive_error,
        }
        factor_rows.append(row)
        print(
            f"N={n:2d} factor={row['factorisation_error']:.3e} "
            f"solve={row['solve_error_vs_dense']:.3e} "
            f"residual={row['relative_residual']:.3e} stage={stage_error:.3e}"
        )
        assert row["manufactured_solution_error"] < 1e-9

    pde_rows = []
    for n in (16, 32, 64, 128):
        j = np.arange(n)
        x = j / n
        forcing = ((1 + 4 * np.pi**2) * np.cos(2 * np.pi * x)
                   + 0.2 * (1 + 16 * np.pi**2) * np.sin(4 * np.pi * x))
        discrete = np.fft.ifft(np.fft.fft(forcing) / symbol(n))
        exact = np.cos(2 * np.pi * x) + 0.2 * np.sin(4 * np.pi * x)
        error = relative(discrete, exact)
        pde_rows.append({"N": n, "relative_discretisation_error": error})
        print(f"N={n:3d} discretisation={error:.8e}")
    ratios = [pde_rows[i]["relative_discretisation_error"] /
              pde_rows[i + 1]["relative_discretisation_error"] for i in range(3)]
    # These are finite-grid ratios approaching four, not an exact algebraic identity.
    assert all(3.9 < ratio < 4.1 for ratio in ratios)

    for name, rows in (("fft_factorisation.csv", factor_rows),
                       ("fft_discretisation.csv", pde_rows)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    checks = {
        "sizes": [8, 16, 32, 64],
        "max_unitary_error": max_unitary,
        "max_sparse_stage_error": max_stage,
        "max_recursive_fft_error": max_recursive,
        "discretisation_error_ratios": ratios,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    (OUT / "fft_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    assert max_unitary < 1e-12
    assert max_stage < 1e-12
    assert max_recursive < 1e-12
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
