"""Reproduce the small DFT rank measurements used in butterfly-key-ideas.pdf.

Run from the project root:
    OPENBLAS_NUM_THREADS=1 python3 results/butterfly-discovery-checks.py

A tile's numerical rank counts singular values with sigma_j >= tol * sigma_1.
The matrix is K[j,k] = exp(2*pi*i*j*k/N). All tiles in each partition are checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bit_reversal(n: int) -> np.ndarray:
    bits = int(np.log2(n))
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])


def numerical_rank(tile: np.ndarray, tol: float) -> int:
    singular_values = np.linalg.svd(tile, compute_uv=False)
    return int(np.count_nonzero(singular_values >= tol * singular_values[0]))


def dft(n: int) -> np.ndarray:
    indices = np.arange(n)
    return np.exp(2j * np.pi * np.outer(indices, indices) / n)


def stage_max_ranks(n: int, tol: float) -> list[int]:
    depth = int(np.log2(n))
    matrix = dft(n)
    result = []
    for level in range(depth + 1):
        row_groups, column_groups = 2**level, 2 ** (depth - level)
        ranks = [
            numerical_rank(
                matrix[
                    a * n // row_groups : (a + 1) * n // row_groups,
                    b * n // column_groups : (b + 1) * n // column_groups,
                ],
                tol,
            )
            for a in range(row_groups)
            for b in range(column_groups)
        ]
        result.append(max(ranks))
    return result


def fixed_versus_middle() -> list[dict[str, int]]:
    rows = []
    for n in (64, 128, 256, 512):
        matrix = dft(n)
        fixed_max = max(
            numerical_rank(
                matrix[a * n // 4 : (a + 1) * n // 4, b * n // 4 : (b + 1) * n // 4],
                1e-6,
            )
            for a in range(4)
            for b in range(4)
        )
        depth = int(np.log2(n))
        middle = depth // 2
        row_groups, column_groups = 2**middle, 2 ** (depth - middle)
        middle_max = max(
            numerical_rank(
                matrix[
                    a * n // row_groups : (a + 1) * n // row_groups,
                    b * n // column_groups : (b + 1) * n // column_groups,
                ],
                1e-6,
            )
            for a in range(row_groups)
            for b in range(column_groups)
        )
        rows.append(
            {
                "N": n,
                "fixed_4_by_4_tiling_max_rank": fixed_max,
                "middle_tile_rows": n // row_groups,
                "middle_tile_columns": n // column_groups,
                "middle_number_of_tiles": row_groups * column_groups,
                "middle_max_rank": middle_max,
            }
        )
    return rows


def ordering_grids() -> dict[str, list[list[int]]]:
    n, groups = 16, 4
    matrix = dft(n)
    natural = np.arange(n)
    reversed_indices = bit_reversal(n)
    orderings = {
        "natural_rows_natural_columns": (natural, natural),
        "bit_reversed_rows_natural_columns": (reversed_indices, natural),
        "bit_reversed_rows_bit_reversed_columns": (reversed_indices, reversed_indices),
    }
    result = {}
    for name, (rows, columns) in orderings.items():
        permuted = matrix[np.ix_(rows, columns)]
        result[name] = [
            [
                numerical_rank(
                    permuted[a * 4 : (a + 1) * 4, b * 4 : (b + 1) * 4],
                    1e-12,
                )
                for b in range(groups)
            ]
            for a in range(groups)
        ]
    return result


def main() -> None:
    data = {
        "matrix": "K[j,k] = exp(2*pi*i*j*k/N)",
        "rank_rule": "count sigma_j >= tol * sigma_1 separately for every tile",
        "tiling_comparison_tolerance": 1e-6,
        "tiling_comparison": fixed_versus_middle(),
        "ordering_tolerance": 1e-12,
        "ordering_N": 16,
        "ordering_middle_rank_grids": ordering_grids(),
        "natural_stage_N": 256,
        "natural_stage_max_ranks_tol_1e_6": stage_max_ranks(256, 1e-6),
        "natural_stage_max_ranks_tol_1e_12": stage_max_ranks(256, 1e-12),
    }
    destination = Path(__file__).with_name("butterfly-discovery-ranks.json")
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
