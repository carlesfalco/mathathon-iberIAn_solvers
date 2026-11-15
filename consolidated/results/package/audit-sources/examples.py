"""Small, self-contained checks for framework-explained.md (NumPy only)."""
import numpy as np


def laplace_example():
    """log(x-y) = log(x) - sum_{m>=1} (y/x)^m/m for 0 <= |y| < x."""
    x, p = 2.0, 3
    y = np.linspace(-0.25, 0.25, 1001)
    exact = np.log(x - y)
    approx = np.log(x) - sum((y / x)**m / m for m in range(1, p + 1))
    actual = float(np.max(np.abs(exact - approx)))
    rho = 0.25 / x
    bound = rho**(p + 1) / ((p + 1) * (1 - rho))
    print("SEPARABLE LAPLACE EXAMPLE")
    print(f"x={x}, |y|<=0.25, terms=p+1={p + 1}")
    print(f"maximum pointwise error = {actual:.6e}")
    print(f"proved tail bound        = {bound:.6e}")
    assert actual <= bound


def dft_matrix(n):
    j = np.arange(n)
    return np.exp(2j * np.pi * np.outer(j, j) / n)


def fourier_examples():
    print("\nDFT RESIDUE-CLASS BLOCKS (N=4, middle level)")
    f = dft_matrix(4)
    for a in range(2):
        for b in range(2):
            rows = np.arange(4)[np.arange(4) % 2 == a]
            cols = np.arange(4)[np.arange(4) % 2 == b]
            block = f[np.ix_(rows, cols)]
            s = np.linalg.svd(block, compute_uv=False)
            print(f"rows j={rows.tolist()}, columns k={cols.tolist()}: "
                  f"singular values {np.array2string(s, precision=2)}")
            assert np.linalg.matrix_rank(block, tol=1e-12) == 1

    # One radix-2 Cooley--Tukey recombination, checked against the dense DFT.
    n, g = 8, np.arange(8) + 1j * np.arange(8)[::-1]
    direct = dft_matrix(n) @ g
    even, odd = dft_matrix(n // 2) @ g[::2], dft_matrix(n // 2) @ g[1::2]
    j = np.arange(n)
    rebuilt = np.tile(even, 2) + np.exp(2j * np.pi * j / n) * np.tile(odd, 2)
    error = float(np.linalg.norm(rebuilt - direct) / np.linalg.norm(direct))
    print(f"\none radix-2 recombination: relative error = {error:.3e}")
    assert error < 1e-13


def transpose_counterexample():
    # Pi = [I; -T], with one scalar redundant and one scalar skeleton unknown.
    t = 1j
    pi = np.array([[1.0 + 0j], [-t]])
    identity = np.eye(2, dtype=complex)  # W = I on the two original unknowns
    transpose_pivot = pi.T @ identity @ pi
    adjoint_pivot = pi.conj().T @ identity @ pi
    print("\nTRANSPOSE VERSUS ADJOINT")
    print(f"Pi^T Pi = {transpose_pivot[0, 0]}")
    print(f"Pi^* Pi = {adjoint_pivot[0, 0]}")
    assert abs(transpose_pivot[0, 0]) < 1e-15
    assert abs(adjoint_pivot[0, 0] - 2) < 1e-15


def congruence_check():
    """With Herm(W)=I, the adjoint row step has Herm(Pi* W Pi)=I+T*T."""
    rng = np.random.default_rng(0)
    n = 20
    worst_transpose = np.inf
    worst_adjoint = np.inf
    for _ in range(2000):
        t = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2 * n)
        s = rng.standard_normal((2 * n, 2 * n)) + 1j * rng.standard_normal((2 * n, 2 * n))
        s = (s + s.conj().T) / 2
        w = np.eye(2 * n) + 1j * s
        pi = np.vstack([np.eye(n), -t])
        plain = pi.T @ w @ pi
        adjoint = pi.conj().T @ w @ pi
        worst_transpose = min(worst_transpose, np.linalg.eigvalsh((plain + plain.conj().T) / 2).min())
        worst_adjoint = min(worst_adjoint, np.linalg.eigvalsh((adjoint + adjoint.conj().T) / 2).min())
    print("\nCOMPLEX CONGRUENCE CHECK (2000 random matrices)")
    print(f"minimum eig Herm(Pi^T W Pi) = {worst_transpose:.3f}")
    print(f"minimum eig Herm(Pi^* W Pi) = {worst_adjoint:.3f} (theoretical lower bound: 1)")
    assert worst_transpose < 0
    assert worst_adjoint >= 1 - 1e-10


if __name__ == "__main__":
    laplace_example()
    fourier_examples()
    transpose_counterexample()
    congruence_check()
