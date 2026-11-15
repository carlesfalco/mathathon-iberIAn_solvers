"""One-dimensional centred-residual butterfly algorithm for oscillatory kernels.

The kernel is exp(2*pi*i*N*x*p), with x_i=p_i=i/N.  Thus Psi(x,p)=xp
and every complementary rectangle has residual (x-a)(p-b).  The local
approximation interpolates exp(2*pi*i*N*residual) on Chebyshev--Lobatto
nodes.  No dense N-by-N matrix is formed.
"""
from functools import lru_cache
import numpy as np


def _lagrange(nodes, targets):
    """Lagrange basis evaluated at targets; shape (len(targets), len(nodes))."""
    values = np.empty((len(targets), len(nodes)))
    for j in range(len(nodes)):
        numerator = np.ones(len(targets))
        denominator = 1.0
        for m in range(len(nodes)):
            if m != j:
                numerator *= targets - nodes[m]
                denominator *= nodes[j] - nodes[m]
        values[:, j] = numerator / denominator
    return values


@lru_cache(maxsize=None)
def _interpolation(q, child=None):
    z = 0.5 * np.cos(np.arange(q) * np.pi / (q - 1))
    if child is None:
        return z
    if child == "output":
        return _lagrange(z, np.array([-0.5]))[0]
    # A child's z-grid written in its parent's coordinates.
    shift = -0.25 if child == 0 else 0.25
    return _lagrange(z, z / 2 + shift)


def butterfly_fio(f, q=8, phase=None):
    """Apply an oscillatory kernel with N=len(f) and q>=2 interpolation nodes.

    phase(x,p,n) must return the phase Psi_scaled(x,p)=N*Psi(x,p) for scalar
    or broadcast-compatible NumPy inputs.  The default is N*x*p, the positive
    DFT.  Increasing q reduces the local Chebyshev interpolation error.
    """
    f = np.asarray(f, dtype=complex)
    n = f.size
    levels = int(np.log2(n))
    if n < 2 or 2**levels != n:
        raise ValueError("len(f) must be a power of two, at least 2")
    if q < 2:
        raise ValueError("q must be at least 2")
    if phase is None:
        phase = lambda x, p, n: n * x * p
    z = _interpolation(q)
    weights = [_interpolation(q, 0), _interpolation(q, 1)]
    left = _interpolation(q, "output")

    # Level 0: the observation box is the root and each source box is a leaf.
    # The j-th source sample lies at the left edge p_j=j/n of its cell.
    width = 1.0 / n
    data = np.empty((1, n, q), dtype=complex)
    for j in range(n):
        centre = (j + 0.5) * width
        interpolation_nodes = centre + width * z
        data[0, j] = (
            left
            * f[j]
            * np.exp(2j * np.pi * (
                phase(0.5, j * width, n) - phase(0.5, interpolation_nodes, n)
            ))
        )

    # Equivalent-source regime: split observations and merge sources.
    middle = levels // 2
    for level in range(middle):
        h = 2.0 ** (-level)
        source_width = 2.0**level / n
        next_data = np.zeros((2 ** (level + 1), n // 2 ** (level + 1), q), complex)
        for new_k in range(next_data.shape[0]):
            parent_k = new_k // 2
            x_centre = (new_k + 0.5) * (h / 2)
            for new_j in range(next_data.shape[1]):
                parent_centre = (new_j + 0.5) * (2 * source_width)
                parent_nodes = parent_centre + 2 * source_width * z
                for child in (0, 1):
                    old_j = 2 * new_j + child
                    child_centre = (old_j + 0.5) * source_width
                    child_nodes = child_centre + source_width * z
                    remodulate = np.exp(2j * np.pi * (
                        phase(x_centre, child_nodes[:, None], n)
                        - phase(x_centre, parent_nodes[None, :], n)
                    ))
                    next_data[new_k, new_j] += data[parent_k, old_j] @ (
                        weights[child] * remodulate
                    )
        data = next_data

    # At the balanced level, replace equivalent sources by potential values.
    h = 2.0 ** (-middle)
    source_width = 2.0**middle / n
    converted = np.empty_like(data)
    for k in range(data.shape[0]):
        x_nodes = (k + 0.5) * h + h * z
        for j in range(data.shape[1]):
            p_nodes = (j + 0.5) * source_width + source_width * z
            converted[k, j] = np.exp(
                2j * np.pi * phase(x_nodes[:, None], p_nodes[None, :], n)
            ) @ data[k, j]
    data = converted

    # Potential-value regime: continue splitting observations and merging sources.
    for level in range(middle, levels):
        h = 2.0 ** (-level)
        source_width = 2.0**level / n
        next_data = np.zeros((2 ** (level + 1), n // 2 ** (level + 1), q), complex)
        for new_k in range(next_data.shape[0]):
            parent_k = new_k // 2
            x_new = (new_k + 0.5) * (h / 2) + (h / 2) * z
            x_old = (parent_k + 0.5) * h + h * z
            interpolation = weights[new_k % 2]
            for new_j in range(next_data.shape[1]):
                for child in (0, 1):
                    old_j = 2 * new_j + child
                    source_centre = (old_j + 0.5) * source_width
                    remodulate = np.exp(2j * np.pi * (
                        phase(x_new[:, None], source_centre, n)
                        - phase(x_old[None, :], source_centre, n)
                    ))
                    next_data[new_k, new_j] += (interpolation * remodulate) @ data[
                        parent_k, old_j
                    ]
        data = next_data

    # Evaluate at x_i=i/n, the left edge of each observation leaf.
    out = np.empty(n, dtype=complex)
    for i in range(n):
        x_nodes = (i + 0.5) / n + z / n
        coefficients = left * np.exp(-2j * np.pi * phase(x_nodes, 0.5, n))
        out[i] = np.exp(2j * np.pi * phase(i / n, 0.5, n)) * (
            coefficients @ data[i, 0]
        )
    return out


def butterfly_dft(f, q=8):
    """Apply sum_j f[j]*exp(+2*pi*i*j*k/N) by the bilinear butterfly case."""
    return butterfly_fio(f, q=q)


def fft_dft(f):
    """The same positive-sign DFT, computed exactly (up to roundoff) by FFT."""
    f = np.asarray(f, dtype=complex)
    return f.size * np.fft.ifft(f)


def dense_dft(f):
    """Explicit dense reference, intended for small correctness tests only."""
    f = np.asarray(f, dtype=complex)
    n = f.size
    kernel = np.exp(2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    return kernel @ f


def dense_fio(f, phase):
    """Dense reference for the same phase convention, for small tests only."""
    f = np.asarray(f, dtype=complex)
    n = f.size
    grid = np.arange(n) / n
    kernel = np.exp(2j * np.pi * phase(grid[:, None], grid[None, :], n))
    return kernel @ f
