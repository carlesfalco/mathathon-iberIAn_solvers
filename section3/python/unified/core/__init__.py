"""Shared core of the unified fast-algorithm code: one tree, one ID, swappable admissibility and compression."""
from .tree import Tree, Node
from .admissibility import Weak, Strong, Complementary
from .compress import interp_decomp, ExactCompressor, ProxyCompressor
from .types import IDResult, Compressor, Admissibility
from . import kernels, types

__all__ = ["Tree", "Node", "Weak", "Strong", "Complementary", "interp_decomp", "ExactCompressor",
           "ProxyCompressor", "IDResult", "Compressor", "Admissibility", "kernels", "types"]
