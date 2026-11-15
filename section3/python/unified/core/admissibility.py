"""Admissibility = the rule that decides which interactions of a box are compressed ("far") and which
are kept exactly ("near"). This is the ONLY thing that differs between HBS-style recursive
skeletonization (weak) and FMM-LU-style strong recursive skeletonization (strong). The butterfly uses
a third rule (complementary), which pairs nodes of two different trees.
"""
from __future__ import annotations

from .tree import Node, Tree
from .types import TilePair as TilePair  # re-exported for drivers that import it from here


class Weak:
    """Compress against everything outside the box. Near field is empty."""
    name: str = "weak"

    def near_nodes(self, tree: Tree, node: Node) -> list[Node]:
        return []


class Strong:
    """Compress only against boxes that do not touch this one (FMM rule). Neighbours stay exact."""
    name: str = "strong"

    def near_nodes(self, tree: Tree, node: Node) -> list[Node]:
        return tree.neighbors(node)


class Complementary:
    """Butterfly rule for two trees with the same depth L: a row node at depth l is paired with every
    column node at depth L - l. All such pairs (one phase-space cell each) are compressed."""
    name: str = "complementary"

    def pairs(self, row_tree: Tree, col_tree: Tree, l: int) -> list[TilePair]:
        L = row_tree.depth
        assert col_tree.depth == L, "complementary admissibility needs two trees of equal depth"
        return [(a, b) for a in row_tree.levels[l] for b in col_tree.levels[L - l]]
