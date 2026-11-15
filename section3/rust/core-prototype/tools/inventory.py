"""Inventory of the Python surface that a Rust port of updated/unified/{core,drivers} must reproduce.

Emits, for every file and every function:
  - physical / code / comment / docstring lines (code = physical - blank - comment - docstring)
  - signature
  - attribute call sites (W[...], F.solve, node.center, ...) found inside the body
and a global tally of distinct numpy / scipy primitives.

Run:  python3 tools/inventory.py <path-to-updated/unified> > inventory.txt
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

FILES = [
    "core/__init__.py", "core/types.py", "core/tree.py", "core/admissibility.py",
    "core/compress.py", "core/kernels.py",
    "drivers/__init__.py", "drivers/rskel.py", "drivers/butterfly_id.py",
]

NUMPY_ATTRS: set[str] = set()
SCIPY_ATTRS: set[str] = set()


def duck_attrs(node: ast.AST) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            v = sub.func.value
            if isinstance(v, ast.Name):
                out.add(f"{v.id}.{sub.func.attr}")
        elif isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
            out.add(f"{sub.value.id}[...]")
    return out


def code_lines(src: str) -> tuple[int, int, int, int]:
    """(physical, blank, comment, docstring) line counts."""
    tree = ast.parse(src)
    lines = src.splitlines()
    blank = sum(1 for ln in lines if not ln.strip())
    comment = 0
    in_str = False
    for ln in lines:
        s = ln.strip()
        if in_str:
            comment += 1
            if s.count('"""') % 2 == 1 or s.count("'''") % 2 == 1:
                in_str = False
            continue
        if s.startswith('"""') or s.startswith("'''"):
            comment += 1
            q = s[:3]
            if s.count(q) == 1:
                in_str = True
            continue
        if s.startswith("#"):
            comment += 1
    doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            b = getattr(node, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                doc += (b[0].end_lineno - b[0].lineno + 1)
    return len(lines), blank, comment, doc


def sig(fn: ast.AST) -> str:
    a = fn.args
    parts = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def body_code(src: str, node: ast.AST) -> int:
    seg = ast.get_source_segment(src, node) or ""
    return len([ln for ln in seg.splitlines() if ln.strip() and not ln.strip().startswith("#")])


def main(root: Path) -> None:
    tot = [0, 0, 0, 0]
    print(f"{'file':28s} {'phys':>5s} {'code':>5s} {'cmt':>5s} {'doc':>5s} {'defs':>5s} {'cls':>4s}")
    for rel in FILES:
        src = (root / rel).read_text()
        p, b, c, d = code_lines(src)
        tree = ast.parse(src)
        ndef = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
        ncls = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        tot = [tot[0] + p, tot[1] + b, tot[2] + c, tot[3] + d]
        print(f"{rel:28s} {p:5d} {p - b - c - d:5d} {c:5d} {d:5d} {ndef:5d} {ncls:4d}")
    print(f"{'TOTAL':28s} {tot[0]:5d} {tot[0] - tot[1] - tot[2] - tot[3]:5d} {tot[2]:5d} {tot[3]:5d}")

    for rel in FILES:
        src = (root / rel).read_text()
        tree = ast.parse(src)
        print(f"\n--- {rel}")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = sorted({c for c in duck_attrs(node) if "." in c})
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        base = sub.func.value
                        if isinstance(base, ast.Name) and base.id == "np":
                            NUMPY_ATTRS.add("np." + sub.func.attr)
                        if isinstance(base, ast.Name) and base.id in ("sla", "sp"):
                            SCIPY_ATTRS.add(base.id + "." + sub.func.attr)
                print(f"  def {sig(node):46s} code={body_code(src, node):3d}  " + " ".join(calls[:12]))
            elif isinstance(node, ast.ClassDef):
                print(f"  class {node.name:44s} code={body_code(src, node):3d}")
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        print(f"      def {sig(sub):42s} code={body_code(src, sub):3d}")

    print("\n== distinct external primitives ==")
    print(f"numpy ({len(NUMPY_ATTRS)}): " + " ".join(sorted(NUMPY_ATTRS)))
    print(f"scipy ({len(SCIPY_ATTRS)}): " + " ".join(sorted(SCIPY_ATTRS)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
