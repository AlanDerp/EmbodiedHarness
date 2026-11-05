"""Generated-program provenance policy adapted from ASPIRE runtime.py (Apache-2.0).

This is an API contract, not an operating-system security sandbox.
"""

import ast
import builtins
import math

ALLOWED_IMPORTS = frozenset({"math"})
FORBIDDEN_ATTRIBUTES = frozenset({"tofile", "dump", "dumps", "ctypes", "base",
                                "native", "sim", "env", "reward", "success", "native_task_success", "take_action"})


class Validator(ast.NodeVisitor):
    def generic_visit(self, node):
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef,
                             ast.Await, ast.With, ast.ImportFrom, ast.Delete)):
            raise ValueError(f"{type(node).__name__} is unavailable to generated programs")
        super().generic_visit(node)

    def visit_Import(self, node):
        if any(alias.name not in ALLOWED_IMPORTS or alias.asname for alias in node.names):
            raise ValueError("generated programs may import math only")

    def visit_Name(self, node):
        if node.id.startswith("_") or node.id in {"open", "eval", "exec", "compile", "getattr", "setattr", "vars", "globals", "locals", "breakpoint"}:
            raise ValueError("private/reflection/file execution names are unavailable")

    def visit_Attribute(self, node):
        if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
            raise ValueError("private/native/file attributes are unavailable")
        self.generic_visit(node)


def validate(code):
    tree = ast.parse(code)
    Validator().visit(tree)
    return compile(tree, "<generated-skill>", "exec")


def safe_builtins():
    names = ("Exception", "RuntimeError", "ValueError", "TypeError", "abs", "all", "any", "bool",
             "dict", "enumerate", "float", "int", "len", "list", "max", "min", "print", "range",
             "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance")
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name != "math" or level != 0:
            raise ImportError("only math is available")
        return math
    return {**{name: getattr(builtins, name) for name in names}, "__import__": restricted_import}
