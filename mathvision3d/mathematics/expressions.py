"""Safe evaluation of small user-typed numeric expressions (e.g. ``z = 0.5*sin(x)``)."""

from __future__ import annotations

import ast

import numpy as np

_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "sqrt": np.sqrt, "abs": np.abs, "arctan": np.arctan, "sinh": np.sinh,
    "cosh": np.cosh, "tanh": np.tanh,
}
_CONSTS = {"pi": np.pi, "e": np.e}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
)


class ExpressionError(ValueError):
    pass


def compile_expression(expr: str, variables: tuple[str, ...] = ("x", "y", "t")):
    """Validate ``expr`` and return a callable ``f(**vars) -> ndarray``."""
    expr = expr.strip().replace("^", "**")
    if not expr:
        raise ExpressionError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(str(exc)) from exc
    allowed_names = set(_FUNCS) | set(_CONSTS) | set(variables)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ExpressionError(f"unknown name: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ExpressionError("only basic math functions may be called")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ExpressionError("only numeric constants are allowed")
    code = compile(tree, "<expression>", "eval")

    def evaluate(**kwargs):
        ns = {**_FUNCS, **_CONSTS, **{k: np.asarray(v, float) for k, v in kwargs.items()}}
        with np.errstate(all="ignore"):
            out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - AST validated above
        ref = next(iter(kwargs.values()))
        return np.broadcast_to(np.asarray(out, float), np.shape(ref)).copy()

    return evaluate
