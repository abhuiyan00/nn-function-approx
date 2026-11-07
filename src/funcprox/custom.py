"""Functions a user brings: a typed formula or measured points, with or without a missing middle.

A formula is parsed into a syntax tree and only a short list of operations is ever
evaluated, so text typed into the interface cannot run code.
"""
from __future__ import annotations

import ast
import math
from dataclasses import replace

import numpy as np

from .data import grid
from .functions import TARGETS, Target, get_target

MAX_FORMULA = 200
MIN_POINTS = 8
MIN_USABLE = 0.05
DEFAULT_RANGE = (-1.0, 1.0)
ALLOWED = "Use numbers, x, y, pi, e, + - * / ^ and functions such as sin(x)."
CONSTANTS = {"pi": np.pi, "e": np.e}
OPERATORS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
FUNCTIONS = {
    "sin": (np.sin, 1), "cos": (np.cos, 1), "tan": (np.tan, 1),
    "asin": (np.arcsin, 1), "acos": (np.arccos, 1), "atan": (np.arctan, 1),
    "sinh": (np.sinh, 1), "cosh": (np.cosh, 1), "tanh": (np.tanh, 1),
    "exp": (np.exp, 1), "log": (np.log, 1), "ln": (np.log, 1), "log10": (np.log10, 1), "sqrt": (np.sqrt, 1),
    "abs": (np.abs, 1), "sign": (np.sign, 1), "floor": (np.floor, 1), "ceil": (np.ceil, 1),
    "step": (lambda v: np.heaviside(v, 1.0), 1),
    "min": (np.minimum, 2), "max": (np.maximum, 2),
}


def _check(node: ast.AST, inputs: set[str]) -> None:
    """Refuse everything that is not plain arithmetic on x and y."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return
    if isinstance(node, ast.Name):
        if node.id in ("x", "y"):
            inputs.add(node.id)
        elif node.id in FUNCTIONS:
            raise ValueError(f"{node.id} needs brackets, as in {node.id}(x).")
        elif node.id not in CONSTANTS:
            raise ValueError(f"Unknown name {node.id!r}. {ALLOWED}")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _check(node.operand, inputs)
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        _check(node.left, inputs)
        _check(node.right, inputs)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id not in FUNCTIONS:
            raise ValueError(f"Unknown function {node.func.id!r}. {ALLOWED}")
        wanted = FUNCTIONS[node.func.id][1]
        if len(node.args) != wanted:
            raise ValueError(f"{node.func.id} takes {wanted} value{'s' if wanted > 1 else ''}.")
        for arg in node.args:
            _check(arg, inputs)
    else:
        raise ValueError(f"This cannot be part of a formula. {ALLOWED}")


def _evaluate(node: ast.AST, values: dict):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _evaluate(node.operand, values)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        return OPERATORS[type(node.op)](_evaluate(node.left, values), _evaluate(node.right, values))
    return FUNCTIONS[node.func.id][0](*(_evaluate(arg, values) for arg in node.args))


def _ranges(ranges, dim: int, label: str) -> tuple:
    out = tuple((float(lo), float(hi)) for lo, hi in ranges)
    if len(out) != dim:
        raise ValueError(f"This function has {dim} input{'s' if dim > 1 else ''}, so {label} needs {dim} range{'s' if dim > 1 else ''}.")
    if not all(math.isfinite(lo) and math.isfinite(hi) and lo < hi for lo, hi in out):
        raise ValueError(f"Each range of {label} must run from a smaller to a larger number.")
    return out


def _from_formula(text: str, domain) -> Target:
    source = text.strip().replace("^", "**")
    if not source:
        raise ValueError(f"Enter a formula, such as sin(3*x). {ALLOWED}")
    if len(text) > MAX_FORMULA:
        raise ValueError(f"This formula is too long: at most {MAX_FORMULA} characters.")
    try:
        tree = ast.parse(source, mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ValueError(f"Could not read this formula. Write multiplication as 2*x and powers as x^2. {ALLOWED}") from None
    inputs: set[str] = set()
    _check(tree, inputs)
    dim = 2 if "y" in inputs else 1
    box = _ranges(domain, dim, "the domain") if domain else (DEFAULT_RANGE,) * dim

    def fn(x: np.ndarray) -> np.ndarray:
        values = dict(CONSTANTS, x=x[:, 0], y=x[:, dim - 1])
        with np.errstate(all="ignore"):  # undefined values are data here: they become holes
            return np.broadcast_to(_evaluate(tree, values), (len(x),)).astype(float)

    return Target("custom", "your formula", text.strip(), dim, box, fn, "your own function")


def _with_gap(target: Target, gap) -> Target:
    if gap is None:
        return target
    if not len(gap):
        return replace(target, gap=None)
    hole = _ranges(gap, target.dim, "the missing middle")
    if any(hi <= box_lo or lo >= box_hi for (lo, hi), (box_lo, box_hi) in zip(hole, target.domain)):
        raise ValueError("The missing middle lies outside the range of the function.")
    return replace(target, gap=hole)


def _from_points(points, gap) -> Target:
    if len(points) < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} points are needed.")
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("Give one x, y pair per point.")
    if not np.isfinite(p).all():
        raise ValueError("Every x and y must be a finite number.")
    lo, hi = float(p[:, 0].min()), float(p[:, 0].max())
    if lo == hi:
        raise ValueError("All points have the same x, so there is nothing to approximate.")
    target = _with_gap(Target("custom", "your points", f"{len(p)} measured points", 1, ((lo, hi),),
                              lambda x: np.full(len(x), np.nan), "your own data", points=p), gap)
    if (~target.in_gap(p[:, :1])).sum() < MIN_POINTS:
        raise ValueError(f"At least {MIN_POINTS} points must lie outside the missing middle.")
    return target


def make_target(target: str = "f1", formula: str | None = None, domain=None, points=None, gap=None) -> Target:
    """The one way in for every function: a built-in key, a formula or points.

    ``gap=None`` keeps the target's own missing middle and an empty list removes it.
    Raises ValueError with a message meant for the person who typed the input.
    """
    if formula is not None and points is not None:
        raise ValueError("Give either a formula or points, not both.")
    if points is not None:
        return _from_points(points, gap)
    if formula is not None:
        made = _with_gap(_from_formula(formula, domain), gap)
    else:
        try:
            base = get_target(target)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        if gap is None:
            return base
        made = _with_gap(base, gap)
    x = grid(made.domain, 2500)
    usable = np.isfinite(made(x).ravel()) & ~made.in_gap(x)
    if usable.mean() < MIN_USABLE:
        raise ValueError(f"This leaves almost nothing to learn from: the function must be defined outside "
                         f"the missing middle on at least {MIN_USABLE:.0%} of its range.")
    return made


def saved_target(meta: dict) -> Target | None:
    """The function a saved model was trained on; None when the file does not say."""
    if "function" not in meta:
        return TARGETS.get(meta.get("target"))
    try:
        return make_target(**meta["function"])
    except (ValueError, TypeError):
        return None
