from __future__ import annotations

import ast
import math
from typing import Any, Dict

_AST_NUM = getattr(ast, "Num", None)

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.FloorDiv,
    ast.Call,
    ast.Load,
    ast.Attribute,
    ast.Name,
)
if _AST_NUM is not None:
    ALLOWED_NODES = ALLOWED_NODES + (_AST_NUM,)

SAFE_FUNCTIONS = {
    name: getattr(math, name)
    for name in [
        "sin",
        "cos",
        "tan",
        "sqrt",
        "log",
        "log10",
        "exp",
        "fabs",
        "ceil",
        "floor",
        "pow",
    ]
}
SAFE_NAMES = {"pi": math.pi, "tau": math.tau, "e": math.e}


def _evaluate(node: ast.AST) -> float:
    if not isinstance(node, ALLOWED_NODES):
        raise ValueError(f"unsupported expression: {type(node).__name__}")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("constants must be numeric")
    if _AST_NUM is not None and isinstance(node, _AST_NUM):  # pragma: no cover - python <3.8 compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise ValueError(f"unsupported operator {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"unsupported unary {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            func = SAFE_FUNCTIONS.get(func_name)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                func_name = node.func.attr
                func = SAFE_FUNCTIONS.get(func_name)
            else:
                func = None
        else:
            func = None
        if func is None:
            raise ValueError("function not allowed")
        args = [_evaluate(arg) for arg in node.args]
        return float(func(*args))
    if isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return float(SAFE_NAMES[node.id])
        raise ValueError(f"name '{node.id}' not allowed")
    raise ValueError(f"unsupported node {type(node).__name__}")


def math_eval_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(payload.get("expr") or payload.get("text") or "")
    if not expr.strip():
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "no expression",
        }
    try:
        tree = ast.parse(expr, mode="eval")
        result = _evaluate(tree)
    except Exception as exc:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": f"error: {exc}",
        }
    return {
        "text": f"{result}",
        "quality_gain": 0.07,
        "faithfulness": 1.0,
        "notes": "math evaluated",
    }


CONVERSION_FACTORS = {
    ("km", "mi"): 0.621371,
    ("mi", "km"): 1.60934,
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 0.453592,
    ("c", "f"): ("c_to_f",),
    ("f", "c"): ("f_to_c",),
    ("l", "gal"): 0.264172,
    ("gal", "l"): 3.78541,
}


def _convert_temperature(value: float, direction: str) -> float:
    if direction == "c_to_f":
        return value * 9.0 / 5.0 + 32.0
    if direction == "f_to_c":
        return (value - 32.0) * 5.0 / 9.0
    raise ValueError("unknown temperature conversion")


def unit_convert_basic(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = float(payload.get("value"))
    except (TypeError, ValueError):
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "value must be numeric",
        }
    from_unit = str(payload.get("from_unit") or "").lower()
    to_unit = str(payload.get("to_unit") or "").lower()
    key = (from_unit, to_unit)
    if key not in CONVERSION_FACTORS:
        return {
            "text": "",
            "quality_gain": 0.0,
            "faithfulness": 0.0,
            "notes": "conversion not supported",
        }
    factor = CONVERSION_FACTORS[key]
    if isinstance(factor, tuple):
        result = _convert_temperature(value, factor[0])
    else:
        result = value * factor
    text = f"{value:g} {from_unit} = {result:.4f} {to_unit}"
    return {
        "text": text,
        "quality_gain": 0.06,
        "faithfulness": 1.0,
        "notes": "conversion completed",
        "result": result,
    }


__all__ = ["math_eval_safe", "unit_convert_basic"]
