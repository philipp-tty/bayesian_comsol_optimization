"""Safe AST-based parameter expression evaluation."""

from __future__ import annotations

import ast
import math

import numpy as np

_ALLOWED_ATTRIBUTE_BASES = {"math", "np", "numpy"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate_expression_ast(
    tree: ast.AST,
    allowed_names: set[str],
    allowed_attribute_bases: set[str],
) -> None:
    """Ensure the parsed expression only contains safe node types and identifiers."""

    def _validate_attribute(node: ast.Attribute) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id not in allowed_attribute_bases:
            raise ValueError(
                "Expressions may only access attributes of the 'math', 'np', or 'numpy' namespaces."
            )
        if node.attr.startswith("_"):
            raise ValueError("Access to private attributes is not permitted in expressions.")

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError("Encountered an unsupported binary operator in the expression.")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise ValueError("Encountered an unsupported unary operator in the expression.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    raise ValueError(f"Function '{node.func.id}' is not permitted in expressions.")
            elif isinstance(node.func, ast.Attribute):
                _validate_attribute(node.func)
            else:
                raise ValueError("Expressions may only call named or module attribute functions.")
        elif isinstance(node, ast.Attribute):
            _validate_attribute(node)
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(
                    f"Name '{node.id}' is not available in expressions; use input parameter names."
                )
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed in expressions.")
        elif isinstance(node, _ALLOWED_BINOPS + _ALLOWED_UNARYOPS):
            continue
        elif isinstance(node, (ast.Load, ast.Expression, ast.keyword)):
            continue
        else:
            raise ValueError("Expressions may only contain arithmetic operations and function calls.")


def evaluate_parameter_expression(
    expression: str,
    parameter_names: list[str],
    physical_samples: np.ndarray,
) -> np.ndarray:
    """Evaluate an arithmetic expression of input parameters for every sample.

    Parameters
    ----------
    expression:
        Arithmetic expression using parameter names, e.g.
        ``"(n_legs^2 * leg_width^2) / leg_length"``.
    parameter_names:
        Ordered list of parameter names corresponding to columns of
        *physical_samples*.
    physical_samples:
        Array of shape ``(n, d)`` with physical parameter values.

    Returns
    -------
    values:
        Array of shape ``(n,)`` with the evaluated expression.
    """
    if physical_samples.shape[0] == 0:
        raise ValueError("Cannot evaluate a parameter expression without any samples.")

    normalized = (expression or "").strip()
    if not normalized:
        raise ValueError("Parameter expression cannot be empty.")
    normalized = normalized.replace("^", "**")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid parameter expression '{expression}': {exc}") from exc

    parameter_context = {
        name: physical_samples[:, axis]
        for axis, name in enumerate(parameter_names)
    }
    allowed_bases = set(_ALLOWED_ATTRIBUTE_BASES)
    allowed_names = set(parameter_context) | allowed_bases | {"pi", "tau", "e"}
    _validate_expression_ast(tree, allowed_names, allowed_bases)

    evaluation_locals = dict(parameter_context)
    evaluation_locals.update({
        "np": np,
        "numpy": np,
        "math": math,
        "pi": math.pi,
        "tau": math.tau,
        "e": math.e,
    })

    try:
        compiled = compile(tree, "<parameter_expression>", "eval")
        values = eval(compiled, {"__builtins__": {}}, evaluation_locals)  # noqa: S307
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate parameter expression '{expression}': {exc}"
        ) from exc

    values_array = np.asarray(values, dtype=float).reshape(-1)
    if values_array.size != physical_samples.shape[0]:
        raise ValueError(
            "Parameter expression must evaluate to exactly one value per sample."
        )
    return values_array
