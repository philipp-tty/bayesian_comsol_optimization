"""Utilities for full-factorial parameter sweeps."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from .parameters import OptimizationParameter


_NUMERIC_VALUE_RE = re.compile(
    r"""
    (?P<value>[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?)
    \s*
    (?:
        \[(?P<bracket_unit>[^\]]+)\]
        |
        (?P<unit>[A-Za-zµμΩ°/%]+)
    )?
    """,
    re.VERBOSE,
)


def build_sweep_grids(
    parameters: Sequence[OptimizationParameter],
    config: dict[str, Any],
    points: int | None = None,
) -> tuple[list[OptimizationParameter], list[list[float]], dict[str, str]]:
    """Build per-parameter value grids for a full-factorial sweep.

    Explicit values are read from ``sweep.values.<parameter_name>`` first, then
    from per-parameter ``sweep_values`` entries. Parameters without explicit
    values fall back to generated grids when *points* is provided.
    """
    sweep_cfg = config.get("sweep", {}) or {}
    if not isinstance(sweep_cfg, dict):
        raise ValueError("Config key 'sweep' must be a mapping when present.")

    values_cfg = sweep_cfg.get("values", {}) or {}
    if not isinstance(values_cfg, dict):
        raise ValueError("Config key 'sweep.values' must be a parameter-to-values mapping.")

    parameter_cfg_by_name = {
        p_cfg["name"]: p_cfg
        for p_cfg in config.get("parameters", [])
        if isinstance(p_cfg, dict) and "name" in p_cfg
    }

    active_params = [p for p in parameters if not p.is_constant]
    if not active_params:
        raise ValueError("No active (non-constant) parameters found in config.")

    parameter_names = {p.name for p in parameters}
    active_names = {p.name for p in active_params}
    unknown_names = sorted(set(values_cfg) - parameter_names)
    if unknown_names:
        raise ValueError(
            "Sweep values reference unknown parameter(s): "
            + ", ".join(unknown_names)
            + "."
        )
    constant_names = sorted(set(values_cfg) - active_names)
    if constant_names:
        raise ValueError(
            "Sweep values were provided for constant parameter(s): "
            + ", ".join(constant_names)
            + ". Remove constant_value to sweep them."
        )

    param_grids: list[list[float]] = []
    grid_sources: dict[str, str] = {}
    missing: list[str] = []

    for p in active_params:
        if p.name in values_cfg:
            values = parse_sweep_values(values_cfg[p.name], p)
            grid_sources[p.name] = "explicit"
        elif "sweep_values" in parameter_cfg_by_name.get(p.name, {}):
            values = parse_sweep_values(parameter_cfg_by_name[p.name]["sweep_values"], p)
            grid_sources[p.name] = "explicit"
        elif points is not None:
            values = _generated_values(p, points)
            grid_sources[p.name] = "generated"
        else:
            missing.append(p.name)
            continue

        if not values:
            raise ValueError(f"Sweep for parameter '{p.name}' has no values.")
        param_grids.append(values)

    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "Missing sweep values for active parameter(s): "
            f"{names}. Add sweep.values entries or pass --points."
        )

    return active_params, param_grids, grid_sources


def parse_sweep_values(raw: Any, parameter: OptimizationParameter) -> list[float]:
    """Parse and coerce explicit sweep values for one parameter.

    Supported input forms include numbers, lists of numbers/strings, and
    compact strings such as ``"4mm 8mm 12 mm"`` or ``"4, 8, 12"``.
    """
    parsed: list[float] = []

    if isinstance(raw, str):
        parsed.extend(_parse_value_string(raw, parameter))
    elif isinstance(raw, (int, float)):
        parsed.append(float(raw))
    elif isinstance(raw, Sequence):
        for item in raw:
            parsed.extend(parse_sweep_values(item, parameter))
    else:
        raise ValueError(
            f"Unsupported sweep values for parameter '{parameter.name}': {raw!r}."
        )

    seen: set[float] = set()
    coerced_values: list[float] = []
    for value in parsed:
        coerced = parameter.coerce_physical_value(value)
        if coerced not in seen:
            seen.add(coerced)
            coerced_values.append(coerced)

    return coerced_values


def combo_key(values: Sequence[float]) -> tuple[float, ...]:
    """Stable key for detecting completed sweep combinations."""
    return tuple(round(float(value), 10) for value in values)


def _parse_value_string(raw: str, parameter: OptimizationParameter) -> list[float]:
    text = raw.strip()
    if not text:
        return []

    values: list[float] = []
    position = 0
    for match in _NUMERIC_VALUE_RE.finditer(text):
        separator = text[position:match.start()]
        if separator.strip(" \t\r\n,;") != "":
            raise ValueError(
                f"Could not parse sweep values for parameter '{parameter.name}': {raw!r}."
            )

        unit = match.group("bracket_unit") or match.group("unit")
        if unit:
            _validate_unit(unit, parameter)
        values.append(float(match.group("value")))
        position = match.end()

    if not values or text[position:].strip(" \t\r\n,;") != "":
        raise ValueError(
            f"Could not parse sweep values for parameter '{parameter.name}': {raw!r}."
        )
    return values


def _validate_unit(unit: str, parameter: OptimizationParameter) -> None:
    if parameter.unit is None:
        raise ValueError(
            f"Sweep value for parameter '{parameter.name}' includes unit '{unit}', "
            "but the parameter has no configured unit."
        )
    if _normalize_unit(unit) != _normalize_unit(parameter.unit):
        raise ValueError(
            f"Sweep value for parameter '{parameter.name}' uses unit '{unit}', "
            f"but the parameter is configured with unit '{parameter.unit}'."
        )


def _normalize_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    normalized = normalized.replace("ω", "ohm")
    normalized = normalized.replace("µ", "u").replace("μ", "u")
    return normalized


def _generated_values(parameter: OptimizationParameter, points: int) -> list[float]:
    if points < 2:
        raise ValueError("--points must be at least 2.")

    lower, upper = parameter.bounds
    if parameter.log_scale:
        if lower <= 0 or upper <= 0:
            raise ValueError(
                f"Log-scale sweep parameter '{parameter.name}' must have positive bounds."
            )
        log_lower = math.log10(lower)
        log_upper = math.log10(upper)
        raw_values = [
            10 ** (log_lower + i * (log_upper - log_lower) / (points - 1))
            for i in range(points)
        ]
    else:
        raw_values = [
            lower + i * (upper - lower) / (points - 1)
            for i in range(points)
        ]

    return _coerce_unique(raw_values, parameter)


def _coerce_unique(
    values: Sequence[float],
    parameter: OptimizationParameter,
) -> list[float]:
    seen: set[float] = set()
    coerced_values: list[float] = []
    for value in values:
        coerced = parameter.coerce_physical_value(value)
        if coerced not in seen:
            seen.add(coerced)
            coerced_values.append(coerced)
    return coerced_values
