"""Data loading, outlier filtering, and GP dataset extraction for analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..state import OptimizationState


def load_state(path: Path) -> OptimizationState:
    """Load an optimization state from a JSON file.

    This is a convenience wrapper around ``OptimizationState.load()``.
    """
    return OptimizationState.load(path)


def load_legacy_results(path: Path) -> dict:
    """Load a legacy results JSON file (v0.2.0 format).

    Returns the raw dict for further processing by ``extract_gp_dataset``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Results file '{path}' does not exist.")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Results file '{path}' must contain a JSON object.")
    return data


def extract_gp_dataset(
    results: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, list[dict], np.ndarray]:
    """Parse a legacy results JSON structure into arrays usable by a GP.

    Returns
    -------
    scaled_samples:
        Unit-hypercube samples of shape ``(n, d)``.
    objectives:
        Objective values of shape ``(n,)``.
    parameter_definitions:
        List of parameter definition dicts.
    objective_definition:
        Dict with objective name/unit/index.
    output_definitions:
        List of output definition dicts.
    outputs_matrix:
        All output values of shape ``(n, n_outputs)``.
    """
    header = results.get("header")
    if not isinstance(header, dict):
        raise ValueError("Results JSON must contain a 'header' section.")

    inputs_meta = header.get("inputs")
    if not isinstance(inputs_meta, list) or not inputs_meta:
        raise ValueError("Results JSON must declare at least one input parameter.")

    outputs_meta = header.get("outputs")
    if not isinstance(outputs_meta, list) or not outputs_meta:
        raise ValueError("Results JSON must declare at least one output/objective.")

    parameter_definitions: list[dict] = []
    for idx, entry in enumerate(inputs_meta):
        if not isinstance(entry, dict):
            raise ValueError(f"Input definition #{idx} is not a JSON object.")

        name = str(entry.get("name") or f"input_{idx}")
        bounds = entry.get("bounds")
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 2
        ):
            raise ValueError(f"Parameter '{name}' must provide numeric [lower, upper] bounds.")

        try:
            lower = float(bounds[0])
            upper = float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bounds for parameter '{name}' must be numeric.") from exc

        if math.isclose(lower, upper):
            span = max(abs(lower), 1.0)
            upper = lower + span
        if lower > upper:
            lower, upper = upper, lower

        parameter_definitions.append({
            "name": name,
            "unit": entry.get("unit"),
            "bounds": (lower, upper),
        })

    output_definitions: list[dict] = []
    for idx, entry in enumerate(outputs_meta):
        if not isinstance(entry, dict):
            raise ValueError(f"Output definition #{idx} is not a JSON object.")
        output_definitions.append({
            "name": str(entry.get("name") or f"output_{idx}"),
            "unit": entry.get("unit"),
            "index": idx,
        })

    objective_definition = output_definitions[0]

    rows = results.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Results JSON does not contain any data rows.")

    scaled_rows: list[list[float]] = []
    objectives: list[float] = []
    outputs_matrix: list[list[float]] = []

    for row in rows:
        if (
            not isinstance(row, Sequence)
            or len(row) < 2
            or isinstance(row, (str, bytes))
        ):
            continue

        raw_inputs, raw_outputs = row[0], row[1]
        if (
            not isinstance(raw_inputs, Sequence)
            or isinstance(raw_inputs, (str, bytes))
            or len(raw_inputs) != len(parameter_definitions)
        ):
            continue
        if (
            not isinstance(raw_outputs, Sequence)
            or not raw_outputs
            or isinstance(raw_outputs, (str, bytes))
            or len(raw_outputs) < len(output_definitions)
        ):
            continue

        try:
            physical_inputs = [float(v) for v in raw_inputs]
            output_values = [float(raw_outputs[i]) for i in range(len(output_definitions))]
        except (TypeError, ValueError, IndexError):
            continue

        if not all(math.isfinite(v) for v in output_values):
            continue

        objective_value = output_values[objective_definition["index"]]
        scaled_inputs = [
            _to_unit(v, parameter_definitions[col]["bounds"])
            for col, v in enumerate(physical_inputs)
        ]
        scaled_rows.append(scaled_inputs)
        objectives.append(objective_value)
        outputs_matrix.append(output_values)

    scaled = (
        np.asarray(scaled_rows, dtype=float)
        if scaled_rows
        else np.empty((0, len(parameter_definitions)), dtype=float)
    )
    objectives_array = (
        np.asarray(objectives, dtype=float) if objectives else np.empty(0, dtype=float)
    )
    outputs_array = (
        np.asarray(outputs_matrix, dtype=float)
        if outputs_matrix
        else np.empty((0, len(output_definitions)), dtype=float)
    )
    return (
        scaled,
        objectives_array,
        parameter_definitions,
        objective_definition,
        output_definitions,
        outputs_array,
    )


def state_to_gp_arrays(
    state: OptimizationState,
    objective_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract successful observations from an OptimizationState as numpy arrays.

    Returns
    -------
    X:
        Unit-hypercube samples of shape ``(n_success, d)``.
    Y:
        Objective values of shape ``(n_success,)``.
    """
    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    if not mask.any():
        d = state.X.shape[1] if state.X.ndim == 2 else 1
        return np.empty((0, d), dtype=float), np.empty(0, dtype=float)
    X = state.X[mask].numpy()
    Y = state.Y[mask, objective_index].numpy()
    return X, Y


def filter_outliers(
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    *,
    method: str = "zscore",
    threshold: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Remove objective outliers using the requested strategy.

    Parameters
    ----------
    scaled_samples:
        Unit-hypercube samples of shape ``(n, d)``.
    objectives:
        Objective values of shape ``(n,)``.
    method:
        One of ``"none"``, ``"mad"``, ``"iqr"``, ``"zscore"``.
    threshold:
        Threshold multiplier for the selected method.

    Returns
    -------
    filtered_samples:
        Retained samples.
    filtered_objectives:
        Retained objectives.
    stats:
        Statistics about the filtering.
    mask:
        Boolean array indicating retained points.
    """
    method = (method or "none").lower()
    threshold = max(0.0, float(threshold))
    total = int(objectives.size)

    stats: dict = {
        "method": method,
        "threshold": threshold,
        "total": total,
        "retained": total,
        "removed": 0,
        "removed_indices": [],
        "removed_values": [],
    }

    mask = np.ones(total, dtype=bool)
    values = objectives

    if method == "none" or total < 3:
        return scaled_samples, objectives, stats, mask

    if method == "mad":
        center = float(np.median(values))
        deviations = np.abs(values - center)
        scale = float(np.median(deviations))
        stats["center"] = center
        stats["scale"] = scale
        if scale > 0.0:
            modified_z = 0.6745 * (values - center) / scale
            mask = np.abs(modified_z) <= threshold
    elif method == "iqr":
        q1, q3 = np.percentile(values, [25.0, 75.0])
        scale = float(q3 - q1)
        stats["q1"] = float(q1)
        stats["q3"] = float(q3)
        stats["scale"] = scale
        if scale > 0.0:
            lower = q1 - threshold * scale
            upper = q3 + threshold * scale
            mask = (values >= lower) & (values <= upper)
    elif method == "zscore":
        center = float(np.mean(values))
        scale = float(np.std(values))
        stats["center"] = center
        stats["scale"] = scale
        if scale > 0.0:
            mask = np.abs((values - center) / scale) <= threshold
    else:
        raise ValueError(f"Unsupported outlier method '{method}'.")

    filtered_scaled = scaled_samples[mask]
    filtered_objectives = objectives[mask]
    stats["removed"] = int(total - filtered_objectives.size)
    stats["retained"] = int(filtered_objectives.size)
    stats["removed_indices"] = np.where(~mask)[0].tolist()
    stats["removed_values"] = objectives[~mask].tolist()
    return filtered_scaled, filtered_objectives, stats, mask


def _to_unit(value: float, bounds: Sequence[float]) -> float:
    low, high = map(float, bounds)
    if math.isclose(high, low):
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def to_physical(unit_values: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Convert unit-hypercube values to physical domain."""
    low, high = map(float, bounds)
    return low + np.clip(unit_values, 0.0, 1.0) * (high - low)
