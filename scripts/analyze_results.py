"""Analyze optimization results stored in JSON and plot GP diagnostics."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Sequence
import warnings

import matplotlib

matplotlib.use("Agg")  # Ensure plotting works in headless environments.
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize optimization results and plot GP mean/uncertainty profiles."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("optimization_results.json"),
        help="Path to the JSON file produced by the optimization workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory where plots and summary artifacts will be stored.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=200,
        help="Number of points per parameter grid when evaluating GP slices.",
    )
    parser.add_argument(
        "--ci-multiplier",
        type=float,
        default=1.96,
        help="Multiplier for the predictive standard deviation (default 1.96 for 95% CI).",
    )
    parser.add_argument(
        "--outlier-method",
        type=str,
        default="zscore",
        choices=("none", "mad", "iqr", "zscore"),
        help="Strategy for removing objective outliers before fitting the GP (default: mad).",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help=(
            "Threshold multiplier used by the selected outlier method "
            "(e.g. 3.5 MAD, 1.5 IQR, or standard deviations)."
        ),
    )
    parser.add_argument(
        "--iso-x-parameter",
        type=str,
        default=None,
        help=(
            "Name of the parameter to use as the x-axis for iso contour plots "
            "(all other parameters will be cycled on the y-axis)."
        ),
    )
    parser.add_argument(
        "--iso-grid-points",
        type=int,
        default=60,
        help="Number of grid steps per axis when evaluating iso contour slices.",
    )
    parser.add_argument(
        "--no-parallel-coordinates",
        action="store_true",
        help="Skip generating the parallel coordinates visualization.",
    )
    parser.add_argument(
        "--parameter-expression",
        type=str,
        default=None,
        help=(
            "Expression involving input parameter names (e.g. '(n_legs^2 * leg_width^2) / ...') "
            "that will be evaluated for each retained sample and used as the x-axis when plotting "
            "all outputs."
        ),
    )
    parser.add_argument(
        "--parameter-expression-label",
        type=str,
        default=None,
        help="Optional custom label for the expression-based x-axis. Defaults to the expression itself.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Results file '{path}' does not exist.")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Results file '{path}' must contain a JSON object.")
    return data


def _extract_gp_dataset(
    results: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, list[dict], np.ndarray]:
    """Parse the COMSOL optimization results structure into arrays usable by the GP."""
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

        parameter_definitions.append(
            {
                "name": name,
                "unit": entry.get("unit"),
                "bounds": (lower, upper),
            }
        )

    output_definitions: list[dict] = []
    for idx, entry in enumerate(outputs_meta):
        if not isinstance(entry, dict):
            raise ValueError(f"Output definition #{idx} is not a JSON object.")
        output_definitions.append(
            {
                "name": str(entry.get("name") or f"output_{idx}"),
                "unit": entry.get("unit"),
                "index": idx,
            }
        )

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
            physical_inputs = [float(value) for value in raw_inputs]
            output_values = [
                float(raw_outputs[idx]) for idx in range(len(output_definitions))
            ]
        except (TypeError, ValueError, IndexError):
            continue

        if not all(math.isfinite(value) for value in output_values):
            continue

        objective_value = output_values[objective_definition["index"]]
        scaled_inputs = [
            _to_unit(value, parameter_definitions[col]["bounds"])
            for col, value in enumerate(physical_inputs)
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


def _resolve_parameter_index(parameter_definitions: Sequence[dict], name: str) -> int:
    lookup = str(name or "").strip().lower()
    if not lookup:
        raise ValueError("Parameter name cannot be empty.")
    for idx, definition in enumerate(parameter_definitions):
        if definition["name"].strip().lower() == lookup:
            return idx
    raise ValueError(f"Unknown parameter '{name}'. Available: {[p['name'] for p in parameter_definitions]}")


def _filter_outliers(
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    *,
    method: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Remove objective outliers using the requested strategy and return the retention mask."""
    method = (method or "none").lower()
    threshold = max(0.0, float(threshold))
    total = int(objectives.size)

    stats = {
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


def _to_physical(unit_values: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    low, high = map(float, bounds)
    return low + np.clip(unit_values, 0.0, 1.0) * (high - low)


def _to_unit(value: float, bounds: Sequence[float]) -> float:
    low, high = map(float, bounds)
    if math.isclose(high, low):
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def _scaled_to_physical(
    scaled_samples: np.ndarray, parameter_definitions: Sequence[dict]
) -> np.ndarray:
    """Convert unit-hypercube samples back into their physical parameter values."""
    if scaled_samples.size == 0:
        return np.empty_like(scaled_samples, dtype=float)
    physical = np.empty_like(scaled_samples, dtype=float)
    for axis, definition in enumerate(parameter_definitions):
        physical[:, axis] = _to_physical(scaled_samples[:, axis], definition["bounds"])
    return physical


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
        elif isinstance(node, (ast.Load, ast.Expression, ast.keyword)):
            continue
        else:
            raise ValueError("Expressions may only contain arithmetic operations and function calls.")


def _evaluate_parameter_expression(
    expression: str,
    parameter_definitions: Sequence[dict],
    scaled_samples: np.ndarray,
) -> np.ndarray:
    """Evaluate an expression of the input parameters for every retained sample."""
    if scaled_samples.shape[0] == 0:
        raise ValueError("Cannot evaluate a parameter expression without any samples.")

    normalized = (expression or "").strip()
    if not normalized:
        raise ValueError("Parameter expression cannot be empty.")
    normalized = normalized.replace("^", "**")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid parameter expression '{expression}': {exc}") from exc

    physical_samples = _scaled_to_physical(scaled_samples, parameter_definitions)
    parameter_context = {
        definition["name"]: physical_samples[:, axis]
        for axis, definition in enumerate(parameter_definitions)
    }
    allowed_bases = set(_ALLOWED_ATTRIBUTE_BASES)
    allowed_names = set(parameter_context) | allowed_bases | {"pi", "tau", "e"}
    _validate_expression_ast(tree, allowed_names, allowed_bases)

    evaluation_locals = dict(parameter_context)
    evaluation_locals.update(
        {
            "np": np,
            "numpy": np,
            "math": math,
            "pi": math.pi,
            "tau": math.tau,
            "e": math.e,
        }
    )

    try:
        compiled = compile(tree, "<parameter_expression>", "eval")
        values = eval(compiled, {"__builtins__": {}}, evaluation_locals)
    except Exception as exc:  # pragma: no cover - expression failures are user-driven
        raise ValueError(f"Failed to evaluate parameter expression '{expression}': {exc}") from exc

    values_array = np.asarray(values, dtype=float).reshape(-1)
    if values_array.size != scaled_samples.shape[0]:
        raise ValueError(
            "Parameter expression must evaluate to exactly one value per retained sample."
        )
    return values_array


def _format_label(name: str, unit: str | None) -> str:
    base = str(name or "value")
    return base if not unit else f"{base} [{unit}]"


def _determine_best_parameters(
    results: dict,
    parameter_definitions: Sequence[dict],
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    best_physical: dict[str, float] = {}

    params_section = results.get("parameters") or results.get("best_parameters")
    if isinstance(params_section, dict) and params_section:
        best_physical = {str(k): float(v) for k, v in params_section.items()}

    if not best_physical and objectives.size and scaled_samples.size:
        best_idx = int(np.nanargmax(objectives))
        best_sample = scaled_samples[best_idx]
        for axis, definition in enumerate(parameter_definitions):
            name = definition["name"]
            best_physical[name] = float(_to_physical(best_sample[axis], definition["bounds"]))

    if not best_physical:
        for definition in parameter_definitions:
            name = definition["name"]
            lower, upper = definition["bounds"]
            best_physical[name] = float((lower + upper) / 2.0)

    best_unit = np.zeros(len(parameter_definitions), dtype=float)
    for axis, definition in enumerate(parameter_definitions):
        value = best_physical.get(definition["name"], float("nan"))
        best_unit[axis] = _to_unit(value, definition["bounds"])

    return best_physical, best_unit


def _fit_gaussian_process(x: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    if x.shape[0] < 2:
        raise ValueError("At least two successful evaluations are required to fit the GP.")

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Enforce 1D targets
    y = np.asarray(y, dtype=float).ravel()

    length_scale = np.full(x.shape[1], 0.5, dtype=float)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=length_scale,
        nu=2.5,
        length_scale_bounds=(1e-3, 100.0),
    ) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    )
    gp.fit(x, y)
    return gp


def _gp_predict_with_std(
    gp: GaussianProcessRegressor, evaluation_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GP mean/std using sklearn's implementation (handles normalization correctly)."""
    if not hasattr(gp, "X_train_"):
        raise ValueError("GaussianProcessRegressor must be fitted before requesting predictions.")

    evaluation_points = np.atleast_2d(np.asarray(evaluation_points, dtype=float))
    mean, std = gp.predict(evaluation_points, return_std=True)
    return mean, std


def _plot_parameter_slice(
    *,
    gp: GaussianProcessRegressor,
    param_def: dict,
    param_index: int,
    best_unit: np.ndarray,
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    output_dir: Path,
    grid_points: int,
    objective_label: str,
    ci_multiplier: float,
) -> Path:
    grid = np.linspace(0.0, 1.0, grid_points)
    evaluation_points = np.tile(best_unit, (grid_points, 1))
    evaluation_points[:, param_index] = grid

    mean, std = _gp_predict_with_std(gp, evaluation_points)
    ci = ci_multiplier * std
    physical_grid = _to_physical(grid, param_def["bounds"])
    observations = _to_physical(scaled_samples[:, param_index], param_def["bounds"])
    best_physical = _to_physical(np.array([best_unit[param_index]]), param_def["bounds"])[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(physical_grid, mean, linewidth=2.0, label="GP mean")
    ax.fill_between(
        physical_grid,
        mean - ci,
        mean + ci,
        alpha=0.25,
        label=f"+/- {ci_multiplier:.2f} sigma band",
    )
    ax.scatter(
        observations,
        objectives,
        s=25,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.85,
        label="Evaluations",
    )
    ax.axvline(best_physical, linestyle="--", linewidth=1.5, label="Best parameter")

    label = _format_label(param_def["name"], param_def.get("unit"))
    ax.set_xlabel(label)
    ax.set_ylabel(objective_label)
    ax.set_title(f"Mean & Uncertainty vs {label}")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / f"mean_uncertainty_{param_def['name']}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_iso_contours(
    *,
    gp: GaussianProcessRegressor,
    x_index: int,
    parameter_definitions: Sequence[dict],
    best_unit: np.ndarray,
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    output_dir: Path,
    grid_points: int,
    objective_label: str,
) -> list[Path]:
    if grid_points < 5:
        raise ValueError("At least 5 grid points per axis are required for iso contour plots.")

    x_definition = parameter_definitions[x_index]
    x_label = _format_label(x_definition["name"], x_definition.get("unit"))
    unit_grid = np.linspace(0.0, 1.0, grid_points)
    xx, yy = np.meshgrid(unit_grid, unit_grid)
    base = np.tile(best_unit, (grid_points * grid_points, 1))

    paths: list[Path] = []
    for y_index, y_definition in enumerate(parameter_definitions):
        if y_index == x_index:
            continue

        evaluation_points = base.copy()
        evaluation_points[:, x_index] = xx.ravel()
        evaluation_points[:, y_index] = yy.ravel()
        mean, _ = _gp_predict_with_std(gp, evaluation_points)
        contour_values = mean.reshape(grid_points, grid_points)

        x_physical = _to_physical(xx, x_definition["bounds"])
        y_physical = _to_physical(yy, y_definition["bounds"])
        observations_x = _to_physical(scaled_samples[:, x_index], x_definition["bounds"])
        observations_y = _to_physical(scaled_samples[:, y_index], y_definition["bounds"])

        best_x = _to_physical(np.array([best_unit[x_index]]), x_definition["bounds"])[0]
        best_y = _to_physical(np.array([best_unit[y_index]]), y_definition["bounds"])[0]

        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        contour = ax.contourf(
            x_physical,
            y_physical,
            contour_values,
            levels=20,
            cmap="viridis",
        )
        fig.colorbar(contour, ax=ax, label=objective_label)
        scatter = ax.scatter(
            observations_x,
            observations_y,
            c=objectives,
            cmap="viridis",
            norm=contour.norm,
            s=25,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.85,
            label="Evaluations",
        )
        ax.scatter(
            best_x,
            best_y,
            marker="*",
            s=150,
            color="red",
            edgecolors="black",
            linewidths=0.8,
            label="Best point",
            zorder=3,
        )
        y_label = _format_label(y_definition["name"], y_definition.get("unit"))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Iso contours: {y_label} vs {x_label}")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        safe_x = x_definition["name"].replace(" ", "_")
        safe_y = y_definition["name"].replace(" ", "_")
        output_path = output_dir / f"iso_contour_{safe_y}_vs_{safe_x}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        paths.append(output_path)
    return paths


def _plot_parallel_coordinates(
    *,
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    parameter_definitions: Sequence[dict],
    objective_label: str,
    output_dir: Path,
) -> Path:
    if scaled_samples.size == 0 or objectives.size == 0:
        raise ValueError("At least one evaluation is required to create the parallel coordinates plot.")

    # Normalize objectives for coloring, falling back to 0.5 if constant.
    obj_min = float(np.min(objectives))
    obj_max = float(np.max(objectives))
    if math.isclose(obj_min, obj_max):
        normalized_objectives = np.full_like(objectives, 0.5, dtype=float)
        vmax = obj_min + 1.0
    else:
        normalized_objectives = (objectives - obj_min) / (obj_max - obj_min)
        vmax = obj_max
    norm = mcolors.Normalize(vmin=obj_min, vmax=vmax)

    data = np.column_stack((scaled_samples, normalized_objectives))
    num_axes = data.shape[1]
    axis_positions = np.arange(num_axes)
    axis_labels = [
        _format_label(definition["name"], definition.get("unit")) for definition in parameter_definitions
    ] + [f"{objective_label} (normalized)"]

    fig, ax = plt.subplots(figsize=(max(8.0, num_axes * 1.3), 5.0))
    colors = plt.cm.viridis(norm(objectives))
    for row, color in zip(data, colors, strict=False):
        ax.plot(axis_positions, row, color=color, linewidth=0.8, alpha=0.7)

    # Highlight the best objective sample if available.
    best_idx = int(np.nanargmax(objectives))
    ax.plot(
        axis_positions,
        data[best_idx],
        color="red",
        linewidth=2.0,
        alpha=0.9,
        label="Best objective",
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar_mappable.set_array(objectives)
    cbar = fig.colorbar(scalar_mappable, ax=ax)
    cbar.set_label(objective_label)

    ax.set_xticks(axis_positions)
    ax.set_xticklabels(axis_labels, rotation=20, ha="right")
    ax.set_xlim(axis_positions[0], axis_positions[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates (parameters normalized to [0, 1])")
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "parallel_coordinates.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_outputs_over_iterations(
    *,
    outputs: np.ndarray,
    output_definitions: Sequence[dict],
    output_dir: Path,
) -> Path:
    if outputs.size == 0 or outputs.shape[1] == 0:
        raise ValueError("At least one output is required to create the iteration plot.")

    iterations = np.arange(1, outputs.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    plotted = False
    for idx, definition in enumerate(output_definitions):
        if idx >= outputs.shape[1]:
            break
        series = outputs[:, idx]
        mask = np.isfinite(series)
        if not np.any(mask):
            continue
        label = _format_label(definition["name"], definition.get("unit"))
        ax.plot(iterations[mask], series[mask], marker="o", linewidth=1.5, label=label)
        plotted = True

    if not plotted:
        raise ValueError("No finite output values available for plotting.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Output value")
    ax.set_title("Outputs vs Iteration")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "outputs_vs_iteration.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_outputs_vs_expression(
    *,
    expression_values: np.ndarray,
    outputs: np.ndarray,
    output_definitions: Sequence[dict],
    output_dir: Path,
    expression_label: str,
) -> Path:
    """Plot each output against an evaluated parameter expression."""
    x_values = np.asarray(expression_values, dtype=float).reshape(-1)
    y_values = np.asarray(outputs, dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("Expression values and outputs must both be provided for plotting.")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError(
            "Expression values must have the same number of entries as the output observations."
        )
    if y_values.ndim == 1:
        y_values = y_values.reshape(-1, 1)

    finite_x = np.isfinite(x_values)
    if not np.any(finite_x):
        raise ValueError("Expression evaluation did not produce any finite values.")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plotted = False
    for idx, definition in enumerate(output_definitions):
        if idx >= y_values.shape[1]:
            break
        series = y_values[:, idx]
        mask = finite_x & np.isfinite(series)
        if not np.any(mask):
            continue
        ordering = np.argsort(x_values[mask])
        x_sorted = x_values[mask][ordering]
        y_sorted = series[mask][ordering]
        label = _format_label(definition["name"], definition.get("unit"))
        ax.plot(x_sorted, y_sorted, marker="o", linewidth=1.5, label=label)
        plotted = True

    if not plotted:
        raise ValueError("No finite output values remained after filtering; cannot plot expression.")

    label = expression_label.strip() or "Parameter expression"
    ax.set_xlabel(label)
    ax.set_ylabel("Output value")
    ax.set_title(f"Outputs vs {label}")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / "outputs_vs_expression.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _write_summary(output_dir: Path, summary: dict) -> Path:
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def main() -> None:
    args = _parse_args()
    results = _load_results(args.results_path)
    (
        scaled_samples,
        objectives,
        parameter_definitions,
        objective_definition,
        output_definitions,
        raw_outputs,
    ) = _extract_gp_dataset(results)
    objective_label = _format_label(
        objective_definition["name"], objective_definition.get("unit")
    )

    if scaled_samples.size == 0 or objectives.size == 0:
        raise ValueError("No valid evaluations found in the results JSON.")

    scaled_samples, objectives, outlier_stats, retained_mask = _filter_outliers(
        scaled_samples,
        objectives,
        method=args.outlier_method,
        threshold=args.outlier_threshold,
    )

    if raw_outputs.size:
        if raw_outputs.shape[0] != retained_mask.size:
            raise ValueError(
                "The number of output rows does not match the number of objective evaluations; "
                "cannot consistently drop filtered points."
            )
        raw_outputs = raw_outputs[retained_mask]

    if objectives.size == 0:
        raise ValueError(
            "All evaluations were filtered as outliers. Relax the threshold or disable outlier filtering."
        )

    if outlier_stats["removed"]:
        print(
            f"Filtered {outlier_stats['removed']} outlier(s) using '{outlier_stats['method']}' "
            f"(threshold={outlier_stats['threshold']}):"
        )
        for idx, value in zip(outlier_stats["removed_indices"], outlier_stats["removed_values"]):
            print(f"  - Point {idx}: objective = {value:.6g}")

    best_physical, best_unit = _determine_best_parameters(
        results, parameter_definitions, scaled_samples, objectives
    )

    best_objective = float(np.max(objectives))

    objective_metadata = {
        "name": objective_definition["name"],
        "unit": objective_definition.get("unit"),
        "index": objective_definition["index"],
        "label": objective_label,
    }

    summary = {
        "results_path": str(args.results_path),
        "num_evaluations_raw": int(outlier_stats["total"]),
        "num_filtered_outliers": int(outlier_stats["removed"]),
        "num_evaluations": int(objectives.size),
        "objective_min": float(np.min(objectives)),
        "objective_max": float(np.max(objectives)),
        "best_parameters": best_physical,
        "objective": objective_metadata,
        "outputs": output_definitions,
        "outlier_filter": outlier_stats,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_summary(args.output_dir, summary)

    print(f"Loaded {summary['num_evaluations_raw']} evaluations from '{args.results_path}'.")
    if outlier_stats["removed"]:
        print(
            f"{summary['num_evaluations']} evaluations retained after filtering "
            f"{summary['num_filtered_outliers']} outlier(s)."
        )
    print(f"Best objective ({objective_label}): {summary['objective_max']:.6g}")
    print(f"Worst objective ({objective_label}): {summary['objective_min']:.6g}")
    print("Best parameters:")
    for name, value in best_physical.items():
        print(f"  - {name}: {value:.6g}")
    print(f"Summary saved to '{summary_path}'.")

    outputs_plot_path = None
    try:
        outputs_plot_path = _plot_outputs_over_iterations(
            outputs=raw_outputs,
            output_definitions=output_definitions,
            output_dir=args.output_dir,
        )
    except ValueError:
        pass

    expression_plot_path = None
    if args.parameter_expression:
        if raw_outputs.size == 0 or raw_outputs.shape[1] == 0:
            print(
                "Skipping parameter expression plot because no raw outputs are available "
                "in the results JSON."
            )
        else:
            try:
                expression_values = _evaluate_parameter_expression(
                    args.parameter_expression,
                    parameter_definitions,
                    scaled_samples,
                )
                expression_label = args.parameter_expression_label or args.parameter_expression
                expression_plot_path = _plot_outputs_vs_expression(
                    expression_values=expression_values,
                    outputs=raw_outputs,
                    output_definitions=output_definitions,
                    output_dir=args.output_dir,
                    expression_label=expression_label,
                )
            except ValueError as exc:
                print(
                    f"Failed to evaluate/plot parameter expression '{args.parameter_expression}': {exc}"
                )

    # GP fitting and plotting
    gp = _fit_gaussian_process(scaled_samples, objectives)
    plot_paths: list[Path] = []
    for index, definition in enumerate(parameter_definitions):
        path = _plot_parameter_slice(
            gp=gp,
            param_def=definition,
            param_index=index,
            best_unit=best_unit,
            scaled_samples=scaled_samples,
            objectives=objectives,
            output_dir=args.output_dir,
            grid_points=max(10, args.grid_points),
            objective_label=objective_label,
            ci_multiplier=max(0.0, args.ci_multiplier),
        )
        plot_paths.append(path)

    iso_plot_paths: list[Path] = []
    if args.iso_x_parameter:
        x_axis_index = _resolve_parameter_index(parameter_definitions, args.iso_x_parameter)
        iso_plot_paths = _plot_iso_contours(
            gp=gp,
            x_index=x_axis_index,
            parameter_definitions=parameter_definitions,
            best_unit=best_unit,
            scaled_samples=scaled_samples,
            objectives=objectives,
            output_dir=args.output_dir,
            grid_points=max(5, args.iso_grid_points),
            objective_label=objective_label,
        )

    parallel_plot_path: Path | None = None
    if not args.no_parallel_coordinates:
        try:
            parallel_plot_path = _plot_parallel_coordinates(
                scaled_samples=scaled_samples,
                objectives=objectives,
                parameter_definitions=parameter_definitions,
                objective_label=objective_label,
                output_dir=args.output_dir,
            )
        except ValueError:
            parallel_plot_path = None

    if outputs_plot_path:
        print(f"Saved output history plot to '{outputs_plot_path}'.")
    if expression_plot_path:
        print(f"Saved output-vs-expression plot to '{expression_plot_path}'.")
    print(f"Generated {len(plot_paths)} mean & uncertainty plots in '{args.output_dir}'.")
    if iso_plot_paths:
        print(
            f"Generated {len(iso_plot_paths)} iso contour plot(s) with x-axis parameter "
            f"'{parameter_definitions[x_axis_index]['name']}' in '{args.output_dir}'."
        )
    if parallel_plot_path:
        print(f"Saved parallel coordinates plot to '{parallel_plot_path}'.")


if __name__ == "__main__":
    main()
