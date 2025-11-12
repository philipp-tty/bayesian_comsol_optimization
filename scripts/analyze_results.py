"""Analyze optimization results stored in JSON and plot GP diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
import warnings

import matplotlib

matplotlib.use("Agg")  # Ensure plotting works in headless environments.
import matplotlib.pyplot as plt
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

    if outputs_plot_path:
        print(f"Saved output history plot to '{outputs_plot_path}'.")
    print(f"Generated {len(plot_paths)} mean & uncertainty plots in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
