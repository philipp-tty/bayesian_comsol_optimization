"""Analyze optimization results stored in JSON and plot GP mean/uncertainty slices."""

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
        default="mad",
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


def _extract_gp_dataset(results: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    gp_data = results.get("gaussian_process")
    if not isinstance(gp_data, dict):
        raise ValueError("Results JSON does not contain a 'gaussian_process' section.")

    parameter_definitions = gp_data.get("parameter_definitions")
    if not isinstance(parameter_definitions, list) or not parameter_definitions:
        raise ValueError("Gaussian process metadata is missing parameter definitions.")

    scaled = np.asarray(gp_data.get("scaled_parameters", []), dtype=float)
    objectives = np.asarray(gp_data.get("objective_observations", []), dtype=float)

    min_len = min(len(scaled), len(objectives))
    scaled = scaled[:min_len]
    objectives = objectives[:min_len]

    if scaled.ndim == 1:
        if scaled.size == 0:
            scaled = scaled.reshape(0, len(parameter_definitions))
        else:
            scaled = scaled.reshape(-1, 1)

    mask = np.isfinite(objectives)
    if scaled.size:
        mask &= np.all(np.isfinite(scaled), axis=1)
    else:
        mask &= False

    scaled = scaled[mask]
    objectives = objectives[mask]
    return scaled, objectives, parameter_definitions


def _filter_outliers(
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
    *,
    method: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove objective outliers using the requested strategy."""
    method = (method or "none").lower()
    threshold = max(0.0, float(threshold))
    total = int(objectives.size)

    stats = {
        "method": method,
        "threshold": threshold,
        "total": total,
        "retained": total,
        "removed": 0,
    }

    if method == "none" or total < 3:
        return scaled_samples, objectives, stats

    mask = np.ones(total, dtype=bool)
    values = objectives

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
    return filtered_scaled, filtered_objectives, stats


def _to_physical(unit_values: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    low, high = map(float, bounds)
    return low + np.clip(unit_values, 0.0, 1.0) * (high - low)


def _to_unit(value: float, bounds: Sequence[float]) -> float:
    low, high = map(float, bounds)
    if math.isclose(high, low):
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def _determine_best_parameters(
    results: dict,
    parameter_definitions: Sequence[dict],
    scaled_samples: np.ndarray,
    objectives: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    best_physical = {}

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
    ci_multiplier: float,
) -> Path:
    grid = np.linspace(0.0, 1.0, grid_points)
    evaluation_points = np.tile(best_unit, (grid_points, 1))
    evaluation_points[:, param_index] = grid

    mean, std = gp.predict(evaluation_points, return_std=True)
    ci = ci_multiplier * std
    physical_grid = _to_physical(grid, param_def["bounds"])
    observations = _to_physical(scaled_samples[:, param_index], param_def["bounds"])
    best_physical = _to_physical(np.array([best_unit[param_index]]), param_def["bounds"])[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(physical_grid, mean, color="#1f77b4", linewidth=2.0, label="GP mean")
    ax.fill_between(
        physical_grid,
        mean - ci,
        mean + ci,
        color="#1f77b4",
        alpha=0.25,
        label=f"+/- {ci_multiplier:.2f} sigma band",
    )
    ax.scatter(
        observations,
        objectives,
        s=25,
        color="#444444",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.85,
        label="Evaluations",
    )
    ax.axvline(best_physical, color="#d62728", linestyle="--", linewidth=1.5, label="Best parameter")

    unit = param_def.get("unit")
    label = param_def["name"] if not unit else f"{param_def['name']} [{unit}]"
    ax.set_xlabel(label)
    ax.set_ylabel("Objective value")
    ax.set_title(f"Mean & Uncertainty vs {label}")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = output_dir / f"mean_uncertainty_{param_def['name']}.png"
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
    scaled_samples, objectives, parameter_definitions = _extract_gp_dataset(results)

    if scaled_samples.size == 0 or objectives.size == 0:
        raise ValueError("No valid evaluations found in the results JSON.")

    scaled_samples, objectives, outlier_stats = _filter_outliers(
        scaled_samples,
        objectives,
        method=args.outlier_method,
        threshold=args.outlier_threshold,
    )

    if objectives.size == 0:
        raise ValueError(
            "All evaluations were filtered as outliers. Relax the threshold or disable outlier filtering."
        )

    if outlier_stats["removed"]:
        print(
            f"Filtered {outlier_stats['removed']} outlier(s) using '{outlier_stats['method']}' "
            f"(threshold={outlier_stats['threshold']})."
        )

    best_physical, best_unit = _determine_best_parameters(results, parameter_definitions, scaled_samples, objectives)

    summary = {
        "results_path": str(args.results_path),
        "num_evaluations_raw": int(outlier_stats["total"]),
        "num_filtered_outliers": int(outlier_stats["removed"]),
        "num_evaluations": int(objectives.size),
        "objective_mean": float(np.mean(objectives)),
        "objective_std": float(np.std(objectives, ddof=1)) if objectives.size > 1 else 0.0,
        "objective_min": float(np.min(objectives)),
        "objective_max": float(np.max(objectives)),
        "best_objective": float(results.get("objective_value", np.max(objectives))),
        "best_parameters": best_physical,
        "outlier_filter": outlier_stats,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write_summary(args.output_dir, summary)

    print(f"Loaded {summary['num_evaluations_raw']} evaluations from '{args.results_path}'.")
    if outlier_stats["removed"]:
        print(f"{summary['num_evaluations']} evaluations retained after outlier filtering.")
    print(f"Best objective: {summary['best_objective']:.6g}")
    print("Best parameters:")
    for name, value in best_physical.items():
        print(f"  - {name}: {value:.6g}")
    print(f"Summary saved to '{summary_path}'.")

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
            ci_multiplier=max(0.0, args.ci_multiplier),
        )
        plot_paths.append(path)

    print(f"Generated {len(plot_paths)} mean & uncertainty plots in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
