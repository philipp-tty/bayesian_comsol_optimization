"""Plotting functions for optimization analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch

from ..state import OptimizationState
from ..surrogate import GPSurrogate
from .dataset import to_physical


def _format_label(name: str, unit: str | None) -> str:
    base = str(name or "value")
    return base if not unit else f"{base} [{unit}]"


# ------------------------------------------------------------------
# 1D parameter slices
# ------------------------------------------------------------------


def plot_parameter_slices(
    state: OptimizationState,
    surrogate: GPSurrogate,
    output_dir: Path,
    *,
    grid_points: int = 200,
    ci_multiplier: float = 1.96,
    objective_index: int = 0,
) -> list[Path]:
    """Plot GP mean +/- CI vs each parameter (1D slices at best point).

    Returns the list of saved plot file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_params = [p for p in state.parameters if not p.is_constant]
    d = len(active_params)
    if d == 0:
        return []

    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    if not mask.any():
        return []

    X = state.X[mask].numpy()
    Y = state.Y[mask, objective_index].numpy()

    # Determine best point in unit space
    obj_name = state.objective_names[objective_index]
    maximize = state.maximize[objective_index] if objective_index < len(state.maximize) else True
    if maximize:
        best_idx = int(np.argmax(Y))
    else:
        best_idx = int(np.argmin(Y))
    best_unit = X[best_idx]

    paths: list[Path] = []
    for axis, param in enumerate(active_params):
        grid = np.linspace(0.0, 1.0, grid_points)
        eval_points = np.tile(best_unit, (grid_points, 1))
        eval_points[:, axis] = grid

        eval_tensor = torch.tensor(eval_points, dtype=torch.double)
        mean_t, var_t = surrogate.predict(eval_tensor)
        mean = mean_t[:, objective_index].detach().numpy()
        std = var_t[:, objective_index].detach().numpy() ** 0.5
        ci = ci_multiplier * std

        physical_grid = param.bounds[0] + grid * (param.bounds[1] - param.bounds[0])
        observations_phys = param.bounds[0] + X[:, axis] * (param.bounds[1] - param.bounds[0])
        best_phys = param.bounds[0] + best_unit[axis] * (param.bounds[1] - param.bounds[0])

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
            observations_phys, Y,
            s=25, edgecolors="white", linewidths=0.4, alpha=0.85,
            label="Evaluations",
        )
        ax.axvline(best_phys, linestyle="--", linewidth=1.5, label="Best parameter")

        label = _format_label(param.name, param.unit)
        ax.set_xlabel(label)
        ax.set_ylabel(_format_label(obj_name, None))
        ax.set_title(f"Mean & Uncertainty vs {label}")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        path = output_dir / f"mean_uncertainty_{param.name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


# ------------------------------------------------------------------
# 2D iso contours
# ------------------------------------------------------------------


def plot_iso_contours(
    state: OptimizationState,
    surrogate: GPSurrogate,
    output_dir: Path,
    x_param_name: str,
    *,
    grid_points: int = 60,
    objective_index: int = 0,
) -> list[Path]:
    """Plot 2D iso-contour plots of the GP mean for each parameter pair.

    The parameter named *x_param_name* is used as the x-axis; all other
    active parameters are cycled on the y-axis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_params = [p for p in state.parameters if not p.is_constant]
    d = len(active_params)
    if d < 2:
        return []

    param_names = [p.name for p in active_params]
    if x_param_name not in param_names:
        raise ValueError(
            f"Unknown parameter '{x_param_name}'. Available: {param_names}"
        )
    x_index = param_names.index(x_param_name)

    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    if not mask.any():
        return []

    X = state.X[mask].numpy()
    Y = state.Y[mask, objective_index].numpy()

    obj_name = state.objective_names[objective_index]
    maximize = state.maximize[objective_index] if objective_index < len(state.maximize) else True
    best_idx = int(np.argmax(Y)) if maximize else int(np.argmin(Y))
    best_unit = X[best_idx]

    unit_grid = np.linspace(0.0, 1.0, grid_points)
    xx, yy = np.meshgrid(unit_grid, unit_grid)
    base = np.tile(best_unit, (grid_points * grid_points, 1))

    x_param = active_params[x_index]
    x_label = _format_label(x_param.name, x_param.unit)
    obj_label = _format_label(obj_name, None)

    paths: list[Path] = []
    for y_index, y_param in enumerate(active_params):
        if y_index == x_index:
            continue

        eval_points = base.copy()
        eval_points[:, x_index] = xx.ravel()
        eval_points[:, y_index] = yy.ravel()

        eval_tensor = torch.tensor(eval_points, dtype=torch.double)
        mean_t, _ = surrogate.predict(eval_tensor)
        contour_values = mean_t[:, objective_index].detach().numpy().reshape(grid_points, grid_points)

        x_phys = x_param.bounds[0] + xx * (x_param.bounds[1] - x_param.bounds[0])
        y_phys = y_param.bounds[0] + yy * (y_param.bounds[1] - y_param.bounds[0])
        obs_x = x_param.bounds[0] + X[:, x_index] * (x_param.bounds[1] - x_param.bounds[0])
        obs_y = y_param.bounds[0] + X[:, y_index] * (y_param.bounds[1] - y_param.bounds[0])
        best_x = x_param.bounds[0] + best_unit[x_index] * (x_param.bounds[1] - x_param.bounds[0])
        best_y = y_param.bounds[0] + best_unit[y_index] * (y_param.bounds[1] - y_param.bounds[0])

        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        contour = ax.contourf(x_phys, y_phys, contour_values, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax, label=obj_label)
        ax.scatter(
            obs_x, obs_y, c=Y, cmap="viridis", norm=contour.norm,
            s=25, edgecolors="white", linewidths=0.4, alpha=0.85,
            label="Evaluations",
        )
        ax.scatter(
            best_x, best_y, marker="*", s=150, color="red",
            edgecolors="black", linewidths=0.8, label="Best point", zorder=3,
        )
        y_label = _format_label(y_param.name, y_param.unit)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Iso contours: {y_label} vs {x_label}")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        safe_x = x_param.name.replace(" ", "_")
        safe_y = y_param.name.replace(" ", "_")
        path = output_dir / f"iso_contour_{safe_y}_vs_{safe_x}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


# ------------------------------------------------------------------
# Parallel coordinates
# ------------------------------------------------------------------


def plot_parallel_coordinates(
    state: OptimizationState,
    output_dir: Path,
    *,
    objective_index: int = 0,
) -> Path:
    """Plot a parallel coordinates chart of all parameters and the objective."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    if not mask.any():
        raise ValueError("No successful evaluations to plot.")

    active_params = [p for p in state.parameters if not p.is_constant]
    X = state.X[mask].numpy()
    Y = state.Y[mask, objective_index].numpy()
    obj_name = state.objective_names[objective_index]
    obj_label = _format_label(obj_name, None)
    maximize = state.maximize[objective_index] if objective_index < len(state.maximize) else True

    obj_min = float(np.min(Y))
    obj_max = float(np.max(Y))
    if math.isclose(obj_min, obj_max):
        norm_Y = np.full_like(Y, 0.5, dtype=float)
        vmax = obj_min + 1.0
    else:
        norm_Y = (Y - obj_min) / (obj_max - obj_min)
        vmax = obj_max
    norm = mcolors.Normalize(vmin=obj_min, vmax=vmax)

    data = np.column_stack((X, norm_Y))
    num_axes = data.shape[1]
    axis_positions = np.arange(num_axes)
    axis_labels = [
        _format_label(p.name, p.unit) for p in active_params
    ] + [f"{obj_label} (normalized)"]

    fig, ax = plt.subplots(figsize=(max(8.0, num_axes * 1.3), 5.0))
    colors = plt.cm.viridis(norm(Y))
    for row, color in zip(data, colors, strict=False):
        ax.plot(axis_positions, row, color=color, linewidth=0.8, alpha=0.7)

    best_idx = int(np.argmax(Y)) if maximize else int(np.argmin(Y))
    ax.plot(
        axis_positions, data[best_idx],
        color="red", linewidth=2.0, alpha=0.9, label="Best objective",
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array(Y)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(obj_label)

    ax.set_xticks(axis_positions)
    ax.set_xticklabels(axis_labels, rotation=20, ha="right")
    ax.set_xlim(axis_positions[0], axis_positions[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates (parameters normalized to [0, 1])")
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = output_dir / "parallel_coordinates.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Convergence plot
# ------------------------------------------------------------------


def plot_convergence(
    state: OptimizationState,
    output_dir: Path,
    *,
    objective_index: int = 0,
) -> Path:
    """Plot objective value vs iteration and cumulative best."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Y_all = state.Y[:, objective_index].numpy()
    success = np.array(state.success_mask, dtype=bool)
    maximize = state.maximize[objective_index] if objective_index < len(state.maximize) else True
    obj_name = state.objective_names[objective_index]
    obj_label = _format_label(obj_name, None)

    iterations = np.arange(1, len(Y_all) + 1)
    Y_success = Y_all.copy()
    Y_success[~success] = np.nan

    # Cumulative best
    cum_best = np.full_like(Y_success, np.nan)
    current_best = float("-inf") if maximize else float("inf")
    for i, (val, ok) in enumerate(zip(Y_success, success)):
        if ok and np.isfinite(val):
            if maximize and val > current_best:
                current_best = val
            elif not maximize and val < current_best:
                current_best = val
        if np.isfinite(current_best) and current_best != float("-inf") and current_best != float("inf"):
            cum_best[i] = current_best

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    valid = np.isfinite(Y_success)
    ax.scatter(iterations[valid], Y_success[valid], s=20, alpha=0.6, label="Evaluations")
    valid_best = np.isfinite(cum_best)
    ax.plot(iterations[valid_best], cum_best[valid_best], color="red", linewidth=2.0, label="Best so far")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(obj_label)
    ax.set_title("Convergence")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = output_dir / "convergence.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Expression plot
# ------------------------------------------------------------------


def plot_expression(
    state: OptimizationState,
    expression_values: np.ndarray,
    output_dir: Path,
    *,
    expression_label: str = "Parameter expression",
    objective_index: int = 0,
) -> Path:
    """Plot objective vs an evaluated parameter expression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    Y = state.Y[mask, objective_index].numpy()
    x_values = expression_values

    if x_values.shape[0] != Y.shape[0]:
        raise ValueError("Expression values must match the number of successful evaluations.")

    finite = np.isfinite(x_values) & np.isfinite(Y)
    if not np.any(finite):
        raise ValueError("No finite values to plot.")

    ordering = np.argsort(x_values[finite])
    x_sorted = x_values[finite][ordering]
    y_sorted = Y[finite][ordering]

    obj_name = state.objective_names[objective_index]
    obj_label = _format_label(obj_name, None)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x_sorted, y_sorted, marker="o", linewidth=1.5)
    ax.set_xlabel(expression_label)
    ax.set_ylabel(obj_label)
    ax.set_title(f"{obj_label} vs {expression_label}")
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout()

    path = output_dir / "outputs_vs_expression.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Pareto front (multi-objective)
# ------------------------------------------------------------------


def plot_pareto_front(
    state: OptimizationState,
    output_dir: Path,
) -> Path | None:
    """Plot the Pareto front for multi-objective optimization.

    Only supports 2- or 3-objective problems.  Returns ``None`` if
    the problem is single-objective.
    """
    if not state.is_multi_objective:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = torch.tensor(state.success_mask, dtype=torch.bool)
    if not mask.any():
        return None

    Y = state.Y[mask].numpy()
    n_obj = Y.shape[1]

    if n_obj > 3:
        return None  # Can't visualize >3 objectives

    pareto_idxs = state.pareto_indices.numpy()
    # Map pareto indices (in full array) to indices in successful-only array
    success_indices = np.where(np.array(state.success_mask))[0]
    success_idx_set = {int(v): i for i, v in enumerate(success_indices)}
    pareto_in_success = [success_idx_set[int(idx)] for idx in pareto_idxs if int(idx) in success_idx_set]

    labels = [_format_label(name, None) for name in state.objective_names]

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.scatter(Y[:, 0], Y[:, 1], s=20, alpha=0.5, label="All evaluations")
        if pareto_in_success:
            pareto_Y = Y[pareto_in_success]
            order = np.argsort(pareto_Y[:, 0])
            ax.plot(
                pareto_Y[order, 0], pareto_Y[order, 1],
                "r-o", markersize=6, linewidth=1.5, label="Pareto front",
            )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title("Pareto Front")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.2)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, alpha=0.5, label="All evaluations")
        if pareto_in_success:
            pareto_Y = Y[pareto_in_success]
            ax.scatter(
                pareto_Y[:, 0], pareto_Y[:, 1], pareto_Y[:, 2],
                c="red", s=60, marker="*", label="Pareto front",
            )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title("Pareto Front")
        ax.legend(fontsize=8)
        fig.tight_layout()

    path = output_dir / "pareto_front.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
