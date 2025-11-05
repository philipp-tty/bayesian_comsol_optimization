"""Visualization utilities for Bayesian optimization progress."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skopt import Optimizer

from .parameters import OptimizationParameter
from .transforms import FillFactorTransform, LinearParameterTransform
from .workflow import FAILED_EVALUATION_PENALTY

logger = logging.getLogger(__name__)

CI_Z_SCORE = 1.96  # Two-sided 95% confidence interval multiplier for Gaussian posterior
FAILED_VALUE_THRESHOLD = FAILED_EVALUATION_PENALTY * 0.5


class GPVisualizer:
    """Visualize Gaussian Process predictions during optimization (supports 1D and 2D)."""

    def __init__(
        self,
        parameters: Sequence[OptimizationParameter],
        transforms: Mapping[str, FillFactorTransform | LinearParameterTransform],
        maximize: bool = True,
    ) -> None:
        self.parameters = list(parameters)
        self.transforms = dict(transforms)
        self.dimension = len(self.parameters)
        self.maximize = bool(maximize)
        self._active = self.dimension in (1, 2)
        self.fig = None
        self.ax_mean = None
        self.ax_std = None
        self._mean_cbar = None
        self._std_cbar = None

        if self._active:
            plt.ion()
        else:
            logger.debug(
                "GPVisualizer instantiated with %s dimensions; visualization disabled.",
                self.dimension,
            )

    @property
    def active(self) -> bool:
        return self._active

    def update_plots(
        self,
        optimizer: Optimizer,
        iteration: int,
        objective_values: Sequence[float] | None = None,
    ) -> None:
        """Update visualization with the current GP model state."""
        if not self._active:
            return

        model = self._get_trained_model(optimizer)
        if model is None:
            return

        x_scaled, _, display_values = self._prepare_training_data(optimizer, objective_values)
        if x_scaled.size == 0:
            logger.debug("No evaluations recorded yet; skipping visualization.")
            return

        self._reset_figure()

        if self.dimension == 1:
            self._plot_1d(model, x_scaled, display_values, iteration)
        else:
            self._plot_2d(model, x_scaled, display_values, iteration)

        plt.draw()
        plt.pause(0.1)

    def _reset_figure(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.ax_mean = None
        self.ax_std = None
        self._mean_cbar = None
        self._std_cbar = None

    def _get_trained_model(self, optimizer: Optimizer):
        model = getattr(optimizer, "base_estimator_", None)
        if model is None:
            model = getattr(optimizer, "base_estimator", None)

        if model is None or not hasattr(model, "predict"):
            logger.warning("Optimizer base estimator lacks predict(); visualization skipped.")
            return None

        if not hasattr(model, "kernel_"):
            logger.debug("Gaussian Process surrogate not yet fitted; visualization skipped.")
            return None

        return model

    def _prepare_training_data(
        self,
        optimizer: Optimizer,
        objective_values: Sequence[float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_samples = np.asarray(optimizer.Xi, dtype=float)
        if x_samples.size == 0:
            return np.empty((0, self.dimension)), np.empty(0), np.empty(0)

        x_samples = x_samples.reshape(len(optimizer.Xi), -1)
        surrogate_values = np.asarray(optimizer.yi, dtype=float).reshape(-1)

        if objective_values is not None:
            obj_values = np.asarray(objective_values, dtype=float)
            if obj_values.shape[0] < surrogate_values.shape[0]:
                pad_len = surrogate_values.shape[0] - obj_values.shape[0]
                obj_values = np.concatenate([obj_values, np.full(pad_len, np.nan)])
            else:
                obj_values = obj_values[: surrogate_values.shape[0]]
            display_values = obj_values
        else:
            display_values = surrogate_values.copy()
            if self.maximize:
                display_values = -display_values

        penalty_mask = surrogate_values >= FAILED_VALUE_THRESHOLD
        display_values = display_values.astype(float)
        display_values[penalty_mask] = np.nan

        return x_samples, surrogate_values, display_values

    def _plot_1d(
        self,
        model,
        x_scaled: np.ndarray,
        display_values: np.ndarray,
        iteration: int,
    ) -> None:
        param = self.parameters[0]
        transform = self.transforms[param.name]

        valid_mask = np.isfinite(display_values)
        if not np.any(valid_mask):
            logger.debug("No finite observations available for 1D visualization.")
            return

        scaled_train = x_scaled[valid_mask, 0].reshape(-1, 1)
        train_values = display_values[valid_mask]
        physical_train = transform.to_physical(scaled_train.reshape(-1))

        n_grid = 200
        grid_scaled = np.linspace(0.0, 1.0, n_grid).reshape(-1, 1)
        mean, std = model.predict(grid_scaled, return_std=True)
        mean = mean.reshape(-1)
        std = std.reshape(-1)
        if self.maximize:
            mean = -mean

        physical_grid = transform.to_physical(grid_scaled.reshape(-1))
        ci_half_width = CI_Z_SCORE * std

        if self.maximize:
            best_idx = np.nanargmax(train_values)
        else:
            best_idx = np.nanargmin(train_values)

        self.fig, ax = plt.subplots(figsize=(10, 5))
        self.ax_mean = ax

        ax.plot(physical_grid, mean, color="navy", label="Posterior mean")
        ax.fill_between(
            physical_grid,
            mean - ci_half_width,
            mean + ci_half_width,
            color="skyblue",
            alpha=0.3,
            label="95% credible interval",
        )
        ax.scatter(
            physical_train,
            train_values,
            color="red",
            edgecolors="black",
            linewidth=0.6,
            s=40,
            label="Observations",
        )
        ax.scatter(
            physical_train[best_idx],
            train_values[best_idx],
            color="gold",
            edgecolors="black",
            linewidth=1.0,
            s=120,
            marker="*",
            label="Best",
        )

        x_label = param.name if not param.unit else f"{param.name} [{param.unit}]"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Objective Value")
        ax.set_title(f"GP Posterior (Iter {iteration})", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        self.fig.tight_layout()

    def _plot_2d(
        self,
        model,
        x_scaled: np.ndarray,
        display_values: np.ndarray,
        iteration: int,
    ) -> None:
        if x_scaled.shape[1] != 2:
            logger.debug(
                "GPVisualizer configured for 2D visualization but received %s-dimensional data.",
                x_scaled.shape[1],
            )
            return

        param_x = self.parameters[0]
        param_y = self.parameters[1]
        transform_x = self.transforms[param_x.name]
        transform_y = self.transforms[param_y.name]

        valid_mask = np.isfinite(display_values)
        if not np.any(valid_mask):
            logger.debug("No finite observations available for 2D visualization.")
            return

        scaled_valid = x_scaled[valid_mask]
        values_valid = display_values[valid_mask]
        x_train_phys = transform_x.to_physical(scaled_valid[:, 0])
        y_train_phys = transform_y.to_physical(scaled_valid[:, 1])

        n_grid = 40
        grid_x = np.linspace(0.0, 1.0, n_grid)
        grid_y = np.linspace(0.0, 1.0, n_grid)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
        grid_points = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])

        mean, std = model.predict(grid_points, return_std=True)
        mean = mean.reshape(n_grid, n_grid)
        std = std.reshape(n_grid, n_grid)
        if self.maximize:
            mean = -mean
        ci_half_width = CI_Z_SCORE * std

        if self.maximize:
            best_idx = np.nanargmax(values_valid)
        else:
            best_idx = np.nanargmin(values_valid)
        best_x = x_train_phys[best_idx]
        best_y = y_train_phys[best_idx]
        best_value = values_valid[best_idx]

        x_grid_phys = transform_x.to_physical(mesh_x)
        y_grid_phys = transform_y.to_physical(mesh_y)

        self.fig = plt.figure(figsize=(14, 5))
        grid_spec = GridSpec(1, 2, figure=self.fig, wspace=0.3)
        self.ax_mean = self.fig.add_subplot(grid_spec[0], projection="3d")
        self.ax_std = self.fig.add_subplot(grid_spec[1], projection="3d")

        mean_surface = self.ax_mean.plot_surface(
            x_grid_phys,
            y_grid_phys,
            mean,
            cmap="viridis",
            alpha=0.8,
            linewidth=0,
        )
        self.ax_mean.scatter(
            x_train_phys,
            y_train_phys,
            values_valid,
            c="red",
            s=40,
            edgecolors="black",
            linewidths=0.6,
            label="Observations",
        )
        self.ax_mean.scatter(
            best_x,
            best_y,
            best_value,
            c="gold",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label="Best",
        )
        x_label = param_x.name if not param_x.unit else f"{param_x.name} [{param_x.unit}]"
        y_label = param_y.name if not param_y.unit else f"{param_y.name} [{param_y.unit}]"
        self.ax_mean.set_xlabel(x_label)
        self.ax_mean.set_ylabel(y_label)
        self.ax_mean.set_zlabel("Objective Value")
        self.ax_mean.set_title(f"GP Mean Surface (Iter {iteration})", fontsize=12, fontweight="bold")
        self.ax_mean.legend(loc="upper left", fontsize=9)
        self.ax_mean.view_init(elev=35, azim=-135)
        self._mean_cbar = self.fig.colorbar(
            mean_surface, ax=self.ax_mean, shrink=0.6, pad=0.1, label="Mean (objective units)"
        )

        ci_surface = self.ax_std.plot_surface(
            x_grid_phys,
            y_grid_phys,
            ci_half_width,
            cmap="plasma",
            alpha=0.85,
            linewidth=0,
        )

        train_std = model.predict(x_scaled, return_std=True)[1]
        train_ci_half = CI_Z_SCORE * train_std
        train_x_phys = transform_x.to_physical(x_scaled[:, 0])
        train_y_phys = transform_y.to_physical(x_scaled[:, 1])

        self.ax_std.scatter(
            train_x_phys,
            train_y_phys,
            train_ci_half,
            c="black",
            s=20,
            alpha=0.8,
            label="Observations",
        )
        self.ax_std.set_xlabel(x_label)
        self.ax_std.set_ylabel(y_label)
        self.ax_std.set_zlabel("95% CI Half-Width")
        self.ax_std.set_title("GP 95% Confidence Interval Surface", fontsize=12, fontweight="bold")
        self.ax_std.view_init(elev=45, azim=-125)
        self.ax_std.legend(loc="upper left", fontsize=9)
        self._std_cbar = self.fig.colorbar(
            ci_surface, ax=self.ax_std, shrink=0.6, pad=0.1, label="95% CI Half-Width"
        )

    def close(self) -> None:
        if not self._active:
            return
        plt.ioff()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def process_events(self, delay: float = 0.05) -> None:
        """Allow Matplotlib to handle GUI events while long tasks execute."""
        if not self._active:
            return
        if self.fig is not None:
            try:
                self.fig.canvas.flush_events()
            except Exception:
                logger.debug("Matplotlib canvas flush failed during event processing.", exc_info=True)
        plt.pause(max(0.001, float(delay)))
