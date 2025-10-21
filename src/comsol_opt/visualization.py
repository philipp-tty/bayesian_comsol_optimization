"""Visualization utilities for Bayesian optimization progress."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from bo import BayesianOptimization, DEVICE, DTYPE

from .parameters import OptimizationParameter
from .transforms import FillFactorTransform, LinearParameterTransform

logger = logging.getLogger(__name__)

CI_Z_SCORE = 1.96  # Two-sided 95% confidence interval multiplier for Gaussian posterior


class GPVisualizer:
    """Visualize Gaussian Process predictions during optimization (supports 1D and 2D)."""

    def __init__(
        self,
        parameters: Sequence[OptimizationParameter],
        transforms: Mapping[str, FillFactorTransform | LinearParameterTransform],
    ):
        self.parameters = list(parameters)
        self.transforms = dict(transforms)
        self.dimension = len(self.parameters)
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

    def update_plots(self, bo: BayesianOptimization, iteration: int) -> None:
        """Update visualization with the current GP model state."""
        if not self._active:
            return

        self._reset_figure()

        if self.dimension == 1:
            self._plot_1d(bo, iteration)
        else:
            self._plot_2d(bo, iteration)

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

    def _plot_1d(self, bo: BayesianOptimization, iteration: int) -> None:
        param = self.parameters[0]
        transform = self.transforms[param.name]

        x_train = bo.x_train.cpu().numpy().reshape(-1)
        y_train = bo.y_train.cpu().numpy().flatten()
        physical_train = transform.to_physical(x_train)

        n_grid = 200
        scaled_grid = torch.linspace(0.0, 1.0, n_grid, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
        physical_grid = transform.to_physical(scaled_grid.cpu().numpy().reshape(-1))

        bo.model.eval()
        with torch.no_grad():
            posterior = bo.model.posterior(scaled_grid, observation_noise=False)
            mean = posterior.mean.cpu().numpy().reshape(-1)
            variance = posterior.variance.cpu().numpy().reshape(-1)
            std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
            ci_half_width = CI_Z_SCORE * std

        best_idx = y_train.argmax()

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
            y_train,
            color="red",
            edgecolors="black",
            linewidth=0.6,
            s=40,
            label="Observations",
        )
        ax.scatter(
            physical_train[best_idx],
            y_train[best_idx],
            color="gold",
            edgecolors="black",
            linewidth=1.0,
            s=120,
            marker="*",
            label="Best",
        )

        x_label = param.name if not param.unit else f"{param.name} [{param.unit}]"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Power Output (mW)")
        ax.set_title(f"GP Posterior (Iter {iteration})", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        self.fig.tight_layout()

    def _plot_2d(self, bo: BayesianOptimization, iteration: int) -> None:
        param_x = self.parameters[0]
        param_y = self.parameters[1]
        transform_x = self.transforms[param_x.name]
        transform_y = self.transforms[param_y.name]

        x_train_scaled = bo.x_train.cpu().numpy()
        y_train = bo.y_train.cpu().numpy().flatten()
        x_train_phys = transform_x.to_physical(x_train_scaled[:, 0])
        y_train_phys = transform_y.to_physical(x_train_scaled[:, 1])

        n_grid = 40
        scaled_x = torch.linspace(0.0, 1.0, n_grid, device=DEVICE, dtype=DTYPE)
        scaled_y = torch.linspace(0.0, 1.0, n_grid, device=DEVICE, dtype=DTYPE)
        mesh_x, mesh_y = torch.meshgrid(scaled_x, scaled_y, indexing="ij")
        X_grid = torch.stack((mesh_x.reshape(-1), mesh_y.reshape(-1)), dim=-1).to(device=DEVICE, dtype=DTYPE)

        bo.model.eval()
        with torch.no_grad():
            posterior = bo.model.posterior(X_grid, observation_noise=False)
            mean = posterior.mean.cpu().numpy().reshape(n_grid, n_grid)
            variance = posterior.variance.cpu().numpy().reshape(n_grid, n_grid)
            std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
            ci_half_width = CI_Z_SCORE * std
            train_posterior = bo.model.posterior(bo.x_train, observation_noise=False)
            train_std = np.sqrt(
                np.clip(train_posterior.variance.cpu().numpy().flatten(), a_min=0.0, a_max=None)
            )
            train_ci_half_width = CI_Z_SCORE * train_std

        best_idx = y_train.argmax()
        best_x = x_train_phys[best_idx]
        best_y = y_train_phys[best_idx]

        x_grid_phys = transform_x.to_physical(mesh_x.cpu().numpy())
        y_grid_phys = transform_y.to_physical(mesh_y.cpu().numpy())

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
            y_train,
            c="red",
            s=40,
            edgecolors="black",
            linewidths=0.6,
            label="Observations",
        )
        self.ax_mean.scatter(
            best_x,
            best_y,
            y_train[best_idx],
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
        self.ax_mean.set_zlabel("Power Output (mW)")
        self.ax_mean.set_title(f"GP Mean Surface (Iter {iteration})", fontsize=12, fontweight="bold")
        self.ax_mean.legend(loc="upper left", fontsize=9)
        self.ax_mean.view_init(elev=35, azim=-135)
        self._mean_cbar = self.fig.colorbar(
            mean_surface, ax=self.ax_mean, shrink=0.6, pad=0.1, label="Mean (mW)"
        )

        ci_surface = self.ax_std.plot_surface(
            x_grid_phys,
            y_grid_phys,
            ci_half_width,
            cmap="plasma",
            alpha=0.85,
            linewidth=0,
        )
        self.ax_std.scatter(
            x_train_phys,
            y_train_phys,
            train_ci_half_width,
            c="black",
            s=20,
            alpha=0.8,
            label="Observations",
        )
        self.ax_std.set_xlabel(x_label)
        self.ax_std.set_ylabel(y_label)
        self.ax_std.set_zlabel("95% CI Half-Width (mW)")
        self.ax_std.set_title("GP 95% Confidence Interval Surface", fontsize=12, fontweight="bold")
        self.ax_std.view_init(elev=45, azim=-125)
        self.ax_std.legend(loc="upper left", fontsize=9)
        self._std_cbar = self.fig.colorbar(
            ci_surface, ax=self.ax_std, shrink=0.6, pad=0.1, label="95% CI Half-Width (mW)"
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
                pass
        plt.pause(max(0.001, float(delay)))
