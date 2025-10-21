"""Visualization utilities for Bayesian optimization progress."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from bo import BayesianOptimization, DEVICE, DTYPE

from .comsol_cli import COMSOLCLIOptimizer
from .transforms import FillFactorTransform, LinearParameterTransform

logger = logging.getLogger(__name__)


class GPVisualizer:
    """
    Visualize Gaussian Process predictions during optimization.
    """

    def __init__(
        self,
        comsol: COMSOLCLIOptimizer,
        fill_transform: FillFactorTransform,
        r_load_transform: LinearParameterTransform,
    ):
        self.comsol = comsol
        self.fill_transform = fill_transform
        self.r_load_transform = r_load_transform
        self.fill_min, self.fill_max = self.fill_transform.bounds
        self.r_min, self.r_max = self.r_load_transform.bounds
        self.fig = plt.figure(figsize=(14, 5))
        grid_spec = GridSpec(1, 2, figure=self.fig, wspace=0.3)
        self.ax_mean = self.fig.add_subplot(grid_spec[0], projection="3d")
        self.ax_std = self.fig.add_subplot(grid_spec[1], projection="3d")
        self._mean_cbar = None
        self._std_cbar = None
        plt.ion()
        plt.show()

    def update_plots(self, bo: BayesianOptimization, iteration: int):
        """Update all plots with current GP state."""
        self.ax_mean.clear()
        self.ax_std.clear()
        if self._mean_cbar is not None:
            self._mean_cbar.remove()
            self._mean_cbar = None
        if self._std_cbar is not None:
            self._std_cbar.remove()
            self._std_cbar = None

        # Get training data
        x_train = bo.x_train.cpu().numpy()
        y_train = bo.y_train.cpu().numpy().flatten()
        fill_factors_train = self.fill_transform.to_physical(x_train[:, 0])
        r_loads_train = self.r_load_transform.to_physical(x_train[:, 1])

        # Create grid for GP predictions
        n_grid = 40
        scaled_fill = torch.linspace(0.0, 1.0, n_grid, device=DEVICE, dtype=DTYPE)
        scaled_r = torch.linspace(0.0, 1.0, n_grid, device=DEVICE, dtype=DTYPE)
        mesh_fill, mesh_r = torch.meshgrid(scaled_fill, scaled_r, indexing="ij")
        X_grid_torch = torch.stack(
            (mesh_fill.reshape(-1), mesh_r.reshape(-1)), dim=-1
        ).to(device=DEVICE, dtype=DTYPE)

        # Get GP predictions without observation noise for sharper variance near observations
        bo.model.eval()
        with torch.no_grad():
            posterior = bo.model.posterior(X_grid_torch, observation_noise=False)
            mean = posterior.mean.cpu().numpy().reshape(n_grid, n_grid)
            variance = posterior.variance.cpu().numpy().reshape(n_grid, n_grid)
            std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
            train_posterior = bo.model.posterior(bo.x_train, observation_noise=False)
            train_std = np.sqrt(
                np.clip(train_posterior.variance.cpu().numpy().flatten(), a_min=0.0, a_max=None)
            )

        best_idx = y_train.argmax()
        best_fill = fill_factors_train[best_idx]
        best_r = r_loads_train[best_idx]

        # Convert meshgrid to physical domain for plotting
        fill_grid = self.fill_transform.to_physical(mesh_fill.cpu().numpy())
        r_grid = self.r_load_transform.to_physical(mesh_r.cpu().numpy())

        # --- Plot 1: GP Mean surface ---
        mean_surface = self.ax_mean.plot_surface(
            fill_grid,
            r_grid,
            mean,
            cmap="viridis",
            alpha=0.8,
            linewidth=0,
        )
        self.ax_mean.scatter(
            fill_factors_train,
            r_loads_train,
            y_train,
            c="red",
            s=40,
            edgecolors="black",
            linewidths=0.6,
            label="Observations",
        )
        self.ax_mean.scatter(
            best_fill,
            best_r,
            y_train[best_idx],
            c="gold",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label="Best",
        )
        self.ax_mean.set_xlabel("Fill Factor (area)")
        self.ax_mean.set_ylabel("R_l")
        self.ax_mean.set_zlabel("Power Output (mW)")
        self.ax_mean.set_title(f"GP Mean Surface (Iter {iteration})", fontsize=12, fontweight="bold")
        self.ax_mean.legend(loc="upper left", fontsize=9)
        self.ax_mean.view_init(elev=35, azim=-135)
        self._mean_cbar = self.fig.colorbar(
            mean_surface, ax=self.ax_mean, shrink=0.6, pad=0.1, label="Mean (mW)"
        )

        # --- Plot 2: GP Standard Deviation surface ---
        std_surface = self.ax_std.plot_surface(
            fill_grid,
            r_grid,
            std,
            cmap="plasma",
            alpha=0.85,
            linewidth=0,
        )
        self.ax_std.scatter(
            fill_factors_train,
            r_loads_train,
            train_std,
            c="black",
            s=20,
            alpha=0.8,
            label="Observations",
        )
        self.ax_std.set_xlabel("Fill Factor (area)")
        self.ax_std.set_ylabel("R_l")
        self.ax_std.set_zlabel("Std Dev (mW)")
        self.ax_std.set_title("GP Standard Deviation Surface", fontsize=12, fontweight="bold")
        self.ax_std.view_init(elev=45, azim=-125)
        self.ax_std.legend(loc="upper left", fontsize=9)
        self._std_cbar = self.fig.colorbar(
            std_surface, ax=self.ax_std, shrink=0.6, pad=0.1, label="Std Dev (mW)"
        )

        plt.draw()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
