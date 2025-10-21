"""Visualization utilities for Bayesian optimization progress."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from bo import BayesianOptimization, DEVICE, DTYPE

from .comsol_cli import COMSOLCLIOptimizer
from .transforms import FillFactorTransform

logger = logging.getLogger(__name__)


class GPVisualizer:
    """
    Visualize Gaussian Process predictions during optimization.
    """

    def __init__(self, comsol: COMSOLCLIOptimizer, fill_transform: FillFactorTransform):
        self.comsol = comsol
        self.fill_transform = fill_transform
        self.fill_min, self.fill_max = self.fill_transform.bounds
        self.fig = plt.figure(figsize=(12, 4.5))
        grid_spec = GridSpec(1, 2, figure=self.fig, wspace=0.3)
        self.ax_surface = self.fig.add_subplot(grid_spec[0])
        self.ax_leg_spacing = self.fig.add_subplot(grid_spec[1])
        plt.ion()
        plt.show()

    def update_plots(self, bo: BayesianOptimization, iteration: int):
        """Update all plots with current GP state."""
        self.ax_surface.clear()
        self.ax_leg_spacing.clear()

        # Get training data
        x_train = bo.x_train.cpu().numpy().reshape(-1, 1)
        y_train = bo.y_train.cpu().numpy().flatten()
        fill_factors_train = self.fill_transform.to_physical(x_train[:, 0])

        # Create grid for GP predictions
        n_grid = 200
        scaled_grid = np.linspace(0.0, 1.0, n_grid)
        fill_grid = self.fill_transform.to_physical(scaled_grid)
        X_grid_torch = torch.tensor(scaled_grid[:, None], device=DEVICE, dtype=DTYPE)

        # Get GP predictions without observation noise for sharper variance near observations
        bo.model.eval()
        with torch.no_grad():
            posterior = bo.model.posterior(X_grid_torch, observation_noise=False)
            mean = posterior.mean.cpu().numpy().flatten()
            variance = posterior.variance.cpu().numpy().flatten()
            std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))

        best_idx = y_train.argmax()
        best_fill = fill_factors_train[best_idx]

        # --- Plot 1: GP Mean vs Fill Factor with Std Dev ---
        self.ax_surface.plot(fill_grid, mean, "b-", linewidth=2, label="GP Mean")
        self.ax_surface.fill_between(
            fill_grid,
            mean - std,
            mean + std,
            color="blue",
            alpha=0.15,
            label="+/- 1 Std Dev",
        )
        self.ax_surface.scatter(
            fill_factors_train,
            y_train,
            c="red",
            s=100,
            edgecolors="black",
            linewidths=1.5,
            label="Observations",
            zorder=5,
        )
        self.ax_surface.scatter(
            best_fill,
            y_train[best_idx],
            c="gold",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Best",
            zorder=10,
        )
        self.ax_surface.set_xlabel("Fill Factor (area)")
        self.ax_surface.set_ylabel("Power Output (mW)")
        self.ax_surface.set_title(f"GP Mean Prediction (Iter {iteration})", fontsize=12, fontweight="bold")
        self.ax_surface.legend(loc="best", fontsize=9)
        self.ax_surface.grid(True, alpha=0.3)

        # --- Plot 2: Leg Spacing vs Fill Factor ---
        leg_spacing_grid = np.array(
            [self.comsol.geometry_from_fill_factor(float(fill))[1] for fill in fill_grid]
        )
        self.ax_leg_spacing.plot(fill_grid, leg_spacing_grid, color="green", linewidth=2)
        self.ax_leg_spacing.set_xlabel("Fill Factor (area)")
        self.ax_leg_spacing.set_ylabel("Leg Spacing (mm)")
        self.ax_leg_spacing.set_title("Leg Spacing vs Fill Factor", fontsize=12, fontweight="bold")
        self.ax_leg_spacing.grid(True, alpha=0.3)

        plt.draw()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

