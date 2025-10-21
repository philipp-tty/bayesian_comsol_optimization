"""High-level optimization routine for the thermoelectric generator."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch

from bo import BayesianOptimization, DEVICE, DTYPE

from .comsol_cli import COMSOLCLIOptimizer
from .visualization import GPVisualizer

logger = logging.getLogger(__name__)


def optimize_thermoelectric_generator(
    model_path: str,
    n_legs: int = 127,
    n_initial: int = 10,
    n_iterations: int = 30,
    fill_factor_bounds: Tuple[float, float] = (0.05, 0.40),
    random_seed: int = 42,
    comsol_exe_path: str | None = None,
    methodcall: str = "methodcall2",
    target_footprint_mm2: float | None = None,
) -> Dict[str, object]:
    """
    Optimize thermoelectric generator using Bayesian Optimization with GP visualization.

    Geometry (leg_width, leg_spacing) is derived solely from the area fill factor and the
    fixed target footprint (no casing in the footprint).
    """
    if target_footprint_mm2 is None or target_footprint_mm2 <= 0:
        raise ValueError("target_footprint_mm2 must be a positive number.")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize COMSOL interface
    comsol = COMSOLCLIOptimizer(
        model_path=model_path,
        n_legs=n_legs,
        comsol_exe_path=comsol_exe_path,
        methodcall=methodcall,
        fill_factor_bounds=fill_factor_bounds,
        target_footprint_mm2=target_footprint_mm2,
    )
    fill_transform = comsol.fill_transform

    bounds = torch.tensor(
        [[0.0], [1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )

    logger.info("\n%s", "=" * 60)
    logger.info("Phase 1: Initial random sampling (%s points)", n_initial)
    logger.info("%s\n", "=" * 60)

    x_init = torch.empty((n_initial, 1), device=DEVICE, dtype=DTYPE)
    y_init = torch.empty((n_initial, 1), device=DEVICE, dtype=DTYPE)

    for i in range(n_initial):
        scaled_fill = np.random.rand()
        fill_factor = float(fill_transform.to_physical(scaled_fill))

        result = comsol.evaluate(fill_factor)
        x_init[i] = torch.tensor([scaled_fill], device=DEVICE, dtype=DTYPE)
        y_init[i] = torch.tensor([result["power"]], device=DEVICE, dtype=DTYPE)

        logger.info(
            "Initial sample %s/%s: fill_factor(area)=%.6f, leg_spacing=%.6f mm, Power = %.6f mW\n",
            i + 1,
            n_initial,
            fill_factor,
            result["leg_spacing"],
            result["power"],
        )

    logger.info("\n%s", "=" * 60)
    logger.info("Phase 2: Bayesian Optimization (%s iterations)", n_iterations)
    logger.info("%s\n", "=" * 60)

    bo = BayesianOptimization(
        x_train=x_init,
        y_train=y_init,
        bounds=bounds,
        maximize=True,
        use_outcome_transform=True,
    )

    # Initialize GP visualizer
    visualizer = GPVisualizer(comsol, fill_transform)

    # Show initial GP state
    visualizer.update_plots(bo, iteration=0)

    for iteration in range(n_iterations):
        logger.info("\n--- BO Iteration %s/%s ---", iteration + 1, n_iterations)
        x_next = bo.get_next_data_points(q=1)
        scaled_fill_next = float(x_next[0, 0].item())
        fill_factor_next = float(fill_transform.to_physical(scaled_fill_next))

        result = comsol.evaluate(fill_factor_next)
        y_next = torch.tensor([[result["power"]]], device=DEVICE, dtype=DTYPE)
        bo.update_model(x_next, y_next)

        # Update visualization
        visualizer.update_plots(bo, iteration=iteration + 1)

        best_idx = bo.y_train.argmax()
        best_scaled = float(bo.x_train[best_idx].item())
        best_fill_factor = float(fill_transform.to_physical(best_scaled))
        best_y = bo.y_train[best_idx]
        best_leg_width, best_leg_spacing = comsol.geometry_from_fill_factor(best_fill_factor)

        logger.info("Current best: Power = %.6f mW", best_y.item())
        logger.info("  fill_factor(area) = %.6f", best_fill_factor)
        logger.info("  leg_width         = %.4f mm", best_leg_width)
        logger.info("  leg_spacing       = %.6f mm\n", best_leg_spacing)

    logger.info("\n%s", "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("%s\n", "=" * 60)

    best_idx = bo.y_train.argmax()
    best_scaled = float(bo.x_train[best_idx].item())
    best_fill_factor = float(fill_transform.to_physical(best_scaled))
    best_y = bo.y_train[best_idx]
    best_leg_width, best_leg_spacing = comsol.geometry_from_fill_factor(best_fill_factor)

    logger.info("Optimal parameters:")
    logger.info("  fill_factor(area) = %.6f", best_fill_factor)
    logger.info("  leg_width         = %.6f mm", best_leg_width)
    logger.info("  leg_spacing       = %.6f mm", best_leg_spacing)
    logger.info("  Power output      = %.6f mW", best_y.item())

    input("\nPress Enter to close plots and exit...")
    visualizer.close()

    return {
        "fill_factor": best_fill_factor,
        "leg_width": best_leg_width,
        "leg_spacing": best_leg_spacing,
        "power": float(best_y.item()),
        "scaled_parameters": bo.x_train.cpu().numpy(),
        "all_fill_factors": fill_transform.to_physical(bo.x_train.cpu().numpy()),
        "all_y": bo.y_train.cpu().numpy(),
    }

