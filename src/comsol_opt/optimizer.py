"""High-level optimization routine for the thermoelectric generator."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from bo import DEVICE, DTYPE

from .comsol_cli import COMSOLCLIOptimizer
from .deterministic_bo import DeterministicBayesianOptimization
from .parameters import OptimizationParameter
from .transforms import FillFactorTransform, LinearParameterTransform
from .visualization import GPVisualizer

logger = logging.getLogger(__name__)


def optimize_thermoelectric_generator(
    model_path: str,
    n_legs: int = 127,
    n_initial: int = 10,
    n_iterations: int = 30,
    fill_factor_bounds: tuple[float, float] = (0.05, 0.40),
    r_load_bounds: tuple[float, float] = (0.0, 5.0),
    random_seed: int = 42,
    comsol_exe_path: str | None = None,
    methodcall: str = "methodcall2",
    target_footprint_mm2: float | None = None,
    parameters: Sequence[OptimizationParameter] | None = None,
) -> Dict[str, object]:
    """
    Optimize the thermoelectric generator using Bayesian Optimization with optional GP visualization.

    Parameters can be configured dynamically via `parameters`. When omitted, the function defaults
    to optimizing the area fill factor (mapped to geometry) and the electrical load resistance.
    """
    if parameters is None:
        parameters = [
            OptimizationParameter(
                name="fill_factor",
                bounds=fill_factor_bounds,
                comsol_name="fill_factor",
                transform="fill_factor",
            ),
            OptimizationParameter(
                name="r_load",
                bounds=r_load_bounds,
                comsol_name="R_l",
                unit="ohm",
                transform="linear",
            ),
        ]
    else:
        parameters = list(parameters)

    if not parameters:
        raise ValueError("At least one optimization parameter must be specified.")

    requires_geometry = any(param.transform == "fill_factor" for param in parameters)
    if requires_geometry and (target_footprint_mm2 is None or target_footprint_mm2 <= 0):
        raise ValueError("target_footprint_mm2 must be a positive number when optimizing fill factor.")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    transforms: Dict[str, FillFactorTransform | LinearParameterTransform] = {}
    for param in parameters:
        if param.transform == "fill_factor":
            transforms[param.name] = FillFactorTransform(param.bounds)
        else:
            transforms[param.name] = LinearParameterTransform(param.bounds)

    dimension = len(parameters)
    bounds = torch.tensor(
        [
            [0.0 for _ in range(dimension)],
            [1.0 for _ in range(dimension)],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    comsol = COMSOLCLIOptimizer(
        model_path=model_path,
        parameters=parameters,
        n_legs=n_legs,
        comsol_exe_path=comsol_exe_path,
        methodcall=methodcall,
        target_footprint_mm2=target_footprint_mm2,
    )

    def project_scaled_values(scaled_values: Iterable[float]) -> tuple[Dict[str, float], List[float]]:
        physical: Dict[str, float] = {}
        adjusted_scaled: List[float] = []
        for value, param in zip(scaled_values, parameters):
            transform = transforms[param.name]
            scaled_float = float(value)
            physical_value = float(transform.to_physical(scaled_float))
            coerced_physical = param.coerce_physical_value(physical_value)
            adjusted_scaled_value = float(transform.to_unit(coerced_physical))
            adjusted_scaled_value = float(transform.ensure_unit(adjusted_scaled_value))
            physical[param.name] = coerced_physical
            adjusted_scaled.append(adjusted_scaled_value)
        return physical, adjusted_scaled

    def scaled_to_physical(scaled_values: Iterable[float]) -> Dict[str, float]:
        physical, _ = project_scaled_values(scaled_values)
        return physical

    def tensor_row_to_list(row: torch.Tensor) -> List[float]:
        return [float(row[idx].item()) for idx in range(row.shape[-1])]

    def format_parameter_value(param: OptimizationParameter, value: float) -> str:
        if param.is_integer:
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.6f}"
        if param.unit:
            return f"{param.name}={formatted_value} [{param.unit}]"
        return f"{param.name}={formatted_value}"

    logger.info("\n%s", "=" * 60)
    logger.info("Phase 1: Initial random sampling (%s points)", n_initial)
    logger.info("%s\n", "=" * 60)

    x_init = torch.empty((n_initial, dimension), device=DEVICE, dtype=DTYPE)
    y_init = torch.empty((n_initial, 1), device=DEVICE, dtype=DTYPE)

    parameter_history: Dict[str, List[float]] = {param.name: [] for param in parameters}
    derived_history: List[Dict[str, float]] = []
    comsol_history: List[Dict[str, Dict[str, float | str | None]]] = []
    power_history: List[float] = []

    for i in range(n_initial):
        scaled_vector = [float(np.random.rand()) for _ in range(dimension)]
        physical_params, adjusted_scaled = project_scaled_values(scaled_vector)
        result = comsol.evaluate(physical_params)

        x_init[i] = torch.tensor(adjusted_scaled, device=DEVICE, dtype=DTYPE)
        y_init[i] = torch.tensor([result["power"]], device=DEVICE, dtype=DTYPE)

        for param in parameters:
            parameter_history[param.name].append(physical_params[param.name])
        derived_history.append(result.get("derived_parameters", {}))
        comsol_history.append(result.get("comsol_parameters", {}))
        power_history.append(result["power"])

        logger.info(
            "Initial sample %s/%s: %s, Power = %.6f mW",
            i + 1,
            n_initial,
            ", ".join(
                format_parameter_value(param, physical_params[param.name]) for param in parameters
            ),
            result["power"],
        )
        if result.get("derived_parameters"):
            derived = result["derived_parameters"]
            logger.info(
                "  Derived geometry: %s",
                ", ".join(f"{key}={value:.6f}" for key, value in derived.items()),
            )

    logger.info("\n%s", "=" * 60)
    logger.info("Phase 2: Bayesian Optimization (%s iterations)", n_iterations)
    logger.info("%s\n", "=" * 60)

    bo = DeterministicBayesianOptimization(
        x_train=x_init,
        y_train=y_init,
        bounds=bounds,
        maximize=True,
        use_outcome_transform=True,
        measurement_noise=0.0,
    )

    visualizer: GPVisualizer | None = None
    if dimension <= 2:
        visualizer = GPVisualizer(parameters, transforms)
        comsol.set_event_pump(lambda: visualizer.process_events(0.02), poll_interval=0.02)
        visualizer.update_plots(bo, iteration=0)
    else:
        comsol.set_event_pump(None)
        logger.info("Skipping GP visualization (only enabled for 1 or 2 parameters).")

    for iteration in range(n_iterations):
        logger.info("\n--- BO Iteration %s/%s ---", iteration + 1, n_iterations)
        x_next = bo.get_next_data_points(q=1)
        scaled_values = tensor_row_to_list(x_next[0])
        physical_params, adjusted_scaled = project_scaled_values(scaled_values)

        result = comsol.evaluate(physical_params)
        y_next = torch.tensor([[result["power"]]], device=DEVICE, dtype=DTYPE)
        x_next_projected = torch.tensor([adjusted_scaled], device=DEVICE, dtype=DTYPE)
        bo.update_model(x_next_projected, y_next)

        for param in parameters:
            parameter_history[param.name].append(physical_params[param.name])
        derived_history.append(result.get("derived_parameters", {}))
        comsol_history.append(result.get("comsol_parameters", {}))
        power_history.append(result["power"])

        if visualizer is not None:
            visualizer.update_plots(bo, iteration=iteration + 1)

        best_idx = bo.y_train.argmax()
        best_scaled = tensor_row_to_list(bo.x_train[best_idx])
        best_parameters = scaled_to_physical(best_scaled)
        best_power = float(bo.y_train[best_idx].item())

        logger.info("Current best: Power = %.6f mW", best_power)
        for param in parameters:
            logger.info("  %s", format_parameter_value(param, best_parameters[param.name]))
        if comsol.fill_parameter is not None:
            fill_name = comsol.fill_parameter.name
            leg_width, leg_spacing = comsol.geometry_from_fill_factor(best_parameters[fill_name])
            logger.info("  leg_width  = %.6f mm", leg_width)
            logger.info("  leg_spacing= %.6f mm", leg_spacing)

    logger.info("\n%s", "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("%s\n", "=" * 60)

    best_idx = bo.y_train.argmax()
    best_scaled = tensor_row_to_list(bo.x_train[best_idx])
    best_parameters = scaled_to_physical(best_scaled)
    best_power = float(bo.y_train[best_idx].item())

    best_derived: Dict[str, float] = {}
    if comsol.fill_parameter is not None:
        fill_name = comsol.fill_parameter.name
        leg_width, leg_spacing = comsol.geometry_from_fill_factor(best_parameters[fill_name])
        best_derived = {"leg_width": leg_width, "leg_spacing": leg_spacing}

    logger.info("Optimal parameters:")
    for param in parameters:
        logger.info("  %s", format_parameter_value(param, best_parameters[param.name]))
    if best_derived:
        logger.info("  leg_width  = %.6f mm", best_derived["leg_width"])
        logger.info("  leg_spacing= %.6f mm", best_derived["leg_spacing"])
    logger.info("  Power output= %.6f mW", best_power)

    if visualizer is not None:
        input("\nPress Enter to close plots and exit...")
        visualizer.close()
    comsol.set_event_pump(None)

    parameter_history_arrays = {
        name: np.asarray(values, dtype=float) for name, values in parameter_history.items()
    }

    return {
        "power": best_power,
        "parameters": best_parameters,
        "derived_parameters": best_derived,
        "scaled_parameters": bo.x_train.cpu().numpy(),
        "parameter_history": parameter_history_arrays,
        "derived_history": derived_history,
        "comsol_parameter_history": comsol_history,
        "power_history": bo.y_train.cpu().numpy(),
    }
