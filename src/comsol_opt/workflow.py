"""High-level optimization workflows built on top of the COMSOL CLI wrapper."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from .comsol_cli import COMSOLCLIOptimizer
from .parameters import OptimizationParameter

logger = logging.getLogger(__name__)

FAILED_EVALUATION_PENALTY = 1e12


def optimize_model(
    *,
    model_path: str,
    parameters: Sequence[OptimizationParameter],
    comsol_exe_path: str,
    n_initial: int = 5,
    n_iterations: int = 20,
    random_seed: int | None = None,
    maximize: bool = True,
    methodcall: str = "methodcall2",
    event_pump: Callable[[], None] | None = None,
    event_poll_interval: float | None = None,
) -> Dict[str, object]:
    """
    Run a model-agnostic optimization loop backed by the COMSOL CLI.

    Parameters
    ----------
    model_path:
        Path to the COMSOL ``.mph`` model evaluated during optimization.
    parameters:
        Sequence of :class:`OptimizationParameter` definitions. Entries with a
        ``constant_value`` are forwarded to COMSOL but excluded from the search
        space, while the remaining parameters are actively optimized.
    comsol_exe_path:
        Absolute path to the COMSOL ``comsolbatch`` executable.
    n_initial:
        Number of purely exploratory evaluations drawn from a random design prior to
        fitting the Gaussian Process surrogate.
    n_iterations:
        Number of additional evaluations after the initial design guided by Expected
        Improvement.
    random_seed:
        Optional random seed forwarded to the optimizer backend for reproducibility.
    maximize:
        Whether the objective read from the COMSOL output should be maximized. Set to
        ``False`` to perform minimization.
    methodcall:
        COMSOL methodcall string passed through to :class:`COMSOLCLIOptimizer`.
    event_pump:
        Optional callback invoked periodically to keep GUIs responsive while COMSOL runs.
    event_poll_interval:
        Optional override for the optimizer's event polling interval.

    Returns
    -------
    Dict[str, object]
        A dictionary summarizing the optimization run. Keys include:

        * ``objective`` – Best objective value found (``nan`` if none succeeded).
        * ``parameters`` / ``best_parameters`` – Mapping of best-known parameter values.
        * ``objective_history`` – NumPy array of objective values per evaluation.
        * ``scaled_samples`` – NumPy array of evaluated points in the unit space (all parameters).
        * ``parameter_history`` – Mapping of parameter names to NumPy arrays of values.
        * ``comsol_parameter_history`` – List of COMSOL parameter payloads per evaluation.
        * ``success_history`` – List of booleans indicating CLI success per evaluation.
        * ``metadata`` – Auxiliary information about the run (seed, iterations, etc.).

        For backwards compatibility with earlier thermoelectric-focused utilities, the
        dictionary also includes the aliases ``power`` and ``power_history``.
    """
    if n_initial < 0:
        raise ValueError("n_initial must be non-negative.")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative.")

    total_evaluations = n_initial + n_iterations
    if total_evaluations <= 0:
        raise ValueError("The optimization loop requires at least one evaluation.")

    optimizer = COMSOLCLIOptimizer(
        model_path=model_path,
        parameters=parameters,
        comsol_exe_path=comsol_exe_path,
        methodcall=methodcall,
    )

    if event_pump is not None or event_poll_interval is not None:
        optimizer.set_event_pump(event_pump, event_poll_interval)

    parameters = list(parameters)
    active_parameters = [param for param in parameters if not param.is_constant]
    active_indices = [index for index, param in enumerate(parameters) if not param.is_constant]
    constant_defaults: Dict[str, float] = {
        param.name: float(param.constant_value) for param in parameters if param.is_constant
    }

    dimension = len(active_parameters)

    scaled_history = np.zeros((total_evaluations, len(parameters)), dtype=float)
    objective_history = np.full(total_evaluations, np.nan, dtype=float)
    parameter_history: Dict[str, np.ndarray] = {
        param.name: np.zeros(total_evaluations, dtype=float) for param in parameters
    }

    comsol_history: list[Mapping[str, Mapping[str, float | str | None]]] = []
    success_history: list[bool] = []
    evaluation_records: list[Mapping[str, object]] = []

    best_index = -1
    best_value = -np.inf if maximize else np.inf
    best_parameters: Dict[str, float] | None = None
    best_comsol_parameters: Mapping[str, Mapping[str, float | str | None]] | None = None

    skopt_optimizer: Optimizer | None = None
    if dimension > 0:
        search_space = [Real(0.0, 1.0, name=param.name) for param in active_parameters]
        initial_points = max(1, n_initial)
        skopt_optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="GP",
            acq_func="EI",
            initial_point_generator="random",
            n_initial_points=initial_points,
            random_state=random_seed,
        )

    for iteration in range(total_evaluations):
        if dimension:
            if skopt_optimizer is not None:
                scaled_candidate_active = np.asarray(
                    skopt_optimizer.ask(), dtype=float
                ).reshape(dimension)
            else:
                scaled_candidate_active = np.empty(0, dtype=float)
            scaled_candidate_active = np.clip(scaled_candidate_active, 0.0, 1.0)
        else:
            scaled_candidate_active = np.empty(0, dtype=float)

        physical_guess: Dict[str, float] = {name: float(value) for name, value in constant_defaults.items()}
        for axis, param in enumerate(active_parameters):
            transform = optimizer.parameter_transforms[param.name]
            physical_value = transform.to_physical(scaled_candidate_active[axis])
            physical_guess[param.name] = float(physical_value)

        evaluation = optimizer.evaluate(physical_guess)
        evaluation_records.append(evaluation)

        objective_value = evaluation.get("objective")
        if objective_value is None:
            objective_value = evaluation.get("power")
        objective_value = float(objective_value) if objective_value is not None else float("nan")
        objective_history[iteration] = objective_value

        actual_params = {
            name: float(value) for name, value in evaluation.get("parameters", physical_guess).items()
        }

        current_scaled_full = np.zeros(len(parameters), dtype=float)
        for axis, param in enumerate(parameters):
            actual_value = actual_params.get(param.name, physical_guess[param.name])
            transform = optimizer.parameter_transforms[param.name]
            scaled_value = transform.to_unit(actual_value)
            scaled_value = transform.ensure_unit(scaled_value)
            current_scaled_full[axis] = float(np.clip(scaled_value, 0.0, 1.0))
            parameter_history[param.name][iteration] = float(actual_value)
        scaled_history[iteration, :] = current_scaled_full

        current_scaled_active = (
            current_scaled_full[active_indices] if active_indices else np.empty(0, dtype=float)
        )

        comsol_payload = evaluation.get("comsol_parameters", {})
        comsol_history.append(comsol_payload)
        success = bool(evaluation.get("success", True))
        success_history.append(success)

        if np.isnan(objective_value):
            continue

        if best_parameters is None:
            best_index = iteration
            best_value = objective_value
            best_parameters = actual_params
            best_comsol_parameters = comsol_payload
            continue

        is_better = (objective_value > best_value) if maximize else (objective_value < best_value)
        if success and is_better:
            best_index = iteration
            best_value = objective_value
            best_parameters = actual_params
            best_comsol_parameters = comsol_payload

        if dimension and skopt_optimizer is not None:
            if not success or not np.isfinite(objective_value):
                surrogate_value = FAILED_EVALUATION_PENALTY
            else:
                surrogate_value = -objective_value if maximize else objective_value
            skopt_optimizer.tell(current_scaled_active.tolist(), surrogate_value)

    if best_parameters is None:
        # No successful evaluations recorded; provide fallbacks.
        best_parameters = {param.name: float("nan") for param in parameters}
        best_value = float("nan")
        best_comsol_parameters = {}

    metadata = {
        "random_seed": random_seed,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "maximize": maximize,
        "total_evaluations": total_evaluations,
    }

    results: Dict[str, object] = {
        "objective": best_value,
        "best_objective": best_value,
        "parameters": best_parameters,
        "best_parameters": best_parameters,
        "best_index": best_index,
        "objective_history": objective_history,
        "power": best_value,
        "power_history": objective_history,
        "parameter_history": parameter_history,
        "scaled_samples": scaled_history,
        "scaled_parameters": scaled_history,
        "scaled_bounds": np.array([[0.0] * len(parameters), [1.0] * len(parameters)], dtype=float),
        "comsol_parameter_history": comsol_history,
        "best_comsol_parameters": best_comsol_parameters or {},
        "success_history": success_history,
        "success": any(success_history),
        "evaluations": evaluation_records,
        "metadata": metadata,
        "derived_history": [],
        "derived_parameters": {},
    }

    return results
