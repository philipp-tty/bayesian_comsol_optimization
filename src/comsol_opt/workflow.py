"""High-level optimization workflows built on top of the COMSOL CLI wrapper."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Mapping, Sequence

import numpy as np

from .comsol_cli import COMSOLCLIOptimizer
from .parameters import OptimizationParameter

logger = logging.getLogger(__name__)

CandidateSampler = Callable[[np.random.Generator, int, np.ndarray | None, int], np.ndarray]


def _default_candidate_sampler(
    rng: np.random.Generator,
    dimension: int,
    best_scaled: np.ndarray | None,
    iteration: int,
) -> np.ndarray:
    """
    Default candidate sampler used by :func:`optimize_model`.

    The sampler alternates between pure exploration (uniform samples in the unit
    hypercube) and mild exploitation by perturbing the best known point.
    """
    if best_scaled is None or iteration < 1 or rng.random() < 0.5:
        return rng.random(dimension)

    noise = rng.normal(scale=0.1, size=dimension)
    candidate = best_scaled + noise
    return np.clip(candidate, 0.0, 1.0)


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
    target_footprint_mm2: float | None = None,
    candidate_sampler: CandidateSampler | None = None,
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
        Number of purely exploratory evaluations sampled uniformly in the unit cube.
    n_iterations:
        Number of additional evaluations after the initial design. The default
        candidate sampler perturbs the incumbent solution with Gaussian noise while
        continuing to explore uniformly.
    random_seed:
        Optional random seed forwarded to the NumPy random generator used for sampling.
    maximize:
        Whether the objective read from the COMSOL output should be maximized. Set to
        ``False`` to perform minimization.
    methodcall:
        COMSOL methodcall string passed through to :class:`COMSOLCLIOptimizer`.
    target_footprint_mm2:
        Optional footprint value required when using a fill-factor parameter transform.
    candidate_sampler:
        Optional sampler that proposes the next point in the unit hypercube. The callable
        receives the NumPy ``Generator`` instance, the problem dimension, the best point in
        the unit space (if one is known), and the zero-based iteration index.
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
        target_footprint_mm2=target_footprint_mm2,
    )

    if event_pump is not None or event_poll_interval is not None:
        optimizer.set_event_pump(event_pump, event_poll_interval)

    rng = np.random.default_rng(random_seed)
    parameters = list(parameters)
    active_parameters = [param for param in parameters if not param.is_constant]
    active_indices = [index for index, param in enumerate(parameters) if not param.is_constant]
    constant_defaults: Dict[str, float] = {
        param.name: float(param.constant_value) for param in parameters if param.is_constant
    }

    dimension = len(active_parameters)
    sampler = candidate_sampler or _default_candidate_sampler

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
    best_scaled_active: np.ndarray | None = None
    best_comsol_parameters: Mapping[str, Mapping[str, float | str | None]] | None = None

    for iteration in range(total_evaluations):
        if iteration < n_initial:
            scaled_candidate_active = rng.random(dimension) if dimension else np.empty(0)
        else:
            scaled_candidate_active = sampler(rng, dimension, best_scaled_active, iteration)

        scaled_candidate_active = np.asarray(scaled_candidate_active, dtype=float).reshape(dimension)
        scaled_candidate_active = np.clip(scaled_candidate_active, 0.0, 1.0)

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
            best_scaled_active = current_scaled_active.copy()
            best_comsol_parameters = comsol_payload
            continue

        is_better = (objective_value > best_value) if maximize else (objective_value < best_value)
        if success and is_better:
            best_index = iteration
            best_value = objective_value
            best_parameters = actual_params
            best_scaled_active = current_scaled_active.copy()
            best_comsol_parameters = comsol_payload

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
