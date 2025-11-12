"""High-level optimization workflows built on top of the COMSOL CLI wrapper."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from .comsol_cli import COMSOLCLIOptimizer
from .parameters import OptimizationParameter

logger = logging.getLogger(__name__)

FAILED_EVALUATION_PENALTY = 1e12


def _ensure_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _to_serializable(data: object) -> object:
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {key: _to_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_serializable(item) for item in data]
    return data


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_parameter_definitions(
    parameters: Sequence[OptimizationParameter],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for param in parameters:
        normalized.append(
            {
                "name": param.name,
                "bounds": [float(param.bounds[0]), float(param.bounds[1])],
                "comsol_name": param.comsol_name,
                "unit": param.unit,
                "transform": param.transform,
                "value_type": param.value_type,
                "is_constant": param.is_constant,
                "constant_value": (
                    float(param.constant_value) if param.constant_value is not None else None
                ),
            }
        )
    return normalized


def _load_resume_results(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Resume file {path} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Resume file {path} must contain a JSON object.")
    required_keys = {"objective_history", "scaled_samples", "parameter_history", "metadata"}
    if not required_keys.issubset(data):
        missing = ", ".join(sorted(required_keys - set(data)))
        raise ValueError(f"Resume file {path} is missing required keys: {missing}")
    return data


def _write_results(path: Path, results: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(results), handle, indent=2)
    tmp_path.replace(path)


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
    results_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    autosave_interval: int = 1,
    progress_callback: Callable[[int, int, Mapping[str, object]], None] | None = None,
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
        Improvement. When resuming from file this value, together with ``n_initial``,
        is treated as the desired total number of evaluations if it exceeds the count
        stored in the resume data.
    random_seed:
        Optional random seed forwarded to the optimizer backend for reproducibility.
    maximize:
        Whether the objective read from the COMSOL output should be maximized. Set to
        ``False`` to perform minimization. When maximizing, negative objective values are
        treated as invalid datapoints and ignored to keep the search moving toward
        better regions.
    methodcall:
        COMSOL methodcall string passed through to :class:`COMSOLCLIOptimizer`.
    event_pump:
        Optional callback invoked periodically to keep GUIs responsive while COMSOL runs.
    event_poll_interval:
        Optional override for the optimizer's event polling interval.
    results_path:
        Optional path to a JSON file that will be rewritten with the current optimization
        snapshot after each autosave interval.
    resume_path:
        Optional path to a JSON file previously produced by :func:`optimize_model`. When
        supplied, the optimizer resumes from that state and continues using the same
        parameterization.
    autosave_interval:
        Number of completed evaluations between consecutive writes to ``results_path``.
        Must be at least one.
    progress_callback:
        Optional callable invoked after each evaluation with arguments
        ``(completed, planned_total, snapshot)`` where ``snapshot`` matches the
        structure written to ``results_path``.

    Returns
    -------
    Dict[str, object]
        A dictionary summarizing the optimization run. Keys include:

        * ``objective`` — Best objective value found (``nan`` if none succeeded).
        * ``parameters`` / ``best_parameters`` — Mapping of best-known parameter values.
        * ``objective_history`` — NumPy array of objective values per evaluation.
        * ``scaled_samples`` — NumPy array of evaluated points in the unit space (all parameters).
        * ``parameter_history`` — Mapping of parameter names to NumPy arrays of values.
        * ``comsol_parameter_history`` — List of COMSOL parameter payloads per evaluation.
        * ``success_history`` — List of booleans indicating CLI success per evaluation.
        * ``gaussian_process`` — JSON-friendly payload with training data for surrogate analysis.
        * ``metadata`` — Auxiliary information about the run (seed, iterations, resume info).
    """
    if n_initial < 0:
        raise ValueError("n_initial must be non-negative.")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative.")
    if autosave_interval < 1:
        raise ValueError("autosave_interval must be at least 1.")

    results_path_obj = _ensure_path(results_path)
    resume_path_obj = _ensure_path(resume_path)

    parameter_definitions = _normalize_parameter_definitions(parameters)

    resume_results: dict[str, object] | None = None
    resume_metadata: dict[str, object] = {}
    if resume_path_obj is not None:
        if not resume_path_obj.exists():
            raise FileNotFoundError(f"Resume file not found: {resume_path_obj}")
        resume_results = _load_resume_results(resume_path_obj)
        resume_metadata = dict(resume_results.get("metadata", {}))
        stored_defs = resume_metadata.get("parameter_definitions")
        if stored_defs is None:
            logger.warning(
                "Resume file %s lacks parameter definitions; skipping strict validation.",
                resume_path_obj,
            )
        elif stored_defs != parameter_definitions:
            raise ValueError(
                "Parameter configuration in resume file does not match provided parameters."
            )

    seed_value = random_seed if random_seed is not None else resume_metadata.get("random_seed")
    if seed_value is not None:
        seed_value = int(seed_value)
    if (
        random_seed is not None
        and resume_metadata.get("random_seed") is not None
        and resume_metadata["random_seed"] != random_seed
    ):
        logger.info(
            "Using random_seed=%s (overriding resume seed %s).",
            random_seed,
            resume_metadata["random_seed"],
        )

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

    scaled_history: list[list[float]] = []
    objective_history: list[float] = []
    parameter_history: Dict[str, list[float]] = {param.name: [] for param in parameters}
    comsol_history: list[Mapping[str, Mapping[str, float | str | None]]] = []
    success_history: list[bool] = []
    evaluation_records: list[Mapping[str, object]] = []

    best_index = -1
    best_value = -np.inf if maximize else np.inf
    best_parameters: Dict[str, float] | None = None
    best_comsol_parameters: Mapping[str, Mapping[str, float | str | None]] | None = None

    start_timestamp = resume_metadata.get("start_timestamp")
    if start_timestamp is None:
        start_timestamp = time.time()

    metadata: Dict[str, object] = {
        "random_seed": seed_value if seed_value is not None else resume_metadata.get("random_seed"),
        "maximize": maximize,
        "autosave_interval": autosave_interval,
        "parameter_definitions": parameter_definitions,
        "start_timestamp": float(start_timestamp),
    }
    if resume_path_obj is not None:
        metadata["resume_source"] = str(resume_path_obj)

    if "n_initial" in resume_metadata:
        metadata["n_initial"] = int(resume_metadata["n_initial"])
    else:
        metadata["n_initial"] = int(n_initial)
    if "n_iterations" in resume_metadata:
        metadata["n_iterations"] = int(resume_metadata["n_iterations"])

    start_offset = 0
    if resume_results is not None:
        scaled_samples_raw = resume_results.get("scaled_samples") or resume_results.get(
            "scaled_parameters"
        ) or []
        objective_history_raw = resume_results.get("objective_history") or []
        parameter_history_raw = resume_results.get("parameter_history") or {}
        success_history_raw = resume_results.get("success_history") or []
        evaluation_records_raw = resume_results.get("evaluations") or []
        comsol_history_raw = resume_results.get("comsol_parameter_history") or []

        candidate_lengths = [
            len(objective_history_raw),
            len(scaled_samples_raw),
        ]
        if success_history_raw:
            candidate_lengths.append(len(success_history_raw))
        if evaluation_records_raw:
            candidate_lengths.append(len(evaluation_records_raw))
        if comsol_history_raw:
            candidate_lengths.append(len(comsol_history_raw))
        for param in parameters:
            values = parameter_history_raw.get(param.name)
            if values is not None:
                candidate_lengths.append(len(values))
        start_offset = min(candidate_lengths) if candidate_lengths else 0

        for idx in range(start_offset):
            scaled_row_raw = scaled_samples_raw[idx]
            scaled_row = [float(value) for value in scaled_row_raw]
            scaled_history.append(scaled_row)

            obj_value_raw = objective_history_raw[idx]
            objective_value = float(obj_value_raw)
            objective_history.append(objective_value)

            for param in parameters:
                values = parameter_history_raw.get(param.name, [])
                if idx < len(values):
                    parameter_history[param.name].append(float(values[idx]))
                else:
                    parameter_history[param.name].append(float("nan"))

            success = True
            if idx < len(success_history_raw):
                success = bool(success_history_raw[idx])
            success_history.append(success)

            comsol_history.append(
                comsol_history_raw[idx] if idx < len(comsol_history_raw) else {}
            )
            evaluation_records.append(
                evaluation_records_raw[idx] if idx < len(evaluation_records_raw) else {}
            )

            if not math.isnan(objective_value) and success:
                if best_parameters is None:
                    best_index = idx
                    best_value = objective_value
                    best_parameters = {
                        param.name: float(parameter_history[param.name][idx]) for param in parameters
                    }
                    best_comsol_parameters = comsol_history[idx]
                else:
                    is_better = (objective_value > best_value) if maximize else (objective_value < best_value)
                    if is_better:
                        best_index = idx
                        best_value = objective_value
                        best_parameters = {
                            param.name: float(parameter_history[param.name][idx]) for param in parameters
                        }
                        best_comsol_parameters = comsol_history[idx]

        metadata["iterations_completed"] = start_offset
        if "planned_total" in resume_metadata:
            metadata["planned_total"] = int(resume_metadata["planned_total"])
        if "last_update" in resume_metadata:
            metadata["last_update"] = resume_metadata["last_update"]
        if "completed_timestamp" in resume_metadata:
            metadata["completed_timestamp"] = resume_metadata["completed_timestamp"]

    planned_from_args = max(n_initial, 0) + max(n_iterations, 0)
    prior_planned_total = int(metadata.get("planned_total", 0))
    total_planned = max(start_offset, prior_planned_total, planned_from_args)
    metadata["planned_total"] = total_planned
    metadata["total_evaluations"] = total_planned
    metadata["iterations_completed"] = int(metadata.get("iterations_completed", start_offset))
    metadata["n_iterations"] = max(total_planned - metadata["n_initial"], 0)

    evaluations_to_run = max(total_planned - start_offset, 0)
    logger.info(
        "Scheduled %d total evaluations (%d already completed, %d queued).",
        total_planned,
        start_offset,
        evaluations_to_run,
    )

    skopt_optimizer: Optimizer | None = None
    if dimension > 0:
        search_space = [Real(0.0, 1.0, name=param.name) for param in active_parameters]
        initial_points = max(1, metadata["n_initial"])
        skopt_optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="GP",
            acq_func="EI",
            initial_point_generator="random",
            n_initial_points=initial_points,
            random_state=seed_value,
        )
        if start_offset:
            offline_points: list[list[float]] = []
            offline_values: list[float] = []
            for idx in range(start_offset):
                scaled_full = scaled_history[idx]
                scaled_active = [scaled_full[index] for index in active_indices]
                if len(scaled_active) != dimension:
                    continue
                objective_value = objective_history[idx]
                success = success_history[idx] if idx < len(success_history) else True
                if not success or not np.isfinite(objective_value):
                    surrogate_value = FAILED_EVALUATION_PENALTY
                else:
                    surrogate_value = -objective_value if maximize else objective_value
                offline_points.append(list(scaled_active))
                offline_values.append(float(surrogate_value))
            if offline_points:
                skopt_optimizer.tell(offline_points, offline_values)

    metadata.setdefault("last_update", metadata["start_timestamp"])
    if results_path_obj is not None:
        initial_snapshot = _assemble_results(
            parameters=parameters,
            parameter_definitions=parameter_definitions,
            scaled_history=scaled_history,
            objective_history=objective_history,
            parameter_history=parameter_history,
            comsol_history=comsol_history,
            success_history=success_history,
            evaluation_records=evaluation_records,
            best_index=best_index,
            best_value=best_value,
            best_parameters=best_parameters,
            best_comsol_parameters=best_comsol_parameters,
            metadata=metadata,
        )
        _write_results(results_path_obj, initial_snapshot)
        if progress_callback is not None:
            progress_callback(len(objective_history), total_planned, _to_serializable(initial_snapshot))

    if evaluations_to_run == 0:
        metadata["completed_timestamp"] = metadata.get("completed_timestamp", time.time())
        final_results = _assemble_results(
            parameters=parameters,
            parameter_definitions=parameter_definitions,
            scaled_history=scaled_history,
            objective_history=objective_history,
            parameter_history=parameter_history,
            comsol_history=comsol_history,
            success_history=success_history,
            evaluation_records=evaluation_records,
            best_index=best_index,
            best_value=best_value,
            best_parameters=best_parameters,
            best_comsol_parameters=best_comsol_parameters,
            metadata=metadata,
        )
        if results_path_obj is not None:
            _write_results(results_path_obj, final_results)
        if progress_callback is not None:
            progress_callback(len(objective_history), total_planned, _to_serializable(final_results))
        return final_results

    start_time = time.monotonic()
    for iteration_offset in range(evaluations_to_run):
        iteration_index = start_offset + iteration_offset
        if dimension and skopt_optimizer is not None:
            scaled_candidate_active = np.asarray(skopt_optimizer.ask(), dtype=float).reshape(dimension)
            scaled_candidate_active = np.clip(scaled_candidate_active, 0.0, 1.0)
        else:
            scaled_candidate_active = np.empty(0, dtype=float)

        physical_guess: Dict[str, float] = {
            name: float(value) for name, value in constant_defaults.items()
        }
        for axis, param in enumerate(active_parameters):
            transform = optimizer.parameter_transforms[param.name]
            physical_value = transform.to_physical(scaled_candidate_active[axis])
            physical_guess[param.name] = float(physical_value)

        evaluation = dict(optimizer.evaluate(physical_guess))

        objective_value_raw = evaluation.get("objective")
        if objective_value_raw is None:
            objective_value_raw = evaluation.get("power")
        objective_value = float(objective_value_raw) if objective_value_raw is not None else float("nan")
        objective_history.append(objective_value)

        actual_params = {
            name: float(value) for name, value in evaluation.get("parameters", physical_guess).items()
        }

        current_scaled_full: list[float] = [0.0] * len(parameters)
        for axis, param in enumerate(parameters):
            actual_value = actual_params.get(param.name, physical_guess[param.name])
            transform = optimizer.parameter_transforms[param.name]
            scaled_value = transform.to_unit(actual_value)
            scaled_value = transform.ensure_unit(scaled_value)
            bounded_value = float(np.clip(scaled_value, 0.0, 1.0))
            current_scaled_full[axis] = bounded_value
            parameter_history[param.name].append(float(actual_value))
        scaled_history.append(current_scaled_full)

        current_scaled_active = [current_scaled_full[index] for index in active_indices] if active_indices else []

        comsol_payload = evaluation.get("comsol_parameters", {})
        comsol_history.append(comsol_payload)
        success = bool(evaluation.get("success", True))
        disregard_negative = maximize and math.isfinite(objective_value) and objective_value < 0.0
        if disregard_negative:
            success = False
            evaluation["disregarded_negative_objective"] = True
            logger.info(
                "Ignoring objective %.6g at iteration %d because it is negative during maximization.",
                objective_value,
                iteration_index,
            )
        success_history.append(success)
        evaluation_records.append(evaluation)

        consider_for_best = success and not math.isnan(objective_value)
        if consider_for_best:
            if best_parameters is None or best_index < 0:
                best_index = iteration_index
                best_value = objective_value
                best_parameters = dict(actual_params)
                best_comsol_parameters = comsol_payload
            else:
                is_better = (objective_value > best_value) if maximize else (objective_value < best_value)
                if is_better:
                    best_index = iteration_index
                    best_value = objective_value
                    best_parameters = dict(actual_params)
                    best_comsol_parameters = comsol_payload

        if dimension and skopt_optimizer is not None and current_scaled_active:
            if not success or not np.isfinite(objective_value):
                surrogate_value = FAILED_EVALUATION_PENALTY
            else:
                surrogate_value = -objective_value if maximize else objective_value
            skopt_optimizer.tell(current_scaled_active, surrogate_value)

        iterations_completed = len(objective_history)
        elapsed = time.monotonic() - start_time
        avg_time = elapsed / (iteration_offset + 1)
        remaining = max(total_planned - iterations_completed, 0)
        eta = avg_time * remaining if iteration_offset + 1 else 0.0
        progress_pct = (iterations_completed / total_planned) * 100 if total_planned else 100.0

        metadata["iterations_completed"] = iterations_completed
        metadata["last_update"] = time.time()
        metadata["planned_total"] = total_planned
        metadata["n_iterations"] = max(total_planned - metadata["n_initial"], 0)

        logger.info(
            "Completed evaluation %d/%d (%.1f%%). Elapsed %s, ETA %s.",
            iterations_completed,
            total_planned,
            progress_pct,
            _format_duration(elapsed),
            _format_duration(eta),
        )

        snapshot = _assemble_results(
            parameters=parameters,
            parameter_definitions=parameter_definitions,
            scaled_history=scaled_history,
            objective_history=objective_history,
            parameter_history=parameter_history,
            comsol_history=comsol_history,
            success_history=success_history,
            evaluation_records=evaluation_records,
            best_index=best_index,
            best_value=best_value,
            best_parameters=best_parameters,
            best_comsol_parameters=best_comsol_parameters,
            metadata=metadata,
        )

        should_autosave = (iteration_offset + 1) % autosave_interval == 0 or iteration_offset == evaluations_to_run - 1
        if results_path_obj is not None and should_autosave:
            _write_results(results_path_obj, snapshot)
        if progress_callback is not None:
            progress_callback(iterations_completed, total_planned, _to_serializable(snapshot))

    metadata["completed_timestamp"] = time.time()

    final_results = _assemble_results(
        parameters=parameters,
        parameter_definitions=parameter_definitions,
        scaled_history=scaled_history,
        objective_history=objective_history,
        parameter_history=parameter_history,
        comsol_history=comsol_history,
        success_history=success_history,
        evaluation_records=evaluation_records,
        best_index=best_index,
        best_value=best_value,
        best_parameters=best_parameters,
        best_comsol_parameters=best_comsol_parameters,
        metadata=metadata,
    )
    if results_path_obj is not None:
        _write_results(results_path_obj, final_results)

    return final_results


def _assemble_results(
    *,
    parameters: Sequence[OptimizationParameter],
    parameter_definitions: list[dict[str, object]],
    scaled_history: list[list[float]],
    objective_history: list[float],
    parameter_history: Dict[str, list[float]],
    comsol_history: list[Mapping[str, Mapping[str, float | str | None]]],
    success_history: list[bool],
    evaluation_records: list[Mapping[str, object]],
    best_index: int,
    best_value: float,
    best_parameters: Dict[str, float] | None,
    best_comsol_parameters: Mapping[str, Mapping[str, float | str | None]] | None,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    total_completed = len(objective_history)
    num_params = len(parameters)

    if total_completed:
        scaled_array = np.asarray(scaled_history[:total_completed], dtype=float)
        objective_array = np.asarray(objective_history[:total_completed], dtype=float)
    else:
        scaled_array = np.empty((0, num_params), dtype=float)
        objective_array = np.empty(0, dtype=float)

    parameter_history_arrays: Dict[str, np.ndarray] = {}
    for param in parameters:
        values = parameter_history.get(param.name, [])
        trimmed = [float(value) for value in values[:total_completed]]
        if len(trimmed) < total_completed:
            trimmed.extend([float("nan")] * (total_completed - len(trimmed)))
        parameter_history_arrays[param.name] = np.asarray(trimmed, dtype=float)

    success_trimmed = [bool(value) for value in success_history[:total_completed]]
    comsol_trimmed = comsol_history[:total_completed]
    evaluation_trimmed = evaluation_records[:total_completed]

    metadata_copy = dict(metadata)
    metadata_copy["iterations_completed"] = total_completed
    planned_total = int(metadata_copy.get("planned_total", total_completed))
    metadata_copy["planned_total"] = max(planned_total, total_completed)
    metadata_copy["total_evaluations"] = metadata_copy["planned_total"]
    metadata_copy.setdefault("n_initial", 0)
    metadata_copy["n_iterations"] = max(metadata_copy["planned_total"] - metadata_copy["n_initial"], 0)
    metadata_copy["parameter_definitions"] = parameter_definitions

    if best_parameters is None:
        best_parameters_local = {param.name: float("nan") for param in parameters}
        best_value_local = float("nan")
        best_index_local = -1
        best_comsol_local: Dict[str, Mapping[str, float | str | None]] = {}
    else:
        best_parameters_local = {name: float(value) for name, value in best_parameters.items()}
        best_value_local = float(best_value)
        best_index_local = best_index
        best_comsol_local = dict(best_comsol_parameters or {})

    results: Dict[str, object] = {
        "objective": best_value_local,
        "best_objective": best_value_local,
        "parameters": best_parameters_local,
        "best_parameters": best_parameters_local,
        "best_index": best_index_local,
        "objective_history": objective_array,
        "power": best_value_local,
        "power_history": objective_array,
        "parameter_history": parameter_history_arrays,
        "scaled_samples": scaled_array,
        "scaled_parameters": scaled_array,
        "scaled_bounds": np.array(
            [[0.0] * num_params, [1.0] * num_params],
            dtype=float,
        ),
        "comsol_parameter_history": comsol_trimmed,
        "best_comsol_parameters": best_comsol_local,
        "success_history": success_trimmed,
        "success": any(success_trimmed),
        "evaluations": evaluation_trimmed,
        "metadata": metadata_copy,
        "derived_history": [],
        "derived_parameters": {},
    }

    results["gaussian_process"] = {
        "scaled_parameters": scaled_array.tolist(),
        "objective_observations": objective_array.tolist(),
        "power_observations": objective_array.tolist(),
        "parameter_history": {name: array.tolist() for name, array in parameter_history_arrays.items()},
        "derived_history": [],
        "comsol_parameter_history": comsol_trimmed,
        "scaled_bounds": [[0.0] * num_params, [1.0] * num_params],
        "parameter_definitions": parameter_definitions,
        "random_seed": metadata_copy.get("random_seed"),
        "n_initial": metadata_copy.get("n_initial"),
        "n_iterations": metadata_copy.get("n_iterations"),
    }

    return results
