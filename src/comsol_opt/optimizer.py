"""Core Bayesian optimization loop built on BoTorch."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

from .acquisition import (
    generate_initial_candidates,
    get_acquisition,
    optimize_acquisition_function,
)
from .objective import EvaluationResult, ObjectiveFunction, wrap_callable
from .parameters import OptimizationParameter
from .state import OptimizationState
from .surrogate import GPSurrogate
from .transforms import FillFactorTransform, LinearParameterTransform

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Bayesian optimization engine backed by BoTorch.

    Parameters
    ----------
    parameters:
        Sequence of parameter definitions.  Parameters with a
        ``constant_value`` are excluded from the search space but forwarded
        to the objective function.
    objective:
        An :class:`ObjectiveFunction` implementation or a plain callable
        ``(dict[str, float]) -> float``.
    objective_names:
        Names of the objectives.  For single-objective pass a single-element
        list (default ``["objective"]``).  For multi-objective pass one name
        per objective returned by the objective function.
    n_initial:
        Number of Sobol quasi-random exploration evaluations before fitting
        the GP surrogate.
    n_iterations:
        Number of GP-guided evaluations after the initial design.
    batch_size:
        Number of candidates per iteration.  ``batch_size > 1`` uses
        q-batch acquisition functions.
    acquisition:
        Acquisition function name.  See :mod:`comsol_opt.acquisition`.
    maximize:
        Whether to maximize the objective(s).  Pass a single bool for all
        objectives or a list of bools for per-objective control.
    seed:
        Random seed for reproducibility.
    state_path:
        Path to auto-checkpoint the state after each evaluation.
    autosave_interval:
        Number of evaluations between autosaves.
    ref_point:
        Reference point for multi-objective hypervolume computation.
    progress_callback:
        Optional callable invoked after each evaluation with
        ``(completed, total, state)``.
    """

    def __init__(
        self,
        parameters: Sequence[OptimizationParameter],
        objective: ObjectiveFunction | Callable,
        *,
        objective_names: list[str] | None = None,
        n_initial: int = 5,
        n_iterations: int = 20,
        batch_size: int = 1,
        acquisition: str = "EI",
        maximize: bool | list[bool] = True,
        seed: int | None = None,
        state_path: Path | None = None,
        autosave_interval: int = 1,
        ref_point: list[float] | None = None,
        progress_callback: Callable[[int, int, OptimizationState], None] | None = None,
    ) -> None:
        if n_initial < 0:
            raise ValueError("n_initial must be non-negative.")
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if autosave_interval < 1:
            raise ValueError("autosave_interval must be at least 1.")

        self._all_parameters = list(parameters)
        self._active_parameters = [p for p in self._all_parameters if not p.is_constant]
        self._constant_defaults: dict[str, float] = {
            p.name: float(p.constant_value)
            for p in self._all_parameters
            if p.is_constant
        }
        self._dimension = len(self._active_parameters)

        if isinstance(objective, ObjectiveFunction):
            self._objective = objective
        else:
            self._objective = wrap_callable(objective)

        self._objective_names = objective_names or ["objective"]
        self._n_objectives = len(self._objective_names)
        self._n_initial = n_initial
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._acquisition_name = acquisition
        self._seed = seed
        self._state_path = Path(state_path) if state_path else None
        self._autosave_interval = autosave_interval
        self._ref_point = (
            torch.tensor(ref_point, dtype=torch.double) if ref_point else None
        )
        self._progress_callback = progress_callback

        if isinstance(maximize, bool):
            self._maximize = [maximize] * self._n_objectives
        else:
            if len(maximize) != self._n_objectives:
                raise ValueError(
                    f"maximize list length ({len(maximize)}) must match "
                    f"number of objectives ({self._n_objectives})."
                )
            self._maximize = list(maximize)

        # Build transforms for active parameters
        self._transforms: dict[str, LinearParameterTransform | FillFactorTransform] = {}
        for p in self._active_parameters:
            if p.transform == "fill_factor":
                self._transforms[p.name] = FillFactorTransform(p.bounds)
            else:
                self._transforms[p.name] = LinearParameterTransform(p.bounds)

        # Unit-space bounds are always [0, 1]^d
        self._unit_bounds = torch.stack([
            torch.zeros(max(self._dimension, 1), dtype=torch.double),
            torch.ones(max(self._dimension, 1), dtype=torch.double),
        ])

        # Internal state lists — populated by tell() or during run()
        self._X_list: list[list[float]] = []
        self._Y_list: list[list[float]] = []
        self._X_physical: dict[str, list[float]] = {
            p.name: [] for p in self._all_parameters
        }
        self._success_mask: list[bool] = []
        self._metadata: dict[str, object] = {
            "seed": seed,
            "n_initial": n_initial,
            "n_iterations": n_iterations,
            "acquisition": acquisition,
            "batch_size": batch_size,
            "start_timestamp": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> OptimizationState:
        """Execute the full optimization loop and return the final state."""
        total = self._n_initial + self._n_iterations
        self._metadata["start_timestamp"] = time.time()

        logger.info(
            "Starting optimization: %d initial + %d iterations = %d total, "
            "%d active parameter(s), %d objective(s).",
            self._n_initial, self._n_iterations, total,
            self._dimension, self._n_objectives,
        )

        for i in range(total):
            if i < self._n_initial:
                candidates = self._generate_initial_candidate(i)
            else:
                candidates = self._generate_bo_candidate()

            for candidate_unit in candidates:
                physical = self._unit_to_physical(candidate_unit)
                result = self._objective.evaluate(physical)
                self.tell(physical, result)

                completed = len(self._X_list)
                if self._progress_callback is not None:
                    self._progress_callback(completed, total, self.state)

                if (
                    self._state_path is not None
                    and completed % self._autosave_interval == 0
                ):
                    self.state.save(self._state_path)

                self._log_progress(completed, total)

        self._metadata["completed_timestamp"] = time.time()
        final_state = self.state
        if self._state_path is not None:
            final_state.save(self._state_path)
        return final_state

    def resume(self, state: OptimizationState) -> OptimizationState:
        """Resume an optimization from a saved state.

        Loads the evaluation history from *state* and continues the
        optimization loop for any remaining iterations.
        """
        # Restore internal history from the state
        n_existing = state.n_completed
        self._X_list = state.X.tolist() if state.X.numel() > 0 else []
        self._Y_list = state.Y.tolist() if state.Y.numel() > 0 else []
        self._X_physical = {k: list(v) for k, v in state.X_physical.items()}
        self._success_mask = list(state.success_mask)
        self._metadata.update(state.metadata)

        total = self._n_initial + self._n_iterations
        remaining = max(total - n_existing, 0)

        logger.info(
            "Resuming optimization: %d already completed, %d remaining.",
            n_existing, remaining,
        )

        for i in range(remaining):
            iteration_index = n_existing + i
            if iteration_index < self._n_initial:
                candidates = self._generate_initial_candidate(iteration_index)
            else:
                candidates = self._generate_bo_candidate()

            for candidate_unit in candidates:
                physical = self._unit_to_physical(candidate_unit)
                result = self._objective.evaluate(physical)
                self.tell(physical, result)

                completed = len(self._X_list)
                if self._progress_callback is not None:
                    self._progress_callback(completed, total, self.state)

                if (
                    self._state_path is not None
                    and completed % self._autosave_interval == 0
                ):
                    self.state.save(self._state_path)

                self._log_progress(completed, total)

        self._metadata["completed_timestamp"] = time.time()
        final_state = self.state
        if self._state_path is not None:
            final_state.save(self._state_path)
        return final_state

    def ask(self, n: int = 1) -> list[dict[str, float]]:
        """Suggest the next *n* candidate parameter sets to evaluate.

        Returns a list of dicts mapping parameter names to physical values.
        """
        results = []
        n_completed = len(self._X_list)

        for i in range(n):
            idx = n_completed + i
            if idx < self._n_initial or n_completed < 2:
                candidates = self._generate_initial_candidate(idx)
            else:
                candidates = self._generate_bo_candidate()
            for candidate_unit in candidates:
                physical = self._unit_to_physical(candidate_unit)
                results.append(physical)
        return results

    def tell(self, params: dict[str, float], result: EvaluationResult) -> None:
        """Record an observed evaluation.

        Parameters
        ----------
        params:
            Physical parameter values that were evaluated.
        result:
            The evaluation result.
        """
        # Map physical values back to unit space for active parameters
        unit_values = []
        for p in self._active_parameters:
            transform = self._transforms[p.name]
            val = params.get(p.name, 0.0)
            coerced = p.coerce_physical_value(val)
            unit_val = float(transform.to_unit(coerced))
            unit_val = max(0.0, min(1.0, unit_val))
            unit_values.append(unit_val)

        self._X_list.append(unit_values)

        # Extract objective values in the configured order
        obj_values = []
        for obj_name in self._objective_names:
            obj_values.append(result.objectives.get(obj_name, float("nan")))
        self._Y_list.append(obj_values)

        # Record physical values for all parameters
        for p in self._all_parameters:
            val = params.get(p.name, self._constant_defaults.get(p.name, float("nan")))
            self._X_physical[p.name].append(float(val))

        self._success_mask.append(result.success)

    @property
    def state(self) -> OptimizationState:
        """Build and return the current optimization state."""
        n = len(self._X_list)
        d = self._dimension
        m = self._n_objectives

        if n > 0:
            X = torch.tensor(self._X_list, dtype=torch.double)
            Y = torch.tensor(self._Y_list, dtype=torch.double)
        else:
            X = torch.empty(0, max(d, 1), dtype=torch.double)
            Y = torch.empty(0, m, dtype=torch.double)

        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        return OptimizationState(
            parameters=list(self._all_parameters),
            objective_names=list(self._objective_names),
            X=X,
            Y=Y,
            X_physical={k: list(v) for k, v in self._X_physical.items()},
            success_mask=list(self._success_mask),
            metadata=dict(self._metadata),
            maximize=list(self._maximize),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_initial_candidate(self, index: int) -> list[list[float]]:
        """Generate a single Sobol quasi-random candidate."""
        if self._dimension == 0:
            return [[]]
        seed = (self._seed or 0) + index
        candidates = generate_initial_candidates(
            bounds=self._unit_bounds, n=1, seed=seed,
        )
        return [candidates[0].tolist()]

    def _generate_bo_candidate(self) -> list[list[float]]:
        """Generate candidate(s) using the GP surrogate and acquisition function."""
        if self._dimension == 0:
            return [[]]

        # Filter to successful observations only
        X_all = torch.tensor(self._X_list, dtype=torch.double)
        Y_all = torch.tensor(self._Y_list, dtype=torch.double)
        if Y_all.ndim == 1:
            Y_all = Y_all.unsqueeze(-1)
        mask = torch.tensor(self._success_mask, dtype=torch.bool)

        if mask.sum() < 2:
            # Not enough data for a GP — fall back to random
            logger.debug("Fewer than 2 successful evaluations; using random candidate.")
            seed = (self._seed or 0) + len(self._X_list)
            candidates = generate_initial_candidates(
                bounds=self._unit_bounds, n=1, seed=seed,
            )
            return [candidates[0].tolist()]

        X_train = X_all[mask]
        Y_train = Y_all[mask]

        # For single-objective, negate Y if maximizing (BoTorch minimizes by default
        # for many acquisition functions, but EI/UCB have a maximize flag)
        surrogate = GPSurrogate(
            n_objectives=self._n_objectives,
            bounds=self._unit_bounds,
        )
        surrogate.fit(X_train, Y_train)

        if self._n_objectives == 1:
            # Single-objective path
            if self._maximize[0]:
                best_f = Y_train[:, 0].max().item()
            else:
                best_f = Y_train[:, 0].min().item()

            acq = get_acquisition(
                name=self._acquisition_name,
                model=surrogate.model,
                best_f=best_f,
                maximize=self._maximize[0],
                batch_size=self._batch_size,
            )
        else:
            # Multi-objective path
            # Negate objectives being minimized for consistent maximization
            Y_for_pareto = Y_train.clone()
            for i, mx in enumerate(self._maximize):
                if not mx:
                    Y_for_pareto[:, i] = -Y_for_pareto[:, i]

            acq = get_acquisition(
                name=self._acquisition_name,
                model=surrogate.model,
                best_f=0.0,  # not used for multi-objective
                maximize=True,
                batch_size=self._batch_size,
                ref_point=self._ref_point,
                Y_successful=Y_for_pareto,
            )

        candidates = optimize_acquisition_function(
            acq=acq,
            bounds=self._unit_bounds,
            batch_size=self._batch_size,
            num_restarts=10,
            raw_samples=256,
        )

        return candidates.tolist()

    def _unit_to_physical(self, unit_values: list[float]) -> dict[str, float]:
        """Convert unit-hypercube values to physical parameter values."""
        physical: dict[str, float] = dict(self._constant_defaults)
        for axis, p in enumerate(self._active_parameters):
            transform = self._transforms[p.name]
            raw_physical = float(transform.to_physical(unit_values[axis]))
            physical[p.name] = p.coerce_physical_value(raw_physical)
        return physical

    def _log_progress(self, completed: int, total: int) -> None:
        if total <= 0:
            return
        pct = (completed / total) * 100

        if self._n_objectives == 1:
            # Report best value
            best_idx = -1
            best_val = float("-inf") if self._maximize[0] else float("inf")
            for i, (success, y_vals) in enumerate(zip(self._success_mask, self._Y_list)):
                if not success:
                    continue
                val = y_vals[0]
                if self._maximize[0] and val > best_val:
                    best_val = val
                    best_idx = i
                elif not self._maximize[0] and val < best_val:
                    best_val = val
                    best_idx = i
            best_str = f", best={best_val:.6g}" if best_idx >= 0 else ""
        else:
            best_str = ""

        logger.info(
            "Evaluation %d/%d (%.1f%%)%s",
            completed, total, pct, best_str,
        )
