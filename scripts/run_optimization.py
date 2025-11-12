"""Command-line entry point for the thermoelectric optimization workflow."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from contextlib import suppress
from pathlib import Path
from typing import Mapping, Sequence

from comsol_opt import OptimizationParameter, optimize_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COMSOL-based Bayesian optimization workflow.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path to store incremental optimization results (defaults to optimization_results.json).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume optimization from a previously saved results JSON file.",
    )
    parser.add_argument(
        "--autosave-interval",
        type=int,
        default=1,
        help="Number of evaluations between progress snapshots (default: 1).",
    )
    return parser.parse_args()


ACTIVE_STATE_PATH = Path("active_optimization.json")


class ActiveOptimizationTracker:
    """Track per-iteration progress in a JSON file for crash-safe monitoring."""

    def __init__(self, path: Path, run_settings: Mapping[str, object]):
        self.path = path
        self.run_settings = dict(run_settings)
        self._iteration_log: list[dict[str, object]] = []
        self._last_logged_iteration = 0
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        log = data.get("iteration_log")
        if isinstance(log, list):
            for entry in log:
                if not isinstance(entry, dict):
                    continue
                iteration_value = entry.get("iteration")
                if isinstance(iteration_value, int):
                    self._last_logged_iteration = max(self._last_logged_iteration, iteration_value)
                self._iteration_log.append(entry)

    def handle_progress(
        self,
        completed: int,
        planned_total: int,
        snapshot: Mapping[str, object],
    ) -> None:
        try:
            self._record_snapshot(completed, planned_total, snapshot)
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"Warning: failed to update '{self.path}': {exc}")

    def _record_snapshot(
        self,
        completed: int,
        planned_total: int,
        snapshot: Mapping[str, object],
    ) -> None:
        if completed < self._last_logged_iteration:
            self._iteration_log = [
                entry for entry in self._iteration_log if entry.get("iteration", 0) <= completed
            ]
            self._last_logged_iteration = completed

        while completed > self._last_logged_iteration:
            iteration_number = self._last_logged_iteration + 1
            entry = self._build_iteration_entry(iteration_number, planned_total, snapshot)
            if entry is None:
                break
            self._iteration_log.append(entry)
            self._last_logged_iteration = iteration_number

        payload = deepcopy(snapshot)
        payload["iteration_log"] = list(self._iteration_log)
        payload["run_settings"] = dict(self.run_settings)
        payload["active_last_update"] = time.time()
        self._write_payload(payload)

    def _build_iteration_entry(
        self,
        iteration_number: int,
        planned_total: int,
        snapshot: Mapping[str, object],
    ) -> dict[str, object] | None:
        if iteration_number <= 0:
            return None
        objective_history = snapshot.get("objective_history")
        if not isinstance(objective_history, list):
            return None
        iteration_idx = iteration_number - 1
        if iteration_idx >= len(objective_history):
            return None

        success_history = snapshot.get("success_history")
        success_value = None
        if isinstance(success_history, list) and iteration_idx < len(success_history):
            success_value = success_history[iteration_idx]

        parameter_history = snapshot.get("parameter_history")
        parameters_at_iteration: dict[str, object] = {}
        if isinstance(parameter_history, Mapping):
            for name, values in parameter_history.items():
                if not isinstance(values, list) or iteration_idx >= len(values):
                    continue
                parameters_at_iteration[name] = values[iteration_idx]

        comsol_history = snapshot.get("comsol_parameter_history")
        comsol_parameters = None
        if isinstance(comsol_history, list) and iteration_idx < len(comsol_history):
            entry = comsol_history[iteration_idx]
            if isinstance(entry, Mapping):
                comsol_parameters = entry

        return {
            "iteration": iteration_number,
            "planned_total": planned_total,
            "objective": objective_history[iteration_idx],
            "success": success_value,
            "parameters": parameters_at_iteration,
            "comsol_parameters": comsol_parameters,
            "best_objective": snapshot.get("best_objective"),
            "timestamp": time.time(),
        }

    def _write_payload(self, payload: Mapping[str, object]) -> None:
        tmp_path = self.path.parent / f"{self.path.name}.tmp"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.path)


def _build_run_settings(
    *,
    model_path: str | Path,
    comsol_exe: str,
    methodcall: str,
    maximize: bool,
    n_initial: int,
    n_iterations: int,
    random_seed: int | None,
    results_path: Path,
    autosave_interval: int,
    parameters: Sequence[OptimizationParameter],
    resume_path: Path | None,
    active_path: Path,
) -> dict[str, object]:
    return {
        "model_path": str(model_path),
        "comsol_exe_path": comsol_exe,
        "methodcall": methodcall,
        "maximize": bool(maximize),
        "n_initial": int(n_initial),
        "n_iterations": int(n_iterations),
        "random_seed": random_seed,
        "results_path": str(results_path),
        "autosave_interval": int(autosave_interval),
        "resume_path": str(resume_path) if resume_path else None,
        "active_state_path": str(active_path),
        "parameters": [
            {
                "name": param.name,
                "bounds": [float(param.bounds[0]), float(param.bounds[1])],
                "value_type": param.value_type,
                "comsol_name": param.comsol_name,
                "unit": param.unit,
                "is_constant": param.is_constant,
                "constant_value": (
                    float(param.constant_value) if param.constant_value is not None else None
                ),
            }
            for param in parameters
        ],
        "created_at": time.time(),
    }


def main() -> None:
    args = _parse_args()

    MODEL_PATH = "teg_no_electrodes.mph"
    N_INITIAL = 10
    N_ITERATIONS = 40
    RANDOM_SEED = 42

    COMSOL_EXE = r"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_NSL\\bin\\win64\\comsolbatch.exe"

    PARAMETERS = [
        OptimizationParameter(
            name="n_legs",
            bounds=(4, 12),
            comsol_name="n_legs",
            value_type="even_integer",
        ),
        OptimizationParameter(
            name="leg_spacing",
            bounds=(0.5, 2.0),
            comsol_name="leg_spacing",
            unit="mm",
        ),
        OptimizationParameter(
            name="leg_width",
            bounds=(0.5, 2.0),
            comsol_name="leg_width",
            unit="mm",
        ),
        OptimizationParameter(
            name="leg_length",
            bounds=(0.5, 4.0),
            comsol_name="leg_length",
            unit="mm",
        ),
        OptimizationParameter(
            name="r_load",
            bounds=(0.5, 10.0),
            comsol_name="r_load",
            unit="ohm",
        ),
    ]

    results_path = args.results_path or args.resume_from or Path("optimization_results.json")
    autosave_interval = max(1, args.autosave_interval)

    resume_path = args.resume_from
    resumed_from_active = False
    if resume_path is None and ACTIVE_STATE_PATH.exists():
        resume_path = ACTIVE_STATE_PATH
        resumed_from_active = True

    if args.resume_from:
        print(f"Resuming optimization from '{args.resume_from}'.")
    elif resumed_from_active:
        print(f"Found '{ACTIVE_STATE_PATH}'. Continuing optimization automatically.")

    run_settings = _build_run_settings(
        model_path=MODEL_PATH,
        comsol_exe=COMSOL_EXE,
        methodcall="methodcall2",
        maximize=True,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        random_seed=RANDOM_SEED,
        results_path=results_path,
        autosave_interval=autosave_interval,
        parameters=PARAMETERS,
        resume_path=resume_path,
        active_path=ACTIVE_STATE_PATH,
    )
    tracker = ActiveOptimizationTracker(ACTIVE_STATE_PATH, run_settings)

    optimization_completed = False
    try:
        results = optimize_model(
            model_path=MODEL_PATH,
            comsol_exe_path=COMSOL_EXE,
            methodcall="methodcall2",
            n_initial=N_INITIAL,
            n_iterations=N_ITERATIONS,
            random_seed=RANDOM_SEED,
            maximize=True,
            parameters=PARAMETERS,
            results_path=results_path,
            resume_path=resume_path,
            autosave_interval=autosave_interval,
            progress_callback=tracker.handle_progress,
        )
        optimization_completed = True
    finally:
        if optimization_completed:
            with suppress(FileNotFoundError):
                ACTIVE_STATE_PATH.unlink()

    objective_value = float(results["objective"])
    best_parameters = results["best_parameters"]

    print("\nOptimization completed.")
    if math.isnan(objective_value):
        print("No valid objective value was obtained.")
    else:
        print(f"Best objective value: {objective_value:.6g}")
        for name, value in best_parameters.items():
            print(f"  {name}: {float(value):.6g}")

    print(f"\nProgress snapshots saved to '{results_path}'.")


if __name__ == "__main__":
    main()
