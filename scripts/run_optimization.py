"""Command-line entry point for the thermoelectric optimization workflow."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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

    if args.resume_from:
        print(f"Resuming optimization from '{args.resume_from}'.")

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
        resume_path=args.resume_from,
        autosave_interval=autosave_interval,
    )

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
