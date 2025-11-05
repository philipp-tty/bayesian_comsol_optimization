"""Command-line entry point for the thermoelectric optimization workflow."""

from __future__ import annotations

import json
from pathlib import Path

from comsol_opt import OptimizationParameter, optimize_model


def main() -> None:
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

    results = optimize_model(
        model_path=MODEL_PATH,
        comsol_exe_path=COMSOL_EXE,
        methodcall="methodcall2",
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        random_seed=RANDOM_SEED,
        maximize=True,
        parameters=PARAMETERS,
    )

    objective_values = results["objective_history"].reshape(-1).tolist()

    gp_training_data = {
        "scaled_parameters": results["scaled_parameters"].tolist(),
        "objective_observations": objective_values,
        "power_observations": objective_values,
        "parameter_history": {
            name: values.tolist() for name, values in results["parameter_history"].items()
        },
        "derived_history": results["derived_history"],
        "comsol_parameter_history": results["comsol_parameter_history"],
        "scaled_bounds": [[0.0 for _ in PARAMETERS], [1.0 for _ in PARAMETERS]],
        "parameter_definitions": [
            {
                "name": param.name,
                "bounds": list(param.bounds),
                "comsol_name": param.comsol_name,
                "unit": param.unit,
                "transform": param.transform,
                "value_type": param.value_type,
            }
            for param in PARAMETERS
        ],
        "random_seed": RANDOM_SEED,
        "n_initial": N_INITIAL,
        "n_iterations": N_ITERATIONS,
    }

    results_path = Path("optimization_results.json")
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "objective_value": results["objective"],
                "power_mw": results["power"],
                "parameters": results["parameters"],
                "derived_parameters": results["derived_parameters"],
                "gaussian_process": gp_training_data,
            },
            handle,
            indent=2,
        )

    print(f"\nResults saved to '{results_path}'")


if __name__ == "__main__":
    main()
