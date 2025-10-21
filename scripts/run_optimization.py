"""Command-line entry point for the thermoelectric optimization workflow."""

from __future__ import annotations

import json
from pathlib import Path

from comsol_opt import OptimizationParameter, optimize_thermoelectric_generator


def main() -> None:
    MODEL_PATH = "teg_no_electrodes.mph"
    N_LEGS = 8
    N_INITIAL = 4
    N_ITERATIONS = 16
    FILL_FACTOR_BOUNDS = (0.01, 0.40)  # area fraction no units
    R_LOAD_BOUNDS = (1.0, 5.0)  # ohms
    TARGET_FOOTPRINT_MM2 = 400

    COMSOL_EXE = r"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_NSL\\bin\\win64\\comsolbatch.exe"

    PARAMETERS = [
        OptimizationParameter(
            name="fill_factor",
            bounds=FILL_FACTOR_BOUNDS,
            comsol_name="fill_factor",
            transform="fill_factor",
        ),
        OptimizationParameter(
            name="n_legs",
            comsol_name="n_legs",
            bounds=(4, 20),
            unit=None,
            value_type="even_integer",
        ),
    ]

    results = optimize_thermoelectric_generator(
        model_path=MODEL_PATH,
        n_legs=N_LEGS,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        random_seed=42,
        comsol_exe_path=COMSOL_EXE,
        methodcall="methodcall2",
        target_footprint_mm2=TARGET_FOOTPRINT_MM2,
        parameters=PARAMETERS,
    )

    gp_training_data = {
        "scaled_parameters": results["scaled_parameters"].tolist(),
        "power_observations": results["power_history"].reshape(-1).tolist(),
        "parameter_history": {
            name: values.tolist() for name, values in results["parameter_history"].items()
        },
        "derived_history": results["derived_history"],
        "comsol_parameter_history": results["comsol_parameter_history"],
        "scaled_bounds": [
            [0.0 for _ in PARAMETERS],
            [1.0 for _ in PARAMETERS],
        ],
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
        "random_seed": 42,
        "n_initial": N_INITIAL,
        "n_iterations": N_ITERATIONS,
    }

    results_path = Path("optimization_results.json")
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
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
