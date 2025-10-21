"""Command-line entry point for the thermoelectric optimization workflow."""

from __future__ import annotations

import json
from pathlib import Path

from comsol_opt import optimize_thermoelectric_generator


def main() -> None:
    MODEL_PATH = "teg_no_electrodes.mph"
    N_LEGS = 8
    N_INITIAL = 4
    N_ITERATIONS = 16
    FILL_FACTOR_BOUNDS = (0.01, 0.40)  # area fraction no units
    R_LOAD_BOUNDS = (1.0, 5.0)  # ohms
    TARGET_FOOTPRINT_MM2 = 400

    COMSOL_EXE = r"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_NSL\\bin\\win64\\comsolbatch.exe"

    results = optimize_thermoelectric_generator(
        model_path=MODEL_PATH,
        n_legs=N_LEGS,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        fill_factor_bounds=FILL_FACTOR_BOUNDS,
        r_load_bounds=R_LOAD_BOUNDS,
        random_seed=42,
        comsol_exe_path=COMSOL_EXE,
        methodcall="methodcall2",
        target_footprint_mm2=TARGET_FOOTPRINT_MM2,
    )

    gp_training_data = {
        "scaled_parameters": results["scaled_parameters"].tolist(),
        "fill_factors": results["all_fill_factors"].tolist(),
        "r_loads": results["all_r_loads"].tolist(),
        "power_observations": results["all_y"].reshape(-1).tolist(),
        "scaled_bounds": [[0.0, 0.0], [1.0, 1.0]],
        "physical_bounds": {
            "fill_factor": list(FILL_FACTOR_BOUNDS),
            "r_load": list(R_LOAD_BOUNDS),
        },
        "random_seed": 42,
        "n_initial": N_INITIAL,
        "n_iterations": N_ITERATIONS,
    }

    results_path = Path("optimization_results.json")
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fill_factor": results["fill_factor"],
                "r_load": results["r_load"],
                "leg_width": results["leg_width"],
                "leg_spacing": results["leg_spacing"],
                "power": results["power"],
                "gaussian_process": gp_training_data,
            },
            handle,
            indent=2,
        )

    print(f"\nResults saved to '{results_path}'")


if __name__ == "__main__":
    main()
