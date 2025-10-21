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
    FILL_FACTOR_BOUNDS = (0.01, 0.40)  # area fraction
    TARGET_FOOTPRINT_MM2 = 400

    COMSOL_EXE = r"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_NSL\\bin\\win64\\comsolbatch.exe"

    results = optimize_thermoelectric_generator(
        model_path=MODEL_PATH,
        n_legs=N_LEGS,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        fill_factor_bounds=FILL_FACTOR_BOUNDS,
        random_seed=42,
        comsol_exe_path=COMSOL_EXE,
        methodcall="methodcall2",
        target_footprint_mm2=TARGET_FOOTPRINT_MM2,
    )

    results_path = Path("optimization_results.json")
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fill_factor": results["fill_factor"],
                "leg_width": results["leg_width"],
                "leg_spacing": results["leg_spacing"],
                "power": results["power"],
            },
            handle,
            indent=2,
        )

    print(f"\nResults saved to '{results_path}'")


if __name__ == "__main__":
    main()

