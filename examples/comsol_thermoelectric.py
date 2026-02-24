#!/usr/bin/env python3
"""Example: Optimize a COMSOL thermoelectric generator model.

This script demonstrates how to use the comsol_opt library to optimize
a COMSOL thermoelectric generator (TEG) model.

Usage:
    python comsol_thermoelectric.py

Or via the CLI with a YAML config:
    comsol-opt run --config config.yaml
"""

from pathlib import Path

from comsol_opt import BayesianOptimizer, COMSOLRunner, OptimizationParameter

# ── Parameter definitions ──────────────────────────────────────────

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

# ── COMSOL configuration ──────────────────────────────────────────

MODEL_PATH = "teg_no_electrodes.mph"
COMSOL_EXE = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics_NSL\bin\win64\comsolbatch.exe"


def main() -> None:
    # Create the COMSOL objective function
    runner = COMSOLRunner(
        model_path=MODEL_PATH,
        parameters=PARAMETERS,
        comsol_exe=COMSOL_EXE,
        objective_name="power",
    )

    # Create the optimizer
    optimizer = BayesianOptimizer(
        parameters=PARAMETERS,
        objective=runner,
        objective_names=["power"],
        n_initial=10,
        n_iterations=50,
        acquisition="EI",
        maximize=True,
        seed=42,
        state_path=Path("optimization_state.json"),
    )

    # Run the optimization
    state = optimizer.run()

    # Print results
    print(f"\nOptimization complete. {state.n_completed} evaluations.")
    print(f"Best power: {state.best_objective:.6g}")
    print("Best parameters:")
    for name, val in state.best_parameters.items():
        print(f"  {name}: {val:.6g}")


if __name__ == "__main__":
    main()
