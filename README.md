# COMSOL Thermoelectric Optimization

This project provides a small Python package and CLI entry point that wraps a COMSOL model
with a scikit-optimize powered Bayesian optimization workflow. It started as a single script and has been refactored into
a modular Python package for easier maintenance.

## Project Layout

- `src/comsol_opt/`: Core package with reusable modules.
- `scripts/run_optimization.py`: CLI entry point mirroring the original script behaviour.

## Usage

1. Install the package in editable mode: `pip install -e .`
2. Run the optimization either via `python -m scripts.run_optimization` or the console script
   `comsol-opt-run`. The CLI accepts `--results-path`, `--resume-from`, and `--autosave-interval`
   flags so runs can emit incremental snapshots and later continue from the same parametrization.

Adjust the constants in `scripts/run_optimization.py` to point at your COMSOL installation and model.

### Incremental Results & Resuming

- `optimize_model` can persist its state after each evaluation by passing `results_path` (and
  optionally `autosave_interval`). The snapshots include all histories plus GP training payloads.
- Supply `resume_path` to continue an interrupted run; the optimizer validates that the parameter
  configuration matches before resuming and automatically replays completed evaluations so the
  surrogate model stays in sync.
- The CLI exposes these features via the flags mentioned above. For example:
  ```bash
  python -m scripts.run_optimization --results-path runs/run01.json --autosave-interval 2
  # later
  python -m scripts.run_optimization --resume-from runs/run01.json
  ```
- Logging now includes timestamps and per-iteration progress with elapsed time and ETA, so long
  optimizations are easier to monitor.

## Parameter Configuration

Optimization variables are declared via `OptimizationParameter`. Continuous parameters are the default, but you can request integral handling by setting `value_type` to `"integer"`, `"even_integer"`, or `"odd_integer"`. Integral parameters are rounded to the nearest in-bounds value that satisfies the requested parity before each COMSOL evaluation.

```python
from comsol_opt import OptimizationParameter

OptimizationParameter(
    name="n_turns",
    bounds=(8, 24),
    comsol_name="coil_turns",
    unit=None,
    value_type="even_integer",
    transform="linear",
)
```

Parameters that should remain fixed during the optimization can be declared with a `constant_value`. They will still be forwarded to COMSOL on every evaluation but are excluded from the search space.

```python
OptimizationParameter(
    name="environment_temp",
    bounds=(293.0, 293.0),
    comsol_name="ambient_temp",
    unit="K",
    constant_value=293.0,
)
```

## Example Optimization Loop

Below is a minimal example that wires up two optimized parameters plus one constant and launches an optimization run. Adjust the COMSOL paths, parameter names, and bounds to match your model.

```python
from comsol_opt import OptimizationParameter, optimize_model

MODEL_PATH = "path/to/your_model.mph"
COMSOL_EXE = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsolbatch.exe"

parameters = [
    OptimizationParameter(
        name="fill_factor",
        bounds=(0.05, 0.35),
        comsol_name="fill_factor",
        transform="fill_factor",
    ),
    OptimizationParameter(
        name="leg_count",
        bounds=(6, 18),
        comsol_name="n_legs",
        value_type="even_integer",  # enforce even number of legs
    ),
    OptimizationParameter(
        name="r_load",
        bounds=(1.0, 5.0),
        comsol_name="r_load",
        unit="ohm",
        constant_value=2.5,  # held fixed while still sent to COMSOL
    ),
]

results = optimize_model(
    model_path=MODEL_PATH,
    n_initial=5,
    n_iterations=20,
    random_seed=123,
    comsol_exe_path=COMSOL_EXE,
    methodcall="methodcall2",
    parameters=parameters,
)

print(results["objective"])
print(results["parameters"])
```
