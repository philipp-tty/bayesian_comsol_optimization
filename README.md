# COMSOL Thermoelectric Optimization

This project provides a small Python package and CLI entry point that wraps a COMSOL model
with Bayesian optimization utilities. It started as a single script and has been refactored into
a modular Python package for easier maintenance.

## Project Layout

- `src/comsol_opt/`: Core package with reusable modules.
- `scripts/run_optimization.py`: CLI entry point mirroring the original script behaviour.

## Usage

1. Install the package in editable mode: `pip install -e .`
2. Run the optimization either via `python -m scripts.run_optimization` or the console script
   `comsol-opt-run`.

Adjust the constants in `scripts/run_optimization.py` to point at your COMSOL installation and model.

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
)
```
