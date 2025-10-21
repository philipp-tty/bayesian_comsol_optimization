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

