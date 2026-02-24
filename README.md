# comsol-opt

Bayesian optimization for black-box objective functions using
[BoTorch](https://botorch.org/), with built-in support for driving COMSOL
Multiphysics simulations.

The optimizer itself is generic: any callable or object implementing the
`ObjectiveFunction` protocol can be optimized. The `comsol.COMSOLRunner`
class is one such implementation that executes COMSOL models via the CLI
and parses their output.

Features:
- Single- and multi-objective optimization
- Batch candidate generation (q-batch acquisition functions)
- Resumable checkpointing via `OptimizationState`
- `ask()`/`tell()` interface for manual loop control
- Config-driven CLI for COMSOL workflows

## Project Layout

```
src/comsol_opt/
├── __init__.py              # Public API exports
├── parameters.py            # OptimizationParameter dataclass
├── transforms.py            # Linear and fill-factor coordinate transforms
├── objective.py             # ObjectiveFunction protocol + EvaluationResult
├── state.py                 # OptimizationState (save/load/resume)
├── surrogate.py             # GP surrogate (BoTorch SingleTaskGP / ModelListGP)
├── acquisition.py           # Acquisition function factory + optimization
├── optimizer.py             # BayesianOptimizer (core optimization loop)
├── cli.py                   # CLI entry point (run / analyze)
├── comsol/
│   ├── runner.py            # COMSOLRunner (ObjectiveFunction for COMSOL)
│   └── parser.py            # Output file parsing
└── analysis/
    ├── dataset.py           # Data loading, outlier filtering
    ├── plots.py             # Slice, contour, convergence, Pareto plots
    └── expressions.py       # Safe AST-based parameter expression evaluation
examples/
└── comsol_thermoelectric.py # Example: COMSOL thermoelectric generator
```

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/philipp-tty/bayesian_comsol_optimization.git
```

Or clone the repository and install in editable mode for development:

```bash
git clone https://github.com/philipp-tty/bayesian_comsol_optimization.git
cd bayesian_comsol_optimization
pip install -e .
```

Dependencies (`botorch`, `gpytorch`, `torch`, `numpy`, `matplotlib`,
`pyyaml`) are installed automatically.

## Quick Start — Custom Objective Function

Any callable that accepts a parameter dict and returns a float works as an
objective:

```python
from pathlib import Path
from comsol_opt import BayesianOptimizer, OptimizationParameter

parameters = [
    OptimizationParameter(name="x", bounds=(-5.0, 10.0)),
    OptimizationParameter(name="y", bounds=(0.0, 15.0)),
]

def branin(params):
    """Branin test function (minimization)."""
    import math
    x, y = params["x"], params["y"]
    a, b, c = 1, 5.1 / (4 * math.pi**2), 5 / math.pi
    r, s, t = 6, 10, 1 / (8 * math.pi)
    return a * (y - b * x**2 + c * x - r)**2 + s * (1 - t) * math.cos(x) + s

optimizer = BayesianOptimizer(
    parameters=parameters,
    objective=branin,
    n_initial=5,
    n_iterations=25,
    maximize=False,
    state_path=Path("branin_state.json"),
)

state = optimizer.run()
print(state.best_objective, state.best_parameters)
```

For more control, implement the `ObjectiveFunction` protocol and return
an `EvaluationResult` with named objectives, a success flag, and arbitrary
metadata.

## Quick Start — COMSOL

```python
from pathlib import Path
from comsol_opt import BayesianOptimizer, COMSOLRunner, OptimizationParameter

parameters = [
    OptimizationParameter(name="n_legs", bounds=(4, 12), value_type="even_integer"),
    OptimizationParameter(name="leg_spacing", bounds=(0.5, 2.0), unit="mm"),
    OptimizationParameter(name="leg_width", bounds=(0.5, 2.0), unit="mm"),
    OptimizationParameter(name="leg_length", bounds=(0.5, 4.0), unit="mm"),
    OptimizationParameter(name="r_load", bounds=(0.5, 10.0), unit="ohm"),
]

runner = COMSOLRunner(
    model_path="model.mph",
    parameters=parameters,
    comsol_exe=r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsolbatch.exe",
    objective_name="power",
)

optimizer = BayesianOptimizer(
    parameters=parameters,
    objective=runner,
    objective_names=["power"],
    n_initial=10,
    n_iterations=50,
    acquisition="EI",
    maximize=True,
    seed=42,
    state_path=Path("optimization_state.json"),
)

state = optimizer.run()
print(state.best_objective, state.best_parameters)
```

### Objective Name: Runner vs Optimizer

Both `COMSOLRunner` and `BayesianOptimizer` accept an objective name, but
they serve different purposes:

- **`COMSOLRunner(objective_name="power")`** — the runner produces an
  `EvaluationResult` whose `objectives` dict is keyed by this name. This is
  the label the runner assigns to the scalar it parses from the COMSOL
  output file. A custom `ObjectiveFunction` implementation would use
  whatever keys make sense for its domain.

- **`BayesianOptimizer(objective_names=["power"])`** — the optimizer
  *consumes* evaluation results and looks up values by these names. It
  needs to know which keys to read from the `objectives` dict returned by
  the objective function, and in what order, so it can build the training
  tensor for the GP surrogate.

The names must match: the optimizer's `objective_names` must be a subset of
the keys produced by the runner (or any other `ObjectiveFunction`). For
multi-objective problems the list contains multiple entries, one per
objective, defining both the names and the column order in the GP training
data.

### Windows Paths

On Windows, use a raw string (`r"..."`) or forward slashes for the COMSOL
executable path to avoid issues with backslash escapes:

```python
# Raw string (recommended)
comsol_exe = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsolbatch.exe"

# Forward slashes (also works)
comsol_exe = "C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolbatch.exe"

# pathlib
from pathlib import Path
comsol_exe = Path("C:/Program Files/COMSOL/COMSOL63/Multiphysics/bin/win64/comsolbatch.exe")
```

In a YAML config file, quoting is sufficient:

```yaml
comsol:
  executable: "C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics\\bin\\win64\\comsolbatch.exe"
```

### CLI

Create a YAML config file:

```yaml
comsol:
  model: path/to/model.mph
  executable: path/to/comsolbatch
  methodcall: methodcall2

optimization:
  n_initial: 10
  n_iterations: 50
  acquisition: EI
  maximize: true
  seed: 42

objectives:
  - name: power
    direction: maximize

parameters:
  - name: n_legs
    bounds: [4, 12]
    value_type: even_integer
  - name: leg_spacing
    bounds: [0.5, 2.0]
    unit: mm
  - name: leg_width
    bounds: [0.5, 2.0]
    unit: mm
  - name: leg_length
    bounds: [0.5, 4.0]
    unit: mm
  - name: r_load
    bounds: [0.5, 10.0]
    unit: ohm
```

Then run:

```bash
comsol-opt run --config config.yaml
comsol-opt run --config config.yaml --resume optimization_state.json
comsol-opt analyze --results optimization_state.json --output-dir analysis/
```

## Parameter Configuration

Optimization variables are declared via `OptimizationParameter`:

```python
OptimizationParameter(
    name="n_turns",            # Internal identifier
    bounds=(8, 24),            # Physical-domain bounds
    comsol_name="coil_turns",  # COMSOL name (defaults to name if omitted)
    unit=None,                 # Optional unit string, e.g. "mm"
    value_type="even_integer", # "continuous", "integer", "even_integer", "odd_integer"
    transform="linear",        # "linear" or "fill_factor"
    log_scale=False,           # Explore in log space
)
```

`comsol_name` is optional. When omitted it defaults to `name`, which is
sufficient when the internal identifier matches the COMSOL model parameter.
Set it explicitly when the two differ.

Parameters with `constant_value` are forwarded to the objective function but
excluded from the search space.

## ask/tell Interface

For manual control over the optimization loop:

```python
optimizer = BayesianOptimizer(parameters=parameters, objective=my_fn, ...)

# Get suggested candidates
candidates = optimizer.ask(n=1)

# Evaluate externally
result = my_fn.evaluate(candidates[0])

# Feed back the result
optimizer.tell(candidates[0], result)

# Access current state at any time
state = optimizer.state
```

## Checkpointing and Resume

Pass `state_path` to auto-save after each evaluation:

```python
optimizer = BayesianOptimizer(..., state_path=Path("state.json"))
state = optimizer.run()  # auto-saves throughout
```

Resume from a saved state:

```python
from comsol_opt import OptimizationState

state = OptimizationState.load(Path("state.json"))
final_state = optimizer.resume(state)
```

## Multi-Objective Optimization

Configure multiple objectives and a reference point:

```python
optimizer = BayesianOptimizer(
    parameters=parameters,
    objective=runner,
    objective_names=["power", "cost"],
    maximize=[True, False],  # maximize power, minimize cost
    acquisition="EHVI",
    ref_point=[0.0, 100.0],  # worst acceptable values per objective
)
```

The objective function must return an `EvaluationResult` whose `objectives`
dict contains all names listed in `objective_names`.

Supported acquisition functions:
- **Single-objective:** `EI`, `UCB`, `qEI`, `qUCB`, `KG`
- **Multi-objective:** `EHVI`, `ParEGO`

## Analysis

Generate diagnostic plots from a saved optimization state:

```bash
comsol-opt analyze --results state.json \
    --output-dir analysis/ \
    --iso-x-parameter leg_spacing \
    --parameter-expression "n_legs * leg_width"
```

Generated plots:
- `mean_uncertainty_*.png` — GP mean +/- CI vs each parameter
- `iso_contour_*.png` — 2D GP mean contour plots
- `convergence.png` — Objective vs iteration with cumulative best
- `parallel_coordinates.png` — All parameters + objective
- `pareto_front.png` — Pareto front (multi-objective only)
- `outputs_vs_expression.png` — Objective vs a custom parameter expression
