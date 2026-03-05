"""CLI entry point for comsol-opt: run optimization and analyze results."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .analysis.dataset import filter_outliers, load_state, state_to_gp_arrays
from .analysis.expressions import evaluate_parameter_expression
from .analysis.plots import (
    plot_convergence,
    plot_expression,
    plot_iso_contours,
    plot_parallel_coordinates,
    plot_parameter_slices,
    plot_pareto_front,
)
from .comsol.runner import COMSOLRunner
from .objective import ObjectiveFunction
from .optimizer import BayesianOptimizer
from .parameters import OptimizationParameter
from .state import OptimizationState
from .surrogate import GPSurrogate

logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="comsol-opt",
        description="Bayesian optimization for COMSOL simulations using BoTorch.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an optimization.")
    run_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML configuration file.",
    )
    run_parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to save optimization state (default: optimization_state.json).",
    )
    run_parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a previous state file to resume from.",
    )

    # --- sweep ---
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run a full parameter sweep without optimization.",
    )
    sweep_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML configuration file.",
    )
    sweep_parser.add_argument(
        "--points", type=int, required=True,
        help="Number of equally spaced points per active parameter.",
    )
    sweep_parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to save sweep state (default: sweep_state.json).",
    )

    # --- analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze optimization results.")
    analyze_parser.add_argument(
        "--results", type=Path, required=True,
        help="Path to the optimization state JSON file.",
    )
    analyze_parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis"),
        help="Directory for output plots and summary.",
    )
    analyze_parser.add_argument(
        "--grid-points", type=int, default=200,
        help="Grid resolution for GP slice plots.",
    )
    analyze_parser.add_argument(
        "--ci-multiplier", type=float, default=1.96,
        help="CI multiplier (default 1.96 for 95%%).",
    )
    analyze_parser.add_argument(
        "--outlier-method", type=str, default="zscore",
        choices=("none", "mad", "iqr", "zscore"),
        help="Outlier filtering method.",
    )
    analyze_parser.add_argument(
        "--outlier-threshold", type=float, default=3.5,
        help="Outlier threshold.",
    )
    analyze_parser.add_argument(
        "--iso-x-parameter", type=str, default=None,
        help="Parameter name for x-axis of iso contour plots.",
    )
    analyze_parser.add_argument(
        "--iso-grid-points", type=int, default=60,
        help="Grid resolution for contour plots.",
    )
    analyze_parser.add_argument(
        "--no-parallel-coordinates", action="store_true",
        help="Skip parallel coordinates plot.",
    )
    analyze_parser.add_argument(
        "--parameter-expression", type=str, default=None,
        help="Expression to evaluate and plot against objective.",
    )
    analyze_parser.add_argument(
        "--parameter-expression-label", type=str, default=None,
        help="Label for the expression axis.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "analyze":
        _cmd_analyze(args)


# ------------------------------------------------------------------
# run command
# ------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    results_path = args.results or Path("optimization_state.json")

    parameters = _build_parameters(config)
    objective = _build_objective(config, parameters)

    opt_cfg = config.get("optimization", {})
    obj_cfg = config.get("objectives", [{"name": "objective", "direction": "maximize"}])

    objective_names = [o["name"] for o in obj_cfg]
    maximize = [o.get("direction", "maximize") == "maximize" for o in obj_cfg]

    ref_point = opt_cfg.get("ref_point")

    optimizer = BayesianOptimizer(
        parameters=parameters,
        objective=objective,
        objective_names=objective_names,
        n_initial=opt_cfg.get("n_initial", 5),
        n_iterations=opt_cfg.get("n_iterations", 20),
        batch_size=opt_cfg.get("batch_size", 1),
        acquisition=opt_cfg.get("acquisition", "EI"),
        maximize=maximize if len(maximize) > 1 else maximize[0],
        seed=opt_cfg.get("seed"),
        state_path=results_path,
        autosave_interval=opt_cfg.get("autosave_interval", 1),
        ref_point=ref_point,
    )

    if args.resume:
        state = OptimizationState.load(args.resume)
        final_state = optimizer.resume(state)
    else:
        final_state = optimizer.run()

    print(f"\nOptimization complete. {final_state.n_completed} evaluations.")
    if not final_state.is_multi_objective:
        print(f"Best objective: {final_state.best_objective:.6g}")
        print("Best parameters:")
        for name, val in final_state.best_parameters.items():
            print(f"  {name}: {val:.6g}")
    print(f"State saved to {results_path}")


def _cmd_sweep(args: argparse.Namespace) -> None:
    import itertools
    import math

    import numpy as np
    import torch

    from .transforms import FillFactorTransform, LinearParameterTransform

    if args.points < 2:
        print("Error: --points must be at least 2.", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    results_path = args.results or Path("sweep_state.json")

    parameters = _build_parameters(config)
    objective = _build_objective(config, parameters)

    obj_cfg = config.get("objectives", [{"name": "objective", "direction": "maximize"}])
    objective_names = [o["name"] for o in obj_cfg]
    maximize = [o.get("direction", "maximize") == "maximize" for o in obj_cfg]

    active_params = [p for p in parameters if not p.is_constant]
    constant_defaults = {p.name: float(p.constant_value) for p in parameters if p.is_constant}

    if not active_params:
        print("Error: no active (non-constant) parameters found in config.", file=sys.stderr)
        sys.exit(1)

    # Build per-parameter grids
    param_grids: list[list[float]] = []
    for p in active_params:
        lo, hi = p.bounds
        if p.log_scale:
            values = np.logspace(math.log10(lo), math.log10(hi), args.points).tolist()
        else:
            values = np.linspace(lo, hi, args.points).tolist()

        # Coerce to integer / parity constraints and deduplicate while preserving order
        if p.is_integer:
            seen: set[float] = set()
            coerced_values: list[float] = []
            for v in values:
                cv = p.coerce_physical_value(v)
                if cv not in seen:
                    seen.add(cv)
                    coerced_values.append(cv)
            values = coerced_values

        param_grids.append(values)

    total = 1
    for g in param_grids:
        total *= len(g)

    print(f"Sweep: {len(active_params)} active parameter(s), {args.points} points each.")
    for p, g in zip(active_params, param_grids):
        print(f"  {p.name}: {len(g)} points from {g[0]:.6g} to {g[-1]:.6g}"
              + (" [log]" if p.log_scale else ""))
    print(f"Total evaluations: {total}")

    if total > 10_000:
        print(
            f"Warning: {total} evaluations is very large. "
            "Consider reducing --points or fixing some parameters as constants.",
        )

    # Build transforms for unit-space conversion
    transforms = {}
    for p in active_params:
        if p.transform == "fill_factor":
            transforms[p.name] = FillFactorTransform(p.bounds)
        else:
            transforms[p.name] = LinearParameterTransform(p.bounds)

    # Run sweep
    X_list: list[list[float]] = []
    Y_list: list[list[float]] = []
    X_physical: dict[str, list[float]] = {p.name: [] for p in parameters}
    success_mask: list[bool] = []

    for i, combo in enumerate(itertools.product(*param_grids), start=1):
        physical: dict[str, float] = dict(constant_defaults)
        for p, val in zip(active_params, combo):
            physical[p.name] = val

        print(
            f"[{i}/{total}] "
            + ", ".join(f"{p.name}={physical[p.name]:.6g}" for p in active_params)
        )

        result = objective.evaluate(physical)

        # Convert to unit space
        unit_values = []
        for p in active_params:
            transform = transforms[p.name]
            unit_val = float(transform.to_unit(physical[p.name]))
            unit_val = max(0.0, min(1.0, unit_val))
            unit_values.append(unit_val)

        X_list.append(unit_values)
        obj_values = [result.objectives.get(name, float("nan")) for name in objective_names]
        Y_list.append(obj_values)

        for p in parameters:
            X_physical[p.name].append(float(physical.get(p.name, constant_defaults.get(p.name, float("nan")))))

        success_mask.append(result.success)

        # Autosave after each evaluation
        state = OptimizationState(
            parameters=parameters,
            objective_names=objective_names,
            X=torch.tensor(X_list, dtype=torch.double),
            Y=torch.tensor(Y_list, dtype=torch.double),
            X_physical={k: list(v) for k, v in X_physical.items()},
            success_mask=list(success_mask),
            metadata={"sweep_points": args.points, "sweep_total": total},
            maximize=maximize,
        )
        state.save(results_path)

    n_success = sum(success_mask)
    print(f"\nSweep complete. {n_success}/{total} evaluations succeeded.")
    print(f"State saved to {results_path}")


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return config


def _build_parameters(config: dict) -> list[OptimizationParameter]:
    params_cfg = config.get("parameters", [])
    if not params_cfg:
        raise ValueError("Config must define at least one parameter.")
    parameters = []
    for p in params_cfg:
        parameters.append(OptimizationParameter(
            name=p["name"],
            bounds=tuple(p["bounds"]),
            comsol_name=p.get("comsol_name"),
            unit=p.get("unit"),
            value_type=p.get("value_type", "continuous"),
            transform=p.get("transform", "linear"),
            constant_value=p.get("constant_value"),
            log_scale=p.get("log_scale", False),
        ))
    return parameters


def _build_objective(
    config: dict,
    parameters: list[OptimizationParameter],
) -> ObjectiveFunction:
    comsol_cfg = config.get("comsol")
    if comsol_cfg is None:
        raise ValueError(
            "Config must include a 'comsol' section with model and executable paths."
        )
    return COMSOLRunner(
        model_path=comsol_cfg["model"],
        parameters=parameters,
        comsol_exe=comsol_cfg["executable"],
        methodcall=comsol_cfg.get("methodcall", "methodcall2"),
        timeout=comsol_cfg.get("timeout", 6000.0),
        objective_name=config.get("objectives", [{"name": "objective"}])[0]["name"],
    )


# ------------------------------------------------------------------
# analyze command
# ------------------------------------------------------------------


def _cmd_analyze(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    state = load_state(args.results)
    print(f"Loaded state with {state.n_completed} evaluations from '{args.results}'.")

    if state.n_completed == 0:
        print("No evaluations to analyze.")
        return

    # Work with the first objective by default
    objective_index = 0
    X_success, Y_success = state_to_gp_arrays(state, objective_index=objective_index)

    if X_success.shape[0] == 0:
        print("No successful evaluations to analyze.")
        return

    # Outlier filtering
    X_filtered, Y_filtered, outlier_stats, retained_mask = filter_outliers(
        X_success, Y_success,
        method=args.outlier_method,
        threshold=args.outlier_threshold,
    )

    if outlier_stats["removed"]:
        print(
            f"Filtered {outlier_stats['removed']} outlier(s) using "
            f"'{outlier_stats['method']}' (threshold={outlier_stats['threshold']})."
        )

    if Y_filtered.size == 0:
        print("All evaluations were filtered as outliers.")
        return

    # Report best
    obj_name = state.objective_names[objective_index]
    maximize = state.maximize[objective_index] if objective_index < len(state.maximize) else True
    if maximize:
        best_idx = int(np.argmax(Y_filtered))
    else:
        best_idx = int(np.argmin(Y_filtered))
    print(f"Best {obj_name}: {Y_filtered[best_idx]:.6g}")

    # Build a filtered state for plotting (only successful + non-outlier)
    # We re-create a state with the filtered data for the surrogate and plots
    mask_full = torch.tensor(state.success_mask, dtype=torch.bool)
    success_indices = torch.where(mask_full)[0]
    retained_full_indices = success_indices[torch.tensor(retained_mask, dtype=torch.bool)]

    # Create filtered state-like object for plotting
    filtered_state = OptimizationState(
        parameters=state.parameters,
        objective_names=state.objective_names,
        X=torch.tensor(X_filtered, dtype=torch.double),
        Y=torch.tensor(Y_filtered, dtype=torch.double).unsqueeze(-1),
        X_physical={},  # not needed for plots that use X
        success_mask=[True] * len(Y_filtered),
        metadata=state.metadata,
        maximize=state.maximize,
    )

    # Fit GP surrogate for analysis
    if X_filtered.shape[0] < 2:
        print("Not enough data points to fit a GP surrogate (need at least 2).")
        return

    surrogate = GPSurrogate(n_objectives=1)
    X_tensor = torch.tensor(X_filtered, dtype=torch.double)
    Y_tensor = torch.tensor(Y_filtered, dtype=torch.double).unsqueeze(-1)
    surrogate.fit(X_tensor, Y_tensor)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    slice_paths = plot_parameter_slices(
        filtered_state, surrogate, args.output_dir,
        grid_points=max(10, args.grid_points),
        ci_multiplier=max(0.0, args.ci_multiplier),
        objective_index=0,
    )
    print(f"Generated {len(slice_paths)} mean & uncertainty plots in '{args.output_dir}'.")

    # Convergence plot (use full state)
    conv_path = plot_convergence(state, args.output_dir, objective_index=objective_index)
    print(f"Saved convergence plot to '{conv_path}'.")

    # Iso contours
    if args.iso_x_parameter:
        iso_paths = plot_iso_contours(
            filtered_state, surrogate, args.output_dir,
            x_param_name=args.iso_x_parameter,
            grid_points=max(5, args.iso_grid_points),
            objective_index=0,
        )
        print(f"Generated {len(iso_paths)} iso contour plot(s).")

    # Parallel coordinates
    if not args.no_parallel_coordinates:
        try:
            pc_path = plot_parallel_coordinates(
                filtered_state, args.output_dir, objective_index=0,
            )
            print(f"Saved parallel coordinates plot to '{pc_path}'.")
        except ValueError:
            pass

    # Expression plot
    if args.parameter_expression:
        active_params = [p for p in state.parameters if not p.is_constant]
        param_names = [p.name for p in active_params]
        # Convert filtered X back to physical
        physical = np.empty_like(X_filtered)
        for axis, p in enumerate(active_params):
            physical[:, axis] = p.bounds[0] + X_filtered[:, axis] * (p.bounds[1] - p.bounds[0])
        try:
            expr_values = evaluate_parameter_expression(
                args.parameter_expression, param_names, physical,
            )
            label = args.parameter_expression_label or args.parameter_expression
            expr_path = plot_expression(
                filtered_state, expr_values, args.output_dir,
                expression_label=label, objective_index=0,
            )
            print(f"Saved expression plot to '{expr_path}'.")
        except ValueError as exc:
            print(f"Failed to plot expression: {exc}")

    # Pareto front (if multi-objective)
    if state.is_multi_objective:
        pareto_path = plot_pareto_front(state, args.output_dir)
        if pareto_path:
            print(f"Saved Pareto front plot to '{pareto_path}'.")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
