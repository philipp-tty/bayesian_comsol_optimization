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
from .sweep import build_sweep_grids, combo_key

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
        "--points", type=int, default=None,
        help=(
            "Number of equally spaced points for active parameters without "
            "explicit sweep values."
        ),
    )
    sweep_parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to save sweep state (default: sweep_state.json).",
    )
    sweep_parser.add_argument(
        "--workers", "--instances", dest="workers", type=int, default=None,
        help="Number of parallel COMSOL instances (default: sweep.workers or 1).",
    )
    sweep_parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a previous sweep state file to resume from.",
    )
    sweep_parser.add_argument(
        "--retry-failed", action="store_true",
        help="When resuming, rerun previously failed sweep points.",
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
    import concurrent.futures
    import itertools
    import os
    import queue as _queue
    import time

    import torch

    from .objective import EvaluationResult
    from .transforms import FillFactorTransform, LinearParameterTransform

    config = _load_config(args.config)
    sweep_cfg = config.get("sweep", {}) or {}
    if not isinstance(sweep_cfg, dict):
        print("Error: config key 'sweep' must be a mapping.", file=sys.stderr)
        sys.exit(1)

    workers = args.workers if args.workers is not None else sweep_cfg.get("workers", 1)
    try:
        workers = int(workers)
    except (TypeError, ValueError):
        print("Error: --workers / sweep.workers must be an integer.", file=sys.stderr)
        sys.exit(1)
    if workers < 1:
        print("Error: --workers / --instances must be at least 1.", file=sys.stderr)
        sys.exit(1)

    points = args.points if args.points is not None else sweep_cfg.get("points")
    if points is not None:
        try:
            points = int(points)
        except (TypeError, ValueError):
            print("Error: --points / sweep.points must be an integer.", file=sys.stderr)
            sys.exit(1)
        if points < 2:
            print("Error: --points / sweep.points must be at least 2.", file=sys.stderr)
            sys.exit(1)

    results_path = args.results or Path("sweep_state.json")

    parameters = _build_parameters(config)

    obj_cfg = config.get("objectives", [{"name": "objective", "direction": "maximize"}])
    objective_names = [o["name"] for o in obj_cfg]
    maximize = [o.get("direction", "maximize") == "maximize" for o in obj_cfg]

    constant_defaults = {p.name: float(p.constant_value) for p in parameters if p.is_constant}

    try:
        active_params, param_grids, grid_sources = build_sweep_grids(
            parameters,
            config,
            points=points,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    total = 1
    for g in param_grids:
        total *= len(g)

    print(
        f"Sweep: {len(active_params)} active parameter(s), "
        f"{workers} COMSOL instance(s)."
    )
    for p, g in zip(active_params, param_grids):
        if grid_sources[p.name] == "explicit":
            preview = ", ".join(f"{value:.6g}" for value in g[:8])
            if len(g) > 8:
                preview += ", ..."
            print(f"  {p.name}: {len(g)} explicit value(s): {preview}")
        else:
            print(
                f"  {p.name}: {len(g)} generated point(s) from {g[0]:.6g} to {g[-1]:.6g}"
                + (" [log]" if p.log_scale else "")
            )
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

    # Result accumulators (written only from the main thread)
    X_list: list[list[float]] = []
    Y_list: list[list[float]] = []
    X_physical: dict[str, list[float]] = {p.name: [] for p in parameters}
    success_mask: list[bool] = []
    evaluation_metadata: list[object] = []

    # Resume: load existing state and record which combos are already done
    done_combos: set[tuple] = set()
    if args.resume:
        if not args.resume.is_file():
            print(f"Error: resume file not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        existing = OptimizationState.load(args.resume)
        X_list = existing.X.tolist() if existing.X.numel() > 0 else []
        Y_list = existing.Y.tolist() if existing.Y.numel() > 0 else []
        X_physical = {k: list(v) for k, v in existing.X_physical.items()}
        success_mask = list(existing.success_mask)
        raw_metadata = existing.metadata.get("evaluation_metadata", [])
        evaluation_metadata = list(raw_metadata) if isinstance(raw_metadata, list) else []
        for i in range(len(success_mask)):
            if args.retry_failed and not success_mask[i]:
                continue
            key = combo_key(existing.X_physical[p.name][i] for p in active_params)
            done_combos.add(key)
        print(
            f"Resuming: {len(done_combos)} evaluation(s) already recorded, "
            f"{total - len(done_combos)} remaining."
        )

    def _record(physical: dict[str, float], result) -> None:
        unit_values = []
        for p in active_params:
            unit_val = float(transforms[p.name].to_unit(physical[p.name]))
            unit_val = max(0.0, min(1.0, unit_val))
            unit_values.append(unit_val)
        X_list.append(unit_values)
        Y_list.append([result.objectives.get(name, float("nan")) for name in objective_names])
        for p in parameters:
            X_physical[p.name].append(
                float(physical.get(p.name, constant_defaults.get(p.name, float("nan"))))
            )
        success_mask.append(result.success)
        evaluation_metadata.append(result.metadata)

    def _save() -> None:
        OptimizationState(
            parameters=parameters,
            objective_names=objective_names,
            X=torch.tensor(X_list, dtype=torch.double),
            Y=torch.tensor(Y_list, dtype=torch.double),
            X_physical={k: list(v) for k, v in X_physical.items()},
            success_mask=list(success_mask),
            metadata={
                "mode": "sweep",
                "sweep_points": points,
                "sweep_total": total,
                "sweep_workers": workers,
                "sweep_grid_sources": grid_sources,
                "evaluation_metadata": list(evaluation_metadata),
                "updated_timestamp": time.time(),
            },
            maximize=maximize,
        ).save(results_path)

    def _physical_from_combo(combo: tuple[float, ...]) -> dict[str, float]:
        physical: dict[str, float] = dict(constant_defaults)
        for p, val in zip(active_params, combo):
            physical[p.name] = p.coerce_physical_value(val)
        return physical

    def _failure_result(error: BaseException | str) -> EvaluationResult:
        return EvaluationResult(
            objectives={name: float("nan") for name in objective_names},
            success=False,
            metadata={"error": str(error)},
        )

    def _iter_pending_combos():
        for combo in itertools.product(*param_grids):
            if combo_key(combo) not in done_combos:
                yield combo

    remaining = sum(1 for _ in _iter_pending_combos())

    total_cores = os.cpu_count() or 1
    cores_per_worker = max(1, total_cores // workers)
    if workers > 1:
        print(
            f"Cores per COMSOL instance: {cores_per_worker} "
            f"(detected {total_cores} logical cores)"
        )

    from datetime import datetime
    import shutil

    sweep_run_dir = Path(f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    sweep_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep working directory: {sweep_run_dir}")

    source_model = Path(config["comsol"]["model"]).resolve()
    if not source_model.is_file():
        print(f"Error: COMSOL model not found: {source_model}", file=sys.stderr)
        sys.exit(1)

    model_copies: list[Path] = []

    def _stage_worker(i: int) -> tuple[Path, Path]:
        # With a single worker, run inside the sweep dir directly to avoid
        # an unnecessary nesting level. Multi-worker runs need per-worker
        # subdirs to keep output.txt / comsol_batch.log from colliding.
        wd = sweep_run_dir if workers == 1 else sweep_run_dir / f"worker_{i}"
        wd.mkdir(parents=True, exist_ok=True)
        model_copy = wd / source_model.name
        shutil.copy2(source_model, model_copy)
        model_copies.append(model_copy)
        return wd, model_copy

    def _cleanup_models() -> None:
        for path in model_copies:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to delete model copy %s: %s", path, exc)

    try:
        if workers == 1:
            wd, model_copy = _stage_worker(0)
            objective = _build_objective(
                config,
                parameters,
                working_dir=wd,
                n_cores=cores_per_worker,
                model_path=model_copy,
            )
            for combo in _iter_pending_combos():
                physical = _physical_from_combo(combo)
                print(
                    f"[{len(success_mask) + 1}/{total}] "
                    + ", ".join(f"{p.name}={physical[p.name]:.6g}" for p in active_params)
                )
                try:
                    result = objective.evaluate(physical)
                except Exception as exc:
                    logger.exception("Sweep evaluation failed unexpectedly.")
                    result = _failure_result(exc)
                _record(physical, result)
                _save()
        else:
            runner_pool: _queue.Queue = _queue.Queue()
            for i in range(workers):
                wd, model_copy = _stage_worker(i)
                runner_pool.put(
                    _build_objective(
                        config,
                        parameters,
                        working_dir=wd,
                        n_cores=cores_per_worker,
                        model_path=model_copy,
                    )
                )

            def _evaluate(combo: tuple[float, ...]) -> tuple[dict[str, float], EvaluationResult]:
                physical = _physical_from_combo(combo)
                runner = runner_pool.get()
                try:
                    try:
                        result = runner.evaluate(physical)
                    except Exception as exc:
                        logger.exception("Sweep evaluation failed unexpectedly.")
                        result = _failure_result(exc)
                finally:
                    runner_pool.put(runner)
                return physical, result

            def _submit_next(
                executor: concurrent.futures.ThreadPoolExecutor,
                combo_iter,
                futures: dict[concurrent.futures.Future, tuple[float, ...]],
            ) -> bool:
                try:
                    combo = next(combo_iter)
                except StopIteration:
                    return False
                futures[executor.submit(_evaluate, combo)] = combo
                return True

            combo_iter = _iter_pending_combos()
            futures: dict[concurrent.futures.Future, tuple[float, ...]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in range(min(workers, remaining)):
                    _submit_next(executor, combo_iter, futures)

                while futures:
                    done, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        combo = futures.pop(future)
                        physical = _physical_from_combo(combo)
                        try:
                            physical, result = future.result()
                        except Exception as exc:
                            logger.exception("Sweep worker future failed unexpectedly.")
                            result = _failure_result(exc)
                        print(
                            f"[{len(success_mask) + 1}/{total}] "
                            + ", ".join(f"{p.name}={physical[p.name]:.6g}" for p in active_params)
                        )
                        _record(physical, result)
                        _save()
                        _submit_next(executor, combo_iter, futures)
    finally:
        _cleanup_models()

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
    working_dir: Path | None = None,
    n_cores: int | None = None,
    model_path: str | Path | None = None,
) -> ObjectiveFunction:
    comsol_cfg = config.get("comsol")
    if comsol_cfg is None:
        raise ValueError(
            "Config must include a 'comsol' section with model and executable paths."
        )
    return COMSOLRunner(
        model_path=model_path if model_path is not None else comsol_cfg["model"],
        parameters=parameters,
        comsol_exe=comsol_cfg["executable"],
        methodcall=comsol_cfg.get("methodcall", "methodcall2"),
        timeout=comsol_cfg.get("timeout", 6000.0),
        objective_name=config.get("objectives", [{"name": "objective"}])[0]["name"],
        working_dir=working_dir,
        n_cores=n_cores,
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
