"""
Simulate building a Gaussian process surrogate while deciding between a slow,
high-fidelity model and a fast, low-fidelity alternative at each step.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return scipy_norm.pdf(np.asarray(x, dtype=float))


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return scipy_norm.cdf(np.asarray(x, dtype=float))


@dataclass(frozen=True)
class FidelityLevel:
    name: str
    noise_variance: float  # Observation noise variance we expect from this model.
    cost: float  # Relative evaluation cost for budgeting decisions.


def true_function(x: np.ndarray) -> np.ndarray:
    """Hidden ground-truth response we want to learn with the surrogate."""
    x = np.asarray(x, dtype=float)
    return (
        np.sin(x)
        + 0.5 * np.sin(2.5 * x)
        + 0.1 * x
    )


def simulate_measurement(x: float, fidelity: FidelityLevel, rng: np.random.Generator) -> Tuple[float, float]:
    """Evaluate the chosen fidelity and return (observed value, true value)."""
    true_val = float(true_function(x))
    noise_std = math.sqrt(fidelity.noise_variance)
    observed = true_val + rng.normal(0.0, noise_std)
    return observed, true_val


def fit_surrogate(xs, ys, alphas) -> GaussianProcessRegressor:
    """Train a Gaussian process surrogate on the accumulated data."""
    X = np.asarray(xs, dtype=float).reshape(-1, 1)
    y = np.asarray(ys, dtype=float)
    alpha = np.asarray(alphas, dtype=float) + 1e-10
    kernel = ConstantKernel(1.0, (1e-2, 10.0)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=2,
        copy_X_train=False,
    )
    gp.fit(X, y)
    return gp


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_reference: float, exploration: float = 1e-3) -> np.ndarray:
    """Expected improvement acquisition function for maximisation."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = mu - float(best_reference) - exploration
    safe_sigma = np.maximum(sigma, 1e-12)
    Z = improvement / safe_sigma
    cdf = normal_cdf(Z)
    pdf = normal_pdf(Z)
    ei = improvement * cdf + safe_sigma * pdf
    ei[sigma <= 1e-12] = 0.0
    return np.maximum(ei, 0.0)


def choose_fidelity(posterior_sigma: float, fidelities: Dict[str, FidelityLevel]) -> Tuple[FidelityLevel, Dict[str, float]]:
    """
    Pick the fidelity that offers the best projected variance reduction per unit cost.

    We approximate the reduction with sigma^2 / (sigma^2 + noise_variance), which captures how
    observation noise limits the benefit of the sample. Higher ratios (per cost) are preferred.
    """
    if posterior_sigma <= 1e-9:
        return fidelities["low"], {name: 0.0 for name in fidelities}

    sigma_sq = posterior_sigma ** 2
    best_choice = None
    best_score = -np.inf
    scores: Dict[str, float] = {}

    for fidelity in fidelities.values():
        info_gain = sigma_sq / (sigma_sq + fidelity.noise_variance)
        score = info_gain / fidelity.cost
        scores[fidelity.name] = score
        if score > best_score or (math.isclose(score, best_score) and fidelity.name == "high"):
            best_score = score
            best_choice = fidelity

    return best_choice, scores


def run_simulation(
    n_iterations: int = 25,
    random_state: int | None = 7,
    grid_size: int = 400,
    seed_mode: str = "low",
) -> Dict[str, Any]:
    """Run the adaptive sampling loop and print progress to the console."""
    rng = np.random.default_rng(random_state)
    bounds = (0.0, 10.0)

    fidelities = {
        "high": FidelityLevel(name="high", noise_variance=5e-4, cost=5.0),
        "low": FidelityLevel(name="low", noise_variance=0.05, cost=1.0),
    }

    xs: List[float] = []
    ys: List[float] = []
    alphas: List[float] = []
    history: List[Dict[str, Any]] = []

    total_cost = 0.0
    high_count = 0
    low_count = 0
    best_high_y = -np.inf
    best_high_x = None

    def add_sample(
        x: float,
        fidelity: FidelityLevel,
        stage: str,
        posterior_sigma: float | None = None,
    ) -> Tuple[float, float]:
        nonlocal total_cost, high_count, low_count, best_high_y, best_high_x
        observed, true_val = simulate_measurement(x, fidelity, rng)
        xs.append(float(x))
        ys.append(observed)
        alphas.append(fidelity.noise_variance)
        step_index = len(history) + 1
        history.append(
            {
                "step": step_index,
                "stage": stage,
                "x": float(x),
                "fidelity": fidelity.name,
                "observed": observed,
                "true": true_val,
                "posterior_sigma": float(posterior_sigma) if posterior_sigma is not None else float("nan"),
                "cost": fidelity.cost,
                "noise_variance": fidelity.noise_variance,
            }
        )
        total_cost += fidelity.cost
        if fidelity.name == "high":
            high_count += 1
            if observed > best_high_y:
                best_high_y = observed
                best_high_x = float(x)
        else:
            low_count += 1
        return observed, true_val

    def bootstrap_surrogate(mode: str) -> None:
        sequences = {
            "mixed": [fidelities["low"], fidelities["low"], fidelities["high"]],
            "low": [fidelities["low"]] * 3,
            "high": [fidelities["high"]] * 3,
        }
        normalized = mode.lower()
        if normalized not in sequences:
            expected = ", ".join(sorted(sequences))
            raise ValueError(f"Unknown seed_mode '{mode}'. Expected one of: {expected}.")

        sequence = sequences[normalized]
        print(f"Bootstrapping surrogate with {normalized} seeding ({len(sequence)} samples)...")
        for index, fidelity in enumerate(sequence, start=1):
            x0 = rng.uniform(*bounds)
            observed, true_val = add_sample(x0, fidelity, stage="seed")
            print(
                f"  Seed {index:02d} [{fidelity.name}] at x={x0:6.3f} -> "
                f"observed={observed:7.3f} (true={true_val:7.3f})"
            )

    bootstrap_surrogate(seed_mode)

    gp = fit_surrogate(xs, ys, alphas)

    for iteration in range(1, n_iterations + 1):
        grid = np.linspace(bounds[0], bounds[1], grid_size).reshape(-1, 1)
        mu, sigma = gp.predict(grid, return_std=True)
        reference = best_high_y if best_high_x is not None else max(ys)
        ei = expected_improvement(mu, sigma, reference)
        best_idx = int(np.argmax(ei))
        x_next = float(grid[best_idx, 0])
        sigma_next = float(sigma[best_idx])
        fidelity, scores = choose_fidelity(sigma_next, fidelities)
        observed, true_val = add_sample(x_next, fidelity, stage="adaptive", posterior_sigma=sigma_next)

        print(
            f"Iteration {iteration:2d} | "
            f"x={x_next:6.3f} | "
            f"fidelity={fidelity.name:4s} | "
            f"EI={ei[best_idx]:7.3f} | "
            f"sigma={sigma_next:6.3f} | "
            f"observed={observed:7.3f} | "
            f"true={true_val:7.3f} | "
            f"score_high={scores['high']:5.3f} | "
            f"score_low={scores['low']:5.3f}"
        )

        gp = fit_surrogate(xs, ys, alphas)

    grid_dense = np.linspace(bounds[0], bounds[1], 2000)
    grid_dense_2d = grid_dense.reshape(-1, 1)
    gp_mean, gp_sigma = gp.predict(grid_dense_2d, return_std=True)
    gp_mean = gp_mean.reshape(-1)
    best_gp_idx = int(np.argmax(gp_mean))
    x_gp = float(grid_dense[best_gp_idx])
    mu_gp = float(gp_mean[best_gp_idx])
    true_at_gp = float(true_function(x_gp))

    true_grid = true_function(grid_dense)
    best_true_idx = int(np.argmax(true_grid))
    x_true = float(grid_dense[best_true_idx])
    true_best_val = float(true_grid[best_true_idx])
    best_high_true = None
    if best_high_x is not None:
        best_high_true = float(true_function(best_high_x))

    print("\nSimulation summary")
    print("------------------")
    print(f"Total evaluations: {len(xs)} (high: {high_count}, low: {low_count})")
    print(f"Total cost units: {total_cost:7.3f}")
    if best_high_x is not None:
        print(
            f"Best high-fidelity sample at x={best_high_x:6.3f}: observed={best_high_y:7.3f}, "
            f"true={best_high_true:7.3f}"
        )
    else:
        print("No high-fidelity samples were collected.")
    print(f"Surrogate optimum estimate x~={x_gp:6.3f}, mean={mu_gp:7.3f}, true={true_at_gp:7.3f}")
    print(f"True optimum on fine grid x~={x_true:6.3f}, true value={true_best_val:7.3f}")

    return {
        "total_cost": total_cost,
        "high_evaluations": high_count,
        "low_evaluations": low_count,
        "surrogate_optimum_x": x_gp,
        "surrogate_optimum_mu": mu_gp,
        "true_at_surrogate_optimum": true_at_gp,
        "true_optimum_x": x_true,
        "true_optimum_value": true_best_val,
        "history": history,
        "grid_dense": grid_dense,
        "gp_mean": gp_mean,
        "gp_sigma": gp_sigma,
        "true_grid": true_grid,
        "best_high_x": best_high_x,
        "best_high_observed": best_high_y if best_high_x is not None else None,
        "best_high_true": best_high_true,
        "bounds": bounds,
    }


def plot_results(results: Dict[str, Any], save_path: str | None = None, show: bool = True) -> None:
    """
    Visualise the surrogate fit and sampling decisions for a completed simulation run.
    """
    history: List[Dict[str, Any]] = results.get("history", [])
    if not history:
        print("No recorded history to plot.")
        return

    grid = np.asarray(results["grid_dense"], dtype=float)
    gp_mean = np.asarray(results["gp_mean"], dtype=float)
    gp_sigma = np.asarray(results["gp_sigma"], dtype=float)
    true_grid = np.asarray(results["true_grid"], dtype=float)

    fidelity_styles = {
        "high": {"color": "tab:red", "marker": "o", "label": "High fidelity"},
        "low": {"color": "tab:blue", "marker": "s", "label": "Low fidelity"},
    }

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for entry in history:
        group = grouped.setdefault(
            entry["fidelity"],
            {"x": [], "observed": [], "step": [], "posterior_sigma": []},
        )
        group["x"].append(float(entry["x"]))
        group["observed"].append(float(entry["observed"]))
        group["step"].append(float(entry["step"]))
        group["posterior_sigma"].append(float(entry["posterior_sigma"]))

    steps = np.array([float(entry["step"]) for entry in history], dtype=float)
    posterior_sigmas = np.array([float(entry["posterior_sigma"]) for entry in history], dtype=float)
    costs = np.array([float(entry["cost"]) for entry in history], dtype=float)
    cumulative_costs = np.concatenate(([0.0], np.cumsum(costs)))
    cumulative_steps = np.concatenate(([0.0], steps))
    xs_all = np.array([float(entry["x"]) for entry in history], dtype=float)
    observed_all = np.array([float(entry["observed"]) for entry in history], dtype=float)
    alphas_all = np.array([float(entry["noise_variance"]) for entry in history], dtype=float)
    best_observed = np.maximum.accumulate(observed_all)

    grid_2d = grid.reshape(-1, 1)
    xs_so_far: List[float] = []
    observed_so_far: List[float] = []
    alphas_so_far: List[float] = []
    best_gp_mean: List[float] = []
    for x_val, y_val, alpha_val in zip(xs_all, observed_all, alphas_all):
        xs_so_far.append(float(x_val))
        observed_so_far.append(float(y_val))
        alphas_so_far.append(float(alpha_val))
        try:
            gp_partial = fit_surrogate(xs_so_far, observed_so_far, alphas_so_far)
            mu_partial = gp_partial.predict(grid_2d, return_std=False)
            best_gp_mean.append(float(np.max(mu_partial)))
        except Exception:
            best_gp_mean.append(float("nan"))

    best_gp_mean_array = np.array(best_gp_mean, dtype=float)

    # Use constrained layout so the twin axis and legend are not clipped.
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    ax_top.plot(grid, true_grid, color="black", linewidth=1.2, label="True function")
    ax_top.plot(grid, gp_mean, color="tab:orange", linewidth=1.4, label="GP mean")
    ax_top.fill_between(
        grid,
        gp_mean - 2.0 * gp_sigma,
        gp_mean + 2.0 * gp_sigma,
        color="tab:orange",
        alpha=0.2,
        label="GP ±2σ",
    )

    for fidelity, values in grouped.items():
        style = fidelity_styles.get(fidelity, {"color": "gray", "marker": "x", "label": fidelity})
        ax_top.scatter(
            values["x"],
            values["observed"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.6,
            s=60,
            zorder=3,
        )

    ax_top.set_ylabel("Response")
    ax_top.set_xlim(grid.min(), grid.max())
    ax_top.set_title("Surrogate fit after adaptive sampling")
    ax_top.legend(loc="upper left", frameon=False)
    ax_top.grid(alpha=0.3)

    valid = ~np.isnan(posterior_sigmas)
    if np.any(valid):
        ax_bottom.plot(
            steps[valid],
            posterior_sigmas[valid],
            color="tab:orange",
            linewidth=1.2,
            alpha=0.7,
            label="Posterior σ trend",
        )

    for fidelity, values in grouped.items():
        style = fidelity_styles.get(fidelity, {"color": "gray", "marker": "x", "label": fidelity})
        sigma_values = np.array(values["posterior_sigma"], dtype=float)
        valid_sigma = ~np.isnan(sigma_values)
        if np.any(valid_sigma):
            ax_bottom.scatter(
                np.array(values["step"], dtype=float)[valid_sigma],
                sigma_values[valid_sigma],
                color=style["color"],
                marker=style["marker"],
                s=55,
                alpha=0.9,
                label=f"{style['label']} σ",
            )

    ax_best = ax_bottom.twinx()
    ax_best.patch.set_visible(False)
    best_observed_line, = ax_best.plot(
        steps,
        best_observed,
        color="tab:red",
        linewidth=1.4,
        alpha=0.85,
        label="Best observed",
    )
    best_gp_line, = ax_best.plot(
        steps,
        best_gp_mean_array,
        color="tab:purple",
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        label="Best GP mean",
    )
    ax_best.set_ylabel("Best response")
    ax_best.grid(False)

    ax_cost = ax_bottom.twinx()
    ax_cost.patch.set_visible(False)
    ax_cost.spines["right"].set_position(("axes", 1.08))
    ax_cost.step(
        cumulative_steps,
        cumulative_costs,
        where="post",
        color="tab:green",
        linewidth=1.4,
        label="Cumulative cost",
    )

    max_step = float(np.nanmax(steps)) if steps.size else 0.0
    if max_step <= 0.0:
        max_step = float(len(history))
    ax_bottom.set_xlim(0.0, max_step + 0.5)
    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_bottom.set_xlabel("Sample index")
    ax_bottom.set_ylabel("Posterior σ at selection")
    ax_bottom.grid(alpha=0.3)
    ax_bottom.set_title("Acquisition choices and information gain proxy")
    ax_cost.set_ylabel("Cumulative cost (units)")
    ax_cost.grid(False)

    handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    handles_best, labels_best = ax_best.get_legend_handles_labels()
    handles_cost, labels_cost = ax_cost.get_legend_handles_labels()
    legend_handles = handles_bottom + handles_best + handles_cost
    legend_labels = labels_bottom + labels_best + labels_cost
    if legend_handles:
        ax_bottom.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=False,
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate adaptive fidelity selection with a Gaussian process.")
    parser.add_argument("--iterations", type=int, default=25, help="Adaptive iterations after seeding (default: 25).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--grid-size", type=int, default=400, help="Resolution used for candidate search.")
    parser.add_argument(
        "--seed-fidelity",
        choices=("mixed", "high", "low"),
        default="mixed",
        help="Fidelity level used for initial seeding (default: mixed).",
    )
    parser.add_argument("--plot", action="store_true", help="Display a matplotlib summary once the run finishes.")
    parser.add_argument("--save-fig", type=str, default=None, help="Path to save the summary figure (requires matplotlib).")
    args = parser.parse_args()

    results = run_simulation(
        n_iterations=args.iterations,
        random_state=args.seed,
        grid_size=args.grid_size,
        seed_mode=args.seed_fidelity,
    )

    if args.plot or args.save_fig:
        show_plot = bool(args.plot)
        plot_results(results, save_path=args.save_fig, show=show_plot or not args.save_fig)


if __name__ == "__main__":
    main()
