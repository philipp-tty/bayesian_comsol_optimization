"""Acquisition function selection and optimization."""

from __future__ import annotations

import logging

import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qKnowledgeGradient,
    qLogExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

logger = logging.getLogger(__name__)

# Supported acquisition function names
SINGLE_OBJECTIVE_ACQFS = {"EI", "logEI", "UCB", "qEI", "qLogEI", "qUCB", "KG"}
MULTI_OBJECTIVE_ACQFS = {"EHVI", "ParEGO"}


def get_acquisition(
    name: str,
    model: Model,
    best_f: float | Tensor,
    *,
    maximize: bool = True,
    batch_size: int = 1,
    ref_point: Tensor | None = None,
    Y_successful: Tensor | None = None,
) -> object:
    """Create an acquisition function by name.

    Parameters
    ----------
    name:
        Acquisition function identifier. One of ``"EI"``, ``"UCB"``,
        ``"qEI"``, ``"qUCB"``, ``"KG"`` for single-objective, or
        ``"EHVI"``, ``"ParEGO"`` for multi-objective.
    model:
        Fitted BoTorch model.
    best_f:
        Best observed objective value (scalar, for single-objective).
    maximize:
        Whether the objective is being maximized.
    batch_size:
        Number of candidates to generate jointly (q-batch).
    ref_point:
        Reference point for hypervolume computation (multi-objective only).
    Y_successful:
        Successful objective observations of shape ``(n, m)`` for
        multi-objective Pareto partitioning.

    Returns
    -------
    AcquisitionFunction
        A BoTorch acquisition function instance.
    """
    name = name.upper()

    if name in ("EHVI", "PAREGO"):
        return _get_multi_objective_acquisition(
            name=name,
            model=model,
            ref_point=ref_point,
            Y_successful=Y_successful,
        )

    if isinstance(best_f, Tensor):
        best_f = best_f.item()

    if name == "EI":
        return ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)

    if name == "LOGEI":
        return LogExpectedImprovement(model=model, best_f=best_f, maximize=maximize)

    if name == "UCB":
        # beta = 2.0 is a common default
        return UpperConfidenceBound(model=model, beta=2.0, maximize=maximize)

    if name == "QEI":
        return qExpectedImprovement(model=model, best_f=best_f)

    if name == "QLOGEI":
        return qLogExpectedImprovement(model=model, best_f=best_f)

    if name == "QUCB":
        return qUpperConfidenceBound(model=model, beta=2.0)

    if name == "KG":
        return qKnowledgeGradient(model=model)

    supported = sorted(SINGLE_OBJECTIVE_ACQFS | MULTI_OBJECTIVE_ACQFS)
    raise ValueError(
        f"Unknown acquisition function '{name}'. Supported: {supported}"
    )


def _get_multi_objective_acquisition(
    name: str,
    model: Model,
    ref_point: Tensor | None,
    Y_successful: Tensor | None,
) -> object:
    """Create a multi-objective acquisition function."""
    if ref_point is None:
        raise ValueError(
            f"Multi-objective acquisition '{name}' requires a ref_point."
        )
    ref_point = ref_point.to(dtype=torch.double)

    if name == "EHVI":
        if Y_successful is None or Y_successful.shape[0] == 0:
            # No observations yet — use a uniform prior partitioning
            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=torch.empty(0, ref_point.shape[0], dtype=torch.double),
            )
        else:
            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=Y_successful.to(dtype=torch.double),
            )
        return qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            objective=IdentityMCMultiOutputObjective(),
        )

    if name == "PAREGO":
        # qParEGO uses random scalarization weights via get_chebyshev_scalarization.
        # Import here to avoid issues if not available in older BoTorch versions.
        from botorch.acquisition.multi_objective.parego import qParEGO
        return qParEGO(model=model, ref_point=ref_point)

    raise ValueError(f"Unknown multi-objective acquisition function '{name}'.")


def optimize_acquisition_function(
    acq: object,
    bounds: Tensor,
    batch_size: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> Tensor:
    """Optimize an acquisition function to find the next candidate(s).

    Parameters
    ----------
    acq:
        A BoTorch acquisition function instance.
    bounds:
        Input bounds tensor of shape ``(2, d)``.
    batch_size:
        Number of candidates to select jointly.
    num_restarts:
        Number of L-BFGS restarts for the optimization.
    raw_samples:
        Number of raw Monte Carlo samples used to seed the optimization.

    Returns
    -------
    candidates:
        Tensor of shape ``(batch_size, d)`` with the optimal candidates
        in the unit hypercube.
    """
    bounds = bounds.to(dtype=torch.double)
    candidates, acq_value = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    logger.debug(
        "Optimized acquisition function: best value %.6f, candidates shape %s.",
        acq_value.item() if acq_value.numel() == 1 else acq_value.max().item(),
        tuple(candidates.shape),
    )
    return candidates


def generate_initial_candidates(
    bounds: Tensor,
    n: int,
    seed: int | None = None,
) -> Tensor:
    """Generate quasi-random Sobol initial candidates.

    Parameters
    ----------
    bounds:
        Input bounds tensor of shape ``(2, d)``.
    n:
        Number of candidates to generate.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    candidates:
        Tensor of shape ``(n, d)`` with Sobol samples within bounds.
    """
    bounds = bounds.to(dtype=torch.double)
    samples = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed)
    return samples.squeeze(1)  # (n, d)
