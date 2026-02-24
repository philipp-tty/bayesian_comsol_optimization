"""Bayesian optimization for COMSOL simulations using BoTorch."""

from __future__ import annotations

import logging

from .comsol.runner import COMSOLRunner
from .objective import EvaluationResult, ObjectiveFunction, wrap_callable
from .optimizer import BayesianOptimizer
from .parameters import OptimizationParameter
from .state import OptimizationState
from .surrogate import GPSurrogate
from .transforms import FillFactorTransform, LinearParameterTransform

# Configure a default logging setup if the host application has not done so.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

__all__ = [
    "BayesianOptimizer",
    "COMSOLRunner",
    "EvaluationResult",
    "FillFactorTransform",
    "GPSurrogate",
    "LinearParameterTransform",
    "ObjectiveFunction",
    "OptimizationParameter",
    "OptimizationState",
    "wrap_callable",
]
