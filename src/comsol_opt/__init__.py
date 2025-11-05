"""High-level package for COMSOL-based thermoelectric optimization."""

from __future__ import annotations

import logging

from .comsol_cli import COMSOLCLIOptimizer
from .parameters import OptimizationParameter
from .transforms import FillFactorTransform, LinearParameterTransform
from .visualization import GPVisualizer
from .workflow import optimize_model

# Configure a default logging setup if the host application has not done so.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(level=logging.INFO)

__all__ = [
    "COMSOLCLIOptimizer",
    "FillFactorTransform",
    "LinearParameterTransform",
    "GPVisualizer",
    "OptimizationParameter",
    "optimize_model",
]
