"""High-level package for COMSOL-based thermoelectric optimization."""

from __future__ import annotations

import logging

from .comsol_cli import COMSOLCLIOptimizer
from .optimizer import optimize_thermoelectric_generator
from .transforms import FillFactorTransform
from .visualization import GPVisualizer

# Configure a default logging setup if the host application has not done so.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(level=logging.INFO)

__all__ = [
    "COMSOLCLIOptimizer",
    "FillFactorTransform",
    "GPVisualizer",
    "optimize_thermoelectric_generator",
]

