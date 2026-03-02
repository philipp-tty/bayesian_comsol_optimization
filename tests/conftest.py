"""pytest configuration: stub heavy optional dependencies (torch, botorch, gpytorch)
so that the lightweight modules (parser, parameters, objective) can be tested
without a full ML stack installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub_package(name: str) -> MagicMock:
    """Register a MagicMock as *name* and all known sub-packages/modules."""
    mock = MagicMock(name=name)
    sys.modules[name] = mock  # type: ignore[assignment]
    return mock


def _ensure_stub(top: str, submodules: list[str]) -> None:
    """Stub *top* and each dotted *submodule* path if the real package is absent."""
    try:
        __import__(top)
        return  # real package present; nothing to do
    except ModuleNotFoundError:
        pass

    _stub_package(top)
    for sub in submodules:
        full = f"{top}.{sub}"
        _stub_package(full)


_TORCH_SUBS = ["nn", "optim", "utils", "utils.data"]
_BOTORCH_SUBS = [
    "acquisition",
    "acquisition.multi_objective",
    "acquisition.multi_objective.objective",
    "fit",
    "models",
    "models.model",
    "models.model_list_gp_regression",
    "models.transforms",
    "models.transforms.input",
    "models.transforms.outcome",
    "optim",
    "utils",
    "utils.multi_objective",
    "utils.multi_objective.box_decompositions",
    "utils.multi_objective.box_decompositions.non_dominated",
    "utils.sampling",
]
_GPYTORCH_SUBS = ["mlls", "mlls.sum_marginal_log_likelihood"]

_ensure_stub("torch", _TORCH_SUBS)
_ensure_stub("botorch", _BOTORCH_SUBS)
_ensure_stub("gpytorch", _GPYTORCH_SUBS)
