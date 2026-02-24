"""Optimization state management with serialization and checkpointing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .parameters import OptimizationParameter

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """Complete state of an optimization run, supporting save/load and resume.

    Attributes
    ----------
    parameters:
        The parameter definitions used in this optimization.
    objective_names:
        Names of the objectives being optimized, e.g. ``["power"]`` or
        ``["power", "cost"]``.
    X:
        Evaluated points in the unit hypercube, shape ``(n, d)``.
    Y:
        Objective values, shape ``(n, m)`` where *m* is the number of
        objectives.
    X_physical:
        Parameter histories in physical space, keyed by parameter name.
    success_mask:
        Boolean list indicating which evaluations succeeded.
    metadata:
        Auxiliary information (seed, timing, config, etc.).
    maximize:
        Per-objective maximization flags.  ``True`` means the optimizer
        seeks to *maximize* the corresponding objective.
    """

    parameters: list[OptimizationParameter]
    objective_names: list[str]
    X: Tensor
    Y: Tensor
    X_physical: dict[str, list[float]]
    success_mask: list[bool]
    metadata: dict[str, object] = field(default_factory=dict)
    maximize: list[bool] = field(default_factory=lambda: [True])

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_objectives(self) -> int:
        return len(self.objective_names)

    @property
    def is_multi_objective(self) -> bool:
        return self.n_objectives > 1

    @property
    def n_completed(self) -> int:
        return self.X.shape[0]

    @property
    def best_index(self) -> int:
        """Index of the best *successful* evaluation (single-objective only)."""
        if self.is_multi_objective:
            raise ValueError("best_index is only defined for single-objective optimization.")
        if self.n_completed == 0:
            return -1
        mask = torch.tensor(self.success_mask, dtype=torch.bool)
        if not mask.any():
            return -1
        values = self.Y[:, 0].clone()
        values[~mask] = float("-inf") if self.maximize[0] else float("inf")
        if self.maximize[0]:
            return int(values.argmax().item())
        return int(values.argmin().item())

    @property
    def best_objective(self) -> float:
        """Best objective value found (single-objective only)."""
        idx = self.best_index
        if idx < 0:
            return float("nan")
        return float(self.Y[idx, 0].item())

    @property
    def best_parameters(self) -> dict[str, float]:
        """Physical parameters at the best evaluation (single-objective only)."""
        idx = self.best_index
        if idx < 0:
            return {p.name: float("nan") for p in self.parameters}
        return {name: values[idx] for name, values in self.X_physical.items()}

    @property
    def pareto_indices(self) -> Tensor:
        """Indices of Pareto-optimal successful evaluations (multi-objective)."""
        if not self.is_multi_objective:
            raise ValueError("pareto_indices is only defined for multi-objective optimization.")
        if self.n_completed == 0:
            return torch.empty(0, dtype=torch.long)

        mask = torch.tensor(self.success_mask, dtype=torch.bool)
        if not mask.any():
            return torch.empty(0, dtype=torch.long)

        Y_successful = self.Y[mask].clone()
        # Negate objectives that are being minimized so we can use
        # standard non-dominated sorting (which assumes maximization).
        for i, mx in enumerate(self.maximize):
            if not mx:
                Y_successful[:, i] = -Y_successful[:, i]

        successful_indices = torch.where(mask)[0]
        from botorch.utils.multi_objective.pareto import is_non_dominated

        pareto_mask = is_non_dominated(Y_successful)
        return successful_indices[pareto_mask]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the optimization state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        param_dicts = []
        for p in self.parameters:
            param_dicts.append({
                "name": p.name,
                "bounds": list(p.bounds),
                "comsol_name": p.comsol_name,
                "unit": p.unit,
                "value_type": p.value_type,
                "transform": p.transform,
                "constant_value": p.constant_value,
                "log_scale": p.log_scale,
            })

        payload = {
            "version": "1.0.0",
            "parameters": param_dicts,
            "objective_names": self.objective_names,
            "maximize": self.maximize,
            "X": self.X.tolist(),
            "Y": self.Y.tolist(),
            "X_physical": {k: list(v) for k, v in self.X_physical.items()},
            "success_mask": self.success_mask,
            "metadata": _make_json_safe(self.metadata),
        }

        tmp_path = path.parent / f"{path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(path)
        logger.info("Saved optimization state to %s (%d evaluations).", path, self.n_completed)

    @classmethod
    def load(cls, path: Path) -> OptimizationState:
        """Load an optimization state from a JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"State file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"State file {path} must contain a JSON object.")

        params = []
        for pd in data["parameters"]:
            params.append(OptimizationParameter(
                name=pd["name"],
                bounds=tuple(pd["bounds"]),
                comsol_name=pd.get("comsol_name"),
                unit=pd.get("unit"),
                value_type=pd.get("value_type", "continuous"),
                transform=pd.get("transform", "linear"),
                constant_value=pd.get("constant_value"),
                log_scale=pd.get("log_scale", False),
            ))

        X = torch.tensor(data["X"], dtype=torch.double)
        Y = torch.tensor(data["Y"], dtype=torch.double)

        if X.ndim == 1 and X.numel() == 0:
            n_params = len([p for p in params if not p.is_constant])
            X = X.reshape(0, max(n_params, 1))
        if Y.ndim == 1 and Y.numel() == 0:
            n_obj = len(data.get("objective_names", ["objective"]))
            Y = Y.reshape(0, n_obj)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        maximize = data.get("maximize", [True])

        state = cls(
            parameters=params,
            objective_names=data.get("objective_names", ["objective"]),
            X=X,
            Y=Y,
            X_physical=data.get("X_physical", {}),
            success_mask=data.get("success_mask", []),
            metadata=data.get("metadata", {}),
            maximize=maximize,
        )
        logger.info("Loaded optimization state from %s (%d evaluations).", path, state.n_completed)
        return state


def _make_json_safe(obj: object) -> object:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    if isinstance(obj, Tensor):
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return obj
