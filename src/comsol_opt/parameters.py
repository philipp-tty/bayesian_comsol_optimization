"""Parameter specifications used to configure the optimization workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


TransformKind = Literal["linear", "fill_factor"]


@dataclass(frozen=True)
class OptimizationParameter:
    """
    Definition of a single optimization variable.

    Attributes
    ----------
    name:
        Identifier used internally when storing parameter values.
    bounds:
        Physical-domain lower and upper bound for the parameter.
    comsol_name:
        Name of the parameter in the COMSOL model. The optimization routine
        will pass values to COMSOL using this identifier.
    unit:
        Optional unit string appended in the COMSOL CLI call, e.g. ``"mm"``.
    transform:
        Which transform to use when mapping between the physical domain and
        the normalized unit interval. ``"fill_factor"`` uses the specialized
        :class:`~comsol_opt.transforms.FillFactorTransform`; ``"linear"``
        applies a simple affine mapping via
        :class:`~comsol_opt.transforms.LinearParameterTransform`.
    """

    name: str
    bounds: Tuple[float, float]
    comsol_name: str
    unit: str | None = None
    transform: TransformKind = "linear"

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if lower >= upper:
            raise ValueError(
                f"Invalid bounds for parameter '{self.name}': lower must be < upper."
            )
        if self.transform not in {"linear", "fill_factor"}:
            raise ValueError(
                f"Unsupported transform '{self.transform}' for parameter '{self.name}'."
            )
