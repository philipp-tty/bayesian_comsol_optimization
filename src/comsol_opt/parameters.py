"""Parameter specifications used to configure the optimization workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple


TransformKind = Literal["linear", "fill_factor"]
ValueType = Literal["continuous", "integer"]


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
    value_type:
        Indicates whether the physical parameter should be treated as
        ``"continuous"`` (default) or ``"integer"``. Integer parameters are
        coerced to the nearest in-bounds integer before being evaluated.
    """

    name: str
    bounds: Tuple[float, float]
    comsol_name: str
    unit: str | None = None
    transform: TransformKind = "linear"
    value_type: ValueType = "continuous"

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
        if self.value_type not in {"continuous", "integer"}:
            raise ValueError(
                f"Unsupported value_type '{self.value_type}' for parameter '{self.name}'."
            )
        if self.value_type == "integer":
            if self.transform != "linear":
                raise ValueError(
                    f"Integer parameter '{self.name}' requires the 'linear' transform."
                )
            lower_int, upper_int = self.integer_bounds
            if lower_int > upper_int:
                raise ValueError(
                    f"Integer bounds for parameter '{self.name}' do not contain any values."
                )

    @property
    def is_integer(self) -> bool:
        return self.value_type == "integer"

    @property
    def integer_bounds(self) -> Tuple[int, int]:
        if not self.is_integer:
            raise ValueError(
                f"Parameter '{self.name}' is not configured as integer-valued."
            )
        lower, upper = self.bounds
        return math.ceil(lower), math.floor(upper)

    def coerce_physical_value(self, value: float) -> float:
        """
        Clamp a physical value to the configured bounds and apply integer coercion if needed.
        """
        if self.is_integer:
            lower_int, upper_int = self.integer_bounds
            if value >= 0:
                rounded = math.floor(value + 0.5)
            else:
                rounded = math.ceil(value - 0.5)
            coerced = int(rounded)
            if coerced < lower_int:
                coerced = lower_int
            elif coerced > upper_int:
                coerced = upper_int
            return float(coerced)

        lower, upper = self.bounds
        if value < lower:
            return float(lower)
        if value > upper:
            return float(upper)
        return float(value)
