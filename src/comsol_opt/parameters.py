"""Parameter specifications used to configure the optimization workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple


TransformKind = Literal["linear", "fill_factor"]
ValueType = Literal["continuous", "integer", "even_integer", "odd_integer"]

INTEGRAL_VALUE_TYPES = {"integer", "even_integer", "odd_integer"}


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
        ``"continuous"`` (default), as a general ``"integer"``, or restricted
        to ``"even_integer"`` / ``"odd_integer"``. Integral parameters are
        coerced to the nearest in-bounds value that satisfies the constraint
        before being evaluated.
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
        if self.value_type not in {"continuous", "integer", "even_integer", "odd_integer"}:
            raise ValueError(
                f"Unsupported value_type '{self.value_type}' for parameter '{self.name}'."
            )
        if self.value_type in INTEGRAL_VALUE_TYPES:
            if self.transform != "linear":
                raise ValueError(
                    f"Integer parameter '{self.name}' requires the 'linear' transform."
                )
            lower_int, upper_int = self.integer_bounds
            if lower_int > upper_int:
                raise ValueError(
                    f"Integer bounds for parameter '{self.name}' do not contain any values."
                )
            parity = self._parity_requirement()
            if parity is not None:
                candidate = self._first_value_with_parity(lower_int, upper_int, parity)
                if candidate is None:
                    raise ValueError(
                        f"Bounds for parameter '{self.name}' do not contain any "
                        f"{'even' if parity == 0 else 'odd'} integers."
                    )

    @property
    def is_integer(self) -> bool:
        return self.value_type in INTEGRAL_VALUE_TYPES

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
            coerced = self._apply_parity_constraint(coerced, lower_int, upper_int, value)
            return float(coerced)

        lower, upper = self.bounds
        if value < lower:
            return float(lower)
        if value > upper:
            return float(upper)
        return float(value)

    def _parity_requirement(self) -> int | None:
        if self.value_type == "even_integer":
            return 0
        if self.value_type == "odd_integer":
            return 1
        return None

    @staticmethod
    def _first_value_with_parity(lower: int, upper: int, parity: int) -> int | None:
        candidate = lower
        if candidate % 2 != parity:
            candidate += 1
        if candidate > upper:
            return None
        return candidate

    def _apply_parity_constraint(
        self, value: int, lower: int, upper: int, reference: float
    ) -> int:
        parity = self._parity_requirement()
        if parity is None:
            return value
        if value % 2 == parity:
            return value

        candidates: list[int] = []
        decrement = value - 1
        increment = value + 1
        if lower <= decrement <= upper and decrement % 2 == parity:
            candidates.append(decrement)
        if lower <= increment <= upper and increment % 2 == parity:
            candidates.append(increment)

        if not candidates:
            # Should not occur thanks to validation, but guard to avoid surprises.
            return value

        best = min(candidates, key=lambda candidate: (abs(candidate - reference), candidate))
        return best
