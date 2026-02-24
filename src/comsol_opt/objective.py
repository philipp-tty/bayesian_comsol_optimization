"""Objective function protocol and evaluation result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@dataclass
class EvaluationResult:
    """Result of a single objective function evaluation.

    Attributes
    ----------
    objectives:
        Named objective values, e.g. ``{"power": 42.5}`` for single-objective
        or ``{"power": 42.5, "cost": 1.2}`` for multi-objective.
    success:
        Whether the evaluation completed successfully.
    metadata:
        Arbitrary metadata from the evaluation (e.g. COMSOL parameters,
        timing info).
    """

    objectives: dict[str, float]
    success: bool
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def objective(self) -> float:
        """Convenience accessor for single-objective: returns the first value.

        Raises ``ValueError`` if there are no objectives.
        """
        if not self.objectives:
            raise ValueError("No objective values recorded in this result.")
        return next(iter(self.objectives.values()))


@runtime_checkable
class ObjectiveFunction(Protocol):
    """Protocol for objective functions that can be optimized."""

    def evaluate(self, parameters: dict[str, float]) -> EvaluationResult: ...


def wrap_callable(
    fn: Callable[..., float | EvaluationResult],
    objective_name: str = "objective",
) -> ObjectiveFunction:
    """Wrap a plain callable as an :class:`ObjectiveFunction`.

    If *fn* returns a float, it is wrapped into an ``EvaluationResult`` with
    a single objective named *objective_name*.  If *fn* returns an
    ``EvaluationResult`` directly it is passed through unchanged.
    """

    class _Wrapper:
        def evaluate(self, parameters: dict[str, float]) -> EvaluationResult:
            try:
                result = fn(parameters)
            except Exception as exc:
                return EvaluationResult(
                    objectives={objective_name: float("nan")},
                    success=False,
                    metadata={"error": str(exc)},
                )
            if isinstance(result, EvaluationResult):
                return result
            return EvaluationResult(
                objectives={objective_name: float(result)},
                success=True,
            )

    return _Wrapper()
