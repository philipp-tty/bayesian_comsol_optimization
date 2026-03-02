"""Tests for comsol_opt.objective.EvaluationResult and wrap_callable."""

from __future__ import annotations

import math

import pytest

from comsol_opt.objective import EvaluationResult, ObjectiveFunction, wrap_callable


# ---------------------------------------------------------------------------
# EvaluationResult tests
# ---------------------------------------------------------------------------


def test_single_objective_success() -> None:
    result = EvaluationResult(objectives={"power_density": 42.5}, success=True)
    assert result.success is True
    assert result.objectives["power_density"] == pytest.approx(42.5)
    assert result.objective == pytest.approx(42.5)


def test_multi_objective_stored() -> None:
    result = EvaluationResult(
        objectives={"power": 10.0, "cost": 2.5},
        success=True,
    )
    assert result.objectives["power"] == pytest.approx(10.0)
    assert result.objectives["cost"] == pytest.approx(2.5)
    # convenience accessor returns first value
    assert result.objective == pytest.approx(10.0)


def test_objective_property_raises_when_empty() -> None:
    result = EvaluationResult(objectives={}, success=False)
    with pytest.raises(ValueError, match="No objective"):
        _ = result.objective


def test_failed_result_with_nan() -> None:
    result = EvaluationResult(
        objectives={"obj": float("nan")},
        success=False,
        metadata={"error": "simulation failed"},
    )
    assert result.success is False
    assert math.isnan(result.objective)
    assert result.metadata["error"] == "simulation failed"


def test_metadata_defaults_to_empty_dict() -> None:
    result = EvaluationResult(objectives={"x": 1.0}, success=True)
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# wrap_callable tests
# ---------------------------------------------------------------------------


def test_wrap_callable_float_return() -> None:
    """A callable returning a plain float is wrapped into EvaluationResult."""
    fn = wrap_callable(lambda params: 3.14, objective_name="my_obj")
    result = fn.evaluate({"x": 0.5})
    assert result.success is True
    assert result.objectives["my_obj"] == pytest.approx(3.14)


def test_wrap_callable_default_objective_name() -> None:
    fn = wrap_callable(lambda params: 7.0)
    result = fn.evaluate({})
    assert "objective" in result.objectives


def test_wrap_callable_passes_parameters() -> None:
    """The wrapped callable receives the parameters dict."""
    received: dict = {}

    def fn(params: dict) -> float:
        received.update(params)
        return params["a"] + params["b"]

    wrapped = wrap_callable(fn)
    result = wrapped.evaluate({"a": 2.0, "b": 3.0})
    assert result.objective == pytest.approx(5.0)
    assert received == {"a": 2.0, "b": 3.0}


def test_wrap_callable_passthrough_evaluation_result() -> None:
    """If the callable already returns an EvaluationResult it is passed through."""
    inner = EvaluationResult(objectives={"power": 99.0}, success=True)
    fn = wrap_callable(lambda params: inner)
    result = fn.evaluate({})
    assert result is inner


def test_wrap_callable_exception_returns_failure() -> None:
    """Exceptions are caught and returned as a failed EvaluationResult with NaN."""
    def bad_fn(params: dict) -> float:
        raise RuntimeError("COMSOL exploded")

    wrapped = wrap_callable(bad_fn, objective_name="obj")
    result = wrapped.evaluate({})
    assert result.success is False
    assert math.isnan(result.objective)
    assert "COMSOL exploded" in str(result.metadata.get("error", ""))


# ---------------------------------------------------------------------------
# ObjectiveFunction protocol check
# ---------------------------------------------------------------------------


def test_wrap_callable_satisfies_protocol() -> None:
    fn = wrap_callable(lambda params: 0.0)
    assert isinstance(fn, ObjectiveFunction)
