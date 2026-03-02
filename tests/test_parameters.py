"""Tests for comsol_opt.parameters.OptimizationParameter."""

from __future__ import annotations

import math

import pytest

from comsol_opt.parameters import OptimizationParameter


# ---------------------------------------------------------------------------
# Construction / attribute tests
# ---------------------------------------------------------------------------


def test_basic_continuous_parameter() -> None:
    p = OptimizationParameter(name="leg_length", bounds=(0.5, 4.0))
    assert p.name == "leg_length"
    assert p.bounds == (0.5, 4.0)
    assert p.value_type == "continuous"
    assert not p.is_integer
    assert not p.is_constant


def test_effective_comsol_name_defaults_to_name() -> None:
    p = OptimizationParameter(name="leg_width", bounds=(0.5, 2.0))
    assert p.effective_comsol_name == "leg_width"


def test_effective_comsol_name_custom() -> None:
    p = OptimizationParameter(name="leg_width", bounds=(0.5, 2.0), comsol_name="w_leg")
    assert p.effective_comsol_name == "w_leg"


def test_unit_stored() -> None:
    p = OptimizationParameter(name="leg_spacing", bounds=(0.5, 2.0), unit="mm")
    assert p.unit == "mm"


# ---------------------------------------------------------------------------
# Bound validation tests
# ---------------------------------------------------------------------------


def test_invalid_bounds_equal_raises() -> None:
    with pytest.raises(ValueError, match="lower must be < upper"):
        OptimizationParameter(name="x", bounds=(1.0, 1.0))


def test_invalid_bounds_reversed_raises() -> None:
    with pytest.raises(ValueError, match="lower must be < upper"):
        OptimizationParameter(name="x", bounds=(2.0, 1.0))


# ---------------------------------------------------------------------------
# Continuous coercion tests
# ---------------------------------------------------------------------------


def test_coerce_continuous_within_bounds() -> None:
    p = OptimizationParameter(name="x", bounds=(0.0, 1.0))
    assert p.coerce_physical_value(0.5) == pytest.approx(0.5)


def test_coerce_continuous_clamps_below() -> None:
    p = OptimizationParameter(name="x", bounds=(0.0, 1.0))
    assert p.coerce_physical_value(-0.1) == pytest.approx(0.0)


def test_coerce_continuous_clamps_above() -> None:
    p = OptimizationParameter(name="x", bounds=(0.0, 1.0))
    assert p.coerce_physical_value(1.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integer coercion tests
# ---------------------------------------------------------------------------


def test_integer_parameter_rounds() -> None:
    p = OptimizationParameter(name="n", bounds=(1.0, 10.0), value_type="integer")
    assert p.coerce_physical_value(3.4) == 3.0
    assert p.coerce_physical_value(3.6) == 4.0


def test_integer_parameter_clamps_below() -> None:
    p = OptimizationParameter(name="n", bounds=(2.0, 8.0), value_type="integer")
    assert p.coerce_physical_value(0.0) == 2.0


def test_integer_parameter_clamps_above() -> None:
    p = OptimizationParameter(name="n", bounds=(2.0, 8.0), value_type="integer")
    assert p.coerce_physical_value(100.0) == 8.0


def test_integer_bounds_property() -> None:
    p = OptimizationParameter(name="n", bounds=(1.5, 7.9), value_type="integer")
    assert p.integer_bounds == (2, 7)


# ---------------------------------------------------------------------------
# Even-integer coercion tests
# ---------------------------------------------------------------------------


def test_even_integer_coercion_exact() -> None:
    p = OptimizationParameter(name="n_legs", bounds=(4.0, 12.0), value_type="even_integer")
    assert p.coerce_physical_value(6.0) == 6.0


def test_even_integer_coercion_rounds_to_nearest_even() -> None:
    p = OptimizationParameter(name="n_legs", bounds=(4.0, 12.0), value_type="even_integer")
    # 7 rounded to nearest even is 8 (odd, so must adjust)
    result = p.coerce_physical_value(7.0)
    assert result % 2 == 0
    assert 4 <= result <= 12


def test_even_integer_coercion_all_values_are_even() -> None:
    p = OptimizationParameter(name="n_legs", bounds=(4.0, 12.0), value_type="even_integer")
    for v in [4.1, 5.0, 6.9, 7.5, 9.9, 11.1]:
        coerced = p.coerce_physical_value(v)
        assert coerced % 2 == 0, f"Expected even, got {coerced} for input {v}"


def test_even_integer_no_even_in_bounds_raises() -> None:
    # Bounds [1, 1] contain only the odd integer 1 — no even integer is available.
    with pytest.raises(ValueError, match="even"):
        OptimizationParameter(name="n", bounds=(1.0, 1.9), value_type="even_integer")


# ---------------------------------------------------------------------------
# Odd-integer coercion tests
# ---------------------------------------------------------------------------


def test_odd_integer_coercion_exact() -> None:
    p = OptimizationParameter(name="n", bounds=(1.0, 9.0), value_type="odd_integer")
    assert p.coerce_physical_value(5.0) == 5.0


def test_odd_integer_coercion_all_values_are_odd() -> None:
    p = OptimizationParameter(name="n", bounds=(1.0, 9.0), value_type="odd_integer")
    for v in [1.1, 2.0, 4.5, 6.9, 8.8]:
        coerced = p.coerce_physical_value(v)
        assert coerced % 2 == 1, f"Expected odd, got {coerced} for input {v}"


# ---------------------------------------------------------------------------
# Constant value tests
# ---------------------------------------------------------------------------


def test_constant_parameter_is_constant() -> None:
    p = OptimizationParameter(name="x", bounds=(0.0, 10.0), constant_value=5.0)
    assert p.is_constant
    assert p.constant_value == pytest.approx(5.0)


def test_constant_value_outside_bounds_raises() -> None:
    with pytest.raises(ValueError, match="outside bounds"):
        OptimizationParameter(name="x", bounds=(0.0, 1.0), constant_value=2.0)


def test_constant_even_integer_must_satisfy_parity() -> None:
    with pytest.raises(ValueError, match="parity|integer"):
        OptimizationParameter(
            name="n", bounds=(2.0, 10.0), value_type="even_integer", constant_value=3.0
        )


# ---------------------------------------------------------------------------
# Unsupported value_type / transform tests
# ---------------------------------------------------------------------------


def test_unsupported_value_type_raises() -> None:
    with pytest.raises((ValueError, TypeError)):
        OptimizationParameter(name="x", bounds=(0.0, 1.0), value_type="fractional")  # type: ignore[arg-type]


def test_integer_bounds_on_continuous_raises() -> None:
    p = OptimizationParameter(name="x", bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="not configured as integer"):
        _ = p.integer_bounds
