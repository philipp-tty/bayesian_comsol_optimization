"""Tests for full-factorial sweep helpers."""

from __future__ import annotations

import pytest

from comsol_opt.parameters import OptimizationParameter
from comsol_opt.sweep import build_sweep_grids, combo_key, parse_sweep_values


def test_parse_compact_values_with_units() -> None:
    parameter = OptimizationParameter(name="width", bounds=(1.0, 20.0), unit="mm")

    assert parse_sweep_values("4mm 8mm 12 mm", parameter) == [4.0, 8.0, 12.0]


def test_parse_list_values_and_coerce_integer_duplicates() -> None:
    parameter = OptimizationParameter(
        name="n_legs",
        bounds=(4.0, 12.0),
        value_type="even_integer",
    )

    assert parse_sweep_values([4, 5, 6, "7 8"], parameter) == [4.0, 6.0, 8.0]


def test_parse_value_unit_mismatch_raises() -> None:
    parameter = OptimizationParameter(name="load", bounds=(1.0, 20.0), unit="ohm")

    with pytest.raises(ValueError, match="configured with unit"):
        parse_sweep_values("4mm", parameter)


def test_build_sweep_grids_uses_explicit_and_generated_values() -> None:
    parameters = [
        OptimizationParameter(name="width", bounds=(4.0, 12.0), unit="mm"),
        OptimizationParameter(name="load", bounds=(4.0, 8.0), unit="ohm"),
    ]
    config = {
        "sweep": {"values": {"width": "4mm 8mm 12 mm"}},
        "parameters": [
            {"name": "width", "bounds": [4.0, 12.0], "unit": "mm"},
            {"name": "load", "bounds": [4.0, 8.0], "unit": "ohm"},
        ],
    }

    active, grids, sources = build_sweep_grids(parameters, config, points=2)

    assert [p.name for p in active] == ["width", "load"]
    assert grids == [[4.0, 8.0, 12.0], [4.0, 8.0]]
    assert sources == {"width": "explicit", "load": "generated"}


def test_build_sweep_grids_requires_values_or_points() -> None:
    parameters = [OptimizationParameter(name="width", bounds=(4.0, 12.0), unit="mm")]

    with pytest.raises(ValueError, match="Missing sweep values"):
        build_sweep_grids(parameters, {"parameters": []})


def test_build_sweep_grids_rejects_unknown_values_parameter() -> None:
    parameters = [OptimizationParameter(name="width", bounds=(4.0, 12.0), unit="mm")]
    config = {
        "sweep": {"values": {"height": [4, 8]}},
        "parameters": [{"name": "width"}],
    }

    with pytest.raises(ValueError, match="unknown parameter"):
        build_sweep_grids(parameters, config, points=2)


def test_combo_key_rounds_float_noise() -> None:
    assert combo_key([4.00000000001, 8.0]) == (4.0, 8.0)
