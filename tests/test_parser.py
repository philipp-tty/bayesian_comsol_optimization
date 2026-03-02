"""Tests for comsol_opt.comsol.parser.parse_output_value."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from comsol_opt.comsol.parser import parse_output_value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMSOL_OUTPUT = """\
% Model:              teg_no_electrodes.mph
% Version:            COMSOL 6.3.0.420
% Date:               Feb 25 2026, 15:53
% Dimension:          3
% Nodes:              1
% Expressions:        5
% Description:        power_density, fill_factor, footprint, v_0, p_0
% Length unit:        mm
% Grid
0
0                        
-15.500000000000004      
% Data
% power_density (W/m^2)
0.009503574379862122
% Data
% fill_factor (1)
0.3357602902367456
% Data
% footprint (m^2)
1.9883757463125308E-4
% Data
% v_0 (V)
-0.001637016299383725
% Data
% p_0 (W)
1.1810423000121872E-7
"""


@pytest.fixture()
def comsol_output_file(tmp_path: Path) -> Path:
    """Write the canonical COMSOL output file and return its path."""
    p = tmp_path / "output.txt"
    p.write_text(COMSOL_OUTPUT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Named-lookup tests (the fix validated here)
# ---------------------------------------------------------------------------


def test_named_lookup_power_density(comsol_output_file: Path) -> None:
    """Parser must skip the Description header and find the value in the Data section."""
    value = parse_output_value(comsol_output_file, objective_name="power_density")
    assert value == pytest.approx(0.009503574379862122)


def test_named_lookup_fill_factor(comsol_output_file: Path) -> None:
    value = parse_output_value(comsol_output_file, objective_name="fill_factor")
    assert value == pytest.approx(0.3357602902367456)


def test_named_lookup_footprint_scientific_notation(comsol_output_file: Path) -> None:
    """Values in 'E-4' scientific notation must be parsed correctly."""
    value = parse_output_value(comsol_output_file, objective_name="footprint")
    assert value == pytest.approx(1.9883757463125308e-4)


def test_named_lookup_negative_value(comsol_output_file: Path) -> None:
    """Negative values must be parsed correctly."""
    value = parse_output_value(comsol_output_file, objective_name="v_0")
    assert value == pytest.approx(-0.001637016299383725)


def test_named_lookup_small_scientific_notation(comsol_output_file: Path) -> None:
    """Very small values in 'E-7' notation must be parsed correctly."""
    value = parse_output_value(comsol_output_file, objective_name="p_0")
    assert value == pytest.approx(1.1810423000121872e-7)


# ---------------------------------------------------------------------------
# Fallback (no objective_name) tests
# ---------------------------------------------------------------------------


def test_fallback_returns_last_float(comsol_output_file: Path) -> None:
    """Without an objective name the parser must return the last float in the file."""
    value = parse_output_value(comsol_output_file)
    # Last numeric line is the p_0 value
    assert value == pytest.approx(1.1810423000121872e-7)


def test_fallback_simple_file(tmp_path: Path) -> None:
    """Fallback must work on a trivial file with a single numeric line."""
    p = tmp_path / "simple.txt"
    p.write_text("42.5\n", encoding="utf-8")
    assert parse_output_value(p) == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


def test_file_not_found_returns_none(tmp_path: Path) -> None:
    """Returns None when the output file does not exist."""
    result = parse_output_value(tmp_path / "nonexistent.txt")
    assert result is None


def test_objective_name_not_in_file_returns_none(comsol_output_file: Path) -> None:
    """Returns None when the requested objective name is absent from the file."""
    result = parse_output_value(comsol_output_file, objective_name="nonexistent_objective")
    assert result is None


def test_empty_file_returns_none(tmp_path: Path) -> None:
    """Returns None when the file is empty."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    assert parse_output_value(p) is None


def test_objective_name_present_but_no_numeric_next_line_returns_none(tmp_path: Path) -> None:
    """Returns None when the objective name appears but is never followed by a number."""
    content = "% power_density (W/m^2)\nno numeric data here\n"
    p = tmp_path / "bad.txt"
    p.write_text(content, encoding="utf-8")
    result = parse_output_value(p, objective_name="power_density")
    assert result is None


def test_named_lookup_skips_description_continues_to_data(tmp_path: Path) -> None:
    """The parser must continue past the Description header to find the Data value."""
    content = (
        "% Description:        power_density, fill_factor\n"
        "% Length unit:        mm\n"
        "% Data\n"
        "% power_density (W/m^2)\n"
        "1.23456\n"
    )
    p = tmp_path / "output.txt"
    p.write_text(content, encoding="utf-8")
    value = parse_output_value(p, objective_name="power_density")
    assert value == pytest.approx(1.23456)
