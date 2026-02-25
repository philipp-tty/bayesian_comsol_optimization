"""COMSOL output file parsing utilities."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_output_value(
    output_file: str | Path = "output.txt",
    objective_name: str | None = None,
) -> float | None:
    """Extract a scalar objective value from a COMSOL output file.

    If *objective_name* is given, the parser searches for a line containing
    that name and reads the floating-point number from the next line.

    If *objective_name* is ``None``, falls back to returning the last
    floating-point literal found anywhere in the file.

    Returns ``None`` if no value can be found.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        logger.error("Output file not found: %s", output_file)
        return None

    try:
        with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception as exc:
        logger.error("Error reading output file %s: %s", output_file, exc)
        return None

    float_pattern = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )

    # --- Name-based lookup: find line mentioning the objective, value is on the next line ---
    if objective_name is not None:
        for i, line in enumerate(lines):
            if objective_name in line:
                # Look at the next line for the numeric value
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    match = float_pattern.search(next_line)
                    if match:
                        value = float(match.group(1))
                        logger.info(
                            "Found objective '%s' value: %.10f (line %d)",
                            objective_name, value, i + 2,
                        )
                        return value
                logger.error(
                    "Found objective name '%s' on line %d but no numeric "
                    "value on the following line in %s",
                    objective_name, i + 1, output_file,
                )
                return None

        logger.error(
            "Objective name '%s' not found in output file %s",
            objective_name, output_file,
        )
        return None

    # --- Fallback: last floating-point number in the file ---
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = float_pattern.search(line)
        if match:
            value = float(match.group(1))
            logger.info("Found objective value: %.10f", value)
            return value

    logger.error("Could not find numeric value in output file %s", output_file)
    return None
