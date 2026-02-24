"""COMSOL output file parsing utilities."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_output_value(output_file: str | Path = "output.txt") -> float | None:
    """Extract a scalar objective value from a COMSOL output file.

    Searches the file for the last line containing a floating-point literal
    and returns it.  Returns ``None`` if no value can be found.
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

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = re.search(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line
        )
        if match:
            value = float(match.group(1))
            logger.info("Found objective value: %.10f", value)
            return value

    logger.error("Could not find numeric value in output file %s", output_file)
    return None
