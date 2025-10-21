"""Wrapper around the COMSOL CLI to evaluate thermoelectric geometries."""

from __future__ import annotations

import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Tuple

from .transforms import FillFactorTransform

logger = logging.getLogger(__name__)


class COMSOLCLIOptimizer:
    """
    Wrapper for COMSOL thermoelectric model optimization using the CLI.

    It evaluates power output for a given area fill factor by deriving the corresponding
    leg width and leg spacing from the target footprint area (without casing).
    """

    def __init__(
        self,
        model_path: str,
        n_legs: int = 127,
        comsol_exe_path: str | None = None,
        methodcall: str = "methodcall2",
        fill_factor_bounds: Tuple[float, float] = (0.05, 0.40),
        target_footprint_mm2: float | None = None,
    ):
        self.n_legs = int(n_legs)
        self.model_path = Path(model_path)
        self.methodcall = methodcall

        if comsol_exe_path is None:
            raise ValueError(
                "Please provide comsol_exe_path, e.g., "
                r'R"C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics_NSL\\bin\\win64\\comsolbatch.exe"'
            )

        self.comsol_exe = Path(comsol_exe_path)
        if not self.comsol_exe.exists():
            raise FileNotFoundError(f"COMSOL executable not found: {self.comsol_exe}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.fill_transform = FillFactorTransform(fill_factor_bounds)
        self.fill_factor_bounds = self.fill_transform.bounds

        if target_footprint_mm2 is None or target_footprint_mm2 <= 0:
            raise ValueError("target_footprint_mm2 must be a positive number.")
        self.target_footprint_mm2 = float(target_footprint_mm2)

        logger.info("Initialized COMSOL CLI wrapper with model: %s", self.model_path)
        logger.info("Using COMSOL executable: %s", self.comsol_exe)
        logger.info("Using COMSOL methodcall: %s", self.methodcall)
        logger.info(
            "Area fill factor bounds: %.1f%% to %.1f%%",
            self.fill_factor_bounds[0] * 100.0,
            self.fill_factor_bounds[1] * 100.0,
        )
        logger.info(
            "Target footprint (no casing): %.3f mm^2 (side %.3f mm)",
            self.target_footprint_mm2,
            math.sqrt(self.target_footprint_mm2),
        )

    # ------------------------------------------------------------
    def calculate_geometry(self, fill_factor: float) -> Tuple[float, float]:
        """
        Given an area fill factor f in (0, 1) and a fixed footprint area (mm^2),
        compute the leg width and leg spacing (both in mm).

        Definitions (no casing):
            L_total = sqrt(target_footprint_mm2)
            A = n * leg_width
            f = (A^2) / (L_total^2)
            L_total = A + (n + 1) * leg_spacing

        Closed-form solution:
            A  = sqrt(f) * L_total
            leg_spacing = (L_total - A) / (n + 1)
            leg_width   = A / n
        """
        f = float(self.fill_transform.clip_physical(fill_factor))
        if not (0.0 < f < 1.0):
            raise ValueError("fill_factor must be in (0, 1).")

        L_total = math.sqrt(self.target_footprint_mm2)
        n = self.n_legs

        A = math.sqrt(f) * L_total  # total leg length along one side
        remaining = L_total - A
        if remaining <= 0:
            raise ValueError(
                "Invalid geometry: legs exceed the total footprint side. Reduce fill_factor."
            )

        leg_spacing = remaining / (n + 1)
        leg_width = A / n

        if leg_width <= 0 or leg_spacing <= 0:
            raise ValueError("Solved non-positive geometry; adjust inputs.")

        return leg_width, leg_spacing

    # ------------------------------------------------------------
    def geometry_from_fill_factor(self, fill_factor: float) -> Tuple[float, float]:
        """Return (leg_width, leg_spacing) for an area fill factor under the fixed footprint."""
        return self.calculate_geometry(fill_factor)

    # ------------------------------------------------------------
    def footprint_side_length(self, leg_width: float, leg_spacing: float) -> float:
        """
        Return the total side length (mm) of the square module for a given geometry.

        Formula (side length, no casing):
            L = n * leg_width + (n + 1) * leg_spacing
        """
        n = self.n_legs
        return n * leg_width + (n + 1) * leg_spacing

    # ------------------------------------------------------------
    def footprint(self, leg_width: float, leg_spacing: float) -> float:
        """Return the footprint area (mm^2), no casing."""
        side_length = self.footprint_side_length(leg_width, leg_spacing)
        return side_length * side_length

    # ------------------------------------------------------------
    def _build_cmd(self, leg_width: float, leg_spacing: float) -> list[str]:
        """Build the COMSOL command line list."""
        return [
            str(self.comsol_exe),
            "-inputfile",
            str(self.model_path),
            "-pname",
            "leg_width,leg_spacing",
            "-plist",
            f"{leg_width}[mm],{leg_spacing}[mm]",
            "-methodcall",
            self.methodcall,
            "-nosave",
        ]

    # ------------------------------------------------------------
    def run_comsol_cli(self, leg_width: float, leg_spacing: float) -> bool:
        """Run COMSOL via CLI. COMSOL will create an output.txt file with results."""
        cmd = self._build_cmd(leg_width, leg_spacing)
        logger.info("Running command:\n  %s", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error("COMSOL returned error code %s", result.returncode)
                if result.stderr:
                    logger.error("stderr: %s", result.stderr)
                return False

            # Check if COMSOL created the output.txt file
            output_file = Path("output.txt")
            if not output_file.exists():
                logger.error("COMSOL did not create output.txt file")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("COMSOL simulation timed out")
            return False
        except Exception as exc:
            logger.error("Error running COMSOL: %s", exc)
            return False

    # ------------------------------------------------------------
    def parse_power_output(self, output_file: str = "output.txt") -> float:
        """
        Extract power output (mW) from COMSOL output.txt file.
        Looks for the last numeric value after the "% realdot(cir.R1.i,cir.R1.v) (mW)" header.
        """
        try:
            output_path = Path(output_file)
            if not output_path.exists():
                logger.error("Output file not found: %s", output_file)
                return -1e6

            with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()

            # Find the line containing the power output header
            header_idx = -1
            for i, line in enumerate(lines):
                if "realdot(cir.R1.i" in line and "(mW)" in line:
                    header_idx = i
                    logger.info("Found power output header at line %s", i)
                    break

            if header_idx == -1:
                logger.error("Could not find power output header in file")
                return -1e6

            # Look for the last numeric value after the header
            power_value = None
            for i in range(header_idx + 1, min(header_idx + 11, len(lines))):
                line = lines[i].strip()
                if not line:
                    continue
                match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line)
                if match:
                    power_value = float(match.group(1))
                    logger.info("Found power value at line %s: %.10f mW", i, power_value)

            if power_value is None:
                logger.error("Could not find numeric power value after header")
                return -1e6

            return power_value

        except Exception as exc:
            logger.error("Error parsing output file: %s", exc)
            return -1e6

    # ------------------------------------------------------------
    def evaluate(self, fill_factor: float) -> dict:
        """
        Run COMSOL simulation and evaluate power output.
        COMSOL creates an output.txt file with the results.
        """
        output_file = "output.txt"
        fill_factor = float(self.fill_transform.clip_physical(fill_factor))
        leg_width, leg_spacing = self.geometry_from_fill_factor(fill_factor)

        # Remove previous output file if it exists
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass

        try:
            logger.info(
                "Evaluating: fill_factor(area)=%.4f, leg_width=%.4f mm, leg_spacing=%.4f mm, footprint=%.3f mm^2",
                fill_factor,
                leg_width,
                leg_spacing,
                self.footprint(leg_width, leg_spacing),
            )
            success = self.run_comsol_cli(leg_width, leg_spacing)

            if not success:
                return {
                    "power": -1e6,
                    "fill_factor": fill_factor,
                    "leg_width": leg_width,
                    "leg_spacing": leg_spacing,
                    "success": False,
                }

            power_value = self.parse_power_output(output_file)

            if power_value == -1e6:
                return {
                    "power": -1e6,
                    "fill_factor": fill_factor,
                    "leg_width": leg_width,
                    "leg_spacing": leg_spacing,
                    "success": False,
                }

            logger.info(
                "Power output: %.6f mW, Fill factor(area): %.6f, Leg spacing: %.6f mm",
                power_value,
                fill_factor,
                leg_spacing,
            )

            return {
                "power": power_value,
                "fill_factor": fill_factor,
                "leg_width": leg_width,
                "leg_spacing": leg_spacing,
                "success": power_value > 0,
            }

        except Exception as exc:
            logger.error("Simulation failed: %s", exc)
            return {
                "power": -1e6,
                "fill_factor": fill_factor,
                "leg_width": leg_width,
                "leg_spacing": leg_spacing,
                "success": False,
            }

