"""Wrapper around the COMSOL CLI to evaluate thermoelectric geometries."""

from __future__ import annotations

import logging
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

from .parameters import OptimizationParameter
from .transforms import FillFactorTransform, LinearParameterTransform

logger = logging.getLogger(__name__)


class COMSOLCLIOptimizer:
    """
    Wrapper for COMSOL thermoelectric model optimization using the CLI.
    """

    def __init__(
        self,
        model_path: str,
        parameters: Sequence[OptimizationParameter],
        comsol_exe_path: str | None = None,
        methodcall: str = "methodcall2",
        target_footprint_mm2: float | None = None,
    ):
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
        if not parameters:
            raise ValueError("Parameter specification list must not be empty.")

        self.parameters: list[OptimizationParameter] = list(parameters)
        self._parameter_by_name: Dict[str, OptimizationParameter] = {}
        self.parameter_transforms: Dict[str, LinearParameterTransform | FillFactorTransform] = {}
        self.fill_parameter: OptimizationParameter | None = None
        self.fill_transform: FillFactorTransform | None = None

        for param in self.parameters:
            if param.name in self._parameter_by_name:
                raise ValueError(f"Duplicate parameter name provided: {param.name!r}.")
            self._parameter_by_name[param.name] = param

            if param.transform == "fill_factor":
                if self.fill_parameter is not None:
                    raise ValueError("Only one fill-factor parameter is supported.")
                transform = FillFactorTransform(param.bounds)
                self.fill_parameter = param
                self.fill_transform = transform
            else:
                transform = LinearParameterTransform(param.bounds)

            self.parameter_transforms[param.name] = transform

        if self.fill_parameter is not None:
            if target_footprint_mm2 is None or target_footprint_mm2 <= 0:
                raise ValueError("target_footprint_mm2 must be a positive number when using fill-factor.")
            self.target_footprint_mm2 = float(target_footprint_mm2)
        else:
            self.target_footprint_mm2 = None

        # GUI integration helpers (set later by optimizer/visualizer)
        self._event_pump: Callable[[], None] | None = None
        self._event_poll_interval = 0.05
        self._cli_timeout = 2000.0

        logger.info("Initialized COMSOL CLI wrapper with model: %s", self.model_path)
        logger.info("Using COMSOL executable: %s", self.comsol_exe)
        logger.info("Using COMSOL methodcall: %s", self.methodcall)
        for param in self.parameters:
            bounds_str = f"{param.bounds[0]} to {param.bounds[1]}"
            logger.info(
                "Parameter '%s' (COMSOL: %s, unit: %s, transform: %s, type: %s) bounds: %s",
                param.name,
                param.comsol_name,
                param.unit or "-",
                param.transform,
                param.value_type,
                bounds_str,
            )
        if self.target_footprint_mm2 is not None:
            logger.info(
                "Target footprint (no casing): %.3f mm^2 (side %.3f mm)",
                self.target_footprint_mm2,
                math.sqrt(self.target_footprint_mm2),
            )

    # ------------------------------------------------------------
    @staticmethod
    def _format_value_for_cli(value: float, unit: str | None) -> str:
        formatted_value = f"{value:.10g}"
        if unit:
            return f"{formatted_value}[{unit}]"
        return formatted_value

    def _build_cmd(self, parameter_names: Sequence[str], parameter_values: Sequence[str]) -> list[str]:
        """Build the COMSOL command line list."""
        if len(parameter_names) != len(parameter_values):
            raise ValueError("Parameter names and values must have the same length.")

        return [
            str(self.comsol_exe),
            "-inputfile",
            str(self.model_path),
            "-pname",
            ",".join(parameter_names),
            "-plist",
            ",".join(parameter_values),
            "-methodcall",
            self.methodcall,
            "-batchlog",
            "comsol_batch.log",
            "-nosave",
        ]

    # ------------------------------------------------------------
    def run_comsol_cli(self, parameter_names: Sequence[str], parameter_values: Sequence[str]) -> bool:
        """Run COMSOL via CLI. COMSOL will create an output.txt file with results."""
        cmd = self._build_cmd(parameter_names, parameter_values)
        logger.info("Running command:\n  %s", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            start_time = time.monotonic()
            stdout_data = ""
            stderr_data = ""

            # Poll until completion, keeping the GUI responsive by pumping events.
            while True:
                poll_result = process.poll()
                if poll_result is not None:
                    stdout_chunk, stderr_chunk = process.communicate()
                    stdout_data = stdout_chunk or ""
                    stderr_data = stderr_chunk or ""
                    break

                if self._event_pump is not None:
                    try:
                        self._event_pump()
                    except Exception:  # pragma: no cover - defensive guard
                        logger.debug("Event pump callback raised", exc_info=True)
                elif self._event_poll_interval > 0:
                    time.sleep(self._event_poll_interval)

                if time.monotonic() - start_time > self._cli_timeout:
                    process.kill()
                    stdout_chunk, stderr_chunk = process.communicate()
                    stdout_data = stdout_chunk or ""
                    stderr_data = stderr_chunk or ""
                    raise subprocess.TimeoutExpired(
                        cmd,
                        self._cli_timeout,
                        output=stdout_data,
                        stderr=stderr_data,
                    )

            if process.returncode != 0:
                logger.error("COMSOL returned error code %s", process.returncode)
                if stderr_data:
                    logger.error("stderr: %s", stderr_data)
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
    def set_event_pump(self, pump: Callable[[], None] | None, poll_interval: float | None = None) -> None:
        """Register a callback used to keep GUIs responsive while COMSOL runs."""
        self._event_pump = pump
        if poll_interval is not None:
            self._event_poll_interval = max(0.0, float(poll_interval))

    # ------------------------------------------------------------
    def parse_power_output(self, output_file: str = "output.txt") -> float:
        """
        Extract power output (mW) from COMSOL output.txt file.
        Reads the last line that contains a float value.
        """
        try:
            output_path = Path(output_file)
            if not output_path.exists():
                logger.error("Output file not found: %s", output_file)
                return -1e6

            with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()

            # Look for the last line containing a float value
            power_value = None
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line)
                if match:
                    power_value = float(match.group(1))
                    logger.info("Found power value: %.10f mW", power_value)
                    break

            if power_value is None:
                logger.error("Could not find numeric value in output file")
                return -1e6

            return power_value

        except Exception as exc:
            logger.error("Error parsing output file: %s", exc)
            return -1e6

    # ------------------------------------------------------------
    def evaluate(self, physical_parameters: Mapping[str, float]) -> dict:
        """
        Run COMSOL simulation and evaluate power output.
        COMSOL creates an output.txt file with the results.
        """
        output_file = "output.txt"
        parameter_values: Dict[str, float] = {}
        for param in self.parameters:
            if param.name not in physical_parameters:
                raise KeyError(f"Missing value for parameter '{param.name}'.")
            transform = self.parameter_transforms[param.name]
            clipped_value = float(transform.clip_physical(physical_parameters[param.name]))
            parameter_values[param.name] = param.coerce_physical_value(clipped_value)

        comsol_names: list[str] = []
        comsol_values: list[str] = []

        for param in self.parameters:
            comsol_names.append(param.comsol_name)
            comsol_values.append(
                self._format_value_for_cli(parameter_values[param.name], param.unit)
            )

        comsol_parameter_payload: Dict[str, Dict[str, float | str | None]] = {}
        for param in self.parameters:
            comsol_parameter_payload[param.comsol_name] = {
                "value": parameter_values[param.name],
                "unit": param.unit,
            }

        # Remove previous output file if it exists
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass

        try:
            logger.info(
                "Evaluating COMSOL at parameters: %s",
                ", ".join(
                    f"{param.name}={parameter_values[param.name]:.6f}"
                    for param in self.parameters
                ),
            )
            success = self.run_comsol_cli(comsol_names, comsol_values)

            if not success:
                return {
                    "power": -1e6,
                    "parameters": parameter_values,
                    "comsol_parameters": comsol_parameter_payload,
                    "success": False,
                }

            power_value = self.parse_power_output(output_file)

            if power_value == -1e6:
                return {
                    "power": -1e6,
                    "parameters": parameter_values,
                    "comsol_parameters": comsol_parameter_payload,
                    "success": False,
                }

            logger.info("Power output: %.6f mW", power_value)

            return {
                "power": power_value,
                "parameters": parameter_values,
                "comsol_parameters": comsol_parameter_payload,
                "success": power_value > 0,
            }

        except Exception as exc:
            logger.error("Simulation failed: %s", exc)
            return {
                "power": -1e6,
                "parameters": parameter_values,
                "comsol_parameters": comsol_parameter_payload,
                "success": False,
            }
