"""COMSOL CLI runner implementing the ObjectiveFunction protocol."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from ..objective import EvaluationResult
from ..parameters import OptimizationParameter
from .parser import parse_output_value

logger = logging.getLogger(__name__)


class COMSOLRunner:
    """Run COMSOL simulations as an objective function.

    Implements the :class:`~comsol_opt.objective.ObjectiveFunction` protocol
    so it can be passed directly to :class:`~comsol_opt.optimizer.BayesianOptimizer`.

    Parameters
    ----------
    model_path:
        Path to the COMSOL ``.mph`` model file.
    parameters:
        Full parameter specification (both active and constant).
    comsol_exe:
        Path to the ``comsolbatch`` executable.
    methodcall:
        COMSOL method call name.
    timeout:
        Maximum wall-clock time (in seconds) for a single COMSOL evaluation.
    working_dir:
        Working directory for COMSOL execution.  Defaults to the current
        working directory.
    objective_name:
        Name assigned to the parsed objective value in the returned
        :class:`EvaluationResult`.
    """

    def __init__(
        self,
        model_path: str | Path,
        parameters: Sequence[OptimizationParameter],
        comsol_exe: str | Path,
        methodcall: str = "methodcall2",
        timeout: float = 6000.0,
        working_dir: Path | None = None,
        objective_name: str = "objective",
    ) -> None:
        self.model_path = Path(model_path)
        self.comsol_exe = Path(comsol_exe)
        self.methodcall = methodcall
        self.timeout = float(timeout)
        self.working_dir = Path(working_dir) if working_dir else None
        self.objective_name = objective_name
        self.parameters: list[OptimizationParameter] = list(parameters)

        if not self.comsol_exe.exists():
            raise FileNotFoundError(f"COMSOL executable not found: {self.comsol_exe}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not parameters:
            raise ValueError("Parameter specification list must not be empty.")

        # Validate no duplicate parameter names
        names_seen: set[str] = set()
        for p in self.parameters:
            if p.name in names_seen:
                raise ValueError(f"Duplicate parameter name: {p.name!r}")
            names_seen.add(p.name)

        # GUI integration helpers
        self._event_pump: Callable[[], None] | None = None
        self._event_poll_interval: float = 0.05

        logger.info("COMSOLRunner initialized: model=%s, exe=%s", self.model_path, self.comsol_exe)

    def set_event_pump(
        self,
        pump: Callable[[], None] | None,
        poll_interval: float | None = None,
    ) -> None:
        """Register a callback to keep GUIs responsive while COMSOL runs."""
        self._event_pump = pump
        if poll_interval is not None:
            self._event_poll_interval = max(0.0, float(poll_interval))

    def evaluate(self, parameters: dict[str, float]) -> EvaluationResult:
        """Run a COMSOL simulation and return the result.

        Parameters
        ----------
        parameters:
            Physical parameter values keyed by parameter name.  Constant
            parameters may be omitted (their configured values are used).
        """
        output_file = "output.txt"

        # Fill in constants for any missing parameters
        provided = dict(parameters)
        for p in self.parameters:
            if p.name not in provided:
                if p.is_constant and p.constant_value is not None:
                    provided[p.name] = p.constant_value
                else:
                    raise KeyError(f"Missing value for parameter '{p.name}'.")

        # Coerce values to satisfy constraints
        coerced: dict[str, float] = {}
        for p in self.parameters:
            coerced[p.name] = p.coerce_physical_value(provided[p.name])

        # Build COMSOL CLI arguments
        comsol_names: list[str] = []
        comsol_values: list[str] = []
        for p in self.parameters:
            comsol_names.append(p.effective_comsol_name)
            comsol_values.append(self._format_value(coerced[p.name], p.unit))

        comsol_payload = {
            p.effective_comsol_name: {"value": coerced[p.name], "unit": p.unit}
            for p in self.parameters
        }

        # Remove previous output file
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass

        logger.info(
            "Evaluating COMSOL: %s",
            ", ".join(f"{p.name}={coerced[p.name]:.6g}" for p in self.parameters),
        )

        success = self._run_cli(comsol_names, comsol_values)
        if not success:
            return EvaluationResult(
                objectives={self.objective_name: float("nan")},
                success=False,
                metadata={"parameters": coerced, "comsol_parameters": comsol_payload},
            )

        value = parse_output_value(output_file)
        if value is None:
            return EvaluationResult(
                objectives={self.objective_name: float("nan")},
                success=False,
                metadata={"parameters": coerced, "comsol_parameters": comsol_payload},
            )

        logger.info("Objective value: %.6f", value)
        return EvaluationResult(
            objectives={self.objective_name: value},
            success=True,
            metadata={"parameters": coerced, "comsol_parameters": comsol_payload},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value(value: float, unit: str | None) -> str:
        formatted = f"{value:.10g}"
        if unit:
            return f"{formatted}[{unit}]"
        return formatted

    def _build_cmd(
        self,
        parameter_names: Sequence[str],
        parameter_values: Sequence[str],
    ) -> list[str]:
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

    def _run_cli(
        self,
        parameter_names: Sequence[str],
        parameter_values: Sequence[str],
    ) -> bool:
        cmd = self._build_cmd(parameter_names, parameter_values)
        logger.info(
            "Running command:\n  %s",
            " ".join(f'"{c}"' if " " in c else c for c in cmd),
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.working_dir) if self.working_dir else None,
            )

            start_time = time.monotonic()
            stdout_data = ""
            stderr_data = ""

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
                    except Exception:
                        logger.debug("Event pump callback raised", exc_info=True)
                elif self._event_poll_interval > 0:
                    time.sleep(self._event_poll_interval)

                if time.monotonic() - start_time > self.timeout:
                    process.kill()
                    process.communicate()
                    logger.error("COMSOL simulation timed out after %.0f s", self.timeout)
                    return False

            if process.returncode != 0:
                logger.error("COMSOL returned error code %s", process.returncode)
                if stderr_data:
                    logger.error("stderr: %s", stderr_data)
                return False

            if not Path("output.txt").exists():
                logger.error("COMSOL did not create output.txt file")
                return False

            return True

        except Exception as exc:
            logger.error("Error running COMSOL: %s", exc)
            return False
