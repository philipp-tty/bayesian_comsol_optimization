"""Deterministic Bayesian optimization helpers."""

from __future__ import annotations

import logging

import torch
from bo import BayesianOptimization as _BayesianOptimization, DEVICE, DTYPE
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

logger = logging.getLogger(__name__)

_MIN_NOISE_FLOOR = 1e-8


class DeterministicBayesianOptimization(_BayesianOptimization):
    """
    Variant of :class:`bo.BayesianOptimization` that treats observations as noise-free.

    We still inject a tiny jitter (`noise_floor`) to satisfy the Gaussian likelihood's
    positivity constraint while keeping the model effectively deterministic.
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        bounds: torch.Tensor,
        maximize: bool = True,
        use_outcome_transform: bool = True,
        *,
        measurement_noise: float = 0.0,
        noise_floor: float = _MIN_NOISE_FLOOR,
    ) -> None:
        self._measurement_noise = max(float(measurement_noise), 0.0)
        self._noise_floor = max(float(noise_floor), _MIN_NOISE_FLOOR)
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            bounds=bounds,
            maximize=maximize,
            use_outcome_transform=use_outcome_transform,
        )

    def _target_noise_level(self) -> float:
        return max(self._measurement_noise, self._noise_floor)

    def _build_and_fit_model(self, use_outcome_transform: bool = True) -> None:  # type: ignore[override]
        if use_outcome_transform:
            model = SingleTaskGP(self.x_train, self.y_train)
        else:
            model = SingleTaskGP(self.x_train, self.y_train, outcome_transform=None)

        model = model.to(DEVICE, DTYPE)
        likelihood = getattr(model, "likelihood", None)

        if likelihood is not None and hasattr(likelihood, "noise_covar"):
            stabilized_noise = self._target_noise_level()
            noise_tensor = torch.as_tensor(
                stabilized_noise,
                dtype=self.y_train.dtype,
                device=self.y_train.device,
            )
            with torch.no_grad():
                try:
                    likelihood.noise = noise_tensor
                except Exception:
                    # Fall back to direct initialization if property assignment is unsupported.
                    try:
                        likelihood.noise_covar.initialize(noise=noise_tensor)
                    except Exception:
                        logger.debug("Could not initialize likelihood noise explicitly.", exc_info=True)

            raw_noise = getattr(likelihood, "raw_noise", None)
            if raw_noise is not None:
                raw_noise.requires_grad_(False)

        self.model = model
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

        logger.info(
            "GP model (re)fit complete with deterministic observation noise level %.3e.",
            self._target_noise_level(),
        )
