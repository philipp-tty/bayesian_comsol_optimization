"""Gaussian Process surrogate model wrapping BoTorch/GPyTorch."""

from __future__ import annotations

import logging

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor

logger = logging.getLogger(__name__)


class GPSurrogate:
    """Gaussian Process surrogate wrapping BoTorch's SingleTaskGP.

    For single-objective problems a single ``SingleTaskGP`` is used.  For
    multi-objective problems an independent ``SingleTaskGP`` per objective is
    composed via ``ModelListGP``.

    Parameters
    ----------
    n_objectives:
        Number of objectives.  When greater than 1, a ``ModelListGP`` is used.
    bounds:
        Input bounds tensor of shape ``(2, d)`` for input normalization.
    """

    def __init__(
        self,
        n_objectives: int = 1,
        bounds: Tensor | None = None,
    ) -> None:
        self._n_objectives = n_objectives
        self._bounds = bounds
        self._model: SingleTaskGP | ModelListGP | None = None
        self._mll: ExactMarginalLogLikelihood | SumMarginalLogLikelihood | None = None

    @property
    def model(self) -> SingleTaskGP | ModelListGP:
        """Return the fitted GP model.

        Raises ``RuntimeError`` if the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._model

    @property
    def n_objectives(self) -> int:
        return self._n_objectives

    def fit(self, X: Tensor, Y: Tensor) -> None:
        """Fit the GP model to observed data.

        Parameters
        ----------
        X:
            Training inputs of shape ``(n, d)`` in the unit hypercube.
        Y:
            Training targets of shape ``(n, m)`` where *m* is the number
            of objectives.
        """
        X = X.detach().clone().to(dtype=torch.double)
        Y = Y.detach().clone().to(dtype=torch.double)

        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        d = X.shape[-1]
        bounds = self._bounds
        if bounds is None:
            bounds = torch.stack([torch.zeros(d, dtype=torch.double),
                                  torch.ones(d, dtype=torch.double)])

        if self._n_objectives == 1:
            self._model = SingleTaskGP(
                train_X=X,
                train_Y=Y,
                input_transform=Normalize(d=d, bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        else:
            models = []
            for i in range(self._n_objectives):
                m = SingleTaskGP(
                    train_X=X,
                    train_Y=Y[:, i : i + 1],
                    input_transform=Normalize(d=d, bounds=bounds),
                    outcome_transform=Standardize(m=1),
                )
                models.append(m)
            self._model = ModelListGP(*models)
            self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        fit_gpytorch_mll(self._mll)
        logger.debug(
            "Fitted GP surrogate on %d observations, %d inputs, %d objective(s).",
            X.shape[0], d, self._n_objectives,
        )

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Compute posterior mean and variance at *X*.

        Parameters
        ----------
        X:
            Evaluation points of shape ``(n, d)``.

        Returns
        -------
        mean:
            Posterior mean, shape ``(n, m)``.
        variance:
            Posterior variance, shape ``(n, m)``.
        """
        model = self.model
        X = X.detach().clone().to(dtype=torch.double)
        posterior = model.posterior(X)
        mean = posterior.mean
        variance = posterior.variance
        return mean, variance
