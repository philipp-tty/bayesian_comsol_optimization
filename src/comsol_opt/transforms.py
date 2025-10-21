"""Normalization utilities for fill-factor parameters."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
import torch


class FillFactorTransform:
    """
    Convert fill-factor values between the physical domain and the normalized unit interval.

    In this context, the fill factor is defined as an area fraction ``f`` that lies in ``(0, 1)``.
    """

    def __init__(self, bounds: Tuple[float, float]):
        if len(bounds) != 2:
            raise ValueError("fill_factor_bounds must be a tuple of length 2.")

        fill_min, fill_max = map(float, bounds)
        if not (0 < fill_min < fill_max < 1):
            raise ValueError("fill_factor_bounds must lie within (0, 1) with min < max.")

        self._min = fill_min
        self._max = fill_max
        self._span = self._max - self._min

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self._min, self._max)

    def to_physical(self, scaled: Union[float, Iterable[float], np.ndarray, torch.Tensor]):
        """
        Map normalized value(s) in [0, 1] to the physical fill-factor domain.
        """
        if isinstance(scaled, torch.Tensor):
            return self._min + self._span * scaled

        scaled_arr = np.asarray(scaled, dtype=float)
        physical = self._min + self._span * scaled_arr
        if np.isscalar(scaled) or np.ndim(scaled_arr) == 0:
            return float(physical)
        return physical

    def to_unit(self, fill_factor: Union[float, Iterable[float], np.ndarray, torch.Tensor]):
        """
        Map physical fill-factor value(s) to the normalized [0, 1] domain.
        """
        if isinstance(fill_factor, torch.Tensor):
            return (fill_factor - self._min) / self._span

        fill_arr = np.asarray(fill_factor, dtype=float)
        scaled = (fill_arr - self._min) / self._span
        if np.isscalar(fill_factor) or np.ndim(fill_arr) == 0:
            return float(scaled)
        return scaled

    def clip_physical(self, fill_factor: Union[float, Iterable[float], np.ndarray, torch.Tensor]):
        """
        Clip physical fill-factor value(s) to stay within the configured bounds.
        """
        if isinstance(fill_factor, torch.Tensor):
            return torch.clamp(fill_factor, self._min, self._max)

        clipped = np.clip(fill_factor, self._min, self._max)
        clipped_arr = np.asarray(clipped, dtype=float)
        if np.isscalar(fill_factor) or np.ndim(clipped_arr) == 0:
            return float(clipped_arr)
        return clipped_arr

    def ensure_unit(self, scaled: Union[float, Iterable[float], np.ndarray, torch.Tensor]):
        """
        Clamp normalized value(s) to [0, 1].
        """
        if isinstance(scaled, torch.Tensor):
            return torch.clamp(scaled, 0.0, 1.0)
        return np.clip(scaled, 0.0, 1.0)

