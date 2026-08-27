from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

__all__ = ["pearson_correlation"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def pearson_correlation(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray: ...


@overload
def pearson_correlation(predictions: Tensor, observations: Tensor) -> Tensor: ...


def pearson_correlation(predictions, observations):
    """Mean Pearson correlation over the target dimensions.

    The correlation is computed per target dimension along the sample axis,
    then averaged over the dimensions.

    Parameters
    ----------
    predictions : np.ndarray or Tensor
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray or Tensor
        Same shape as `predictions`.

    Returns
    -------
    np.ndarray or Tensor
        ``()`` for 1D/2D input, ``(B,)`` for 3D input.

    Raises
    ------
    ValueError
        - If `predictions` and `observations` have different shapes.
        - If the inputs are not 1D, 2D, or 3D.
    """
    if isinstance(predictions, np.ndarray):
        return _numpy(predictions, observations)

    return _tensor(predictions, observations)


def _numpy(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    if predictions.shape != observations.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}")
    if predictions.ndim not in (1, 2, 3):
        raise ValueError(f"Expected 1D, 2D, or 3D arrays, got {predictions.ndim}D")

    # treat a 1D series as a single target dimension
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]

    predictions_centered = predictions - predictions.mean(axis=-2, keepdims=True)
    observations_centered = observations - observations.mean(axis=-2, keepdims=True)

    numerator = (predictions_centered * observations_centered).sum(axis=-2)
    denominator = np.sqrt((predictions_centered**2).sum(axis=-2) * (observations_centered**2).sum(axis=-2))

    denominator = np.where(denominator > 0, denominator, np.nan)

    # 2D: (D,) -> ()
    # 3D: (B, D) -> (B,)
    return (numerator / denominator).mean(axis=-1)


def _tensor(predictions: Tensor, observations: Tensor) -> Tensor:
    if predictions.shape != observations.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}")
    if predictions.ndim not in (1, 2, 3):
        raise ValueError(f"Expected 1D, 2D, or 3D tensors, got {predictions.ndim}D")

    # treat a 1D series as a single target dimension
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(-1)
        observations = observations.unsqueeze(-1)

    predictions_centered = predictions - predictions.mean(axis=-2, keepdim=True)
    observations_centered = observations - observations.mean(axis=-2, keepdim=True)

    numerator = (predictions_centered * observations_centered).sum(axis=-2)
    denominator = predictions_centered.pow(2).sum(axis=-2) * observations_centered.pow(2).sum(axis=-2)

    denominator = (denominator > 0).where(denominator.clamp(min_=1e-12).sqrt(), float("nan"))

    # 2D: (D,) -> ()
    # 3D: (B, D) -> (B,)
    return (numerator / denominator).mean(axis=-1)
