from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

__all__ = ["mae"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def mae(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray: ...


@overload
def mae(predictions: Tensor, observations: Tensor) -> Tensor: ...


def mae(predictions, observations):
    """Mean Absolute Error.

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

    # 2D: (N, D) -> (N,) -> ()
    # 3D: (B, N, D) -> (B, N) -> (B,)
    return np.abs(predictions - observations).mean(axis=-1).mean(axis=-1)


def _tensor(predictions: Tensor, observations: Tensor) -> Tensor:
    if predictions.shape != observations.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}")
    if predictions.ndim not in (1, 2, 3):
        raise ValueError(f"Expected 1D, 2D, or 3D tensors, got {predictions.ndim}D")

    # treat a 1D series as a single target dimension
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(-1)
        observations = observations.unsqueeze(-1)

    # 2D: (N, D) -> (N,) -> ()
    # 3D: (B, N, D) -> (B, N) -> (B,)
    return (predictions - observations).abs().mean(axis=-1).mean(axis=-1)
