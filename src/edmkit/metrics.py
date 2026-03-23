from typing import TYPE_CHECKING, Callable, TypeAlias

import numpy as np

MetricFunc: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""MetricFunc is a function that takes (predictions, observations) and returns a metric value."""


def validate_and_promote(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shape match and promote 1D to 2D."""
    if predictions.shape != observations.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs observations {observations.shape}")
    if predictions.ndim not in (1, 2, 3):
        raise ValueError(f"Expected 1D, 2D, or 3D arrays, got {predictions.ndim}D")
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]
    return predictions, observations


def rhos(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Pearson correlation per dimension.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as predictions.

    Returns
    -------
    np.ndarray
        ``(1,)`` for 1D input, ``(D,)`` for 2D, ``(B, D)`` for 3D.
    """
    predictions, observations = validate_and_promote(predictions, observations)

    p_centered = predictions - predictions.mean(axis=-2, keepdims=True)
    o_centered = observations - observations.mean(axis=-2, keepdims=True)
    num = (p_centered * o_centered).sum(axis=-2)
    denom = np.sqrt((p_centered**2).sum(axis=-2) * (o_centered**2).sum(axis=-2))
    safe_denom = np.where(denom > 0, denom, 1.0)

    return np.where(denom > 0, num / safe_denom, 0.0)


def mean_rho(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Mean Pearson correlation.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as predictions.

    Returns
    -------
    np.ndarray
        ``()`` for 1D/2D input, ``(B,)`` for 3D input.
    """
    return rhos(predictions, observations).mean(axis=-1)


def rmse(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Root Mean Squared Error.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    np.ndarray
        ``()`` for 1D/2D input, ``(B,)`` for 3D input.
    """
    predictions, observations = validate_and_promote(predictions, observations)

    # 2D: (N, D) -> (N,) -> ()
    # 3D: (B, N, D) -> (B, N) -> (B,)
    return np.sqrt(((predictions - observations) ** 2).mean(axis=-1).mean(axis=-1))


def mae(
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Error.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as predictions.

    Returns
    -------
    np.ndarray
        ``()`` for 1D/2D input, ``(B,)`` for 3D input.
    """
    predictions, observations = validate_and_promote(predictions, observations)

    # 2D: (N, D) -> (N,) -> ()
    # 3D: (B, N, D) -> (B, N) -> (B,)
    return np.abs(predictions - observations).mean(axis=-1).mean(axis=-1)


if TYPE_CHECKING:
    func: MetricFunc

    func = rhos
    func = mean_rho
    func = rmse
    func = mae
