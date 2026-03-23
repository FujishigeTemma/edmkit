"""Prediction evaluation metrics.

Supports 1D ``(N,)``, 2D ``(N, D)``, and 3D ``(B, N, D)`` inputs.

- 1D inputs are auto-promoted to ``(N, 1)`` before processing.
- 3D inputs are handled per-batch-element and results are stacked.
- Aggregate functions (``mean_rho``, ``rmse``, ``mae``) return ``float``
  for 1D/2D input and ``(B,)`` array for 3D input.
- Per-dim functions (``rho_per_dim``, ``rmse_per_dim``, ``mae_per_dim``)
  return ``(D,)`` for 1D/2D input and ``(B, D)`` for 3D input.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Validation / promotion
# ---------------------------------------------------------------------------

def _validate_and_promote(
    predictions: np.ndarray, observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shape match and promote 1D to 2D."""
    if predictions.shape != observations.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"vs observations {observations.shape}"
        )
    if predictions.ndim not in (1, 2, 3):
        raise ValueError(
            f"Expected 1D, 2D, or 3D arrays, got {predictions.ndim}D"
        )
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]
    return predictions, observations


# ---------------------------------------------------------------------------
# Per-dim metrics
# ---------------------------------------------------------------------------

def rho_per_dim(
    predictions: np.ndarray, observations: np.ndarray,
) -> np.ndarray:
    """Pearson correlation per target dimension.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    np.ndarray
        ``(1,)`` for 1D input, ``(D,)`` for 2D, ``(B, D)`` for 3D.
    """
    predictions, observations = _validate_and_promote(predictions, observations)

    if predictions.ndim == 3:
        # (B, N, D) → per-column correlation → (B, D)
        p_centered = predictions - predictions.mean(axis=1, keepdims=True)
        o_centered = observations - observations.mean(axis=1, keepdims=True)
        num = (p_centered * o_centered).sum(axis=1)  # (B, D)
        denom = np.sqrt(
            (p_centered**2).sum(axis=1) * (o_centered**2).sum(axis=1)
        )  # (B, D)
        safe_denom = np.where(denom > 0, denom, 1.0)
        return np.where(denom > 0, num / safe_denom, 0.0)

    # 2D path: (N, D) → (D,)
    p_centered = predictions - predictions.mean(axis=0)
    o_centered = observations - observations.mean(axis=0)
    cov = (p_centered * o_centered).sum(axis=0)
    denom = np.sqrt((p_centered**2).sum(axis=0) * (o_centered**2).sum(axis=0))
    result = np.zeros_like(cov)
    mask = denom > 0
    np.divide(cov, denom, out=result, where=mask)
    return result


def rmse_per_dim(
    predictions: np.ndarray, observations: np.ndarray,
) -> np.ndarray:
    """Root Mean Squared Error per target dimension.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    np.ndarray
        ``(1,)`` for 1D input, ``(D,)`` for 2D, ``(B, D)`` for 3D.
    """
    predictions, observations = _validate_and_promote(predictions, observations)

    # axis=1 for 3D (B, N, D) → (B, D); axis=0 for 2D (N, D) → (D,)
    ax = 1 if predictions.ndim == 3 else 0
    return np.sqrt(np.mean((predictions - observations) ** 2, axis=ax))


def mae_per_dim(
    predictions: np.ndarray, observations: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Error per target dimension.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    np.ndarray
        ``(1,)`` for 1D input, ``(D,)`` for 2D, ``(B, D)`` for 3D.
    """
    predictions, observations = _validate_and_promote(predictions, observations)

    ax = 1 if predictions.ndim == 3 else 0
    return np.mean(np.abs(predictions - observations), axis=ax)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def mean_rho(
    predictions: np.ndarray, observations: np.ndarray,
) -> float | np.ndarray:
    """Mean Pearson correlation across targets.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    float or np.ndarray
        ``float`` for 1D/2D input, ``(B,)`` array for 3D input.
    """
    # Peek at ndim before promote to decide return type.
    is_batch = predictions.ndim == 3
    per_dim = rho_per_dim(predictions, observations)
    if is_batch:
        return per_dim.mean(axis=-1)  # (B, D) → (B,)
    return float(per_dim.mean())


def rmse(
    predictions: np.ndarray, observations: np.ndarray,
) -> float | np.ndarray:
    """Root Mean Squared Error over all elements.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    float or np.ndarray
        ``float`` for 1D/2D input, ``(B,)`` array for 3D input.
    """
    is_batch = predictions.ndim == 3
    per_dim = rmse_per_dim(predictions, observations)
    if is_batch:
        # (B, D) → (B,): RMS over dims = sqrt(mean(per_dim²))
        return np.sqrt((per_dim**2).mean(axis=-1))
    # scalar: sqrt(mean(per_dim²))
    return float(np.sqrt((per_dim**2).mean()))


def mae(
    predictions: np.ndarray, observations: np.ndarray,
) -> float | np.ndarray:
    """Mean Absolute Error over all elements.

    Parameters
    ----------
    predictions : np.ndarray
        ``(N,)``, ``(N, D)``, or ``(B, N, D)``.
    observations : np.ndarray
        Same shape as *predictions*.

    Returns
    -------
    float or np.ndarray
        ``float`` for 1D/2D input, ``(B,)`` array for 3D input.
    """
    is_batch = predictions.ndim == 3
    per_dim = mae_per_dim(predictions, observations)
    if is_batch:
        return per_dim.mean(axis=-1)  # (B, D) → (B,)
    return float(per_dim.mean())
