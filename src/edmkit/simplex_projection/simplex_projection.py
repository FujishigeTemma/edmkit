from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.simplex_projection.knn import knn
from edmkit.util import pairwise_distance

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def simplex_projection(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor: ...


def simplex_projection(
    X,
    Y,
    Q,
    *,
    mask=None,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`.

    Parameters
    ----------
    X : np.ndarray or Tensor
        The input data of shape (N,) or (N, E) or (B, N, E)
    Y : np.ndarray or Tensor
        The target data of shape (N,) or (N, E') or (B, N, E')
    Q : np.ndarray or Tensor
        The query points of shape (M,) or (M, E) or (B, M, E) for which to find the nearest neighbors in `X`.
    mask : np.ndarray or Tensor or None
        Boolean mask of shape (N,) or (B, N) indicating which library points to include when finding nearest neighbors for the queries in `Q`.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import lagged_embed
    from edmkit.simplex_projection import simplex_projection

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    embedding = lagged_embed(x, tau=tau, e=E)
    shift = tau * (E - 1)

    lib_size = 200
    Tp = 1
    X = embedding[:lib_size - shift]
    Y = x[shift + Tp : lib_size + Tp]
    Q = embedding[lib_size - shift : -Tp]
    actual = x[lib_size + Tp :]

    predictions = simplex_projection(X, Y, Q)

    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"Correlation: {correlation:.3f}")
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Y, Q, mask=mask)

    return _tensor(X, Y, Q, mask=mask)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
):
    # ensure 2D or 3D arrays
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    # X (N, E), Y (N, E'), Q (M, E), mask (N,)
    if X.ndim == 2 and Y.ndim == 2 and Q.ndim == 2:
        N, E = X.shape
        if Y.shape[0] != N:
            raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

        k: int = E + 1

        if mask is not None:
            X = X[mask]
            Y = Y[mask]

        distances, indices = knn(X, Q, k)
        Y_neighbors = Y[indices]  # (M, k, E')

        # clamp to avoid division by zero
        d_min = np.fmax(distances.min(axis=1, keepdims=True), 1e-6)  # (M, 1)
        weights = np.exp(-distances / d_min)  # (M, k)

        weighted_sum = np.matmul(weights[:, None, :], Y_neighbors).squeeze(-2)  # (M, E')
        predictions = weighted_sum / weights.sum(axis=1, keepdims=True)

        return predictions.squeeze()  # (M,) or (M, E')
    # X (B, N, E), Y (B, N, E'), Q (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and Q.ndim == 3:
        B, N, E = X.shape
        if Y.shape[0] != B or Y.shape[1] != N:
            raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
        if Q.shape[0] != B or Q.shape[2] != E:
            raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")

        k: int = E + 1
        M = Q.shape[1]

        distances = np.empty((B, M, k))
        indices = np.empty((B, M, k), dtype=np.intp)
        if mask is None:
            for b in range(B):
                distances[b], indices[b] = knn(X[b], Q[b], k)
        else:
            for b in range(B):
                positions = np.flatnonzero(mask[b])
                distances[b], _indices = knn(X[b, positions], Q[b], k)
                indices[b] = positions[_indices]

        batch_idx = np.arange(B)[:, None, None]  # (B, 1, 1)
        Y_neighbors = Y[batch_idx, indices]  # (B, M, k, E')

        # clamp to avoid division by zero
        d_min = np.fmax(distances.min(axis=2, keepdims=True), 1e-6)  # (B, M, 1)
        weights = np.exp(-distances / d_min)  # (B, M, k)

        weighted_sum = np.matmul(weights[..., None, :], Y_neighbors).squeeze(-2)  # (B, M, E')
        predictions = weighted_sum / weights.sum(axis=2, keepdims=True)  # (B, M, E')

        return predictions
    else:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    mask: Tensor | None = None,
):
    # Lazy import: tinygrad starts a per-CPU async-executor pool at import time
    # Loading it only when needed keeps the numpy path free of that scheduler pressure.
    from tinygrad import Tensor, dtypes

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    # X (N, E), Y (N, E'), Q (M, E)
    if X.ndim == 2 and Y.ndim == 2 and Q.ndim == 2:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

        if mask is not None:
            X = X[mask]
            Y = Y[mask]

        D = pairwise_distance(Q, X).sqrt()  # (M, N)

        k: int = int(X.shape[1]) + 1

        distances, indices = D.topk(k, dim=1, largest=False, sorted_=True)  # (M, k)
        Y_neighbors = Y[indices]  # (M, k, E')

        d_min = distances[:, :1].clip(min_=1e-6)  # (M, 1)
        weights: Tensor = (-distances / d_min).exp()  # (M, k)

        weighted_sum: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=1)  # (M, E')
        predictions: Tensor = weighted_sum / weights.sum(axis=1, keepdim=True)  # (M, E')

        return predictions.squeeze()
    # X (B, N, E), Y (B, N, E'), Q (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and Q.ndim == 3:
        if mask is not None:
            raise NotImplementedError("Tensor-based 3D simplex_projection with mask is not supported.")
        B, N, E = (int(X.shape[0]), int(X.shape[1]), int(X.shape[2]))
        if Y.shape[0] != B or Y.shape[1] != N:
            raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
        if Q.shape[0] != B or Q.shape[2] != E:
            raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")

        D = pairwise_distance(Q, X).sqrt()  # (B, M, N)

        k: int = E + 1

        distances, indices = D.topk(k, dim=2, largest=False, sorted_=True)  # (B, M, k)

        offsets = Tensor.arange(B, dtype=dtypes.int32).reshape(B, 1, 1) * N
        flat_indices = (indices + offsets).reshape(B * Q.shape[1], k)
        Y_flat = Y.reshape(B * N, Y.shape[-1])
        Y_neighbors = Y_flat[flat_indices].reshape(B, Q.shape[1], k, Y.shape[-1])

        d_min = distances[:, :, :1].clip(min_=1e-6)  # (B, M, 1)
        weights: Tensor = (-distances / d_min).exp()  # (B, M, k)

        weighted_sum: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=2)  # (B, M, E')
        predictions: Tensor = weighted_sum / weights.sum(axis=2, keepdim=True)  # (B, M, E')

        return predictions
    else:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")


if TYPE_CHECKING:
    from edmkit.types import PredictFunc

    f: PredictFunc[np.ndarray] = simplex_projection
    g: PredictFunc[Tensor] = simplex_projection
