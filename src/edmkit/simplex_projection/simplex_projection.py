from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.simplex_projection.knn import knn
from edmkit.util import pairwise_distance

__all__ = ["simplex_projection"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def simplex_projection(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
) -> Tensor: ...


def simplex_projection(
    X,
    Y,
    Q,
    *,
    k=None,
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
    k : int or None, default None
        The number of nearest neighbors to use. If None, uses E + 1, where E is the dimension of `X`.
    mask : np.ndarray or Tensor or None
        Boolean mask of shape (N,) or (B, N) indicating which library points to include when finding nearest neighbors for the queries in `Q`.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If `k` is not positive.
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
        return _numpy(X, Y, Q, k=k, mask=mask)

    return _tensor(X, Y, Q, k=k, mask=mask)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
):
    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Y.ndim == Q.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
    if Q.shape[:-2] != X.shape[:-2] or Q.shape[-1] != X.shape[-1]:
        raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")

    # treat the unbatched case as a single batch
    batched = X.ndim == 3
    if not batched:
        X, Y, Q = X[None], Y[None], Q[None]
        if mask is not None:
            mask = mask[None]

    B, _, E = X.shape
    M = Q.shape[1]
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    distances = np.empty((B, M, k))
    indices = np.empty((B, M, k), dtype=np.intp)
    for b in range(B):
        if mask is None:
            distances[b], indices[b] = knn(X[b], Q[b], k)
        else:
            positions = np.flatnonzero(mask[b])
            distances[b], neighbors = knn(X[b, positions], Q[b], k)
            indices[b] = positions[neighbors]

    batch_index = np.arange(B)[:, None, None]  # (B, 1, 1)
    Y_neighbors = Y[batch_index, indices]  # (B, M, k, E')

    # clamp to avoid division by zero
    d_min = np.fmax(distances.min(axis=-1, keepdims=True), 1e-6)  # (B, M, 1)
    weights = np.exp(-distances / d_min)  # (B, M, k)

    weighted_sum = np.matmul(weights[..., None, :], Y_neighbors).squeeze(-2)  # (B, M, E')
    predictions = weighted_sum / weights.sum(axis=-1, keepdims=True)  # (B, M, E')

    if not batched:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
):
    # Lazy import: tinygrad starts a per-CPU async-executor pool at import time
    # Loading it only when needed keeps the numpy path free of that scheduler pressure.
    from tinygrad import Tensor, dtypes

    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Y.ndim == Q.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D tensors, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
    if Q.shape[:-2] != X.shape[:-2] or Q.shape[-1] != X.shape[-1]:
        raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")

    # treat the unbatched case as a single batch
    batched = X.ndim == 3
    if not batched:
        X, Y, Q = X[None], Y[None], Q[None]
        if mask is not None:
            mask = mask[None]

    B, N, E = (int(X.shape[0]), int(X.shape[1]), int(X.shape[2]))
    M = int(Q.shape[1])
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    # clamp to avoid NaN gradient
    D = pairwise_distance(Q, X).clamp(min_=1e-12).sqrt()  # (B, M, N)

    if mask is not None:
        D = mask.unsqueeze(-2).where(D, float("inf"))

    distances, indices = D.topk(k, dim=-1, largest=False, sorted_=True)  # (B, M, k)

    offsets = Tensor.arange(B, dtype=dtypes.int32).reshape(B, 1, 1) * N
    flat_indices = (indices + offsets).reshape(B * M, k)
    Y_neighbors = Y.reshape(B * N, -1)[flat_indices].reshape(B, M, k, -1)  # (B, M, k, E')

    # neighbors are sorted, so the first is the nearest; clamp to avoid division by zero
    d_min = distances[..., :1].clamp(min_=1e-6)  # (B, M, 1)
    weights: Tensor = (-distances / d_min).exp()  # (B, M, k)

    weighted_sum: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=-2)  # (B, M, E')
    predictions: Tensor = weighted_sum / weights.sum(axis=-1, keepdim=True)  # (B, M, E')

    if not batched:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions


if TYPE_CHECKING:
    from edmkit.types import PredictFunc

    f: PredictFunc[np.ndarray] = simplex_projection
    g: PredictFunc[Tensor] = simplex_projection
