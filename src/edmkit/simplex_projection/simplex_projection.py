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
        Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include when finding nearest neighbors.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If `k` is not positive.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `mask` does not have shape (M, N) or (B, M, N).
        - If fewer than `k` library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import embed
    from edmkit.simplex_projection import simplex_projection

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    coordinates = np.array([[0, tau * j] for j in range(E)])
    embedding, _ = embed(coordinates, x)
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
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

    # treat the unbatched case as a single batch
    batched = X.ndim == 3
    if not batched:
        X, Y, Q = X[None], Y[None], Q[None]
        if mask is not None:
            mask = mask[None]

    B, N, E = X.shape
    M = Q.shape[1]
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    distances = np.empty((B, M, k))
    indices = np.empty((B, M, k), dtype=np.intp)
    for b in range(B):
        if mask is None:
            distances[b], indices[b] = knn(X[b], Q[b], k)
        else:  # over-fetch, then drop the masked-out candidates
            n_exclude = int((~mask[b]).sum(axis=-1).max())
            if N - n_exclude < k:
                raise ValueError(
                    f"Not enough unmasked points in X to find {k} neighbors, got N={N} with up to {n_exclude} points masked out per query"
                )
            _distances, _indices = knn(X[b], Q[b], k + n_exclude)
            _distances = np.where(np.take_along_axis(mask[b], _indices, axis=-1), _distances, np.inf)
            top_k = np.argsort(_distances, axis=-1)[..., :k]
            distances[b] = np.take_along_axis(_distances, top_k, axis=-1)
            indices[b] = np.take_along_axis(_indices, top_k, axis=-1)

    batch_index = np.arange(B)[:, None, None]  # (B, 1, 1)
    Y_neighbors = Y[batch_index, indices]  # (B, M, k, E')

    # clamp to avoid division by zero
    d_min = np.fmax(distances.min(axis=-1, keepdims=True), 1e-6)  # (B, M, 1)
    weights = np.exp(-distances / d_min)  # (B, M, k)

    predictions = np.matmul(weights[..., None, :], Y_neighbors).squeeze(-2) / weights.sum(axis=-1, keepdims=True)  # (B, M, E')

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
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

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
        D = mask.where(D, float("inf"))

    distances, indices = D.topk(k, dim=-1, largest=False, sorted_=True)  # (B, M, k)

    offsets = Tensor.arange(B, dtype=dtypes.int32).reshape(B, 1, 1) * N
    flat_indices = (indices + offsets).reshape(B * M, k)
    Y_neighbors = Y.reshape(B * N, -1)[flat_indices].reshape(B, M, k, -1)  # (B, M, k, E')

    # neighbors are sorted, so the first is the nearest; clamp to avoid division by zero
    d_min = distances[..., :1].clamp(min_=1e-6)  # (B, M, 1)
    weights: Tensor = (-distances / d_min).exp()  # (B, M, k)

    predictions: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=-2) / weights.sum(axis=-1, keepdim=True)  # (B, M, E')

    if not batched:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions


if TYPE_CHECKING:
    from edmkit.types import PredictFunc

    func: PredictFunc = simplex_projection
