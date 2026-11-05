from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.simplex_projection.knn import knn
from edmkit.util import materialize, pairwise_distance, topk

__all__ = ["simplex_projection"]

if TYPE_CHECKING:
    from tinygrad import Tensor

    from edmkit.types import Mask


@overload
def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: Mask | None = None,
) -> np.ndarray: ...


@overload
def simplex_projection(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Mask | None = None,
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
    mask : Mask or None
        ``mask(i, j)``, or ``mask(b, i, j)``, indicating for each query in `Q` which library points to include when finding nearest neighbors.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.
        Shape (M,) if `Y` is 1D, (M, E') if `Y` is 2D, or (B, M, E') if `Y` is 3D.

    Raises
    ------
    ValueError
        - If `k` is not positive.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If the batch size or dimension of `X` and `Q` do not match.
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
    X = embedding[: lib_size - shift]
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
    mask: Mask | None = None,
):
    return_ndim = Y.ndim

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
    if X.ndim == 2:
        X, Y, Q = X[None], Y[None], Q[None]

    B, _, E = X.shape
    M = Q.shape[1]
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    predictions = np.empty((B, M, Y.shape[-1]), dtype=np.result_type(np.float64, Y.dtype))
    for b in range(B):
        distances, indices = knn(X[b], Q[b], k, partial(mask, b) if return_ndim == 3 and mask is not None else mask)

        # Neighbors are sorted. Clamp the scale and reuse the distance buffer.
        d_min = np.fmax(distances[:, :1], 1e-6)
        np.divide(distances, -d_min, out=distances)
        weights = np.exp(distances, out=distances)

        prediction = predictions[b]
        np.matmul(weights[:, None, :], Y[b, indices], out=prediction[:, None, :])
        prediction /= weights.sum(axis=-1, keepdims=True)

    if return_ndim == 1:
        return predictions[0, :, 0]  # (M,)
    if return_ndim == 2:
        return predictions[0]  # (M, E')
    return predictions  # (B, M, E')


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Mask | None = None,
):
    # Lazy import: tinygrad starts a per-CPU async-executor pool at import time
    # Loading it only when needed keeps the numpy path free of that scheduler pressure.
    from tinygrad import Tensor, dtypes

    return_ndim = Y.ndim

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
    _mask = None if mask is None else Tensor(materialize(mask, Q.shape[:-1] + X.shape[-2:-1]))

    # treat the unbatched case as a single batch
    if X.ndim == 2:
        X, Y, Q = X[None], Y[None], Q[None]
        if _mask is not None:
            _mask = _mask[None]

    B, N, E = (int(X.shape[0]), int(X.shape[1]), int(X.shape[2]))
    M = int(Q.shape[1])
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    # clamp to avoid NaN gradient
    D = pairwise_distance(Q, X).clamp(min_=1e-12).sqrt()  # (B, M, N)

    if _mask is not None:
        D = _mask.where(D, float("inf"))

    distances, indices = topk(D, k, largest=False)  # (B, M, k)

    offsets = Tensor.arange(B, dtype=dtypes.int32).reshape(B, 1, 1) * N
    flat_indices = (indices + offsets).reshape(B * M, k)
    Y_neighbors = Y.reshape(B * N, -1)[flat_indices].reshape(B, M, k, -1)  # (B, M, k, E')

    # neighbors are sorted, so the first is the nearest; clamp to avoid division by zero
    d_min = distances[..., :1].clamp(min_=1e-6)  # (B, M, 1)
    weights: Tensor = (-distances / d_min).exp()  # (B, M, k)

    predictions: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=-2) / weights.sum(axis=-1, keepdim=True)  # (B, M, E')

    if return_ndim == 1:
        return predictions[0, :, 0]  # (M,)
    if return_ndim == 2:
        return predictions[0]  # (M, E')
    return predictions  # (B, M, E')


if TYPE_CHECKING:
    from edmkit.types import PredictFunc

    func: PredictFunc = simplex_projection
