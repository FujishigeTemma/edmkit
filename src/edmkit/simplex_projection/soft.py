from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.special import expit

from edmkit.util import pairwise_distance, pairwise_distance_np

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def soft_simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> np.ndarray: ...


@overload
def soft_simplex_projection(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    mask: Tensor | None = None,
    softness: float = 0.02,
) -> Tensor: ...


def soft_simplex_projection(
    X,
    Y,
    Q,
    *,
    mask=None,
    softness=0.02,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`, with a soft boundary between neighbors and non-neighbors.

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
    softness : float, default 0.02
        Width of the boundary between neighbors and non-neighbors, as a fraction of the neighborhood radius.
        For distinct boundary distances:
        ``soft_simplex_projection(X, Y, Q, softness) -> simplex_projection(X, Y, Q) as softness -> 0``

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If `softness` is not positive.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `X` does not contain at least `E + 2` points.

    Examples
    --------
    ```python
    import numpy as np
    from tinygrad import Tensor

    from edmkit.embedding import lagged_embed
    from edmkit.simplex_projection import soft_simplex_projection

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N, dtype=np.float32)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    embedding = lagged_embed(x, tau=tau, e=E)
    shift = tau * (E - 1)

    lib_size = 200
    Tp = 1
    X = Tensor(embedding[:lib_size - shift])
    Y = Tensor(x[shift + Tp : lib_size + Tp])
    Q = Tensor(embedding[lib_size - shift : -Tp])
    actual = x[lib_size + Tp :]

    predictions = soft_simplex_projection(X, Y, Q).numpy()

    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"Correlation: {correlation:.3f}")
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Y, Q, mask=mask, softness=softness)

    return _tensor(X, Y, Q, mask=mask, softness=softness)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> np.ndarray:
    if softness <= 0:
        raise ValueError(f"softness must be positive, got softness={softness}")

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

    _, N, E = X.shape
    k: int = E + 1
    if N < k + 1:
        raise ValueError(f"Not enough points in X to find {k + 1} neighbors, got N={N}")

    # clamp to align with _tensor()
    D = np.sqrt(np.maximum(pairwise_distance_np(Q, X), 1e-12))  # (B, M, N)

    if mask is not None:
        # a finite sentinel instead of inf to avoid NaN
        D = np.where(mask[:, None, :], D, D.max(axis=-1, keepdims=True) * 2 + 1)

    neighbors = np.sort(np.partition(D, k, axis=-1)[..., : k + 1], axis=-1)  # (B, M, k + 1)

    # clamp to avoid division by zero
    d_min = np.maximum(neighbors[..., :1], 1e-6)  # (B, M, 1)
    # radius is the midpoint between the k-th and (k+1)-th nearest neighbors
    radius = (neighbors[..., k - 1 : k] + neighbors[..., k : k + 1]) / 2  # (B, M, 1)
    radius = np.maximum(radius, 1e-6)

    weights = np.exp(-D / d_min) * expit((radius - D) / (softness * radius))  # (B, M, N)
    if mask is not None:
        weights = np.where(mask[:, None, :], weights, 0.0)

    predictions = np.matmul(weights, Y) / weights.sum(axis=-1, keepdims=True)  # (B, M, E')

    if not batched:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    mask: Tensor | None = None,
    softness: float = 0.02,
) -> Tensor:
    if softness <= 0:
        raise ValueError(f"softness must be positive, got softness={softness}")

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

    N = int(X.shape[1])
    E = int(X.shape[2])
    k: int = E + 1
    if N < k + 1:
        raise ValueError(f"Not enough points in X to find {k + 1} neighbors, got N={N}")

    # clamp to avoid NaN gradient
    D = pairwise_distance(Q, X).clamp(min_=1e-12).sqrt()  # (B, M, N)

    if mask is not None:
        # a finite sentinel instead of inf to avoid NaN
        D = mask.unsqueeze(-2).where(D, D.max(axis=-1, keepdim=True) * 2 + 1)

    neighbors = D.topk(k + 1, dim=-1, largest=False, sorted_=True)[0]  # (B, M, k + 1)

    # clamp to avoid division by zero
    d_min = neighbors[..., :1].clamp(min_=1e-6)  # (B, M, 1)
    # radius is the midpoint between the k-th and (k+1)-th nearest neighbors
    radius = (neighbors[..., k - 1 : k] + neighbors[..., k : k + 1]) / 2  # (B, M, 1)
    radius = radius.clamp(min_=1e-6)

    gate = ((radius - D) / (softness * radius)).sigmoid()  # (B, M, N)
    weights = (-D / d_min).exp() * gate  # (B, M, N)
    if mask is not None:
        weights = mask.unsqueeze(-2).where(weights, 0)

    predictions: Tensor = weights.matmul(Y) / weights.sum(axis=-1, keepdim=True)  # (B, M, E')

    if not batched:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions
