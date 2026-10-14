from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.simplex_projection.knn import knn
from edmkit.util import pairwise_distance

__all__ = ["linear_operator"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def linear_operator(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def linear_operator(
    X: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
) -> Tensor: ...


def linear_operator(
    X,
    Q,
    *,
    k=None,
    mask=None,
):
    """
    Build the linear operator `S` of simplex projection, such that ``S @ Y`` equals ``simplex_projection(X, Y, Q)`` for any target `Y`.

    Parameters
    ----------
    X : np.ndarray or Tensor
        The input data of shape (N,) or (N, E) or (B, N, E)
    Q : np.ndarray or Tensor
        The query points of shape (M,) or (M, E) or (B, M, E) for which to find the nearest neighbors in `X`.
    k : int or None, default None
        The number of nearest neighbors to use. If None, uses E + 1, where E is the dimension of `X`.
    mask : np.ndarray or Tensor or None
        Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include when finding nearest neighbors.

    Returns
    -------
    S : np.ndarray or Tensor
        The operator of shape (M, N) or (B, M, N).

    Raises
    ------
    ValueError
        - If `k` is not positive.
        - If the batch size or dimension of `X` and `Q` do not match.
        - If `mask` does not have shape (M, N) or (B, M, N).
        - If fewer than `k` library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.simplex_projection import linear_operator, simplex_projection

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Q = rng.normal(size=(50, 3))
    Y = rng.normal(size=(200, 4))

    assert np.allclose(linear_operator(X, Q) @ Y, simplex_projection(X, Y, Q))
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Q, k=k, mask=mask)

    return _tensor(X, Q, k=k, mask=mask)


def _numpy(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Q.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X and Q must both be 2D or both be 3D arrays, got X.ndim={X.ndim}, Q.ndim={Q.ndim}")
    if Q.shape[:-2] != X.shape[:-2] or Q.shape[-1] != X.shape[-1]:
        raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

    return_ndim = X.ndim

    # treat the unbatched case as a single batch
    if X.ndim == 2:
        X, Q = X[None], Q[None]
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

    # clamp to avoid division by zero
    d_min = np.fmax(distances.min(axis=-1, keepdims=True), 1e-6)  # (B, M, 1)
    weights = np.exp(-distances / d_min)  # (B, M, k)
    weights /= weights.sum(axis=-1, keepdims=True)

    S = np.zeros((B, M, N))
    np.put_along_axis(S, indices, weights, axis=-1)

    if return_ndim == 2:
        return S[0]  # (M, N)
    return S  # (B, M, N)


def _tensor(
    X: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Q.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X and Q must both be 2D or both be 3D tensors, got X.ndim={X.ndim}, Q.ndim={Q.ndim}")
    if Q.shape[:-2] != X.shape[:-2] or Q.shape[-1] != X.shape[-1]:
        raise ValueError(f"batch size and dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

    return_ndim = X.ndim

    # treat the unbatched case as a single batch
    if X.ndim == 2:
        X, Q = X[None], Q[None]
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

    # neighbors are sorted, so the first is the nearest; clamp to avoid division by zero
    d_min = distances[..., :1].clamp(min_=1e-6)  # (B, M, 1)
    weights = (-distances / d_min).exp()  # (B, M, k)
    weights = weights / weights.sum(axis=-1, keepdim=True)

    S = weights.zeros(B, M, N, dtype=weights.dtype, device=weights.device).scatter(-1, indices, weights)  # (B, M, N)

    if return_ndim == 2:
        return S[0]  # (M, N)
    return S  # (B, M, N)
