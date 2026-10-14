from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.special import expit

from edmkit.util import pairwise_distance, pairwise_distance_np

__all__ = ["soft_linear_operator"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def soft_linear_operator(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> np.ndarray: ...


@overload
def soft_linear_operator(
    X: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
    softness: float = 0.02,
) -> Tensor: ...


def soft_linear_operator(
    X,
    Q,
    *,
    k=None,
    mask=None,
    softness=0.02,
):
    """
    Build the linear operator `S` of soft simplex projection, such that ``S @ Y`` equals ``soft_simplex_projection(X, Y, Q)`` for any target `Y`.

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
    softness : float, default 0.02
        Width of the boundary between neighbors and non-neighbors, as a fraction of the neighborhood radius.

    Returns
    -------
    S : np.ndarray or Tensor
        The operator of shape (M, N) or (B, M, N).

    Raises
    ------
    ValueError
        - If `k` is not positive.
        - If `softness` is not positive.
        - If the batch size or dimension of `X` and `Q` do not match.
        - If `mask` does not have shape (M, N) or (B, M, N).
        - If fewer than `k + 1` library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.simplex_projection import soft_linear_operator, soft_simplex_projection

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Q = rng.normal(size=(50, 3))
    Y = rng.normal(size=(200, 4))

    assert np.allclose(soft_linear_operator(X, Q) @ Y, soft_simplex_projection(X, Y, Q))
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Q, k=k, mask=mask, softness=softness)

    return _tensor(X, Q, k=k, mask=mask, softness=softness)


def _numpy(
    X: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> np.ndarray:
    if softness <= 0:
        raise ValueError(f"softness must be positive, got softness={softness}")

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

    N, E = X.shape[-2:]
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    n_exclude = 0 if mask is None else int((~mask).sum(axis=-1).max())
    if N - n_exclude < k + 1:
        raise ValueError(f"Not enough unmasked points in X to find {k + 1} neighbors, got N={N} with up to {n_exclude} points masked out per query")

    # clamp to align with _tensor()
    D = np.sqrt(np.maximum(pairwise_distance_np(Q, X), 1e-12))  # (M, N) or (B, M, N)

    if mask is not None:
        # a finite sentinel instead of inf to avoid NaN
        D = np.where(mask, D, D.max(axis=-1, keepdims=True) * 2 + 1)

    neighbors = np.sort(np.partition(D, k, axis=-1)[..., : k + 1], axis=-1)  # (M, k + 1) or (B, M, k + 1)

    # clamp to avoid division by zero
    d_min = np.maximum(neighbors[..., :1], 1e-6)  # (M, 1) or (B, M, 1)
    # radius is the midpoint between the k-th and (k+1)-th nearest neighbors
    radius = (neighbors[..., k - 1 : k] + neighbors[..., k : k + 1]) / 2  # (M, 1) or (B, M, 1)
    radius = np.maximum(radius, 1e-6)

    weights = np.exp(-D / d_min) * expit((radius - D) / (softness * radius))  # (M, N) or (B, M, N)
    if mask is not None:
        weights = np.where(mask, weights, 0.0)

    return weights / weights.sum(axis=-1, keepdims=True)


def _tensor(
    X: Tensor,
    Q: Tensor,
    *,
    k: int | None = None,
    mask: Tensor | None = None,
    softness: float = 0.02,
) -> Tensor:
    if softness <= 0:
        raise ValueError(f"softness must be positive, got softness={softness}")

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

    N = int(X.shape[-2])
    E = int(X.shape[-1])
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    if N < k + 1:
        raise ValueError(f"Not enough points in X to find {k + 1} neighbors, got N={N}")

    # clamp to avoid NaN gradient
    D = pairwise_distance(Q, X).clamp(min_=1e-12).sqrt()  # (M, N) or (B, M, N)

    if mask is not None:
        # a finite sentinel instead of inf to avoid NaN
        D = mask.where(D, D.max(axis=-1, keepdim=True) * 2 + 1)

    neighbors = D.topk(k + 1, dim=-1, largest=False, sorted_=True)[0]  # (M, k + 1) or (B, M, k + 1)

    # clamp to avoid division by zero
    d_min = neighbors[..., :1].clamp(min_=1e-6)  # (M, 1) or (B, M, 1)
    # radius is the midpoint between the k-th and (k+1)-th nearest neighbors
    radius = (neighbors[..., k - 1 : k] + neighbors[..., k : k + 1]) / 2  # (M, 1) or (B, M, 1)
    radius = radius.clamp(min_=1e-6)

    gate = ((radius - D) / (softness * radius)).sigmoid()  # (M, N) or (B, M, N)
    weights = (-D / d_min).exp() * gate  # (M, N) or (B, M, N)
    if mask is not None:
        weights = mask.where(weights, 0)

    return weights / weights.sum(axis=-1, keepdim=True)
