from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.util import pairwise_distance_np

__all__ = ["coefficients"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def coefficients(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def coefficients(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
) -> Tensor: ...


def coefficients(
    X,
    Y,
    Q,
    *,
    theta,
    alpha=1e-10,
    mask=None,
):
    """
    Compute the S-Map coefficients `C` of the local linear regression from `X` to `Y` around each query in `Q`.

    ``C[m, 0]`` is the intercept and ``C[m, 1 + i, j]`` is the slope of target `j` with respect to coordinate `i` around ``Q[m]``,
    so ``smap(X, Y, Q)[m]`` equals ``C[m, 0] + Q[m] @ C[m, 1:]``.
    The slopes estimate the local Jacobian of the map from `X` to `Y`, shrunk by the regularization.

    Parameters
    ----------
    X : np.ndarray or Tensor
        The input data of shape (N,) or (N, E) or (B, N, E)
    Y : np.ndarray or Tensor
        The target data of shape (N,) or (N, E') or (B, N, E')
    Q : np.ndarray or Tensor
        The query points of shape (M,) or (M, E) or (B, M, E) around which to fit the regressions.
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.
    mask : np.ndarray or Tensor or None
        Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include in the regression.

    Returns
    -------
    C : np.ndarray or Tensor
        The coefficients of shape (M, E+1) if `Y` is 1D, (M, E+1, E') if `Y` is 2D, or (B, M, E+1, E') if `Y` is 3D.

    Raises
    ------
    ValueError
        - If `theta` is negative.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If the batch size or dimension of `X` and `Q` do not match.
        - If `mask` does not have shape (M, N) or (B, M, N).
        - If fewer than E + 1 library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.smap import coefficients, smap

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    Q = rng.normal(size=(50, 3))
    Y = rng.normal(size=(200, 4))

    C = coefficients(X, Y, Q, theta=4.0)  # (50, 4, 4)
    jacobians = C[:, 1:].swapaxes(-1, -2)  # (50, 4, 3): d Y_j / d X_i around each query

    assert np.allclose(C[:, 0] + np.einsum("mi,mij->mj", Q, C[:, 1:]), smap(X, Y, Q, theta=4.0))
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Y, Q, theta=theta, alpha=alpha, mask=mask)
    return _tensor(X, Y, Q, theta=theta, alpha=alpha, mask=mask)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got theta={theta}")

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
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

    N, E = X.shape[-2:]

    n_exclude = 0 if mask is None else int((~mask).sum(axis=-1).max())
    if N - n_exclude < E + 1:
        raise ValueError(
            f"Not enough unmasked points in X to fit S-Map with {E + 1} coefficients, got N={N} with up to {n_exclude} points masked out per query"
        )

    D = np.sqrt(pairwise_distance_np(Q, X))  # (M, N) or (B, M, N)

    # Exponential weights, scaled by the mean distance over valid library points
    if theta == 0.0:
        W = np.ones_like(D)
    else:
        if mask is None:
            d_mean = D.mean(axis=-1, keepdims=True)  # (M, 1) or (B, M, 1)
        else:
            n_valid = mask.sum(axis=-1, keepdims=True)  # (M, 1) or (B, M, 1)
            d_mean = np.where(mask, D, 0.0).sum(axis=-1, keepdims=True) / n_valid
        W = np.exp(-theta * D / np.maximum(d_mean, 1e-6))

    # Zero out masked-out library points
    if mask is not None:
        W = np.where(mask, W, 0.0)

    # Add intercept term
    X_aug = np.concatenate([np.ones_like(X[..., :1]), X], axis=-1)  # (N, E+1) or (B, N, E+1)

    # Weighted design matrices for all query points: A^T @ W @ A and A^T @ W @ Y
    XTX = np.einsum("...pn,...ni,...nj->...pij", W, X_aug, X_aug)  # (M, E+1, E+1) or (B, M, E+1, E+1)
    XTY = np.einsum("...pn,...ni,...nj->...pij", W, X_aug, Y)  # (M, E+1, E') or (B, M, E+1, E')

    # Tikhonov regularization
    eye = np.eye(E + 1)
    eye[0, 0] = 0  # Do not regularize intercept term
    trace = np.maximum(np.trace(XTX, axis1=-2, axis2=-1), 1e-12)  # (M,) or (B, M)
    reg_term = (alpha * trace)[..., None, None] * eye
    XTX = XTX + reg_term

    C = np.linalg.solve(XTX, XTY)  # (M, E+1, E') or (B, M, E+1, E')

    if return_ndim == 1:
        return C[..., 0]  # (M, E+1)
    return C


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
) -> Tensor:
    # Lazy import: tinygrad starts a per-CPU async-executor pool at import time
    # Loading it only on the tensor path keeps the numpy path free of that scheduler pressure.
    from tinygrad import Tensor
    from tinysolve import solve

    from edmkit.util import pairwise_distance

    if theta < 0:
        raise ValueError(f"theta must be non-negative, got theta={theta}")

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
    if mask is not None and mask.shape != Q.shape[:-1] + X.shape[-2:-1]:
        raise ValueError(f"mask shape must be {Q.shape[:-1] + X.shape[-2:-1]}, got mask.shape={mask.shape}")

    E = int(X.shape[-1])

    # clamp to avoid NaN gradient
    D = pairwise_distance(Q, X).clamp(min_=1e-12).sqrt()  # (M, N) or (B, M, N)

    # Exponential weights, scaled by the mean distance over valid library points
    if theta == 0.0:
        W = Tensor.ones_like(D)
    else:
        if mask is None:
            d_mean = D.mean(axis=-1, keepdim=True)  # (M, 1) or (B, M, 1)
        else:
            valid = mask.cast(D.dtype)  # (M, N) or (B, M, N)
            n_valid = valid.sum(axis=-1, keepdim=True)  # (M, 1) or (B, M, 1)
            d_mean = (D * valid).sum(axis=-1, keepdim=True) / n_valid.clamp(min_=1)
        W = (-theta * D / d_mean.clamp(min_=1e-6)).exp()

    # Zero out masked-out library points
    if mask is not None:
        W = mask.where(W, 0)

    # Add intercept term
    X_aug = Tensor.ones_like(X[..., :1]).cat(X, dim=-1)  # (N, E+1) or (B, N, E+1)

    # Weighted design matrices for all query points: A^T @ W @ A and A^T @ W @ Y, without materializing diag(W)
    X_augT = X_aug.transpose(-1, -2).unsqueeze(-3)  # (1, E+1, N) or (B, 1, E+1, N)
    XTX = X_augT.matmul(W.unsqueeze(-1) * X_aug.unsqueeze(-3))  # (M, E+1, E+1) or (B, M, E+1, E+1)
    XTY = X_augT.matmul(W.unsqueeze(-1) * Y.unsqueeze(-3))  # (M, E+1, E') or (B, M, E+1, E')

    # Tikhonov regularization
    eye = Tensor.eye(E + 1, dtype=X.dtype).to(X.device)
    trace = (XTX * eye).sum(axis=(-2, -1)).clamp(min_=1e-12)  # (M,) or (B, M)
    eye = (Tensor.arange(E + 1).to(X.device) > 0).where(eye, 0)  # Do not regularize intercept term
    reg_term = (alpha * trace).unsqueeze(-1).unsqueeze(-1) * eye
    XTX = XTX + reg_term

    C = solve(XTX, XTY)  # (M, E+1, E') or (B, M, E+1, E')

    if return_ndim == 1:
        return C[..., 0]  # (M, E+1)
    return C
