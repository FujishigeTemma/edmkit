from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.util import pairwise_distance_np

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def smap(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def smap(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
) -> Tensor: ...


def smap(
    X,
    Y,
    Q,
    *,
    theta,
    alpha=1e-10,
    mask=None,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    X : np.ndarray or Tensor
        The input data
    Y : np.ndarray or Tensor
        The target data
    Q : np.ndarray or Tensor
        The query points for which to make predictions.
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted linear regression.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `theta` is negative.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import lagged_embed
    from edmkit.smap import smap

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

    # Local linear with theta=4.0
    predictions = smap(X, Y, Q, theta=4.0)
    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"Correlation (theta=4.0): {correlation:.3f}")

    # Global linear with theta=0.0
    predictions_global = smap(X, Y, Q, theta=0.0)
    correlation_global = np.corrcoef(predictions_global, actual)[0, 1]
    print(f"Correlation (theta=0.0): {correlation_global:.3f}")
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Y, Q, theta=theta, alpha=alpha, mask=mask)
    return _tensor(X, Y, Q, theta=theta, alpha=alpha, mask=mask)


def weights(
    D: np.ndarray,
    theta: float,
    *,
    mask: np.ndarray | None = None,
    min_points: int,
) -> np.ndarray:
    """Compute S-Map exponential weights, zeroing out masked library points.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix — (M, N) or (B, M, N).
    theta : float
        Locality parameter.
    mask : np.ndarray | None
        Boolean mask over the library axis — (N,) or (B, N).
    min_points : int
        Minimum number of valid library points required.
    """
    valid = np.isfinite(D) if mask is None else np.isfinite(D) & mask[..., None, :]  # mask[..., None, :].shape == (1, N) or (B, 1, N)

    n_valid = valid.sum(axis=-1, keepdims=True)  # (M, 1) or (B, M, 1)
    if int(n_valid.min()) < min_points:
        raise ValueError(f"Not enough valid library points to fit S-Map: need at least {min_points}, got {int(n_valid.min())}")

    if theta == 0.0:
        return np.where(valid, 1.0, 0.0)

    d_sum = np.where(valid, D, 0.0).sum(axis=-1, keepdims=True)
    d_mean = np.maximum(d_sum / np.maximum(n_valid, 1), 1e-6)
    w = np.exp(-theta * D / d_mean)
    return np.where(valid, w, 0.0)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    X : np.ndarray
        (N,) or (N, E) or (B, N, E)
    Y : np.ndarray
        (N,) or (N, E') or (B, N, E')
    Q : np.ndarray
        The query points for which to make predictions.
        (M,) or (M, E) or (B, M, E)
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.

    Returns
    -------
    predictions : np.ndarray
        The predicted values based on the weighted linear regression.
        (M, E') or (B, M, E')

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `theta` is negative.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got theta={theta}")

    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    # X (N, E), Y (N, E'), Q (M, E)
    if X.ndim == 2 and Y.ndim == 2 and Q.ndim == 2:
        D = np.sqrt(pairwise_distance_np(Q, X))  # (M, N)
        W = weights(D, theta, mask=mask, min_points=X.shape[1] + 1)

        # Add intercept term
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # (N, E+1)
        Q_aug = np.hstack([np.ones((Q.shape[0], 1)), Q])  # (M, E+1)

        # Create weighted design matrices for all query points
        # A^T @ W @ A
        XTX = np.einsum("pn,ni,nj->pij", W, X_aug, X_aug)  # (M, E+1, E+1)
        XTY = np.einsum("pn,ni,nj->pij", W, X_aug, Y)  # (M, E+1, E')

        # Tikhonov regularization
        eye = np.eye(XTX.shape[1])
        eye[0, 0] = 0  # Do not regularize intercept term
        trace = np.maximum(np.trace(XTX, axis1=1, axis2=2), 1e-12)
        reg_term = (alpha * trace)[:, None, None] * eye
        XTX = XTX + reg_term

        C = np.linalg.solve(XTX, XTY)  # (M, E+1, E')

        predictions = np.einsum("pi,pij->pj", Q_aug, C)

        return predictions.squeeze()  # (M,) or (M, E')
    # X (B, N, E), Y (B, N, E'), Q (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and Q.ndim == 3:
        B, N, E = X.shape
        M = Q.shape[1]

        D = np.sqrt(pairwise_distance_np(Q, X))  # (B, M, N)
        W = weights(D, theta, mask=mask, min_points=E + 1)

        # Add intercept term
        X_aug = np.concatenate([np.ones((B, N, 1)), X], axis=2)  # (B, N, E+1)
        Q_aug = np.concatenate([np.ones((B, M, 1)), Q], axis=2)  # (B, M, E+1)

        # Weighted design matrices: A^T @ W @ A
        XTX = np.einsum("bpn,bni,bnj->bpij", W, X_aug, X_aug)  # (B, M, E+1, E+1)
        XTY = np.einsum("bpn,bni,bnj->bpij", W, X_aug, Y)  # (B, M, E+1, E')

        # Tikhonov regularization
        eye = np.eye(E + 1)
        eye[0, 0] = 0
        trace = np.maximum(np.trace(XTX, axis1=2, axis2=3), 1e-12)  # (B, M)
        reg_term = (alpha * trace)[..., None, None] * eye  # (B, M, E+1, E+1)
        XTX = XTX + reg_term

        C = np.linalg.solve(XTX, XTY)  # (B, M, E+1, E')

        predictions = np.einsum("bpi,bpij->bpj", Q_aug, C)  # (B, M, E')

        return predictions
    else:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    X : Tensor
        The input data
        (N,) or (N, E) or (B, N, E)
    Y : Tensor
        The target data
        (N,) or (N, E') or (B, N, E')
    Q : Tensor
        The query points for which to make predictions.
        (M,) or (M, E) or (B, M, E)
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.
    mask : Tensor | None
        Boolean mask over the library axis — (N,) or (B, N). Masked-out points get zero weight.

    Returns
    -------
    predictions : Tensor
        The predicted values based on the weighted linear regression.
        (M, E') or (B, M, E')

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `theta` is negative.
    """
    # Lazy import: tinygrad starts a per-CPU async-executor pool at import time
    # Loading it only on the tensor path keeps the numpy path free of that scheduler pressure.
    from tinygrad import Tensor
    from tinysolve import solve

    from edmkit.util import pairwise_distance

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got theta={theta}")

    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Y.ndim == Q.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D tensors, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")

    E = int(X.shape[-1])

    D = pairwise_distance(Q, X).sqrt()  # (M, N) or (B, M, N)

    # Exponential weights, scaled by the mean distance over valid library points
    if theta == 0.0:
        W = Tensor.ones_like(D)
    else:
        if mask is None:
            d_mean = D.mean(axis=-1, keepdim=True)  # (M, 1) or (B, M, 1)
        else:
            valid = mask.unsqueeze(-2).cast(D.dtype)  # (1, N) or (B, 1, N)
            n_valid = valid.sum(axis=-1, keepdim=True)  # (1, 1) or (B, 1, 1)
            d_mean = (D * valid).sum(axis=-1, keepdim=True) / n_valid.clip(min_=1)
        W = (-theta * D / d_mean.clip(min_=1e-6)).exp()

    # Zero out masked-out library points
    if mask is not None:
        W = mask.unsqueeze(-2).where(W, 0)

    # Add intercept term
    X_aug = Tensor.ones_like(X[..., :1]).cat(X, dim=-1)  # (N, E+1) or (B, N, E+1)
    Q_aug = Tensor.ones_like(Q[..., :1]).cat(Q, dim=-1)  # (M, E+1) or (B, M, E+1)

    # Weighted design matrices for all query points: A^T @ W @ A, without materializing diag(W)
    X_augT = X_aug.transpose(-1, -2).unsqueeze(-3)  # (1, E+1, N) or (B, 1, E+1, N)
    XTX = X_augT.matmul(W.unsqueeze(-1) * X_aug.unsqueeze(-3))  # (M, E+1, E+1) or (B, M, E+1, E+1)
    XTY = X_augT.matmul(W.unsqueeze(-1) * Y.unsqueeze(-3))  # (M, E+1, E') or (B, M, E+1, E')

    # Tikhonov regularization
    eye = Tensor.eye(E + 1, dtype=X.dtype, device=X.device)
    trace = (XTX * eye).sum(axis=(-2, -1)).clip(min_=1e-12)  # (M,) or (B, M)
    eye = (Tensor.arange(E + 1, device=X.device) > 0).where(eye, 0)  # Do not regularize intercept term
    reg_term = (alpha * trace).unsqueeze(-1).unsqueeze(-1) * eye
    XTX = XTX + reg_term

    C = solve(XTX, XTY)  # (M, E+1, E') or (B, M, E+1, E')

    predictions = Q_aug.unsqueeze(-2).matmul(C).squeeze(-2)  # (M, E') or (B, M, E')

    if X.ndim == 2:
        return predictions.squeeze()  # (M,) or (M, E')
    return predictions


if TYPE_CHECKING:
    from functools import partial

    from tinygrad import Tensor

    from edmkit.types import PredictFunc

    f: PredictFunc[np.ndarray] = partial(smap, theta=4.0)
    g: PredictFunc[Tensor] = partial(smap, theta=0.0)
