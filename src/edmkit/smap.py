from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist
from tinygrad import Tensor, dtypes

from edmkit.util import pairwise_distance, pairwise_distance_np


def smap(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
    use_tensor: bool = False,
) -> np.ndarray:
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    X : np.ndarray
        The input data
    Y : np.ndarray
        The target data
    Q : np.ndarray
        The query points for which to make predictions.
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.
    use_tensor : bool, default False
        Whether to use `tinygrad.Tensor` for computation.
        The tensor path computes distances and weighted normal equations on
        `Tensor` and falls back to NumPy for the small `(E+1) x (E+1)` solve;
        it is most useful when the library and query sets are large.

    Returns
    -------
    predictions : np.ndarray
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
    return _numpy(X, Y, Q, theta=theta, alpha=alpha, mask=mask) if not use_tensor else _tensor(X, Y, Q, theta=theta, alpha=alpha, mask=mask)


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
        D = cdist(Q, X, metric="euclidean")  # (M, N)
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


def _weights_tensor(
    D: Tensor,
    theta: float,
    *,
    mask: np.ndarray | None,
    min_points: int,
) -> Tensor:
    """Tensor counterpart of `weights`. Returns weights with masked / non-finite entries zeroed out."""
    valid = D.isfinite().cast(dtypes.float32)  # (M, N) or (B, M, N)
    if mask is not None:
        valid = valid * Tensor(mask.astype(np.float32), dtype=dtypes.float32).unsqueeze(-2)

    n_valid = valid.sum(axis=-1, keepdim=True)  # (M, 1) or (B, M, 1)
    n_valid_min = int(n_valid.min().numpy())
    if n_valid_min < min_points:
        raise ValueError(f"Not enough valid library points to fit S-Map: need at least {min_points}, got {n_valid_min}")

    if theta == 0.0:
        return valid

    d_sum = (D * valid).sum(axis=-1, keepdim=True)
    d_mean = (d_sum / n_valid.clip(min_=1.0)).clip(min_=1e-6)
    return (-theta * D / d_mean).exp() * valid


def _intercept_skip_eye(size: int) -> Tensor:
    """Identity of `size` with the intercept slot zeroed — for Tikhonov regularization that leaves the intercept untouched."""
    eye = np.eye(size, dtype=np.float32)
    eye[0, 0] = 0.0
    return Tensor(eye)


def _solve_upper(U: Tensor, B: Tensor) -> Tensor:
    """Back-substitute an upper-triangular `U @ X = B` over leading batch dims.

    `U` is `(..., n, n)`, `B` is `(..., n, k)`. Builds `X` one row at a time
    from the bottom up; the Python loop is over `n` (small for S-Map: `E + 1`),
    while every op inside is fully batched.
    """
    n = U.shape[-1]
    rows: list[Tensor] = []  # rows in natural order: X[i], X[i+1], ..., X[n-1]
    for i in range(n - 1, -1, -1):
        u_ii = U[..., i : i + 1, i : i + 1]
        if rows:
            below = rows[0] if len(rows) == 1 else rows[0].cat(*rows[1:], dim=-2)  # (..., n-1-i, k)
            x_i = (B[..., i : i + 1, :] - U[..., i : i + 1, i + 1 :] @ below) / u_ii
        else:
            x_i = B[..., i : i + 1, :] / u_ii
        rows.insert(0, x_i)
    return rows[0] if len(rows) == 1 else rows[0].cat(*rows[1:], dim=-2)


def _solve_spd(A: Tensor, B: Tensor) -> Tensor:
    """Solve `A @ X = B` for batched symmetric positive-definite `A` via QR.

    Tinygrad has no `solve`/`cholesky`, but `qr` is batched. With `A = Q R`,
    `R` is upper-triangular and `Q` is orthogonal, so `X = R^{-1} Q^T B`.
    """
    Q, R = A.qr()
    return _solve_upper(R, Q.transpose(-1, -2) @ B)


def _tensor(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y` using `tinygrad.Tensor`.

    The entire pipeline — pairwise distance, weighting, the weighted normal
    equations `X^T W X` / `X^T W Y`, the per-query SPD solve (QR + triangular
    back-substitution), and the final prediction — stays on `Tensor`. Only the
    final result is materialized via `.numpy()`. Suitable for large libraries
    and query sets when running on a GPU backend.
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
        N, E = X.shape
        M = Q.shape[0]

        X_t = Tensor(X, dtype=dtypes.float32)
        Y_t = Tensor(Y, dtype=dtypes.float32)
        Q_t = Tensor(Q, dtype=dtypes.float32)

        D = pairwise_distance(Q_t, X_t).sqrt()  # (M, N)
        W = _weights_tensor(D, theta, mask=mask, min_points=E + 1)  # (M, N)

        # Add intercept term
        X_aug = Tensor.ones(N, 1, dtype=dtypes.float32).cat(X_t, dim=-1)  # (N, E+1)
        Q_aug = Tensor.ones(M, 1, dtype=dtypes.float32).cat(Q_t, dim=-1)  # (M, E+1)

        # Weighted normal equations: A^T @ W @ A and A^T @ W @ Y
        XTX = Tensor.einsum("pn,ni,nj->pij", W, X_aug, X_aug)  # (M, E+1, E+1)
        XTY = Tensor.einsum("pn,ni,nj->pij", W, X_aug, Y_t)  # (M, E+1, E')

        # Tikhonov regularization (intercept slot left untouched)
        trace = Tensor.einsum("pii->p", XTX).clip(min_=1e-12)  # (M,)
        XTX = XTX + (alpha * trace).reshape(M, 1, 1) * _intercept_skip_eye(E + 1)

        C = _solve_spd(XTX, XTY)  # (M, E+1, E')
        predictions = (Q_aug.unsqueeze(-2) @ C).squeeze(-2)  # (M, E')

        return predictions.numpy().squeeze()  # (M,) or (M, E')
    # X (B, N, E), Y (B, N, E'), Q (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and Q.ndim == 3:
        B, N, E = X.shape
        M = Q.shape[1]

        X_t = Tensor(X, dtype=dtypes.float32)
        Y_t = Tensor(Y, dtype=dtypes.float32)
        Q_t = Tensor(Q, dtype=dtypes.float32)

        D = pairwise_distance(Q_t, X_t).sqrt()  # (B, M, N)
        W = _weights_tensor(D, theta, mask=mask, min_points=E + 1)  # (B, M, N)

        # Add intercept term
        X_aug = Tensor.ones(B, N, 1, dtype=dtypes.float32).cat(X_t, dim=-1)  # (B, N, E+1)
        Q_aug = Tensor.ones(B, M, 1, dtype=dtypes.float32).cat(Q_t, dim=-1)  # (B, M, E+1)

        # Weighted normal equations: A^T @ W @ A and A^T @ W @ Y
        XTX = Tensor.einsum("bpn,bni,bnj->bpij", W, X_aug, X_aug)  # (B, M, E+1, E+1)
        XTY = Tensor.einsum("bpn,bni,bnj->bpij", W, X_aug, Y_t)  # (B, M, E+1, E')

        # Tikhonov regularization
        trace = Tensor.einsum("bpii->bp", XTX).clip(min_=1e-12)  # (B, M)
        XTX = XTX + (alpha * trace).reshape(B, M, 1, 1) * _intercept_skip_eye(E + 1)

        C = _solve_spd(XTX, XTY)  # (B, M, E+1, E')
        predictions = (Q_aug.unsqueeze(-2) @ C).squeeze(-2)  # (B, M, E')

        return predictions.numpy()
    else:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")


if TYPE_CHECKING:
    from functools import partial

    from edmkit.types import PredictFunc

    func: PredictFunc
    func = partial(smap, theta=4.0)
