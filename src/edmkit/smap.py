import numpy as np
from scipy.spatial.distance import cdist

from edmkit.util import pairwise_distance_np


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
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `Q` : `np.ndarray`
        The query points for which to make predictions.
    `theta` : `float`
        Locality parameter. (0: global linear, >0: local linear)
    `alpha` : `float`, default `1e-10`
        Regularization parameter to stabilize the inversion.
    `use_tensor` : `bool`, default `False`
        Whether to use `tinygrad.Tensor` for computation.
        **This may be slower than the NumPy implementation in most cases for now.**

    Returns
    -------
    predictions : `np.ndarray`
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
    return _numpy(X, Y, Q, theta=theta, alpha=alpha, mask=mask) if not use_tensor else _tensor(X, Y, Q, theta=theta, alpha=alpha)


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
    `X` : `np.ndarray`
        (N,) or (N, E) or (B, N, E)
    `Y` : `np.ndarray`
        (N,) or (N, E') or (B, N, E')
    `Q` : `np.ndarray`
        The query points for which to make predictions.
        (M,) or (M, E) or (B, M, E)
    `theta` : `float`
        Locality parameter. (0: global linear, >0: local linear)
    `alpha` : `float`, default `1e-10`
        Regularization parameter to stabilize the inversion.

    Returns
    -------
    predictions : `np.ndarray`
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

        if mask is not None:
            D[:, ~mask] = np.inf

        if theta == 0:
            weights = np.ones_like(D)
            if mask is not None:
                weights[:, ~mask] = 0.0
        else:
            if mask is not None:
                n_valid = mask.sum()
                d_sum = np.where(mask[None, :], D, 0.0).sum(axis=1, keepdims=True)
                d_mean = np.maximum(d_sum / n_valid, 1e-6)
            else:
                d_mean = np.maximum(D.mean(axis=1, keepdims=True), 1e-6)
            weights = np.exp(-theta * D / d_mean)  # inf → 0

        # Add intercept term
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # (N, E+1)
        Q_aug = np.hstack([np.ones((Q.shape[0], 1)), Q])  # (M, E+1)

        # Create weighted design matrices for all query points
        # A^T @ W @ A
        XTX = np.einsum("pn,ni,nj->pij", weights, X_aug, X_aug)  # (M, E+1, E+1)
        XTY = np.einsum("pn,ni,nj->pij", weights, X_aug, Y)  # (M, E+1, E')

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

        if mask is not None:
            inv_mask = ~mask[:, None, :]  # (B, 1, N)
            D = np.where(inv_mask, np.inf, D)

        if theta == 0:
            weights = np.ones_like(D)  # (B, M, N)
            if mask is not None:
                weights = np.where(inv_mask, 0.0, weights)
        else:
            if mask is not None:
                n_valid = mask.sum(axis=1)[:, None, None].astype(float)  # (B, 1, 1)
                d_sum = np.where(inv_mask, 0.0, D).sum(axis=2, keepdims=True)
                d_mean = np.maximum(d_sum / n_valid, 1e-6)
            else:
                d_mean = np.maximum(D.mean(axis=2, keepdims=True), 1e-6)  # (B, M, 1)
            weights = np.exp(-theta * D / d_mean)  # (B, M, N)

        # Add intercept term
        X_aug = np.concatenate([np.ones((B, N, 1)), X], axis=2)  # (B, N, E+1)
        Q_aug = np.concatenate([np.ones((B, M, 1)), Q], axis=2)  # (B, M, E+1)

        # Weighted design matrices: A^T @ W @ A
        XTX = np.einsum("bpn,bni,bnj->bpij", weights, X_aug, X_aug)  # (B, M, E+1, E+1)
        XTY = np.einsum("bpn,bni,bnj->bpij", weights, X_aug, Y)  # (B, M, E+1, E')

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
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `Q` : `np.ndarray`
        The query points for which to make predictions.
    `theta` : `float`
        Locality parameter. (0: global linear, >0: local linear)
    `alpha` : `float`, default `1e-10`
        Regularization parameter to stabilize the inversion.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted linear regression.

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

    raise NotImplementedError("Tensor-based S-Map is not implemented yet.")
