import numpy as np
from scipy.spatial.distance import cdist


def smap(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
    alpha: float = 1e-10,
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
    `query_points` : `np.ndarray`
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
    """
    return _numpy(X, Y, query_points, theta, alpha) if not use_tensor else _tensor(X, Y, query_points, theta, alpha)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
    alpha: float = 1e-10,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    `X` : `np.ndarray`
        (N,) or (N, E)
    `Y` : `np.ndarray`
        (N,) or (N, E')
    `query_points` : `np.ndarray`
        The query points for which to make predictions.
        (M,) or (M, E)
    `theta` : `float`
        Locality parameter. (0: global linear, >0: local linear)
    `alpha` : `float`, default `1e-10`
        Regularization parameter to stabilize the inversion.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted linear regression.
        (M, E')

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

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    query_points = query_points.reshape(query_points.shape[0], -1)
    squeeze_output = Y.shape[1] == 1

    D = cdist(query_points, X, metric="euclidean")

    if theta == 0:
        weights = np.ones_like(D)
    else:
        d_mean = np.maximum(D.mean(axis=1, keepdims=True), 1e-6)
        weights = np.exp(-theta * D / d_mean)

    # Add intercept term
    ones_X = np.ones((X.shape[0], 1))
    ones_query_points = np.ones((query_points.shape[0], 1))
    X_aug = np.hstack([ones_X, X])
    query_points_aug = np.hstack([ones_query_points, query_points])

    # Create weighted design matrices for all query points
    # A^T @ W @ A
    XTX = np.einsum("pn,ni,nj->pij", weights, X_aug, X_aug)  # (N_pred, E+1, E+1)
    XTY = np.einsum("pn,ni,nj->pij", weights, X_aug, Y)  # (N_pred, E+1, E')

    # Tikhonov regularization
    eye = np.eye(XTX.shape[1])
    eye[0, 0] = 0  # Do not regularize intercept term
    trace = np.maximum(np.trace(XTX, axis1=1, axis2=2), 1e-12)
    reg_term = (alpha * trace)[:, None, None] * eye
    XTX = XTX + reg_term

    C = np.linalg.solve(XTX, XTY)  # (N_pred, E+1, E')

    predictions = np.einsum("pi,pij->pj", query_points_aug, C)

    if squeeze_output:
        predictions = predictions.squeeze(axis=1)

    return predictions


def _tensor(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
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
    `query_points` : `np.ndarray`
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
