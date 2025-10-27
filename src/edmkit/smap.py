import numpy as np
from scipy.spatial.distance import cdist


def smap(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
    rcond: float = 1e-10,
    use_tensor: bool = False,
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
    `rcond` : `float`, optional
        Cutoff for small singular values (relative to the largest singular value).
        Default is 1e-10.
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
    """
    return _numpy(X, Y, query_points, theta, rcond) if not use_tensor else _tensor(X, Y, query_points, theta, rcond)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
    rcond: float = 1e-10,
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
    `rcond` : `float`, optional
        Cutoff for small singular values (relative to the largest singular value).
        Default is 1e-10.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted linear regression.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

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
    reg_term = rcond * np.trace(XTX, axis1=1, axis2=2)[:, np.newaxis, np.newaxis] * eye
    XTX_reg = XTX + reg_term

    # Solve linear system in batch
    C = np.linalg.solve(XTX_reg, XTY)  # (N_pred, E+1, E')

    predictions = np.einsum("pi,pij->pj", query_points_aug, C)

    if squeeze_output:
        predictions = predictions.squeeze(axis=1)

    return predictions


def _tensor(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
    rcond: float = 1e-10,
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
    `rcond` : `float`, optional
        Cutoff for small singular values (relative to the largest singular value).
        Default is 1e-10.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted linear regression.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

    raise NotImplementedError("S-Map with tinygrad.Tensor is not implemented yet.")
