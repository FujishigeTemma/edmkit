import numpy as np
from scipy.spatial.distance import cdist
from tinygrad import Tensor, dtypes

from edmkit.util import pairwise_distance


def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    use_tensor: bool = False,
) -> np.ndarray:
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.
    `use_tensor` : `bool`, default `False`
        Whether to use `tinygrad.Tensor` for computation.
        **This may be slower than the NumPy implementation in most cases for now.**

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    return _numpy(X, Y, query_points) if not use_tensor else _tensor(X, Y, query_points)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

    D = cdist(query_points, X, metric="euclidean")

    k: int = X.shape[1] + 1

    indices = np.argpartition(D, k - 1, axis=1)[:, :k]
    distances = np.take_along_axis(D, indices, axis=1)

    Y_neighbors = Y[indices]  # shape: (len(query_points), k)

    # clamp to avoid division by zero
    d_min = np.fmax(distances[:, 0:1], 1e-6)
    weights = np.exp(-distances / d_min)

    weighted_sum = np.sum(weights * Y_neighbors, axis=1)
    predictions = weighted_sum / np.sum(weights, axis=1)

    return predictions


def _tensor(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

    D = pairwise_distance(Tensor(query_points, dtype=dtypes.float32), Tensor(X, dtype=dtypes.float32)).sqrt()

    k: int = X.shape[1] + 1

    # find k nearest neighbors for all query points
    distances, indices = D.topk(k, dim=1, largest=False, sorted_=True)

    Y_neighbors = Tensor(Y, dtype=dtypes.float32)[indices]

    # clamp to avoid division by zero
    d_min = distances[:, 0:1].clip(min_=1e-6)
    weights: Tensor = (-distances / d_min).exp()  # type: ignore

    weighted_sum: Tensor = (weights * Y_neighbors).sum(axis=1)  # type: ignore
    predictions: Tensor = weighted_sum / weights.sum(axis=1)  # type: ignore

    return predictions.numpy()
