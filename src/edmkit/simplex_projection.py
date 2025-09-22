import numpy as np

from edmkit.tensor import Tensor, dtypes
from edmkit.util import pairwise_distance, topk


def simplex_projection(
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
    AssertionError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    assert X.shape[0] == Y.shape[0], f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}"

    D = pairwise_distance(Tensor(query_points, dtype=dtypes.float32), Tensor(X, dtype=dtypes.float32)).numpy()
    D = np.sqrt(D)

    k: int = X.shape[1] + 1
    predictions = np.zeros(len(query_points))

    for i in range(len(query_points)):
        # find k nearest neighbors
        indices, distances = topk(D[i], k, largest=False)

        d_min = np.fmax(distances[0], 1e-6)  # clamp to avoid division by zero
        weights = np.exp(-distances / d_min)

        predictions[i] = np.sum(weights * Y[indices]) / np.sum(weights)

    return predictions
