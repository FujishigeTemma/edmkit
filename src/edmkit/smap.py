import numpy as np

from edmkit.tensor import Tensor, dtypes
from edmkit.util import pairwise_distance


def smap(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    theta: float,
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

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted linear regression.

    Raises
    ------
    AssertionError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    assert X.shape[0] == Y.shape[0], f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}"
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    query_points = query_points[:, None] if query_points.ndim == 1 else query_points

    D = pairwise_distance(Tensor(query_points, dtype=dtypes.float32), Tensor(X, dtype=dtypes.float32)).numpy()
    D = np.sqrt(D)

    N_pred = len(query_points)
    predictions = np.zeros(N_pred)

    X = np.insert(X, 0, 1, axis=1)  # add intercept term
    query_points = np.insert(query_points, 0, 1, axis=1)  # add intercept term

    for i in range(N_pred):
        distances = D[i]

        if theta == 0:
            weights = np.ones(X.shape[0])
        else:
            d_mean = np.fmax(np.mean(distances), 1e-6)  # clamp to avoid division by zero
            weights = np.exp(-theta * distances / d_mean)
        weights = weights[:, None]

        A = weights * X  # A.shape = (N, E+1)
        B = weights * Y  # B.shape = (N, E')

        try:
            C, residuals, rank, s = np.linalg.lstsq(A, B)
        except np.linalg.LinAlgError:
            # If singular, fallback to pseudo-inverse
            C = np.linalg.pinv(A) @ B

        predictions[i] = query_points[i] @ C

    return predictions
