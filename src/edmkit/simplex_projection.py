import numpy as np

from edmkit.tensor import Tensor, dtypes
from edmkit.util import pairwise_distance, topk


def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    t: np.ndarray,
    partition: int | None = None,
    exclusion_radius: int | None = None,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `t`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `t` : `np.ndarray`
        The indices of the points to be predicted.
    `partition` : int, optional
        The index to partition the data into library and test sets. If specified, only the library set is used for finding neighbors.
    `exclusion_radius` : int, optional
        The radius around each point to exclude from the neighbor search to avoid trivial solutions.

    Returns
    -------
    observations : `np.ndarray`
        The observed values at the indices specified by `t` in `X`.
    predictions : `np.ndarray`
        The predicted values based on the mean of the nearest neighbors in `Y`.

    Raises
    ------
    AssertionError
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If `partition` is specified and is greater than or equal to the number of points.
        - If any value in `t` is less than `partition` when `partition` is specified.
    """
    assert X.shape[0] == Y.shape[0], f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}"

    D = pairwise_distance(Tensor(X, dtype=dtypes.float32)).numpy()

    N = D.shape[0]
    seq = np.arange(N)

    k: int = X.shape[1] + 1
    neighbors = []

    for i in t:
        mask = np.ones(N, dtype=bool)

        # exclude self
        mask[i] = False

        # split into lib and test
        if partition is not None:
            assert partition < N, f"partition must be less than the number of points, got partition={partition} and N={N}"
            assert (t >= partition).all(), (
                f"t must be greater than or equal to partition when partition is specified, got t={t} and partition={partition}"
            )
            mask[partition:] = False

        # exclude neighbors within exclusion_radius
        if exclusion_radius is not None:
            mask[max(0, i - exclusion_radius) : min(N, i + exclusion_radius + 1)] = False

        # find k nearest neighbors
        indices_masked, _ = topk(D[i][mask], k, largest=False)
        indices = seq[mask][indices_masked]
        neighbors.append(indices)

    observations = Y[t]
    predictions = np.array([Y[indices].mean(axis=0) for indices in neighbors])

    return observations, predictions
