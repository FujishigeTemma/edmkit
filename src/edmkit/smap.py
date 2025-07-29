import numpy as np

from edmkit.tensor import Tensor, dtypes
from edmkit.util import pairwise_distance


def smap(
    X: np.ndarray,
    Y: np.ndarray,
    t: np.ndarray,
    theta: float,
    partition: int | None = None,
    exclusion_radius: int | None = None,
):
    assert X.shape[0] == Y.shape[0], "X and Y must have the same length"

    D = pairwise_distance(Tensor(X, dtype=dtypes.float32))
    d_mean = D.mean().numpy()
    D = D.numpy()

    W = np.exp(-theta * D / d_mean)

    C = np.linalg.lstsq(W[t] @ X[:partition], Y[t], rcond=1e-5)[0]

    D = pairwise_distance(Tensor(Y, dtype=dtypes.float32))
    d_mean = D.mean().numpy()
    D = D.numpy()

    W = np.exp(-theta * D / d_mean)

    observations = Y[t]
    predictions = W @ C @ X[t]

    return observations, predictions, C
