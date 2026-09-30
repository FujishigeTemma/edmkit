"""Judge input coordinates on a held-out trajectory, by quantities neither selection method optimizes."""

import numpy as np

from edmkit.embedding import embed
from edmkit.simplex_projection import simplex_projection
from edmkit.simplex_projection.knn import knn
from edmkit.theiler_window import theiler_window

__all__ = ["forecast_error", "neighbour_inflation"]


def _neighbours(X: np.ndarray, t: np.ndarray, k: int, width: int) -> np.ndarray:
    """Indices ``(M, k)`` of the `k` nearest rows of `X` to each row, outside the Theiler window."""
    distances, indices = knn(X, X, k + 2 * width + 1)
    distances = np.where(np.abs(t[indices] - t[:, None]) > width, distances, np.inf)
    return np.take_along_axis(indices, np.argsort(distances, axis=1)[:, :k], axis=1)


def neighbour_inflation(
    coordinates: np.ndarray, x: np.ndarray, state: np.ndarray, t: np.ndarray, *, k: int, width: int, shift: int = 0
) -> np.ndarray:
    """Per-point ratio of true-state distances: neighbours found in the delay vectors over true nearest neighbours.

    The ratio is 1 where the `k` nearest delay vectors are the `k` nearest states, and grows where
    the delay vectors place distant states close together. The delay vector at time ``t`` is compared
    with the state at time ``t - shift``: nearby states separate over time, so a delay vector whose
    coordinates lie far in the past is judged unfairly against the state at its latest time.
    """
    X = x[t[:, None] - coordinates[:, 1], coordinates[:, 0]]
    Z = state[t - shift]
    found = _neighbours(X, t, k, width)
    true = _neighbours(Z, t, k, width)
    distance = lambda indices: np.linalg.norm(Z[indices] - Z[:, None, :], axis=2).mean(axis=1)
    return distance(found) / distance(true)


def forecast_error(coordinates: np.ndarray, x: np.ndarray, clean: np.ndarray, t: np.ndarray, steps: np.ndarray, *, width: int) -> np.ndarray:
    """Leave-one-out simplex RMSE of variable 0 at each horizon in `steps`, against the noise-free series."""
    X = x[t[:, None] - coordinates[:, 1], coordinates[:, 0]]
    Y = x[t[:, None] + steps, 0]
    predictions = simplex_projection(X, Y, X, mask=theiler_window(t, t, width)).reshape(len(t), -1)
    return np.sqrt(((predictions - clean[t[:, None] + steps]) ** 2).mean(axis=0))


def common_times(x: np.ndarray, max_lag: int, max_step: int) -> np.ndarray:
    """Times at which every lag up to `max_lag` and every horizon up to `max_step` is available."""
    return embed(np.array([[0, max_lag], [0, -max_step]]), x)[1]
