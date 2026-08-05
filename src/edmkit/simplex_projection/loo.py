import numpy as np

from edmkit.simplex_projection.knn import knn


def loo(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    theiler_window: int,
) -> np.ndarray:
    """
    Leave-one-out simplex projection: predict each point in `X` from its neighbors, excluding temporally close points.

    Equivalent to ``simplex_projection(X, Y, X)`` with Theiler window exclusion,
    but with the correct temporal index handling.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (N,) or (N, E) or (B, N, E).
    Y : np.ndarray
        The target data of shape (N,) or (N, E') or (B, N, E').
    theiler_window : int
        Theiler window half-width. Library points ``j`` where
        ``|i - j| <= theiler_window`` are excluded when predicting point ``i``.
        For lagged embedding, use ``(E - 1) * tau``.

    Returns
    -------
    predictions : np.ndarray
        The predicted values of shape (N,) or (N, E') or (B, N, E').

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If there are not enough library points outside the Theiler window.
    """
    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if not (X.ndim == Y.ndim and X.ndim in (2, 3)):
        raise ValueError(f"X and Y must both be 2D or both be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")

    # treat the unbatched case as a single batch
    batched = X.ndim == 3
    if not batched:
        X, Y = X[None], Y[None]

    B, N, E = X.shape
    k: int = E + 1
    n_exclude = 2 * theiler_window + 1

    if N - n_exclude < k:
        raise ValueError(
            f"Not enough library points outside Theiler window: need at least k={k} points, but only {N - n_exclude} available, N={N}, theiler_window={theiler_window}"
        )

    distances = np.empty((B, N, k + n_exclude))
    indices = np.empty((B, N, k + n_exclude), dtype=np.intp)
    for b in range(B):
        distances[b], indices[b] = knn(X[b], X[b], k + n_exclude)

    distances = np.where(np.abs(indices - np.arange(N)[:, None]) <= theiler_window, np.inf, distances)

    top_k = np.argsort(distances, axis=-1)[..., :k]
    distances = np.take_along_axis(distances, top_k, axis=-1)
    indices = np.take_along_axis(indices, top_k, axis=-1)

    batch_index = np.arange(B)[:, None, None]  # (B, 1, 1)
    Y_neighbors = Y[batch_index, indices]  # (B, N, k, E')

    # clamp to avoid division by zero
    d_min = np.fmax(distances.min(axis=-1, keepdims=True), 1e-6)  # (B, N, 1)
    weights = np.exp(-distances / d_min)  # (B, N, k)

    # matmul avoids the (B, N, k, E') temporary that `weights[..., None] * Y_neighbors` would build
    weighted_sum = np.matmul(weights[..., None, :], Y_neighbors).squeeze(-2)  # (B, N, E')
    predictions = weighted_sum / weights.sum(axis=-1, keepdims=True)  # (B, N, E')

    if not batched:
        return predictions.squeeze()  # (N,) or (N, E')
    return predictions
