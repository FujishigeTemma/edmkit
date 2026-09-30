import numpy as np

from edmkit.simplex_projection.knn import knn

__all__ = ["spread"]


def spread(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    k: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Variance of the outputs at the `k` nearest neighbours of each query point.

    Simplex projection predicts an output from the outputs at the nearest inputs. This function measures
    how much those outputs disagree. The neighbours carry equal weights, unlike in `simplex_projection`:
    its weights ``exp(-d / d_min)`` concentrate on one neighbour when the nearest distance is small,
    which makes the variance vanish for a reason unrelated to the outputs.

    One value is returned per query and output coordinate, without averaging over queries.
    With outputs at several horizons as columns of `Y`, row ``m`` is the spread profile of query ``m``.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (N,) or (N, E).
    Y : np.ndarray
        The target data of shape (N,) or (N, E').
    Q : np.ndarray
        The query points of shape (M,) or (M, E).
    k : int or None, default None
        The number of nearest neighbors to use. If None, uses E + 1, where E is the dimension of `X`.
    mask : np.ndarray or None
        Boolean mask of shape (M, N) indicating, for each query in `Q`, which library points to include when finding nearest neighbors.

    Returns
    -------
    V : np.ndarray
        Variances of shape (M, E').

    Raises
    ------
    ValueError
        - If `X`, `Y` and `Q` are not 1D or 2D arrays with matching lengths and dimensions.
        - If `k` is not positive.
        - If `mask` does not have shape (M, N).
        - If fewer than `k` library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import embed
    from edmkit.spread import spread
    from edmkit.theiler_window import theiler_window

    x = np.zeros(500)
    x[0] = 0.4
    for i in range(1, 500):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    inputs = np.array([[0, 0], [0, 1]])
    outputs = np.array([[0, -1], [0, -2], [0, -4]])  # horizons 1, 2 and 4

    # Times at which the input and every output are available.
    _, t = embed(np.vstack([inputs, outputs]), x)
    X = x[t[:, None] - inputs[:, 1]]
    Y = x[t[:, None] - outputs[:, 1]]

    V = spread(X, Y, X, k=8, mask=theiler_window(t, t, 4))  # (len(t), 3)
    ```
    """
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if not (X.ndim == Y.ndim == Q.ndim == 2):
        raise ValueError(f"X, Y, and Q must be 1D or 2D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
    if Q.shape[1] != X.shape[1]:
        raise ValueError(f"dimension of X and Q must match, got X.shape={X.shape} and Q.shape={Q.shape}")
    if mask is not None and mask.shape != (Q.shape[0], X.shape[0]):
        raise ValueError(f"mask shape must be {(Q.shape[0], X.shape[0])}, got mask.shape={mask.shape}")

    N, E = X.shape
    k = E + 1 if k is None else k
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    if mask is None:
        _, indices = knn(X, Q, k)
    else:  # over-fetch, then drop the masked-out candidates
        n_exclude = int((~mask).sum(axis=-1).max())
        if N - n_exclude < k:
            raise ValueError(f"Not enough unmasked points in X to find {k} neighbors, got N={N} with up to {n_exclude} points masked out per query")
        distances, indices = knn(X, Q, k + n_exclude)
        distances = np.where(np.take_along_axis(mask, indices, axis=-1), distances, np.inf)
        indices = np.take_along_axis(indices, np.argsort(distances, axis=-1)[:, :k], axis=-1)

    return Y[indices].var(axis=1)  # (M, k, E') -> (M, E')
