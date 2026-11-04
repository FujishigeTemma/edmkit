import numpy as np
from kdtree import KDTree

from edmkit.types import Mask

__all__: list[str] = []


def knn(X: np.ndarray, Q: np.ndarray, k: int, mask: Mask | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Find the k-nearest neighbors of `Q` in `X` using `kdtree.KDTree`, optionally restricted by `mask`.

    Parameters
    ----------
    X : np.ndarray
        The input data (N, E)
    Q : np.ndarray
        The query points (M, E)
    k : int
        The number of nearest neighbors to find (typically E+1 for simplex projection).
    mask : Mask or None
        ``mask(i, j)`` is True where X[j] is a valid point for Q[i].

    Returns
    -------
    distances : np.ndarray
        The distances from each query point in `Q` to its k nearest neighbors in `X`, sorted in ascending order (M, k)
    indices : np.ndarray
        The indices of the k nearest neighbors in `X` for each query point in `Q` (M, k)
    """
    N = X.shape[0]

    if N < k:
        raise ValueError(f"Not enough points in X to find {k} neighbors, got N={N}")

    tree = KDTree(X)
    if mask is None:
        return tree.query(Q, k=k)

    distances = np.empty((Q.shape[0], k))
    indices = np.empty((Q.shape[0], k), dtype=np.intp)

    def fetch(pending: np.ndarray, k_query: int) -> np.ndarray:
        """Fill `distances` and `indices` with the k nearest neighbors for the `pending` queries, returning the queries that still need more neighbors."""

        # over fetch (k_query > k)
        _distances, _indices = tree.query(Q[pending], k=k_query)

        # evaluate the mask and filter the resultss
        valid = np.broadcast_to(np.asarray(mask(pending[:, None], _indices), dtype=bool), _indices.shape)
        count = valid.sum(axis=-1)

        if k_query == N and count.min() < k:
            raise ValueError(f"Not enough unmasked points in X to find {k} neighbors, got {count.min()} for some query")

        # take the first k valid neighbors for the queries that have them
        done = count >= k
        first_k = valid[done] & (np.cumsum(valid[done], axis=-1) <= k)
        distances[pending[done]] = _distances[done][first_k].reshape(-1, k)
        indices[pending[done]] = _indices[done][first_k].reshape(-1, k)

        return pending[~done]

    k_query = min(k + 1, N)
    pending = fetch(np.arange(Q.shape[0]), k_query)
    while pending.size > 0:
        k_query = min(2 * k_query, N)
        pending = fetch(pending, k_query)

    return distances, indices
