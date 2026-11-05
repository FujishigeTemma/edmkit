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

    # smaller leaves improve pruning in low dimensions.
    tree = KDTree(X, leafsize=16 if X.shape[-1] <= 8 else 32)
    if mask is None or len(Q) == 0:
        return tree.query(Q, k=k)

    distances = np.empty((Q.shape[0], k))
    indices = np.empty((Q.shape[0], k), dtype=np.intp)

    pending = np.arange(Q.shape[0])
    k_query = min(k + 1, N)

    while pending.size > 0:
        _distances, _indices = tree.query(Q if pending.size == len(Q) else Q[pending], k=k_query)

        # evaluate the mask and keep the first k valid candidates.
        valid = np.broadcast_to(np.asarray(mask(pending[:, None], _indices), dtype=bool), _indices.shape)
        ranks = valid.cumsum(axis=-1, dtype=np.uint32)
        count = ranks[:, -1]

        if k_query == N and count.min() < k:
            raise ValueError(f"Not enough unmasked points in X to find {k} neighbors, got {count.min()} for some query")

        done = count >= k
        if done.any():
            first_k = valid & (ranks <= k) & done[:, None]
            distances[pending[done]] = _distances[first_k].reshape(-1, k)
            indices[pending[done]] = _indices[first_k].reshape(-1, k)

        pending = pending[~done]
        k_query = min(2 * k_query, N)

    return distances, indices
