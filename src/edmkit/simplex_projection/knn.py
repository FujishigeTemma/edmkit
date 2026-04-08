import numpy as np
from scipy.spatial import KDTree
from usearch.index import Index


def knn(X: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the k-nearest neighbors of `Q` in `X` using either `usearch` or `scipy.spatial.KDTree` depending on the size and dimensionality of the data.

    Parameters
    ----------
    X : np.ndarray
        The input data (N, E)
    Q : np.ndarray
        The query points (M, E)
    k : int
        The number of nearest neighbors to find (typically E+1 for simplex projection).

    Returns
    -------
    distances : np.ndarray
        The distances from each query point in `Q` to its k nearest neighbors in `X` (M, k)
    indices : np.ndarray
        The indices of the k nearest neighbors in `X` for each query point in `Q` (M, k)
    """

    N, E = X.shape

    if N < k:
        raise ValueError(f"Not enough points in X to find {k} neighbors, got N={N}")

    if E >= 15 and N >= 10_000:
        index = Index(ndim=E, metric="l2sq")
        index.add(np.arange(len(X)), np.ascontiguousarray(X, dtype=np.float32))
        matches = index.search(np.ascontiguousarray(Q, dtype=np.float32), k)
        distances = np.atleast_2d(np.sqrt(np.asarray(matches.distances)))
        indices = np.atleast_2d(np.asarray(matches.keys).astype(np.intp))
        return distances, indices
    else:
        tree = KDTree(X)
        return tree.query(Q, k=k)
