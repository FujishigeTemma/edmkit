import numpy as np
from kdtree import KDTree


def knn(X: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the k-nearest neighbors of `Q` in `X` using `kdtree.KDTree`.

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

    N = X.shape[0]

    if N < k:
        raise ValueError(f"Not enough points in X to find {k} neighbors, got N={N}")

    tree = KDTree(X)
    return tree.query(Q, k=k)
