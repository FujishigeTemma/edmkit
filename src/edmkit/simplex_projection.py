import numpy as np
from scipy.spatial import KDTree
from tinygrad import Tensor, dtypes
from usearch.index import Index

from edmkit.util import pairwise_distance


def simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
    use_tensor: bool = False,
) -> np.ndarray:
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        The input data
    `Y` : `np.ndarray`
        The target data
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.
    `use_tensor` : `bool`, default `False`
        Whether to use `tinygrad.Tensor` for computation.
        **This may be slower than the NumPy implementation in most cases for now.**

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import lagged_embed
    from edmkit.simplex_projection import simplex_projection

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    embedding = lagged_embed(x, tau=tau, e=E)
    shift = tau * (E - 1)

    lib_size = 200
    Tp = 1
    X = embedding[:lib_size - shift]
    Y = x[shift + Tp : lib_size + Tp]
    query_points = embedding[lib_size - shift : -Tp]
    actual = x[lib_size + Tp :]

    predictions = simplex_projection(X, Y, query_points)

    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"Correlation: {correlation:.3f}")
    ```
    """
    return _numpy(X, Y, query_points) if not use_tensor else _tensor(X, Y, query_points)


def knn(X: np.ndarray, query_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    N, E = X.shape
    if E >= 15 and N >= 10_000:
        index = Index(ndim=E, metric="l2sq")
        index.add(np.arange(len(X)), np.ascontiguousarray(X, dtype=np.float32))
        matches = index.search(np.ascontiguousarray(query_points, dtype=np.float32), k)
        distances = np.atleast_2d(np.sqrt(np.asarray(matches.distances)))
        indices = np.atleast_2d(np.asarray(matches.keys).astype(np.intp))
        return distances, indices
    else:
        tree = KDTree(X)
        return tree.query(query_points, k=k)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        (N,) or (N, E) or (B, N, E)
    `Y` : `np.ndarray`
        (N,) or (N, E') or (B, N, E')
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.
        (M,) or (M, E) or (B, M, E)

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.
        (M,) or (M, E') or (B, M, E')

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    # ensure 2D or 3D arrays
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if query_points.ndim == 1:
        query_points = query_points[:, None]

    # X (N, E), Y (N, E'), query_points (M, E)
    if X.ndim == 2 and Y.ndim == 2 and query_points.ndim == 2:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

        k: int = X.shape[1] + 1

        distances, indices = knn(X, query_points, k)

        Y_neighbors = Y[indices]  # (M, k, E')

        # clamp to avoid division by zero
        d_min = np.fmax(distances.min(axis=1, keepdims=True), 1e-6)  # (M, 1)
        weights = np.exp(-distances / d_min)  # (M, k)

        weighted_sum = np.sum(weights[..., None] * Y_neighbors, axis=1)
        predictions = weighted_sum / np.sum(weights, axis=1, keepdims=True)

        return predictions.squeeze()  # (M,) or (M, E')
    # X (B, N, E), Y (B, N, E'), query_points (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and query_points.ndim == 3:
        B, N, E = X.shape
        if Y.shape[0] != B or Y.shape[1] != N:
            raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
        if query_points.shape[0] != B or query_points.shape[2] != E:
            raise ValueError(
                f"batch size and dimension of X and query_points must match, got X.shape={X.shape} and query_points.shape={query_points.shape}"
            )

        k: int = E + 1
        M = query_points.shape[1]

        distances = np.empty((B, M, k))
        indices = np.empty((B, M, k), dtype=np.intp)
        for b in range(B):
            distances[b], indices[b] = knn(X[b], query_points[b], k)

        batch_idx = np.arange(B)[:, None, None]  # (B, 1, 1)
        Y_neighbors = Y[batch_idx, indices]  # (B, M, k, E')

        # clamp to avoid division by zero
        d_min = np.fmax(distances.min(axis=2, keepdims=True), 1e-6)  # (B, M, 1)
        weights = np.exp(-distances / d_min)  # (B, M, k)

        weighted_sum = np.sum(weights[..., None] * Y_neighbors, axis=2)  # (B, M, E')
        predictions = weighted_sum / np.sum(weights, axis=2, keepdims=True)  # (B, M, E')

        return predictions
    else:
        raise ValueError(
            f"X, Y, and query_points must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, query_points.ndim={query_points.ndim}"
        )


def _tensor(
    X: np.ndarray,
    Y: np.ndarray,
    query_points: np.ndarray,
):
    """
    Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `query_points`.

    Parameters
    ----------
    `X` : `np.ndarray`
        (N,) or (N, E) or (B, N, E)
    `Y` : `np.ndarray`
        (N,) or (N, E') or (B, N, E')
    `query_points` : `np.ndarray`
        The query points for which to find the nearest neighbors in `X`.
        (M,) or (M, E) or (B, M, E)

    Returns
    -------
    predictions : `np.ndarray`
        The predicted values based on the weighted mean of the nearest neighbors in `Y`.
        (M,) or (M, E') or (B, M, E')

    Raises
    ------
    ValueError
        - If the input arrays `X` and `Y` do not have the same number of points.
    """
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if query_points.ndim == 1:
        query_points = query_points[:, None]

    # X (N, E), Y (N, E'), query_points (M, E)
    if X.ndim == 2 and Y.ndim == 2 and query_points.ndim == 2:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same length, got X.shape={X.shape} and Y.shape={Y.shape}")

        X_tensor = Tensor(X, dtype=dtypes.float32)
        Y_tensor = Tensor(Y, dtype=dtypes.float32)
        query_tensor = Tensor(query_points, dtype=dtypes.float32)

        D = pairwise_distance(query_tensor, X_tensor).sqrt()  # (M, N)

        k: int = X.shape[1] + 1

        distances, indices = D.topk(k, dim=1, largest=False, sorted_=True)  # (M, k)
        Y_neighbors = Y_tensor[indices]  # (M, k, E')

        d_min = distances[:, :1].clip(min_=1e-6)  # (M, 1)
        weights: Tensor = (-distances / d_min).exp()  # (M, k)

        weighted_sum: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=1)  # (M, E')
        predictions: Tensor = weighted_sum / weights.sum(axis=1, keepdim=True)  # (M, E')

        return predictions.numpy().squeeze()
    # X (B, N, E), Y (B, N, E'), query_points (B, M, E)
    elif X.ndim == 3 and Y.ndim == 3 and query_points.ndim == 3:
        B, N, E = X.shape
        if Y.shape[0] != B or Y.shape[1] != N:
            raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")
        if query_points.shape[0] != B or query_points.shape[2] != E:
            raise ValueError(
                f"batch size and dimension of X and query_points must match, got X.shape={X.shape} and query_points.shape={query_points.shape}"
            )

        Y_tensor = Tensor(Y, dtype=dtypes.float32)

        D = pairwise_distance(Tensor(query_points, dtype=dtypes.float32), Tensor(X, dtype=dtypes.float32)).sqrt()  # (B, M, N)

        k: int = E + 1

        distances, indices = D.topk(k, dim=2, largest=False, sorted_=True)  # (B, M, k)

        # --- Neighbor lookup -------------------------------------------------------
        # Purpose:
        #   `indices` contains the k-nearest-neighbor indices in X for each batch and each query point. (B, M, k)
        #   However, we need to gather the corresponding Y values from (B, N, E'),
        #   and tinygrad currently doesn’t support a batched gather operation like PyTorch does.
        #   Therefore, we flatten the batch dimension so we can perform a single gather
        #   from a flattened (B*N, E') tensor.
        #
        # Notation:
        #   B = batch size, M = number of query points, N = number of library points,
        #   k = number of neighbors, E' = output dimension
        #
        # Steps:
        #   1) Create per-batch offsets [0*N, 1*N, ..., (B-1)*N]
        #   2) Add these offsets to the neighbor indices (B, M, k)
        #      -> converts them to flattened indices relative to (B*N)
        #   3) Reshape Y into (B*N, E') and gather using the flattened indices
        #   4) Reshape the gathered results back to (B, M, k, E') to continue computation
        offsets = Tensor.arange(B, dtype=dtypes.int32).reshape(B, 1, 1) * N  # (B,1,1) create per-batch offsets spaced by N
        flat_indices = (indices + offsets).reshape(B * query_points.shape[1], k)  # (B*M, k) flatten batch and query dimensions
        Y_flat = Y_tensor.reshape(B * N, Y_tensor.shape[-1])  # (B*N, E') flatten batch and library points
        Y_neighbors = Y_flat[flat_indices].reshape(B, query_points.shape[1], k, Y_tensor.shape[-1])  # (B, M, k, E') restore shape
        # ---------------------------------------------------------------------------

        d_min = distances[:, :, :1].clip(min_=1e-6)  # (B, M, 1)
        weights: Tensor = (-distances / d_min).exp()  # (B, M, k)

        weighted_sum: Tensor = (weights.unsqueeze(-1) * Y_neighbors).sum(axis=2)  # (B, M, E')
        predictions: Tensor = weighted_sum / weights.sum(axis=2, keepdim=True)  # (B, M, E')

        return predictions.numpy()
    else:
        raise ValueError(
            f"X, Y, and query_points must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, query_points.ndim={query_points.ndim}"
        )
