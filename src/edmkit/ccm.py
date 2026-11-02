from collections.abc import Callable, Iterator
from functools import partial

import numpy as np

from edmkit.types import PredictFunc

__all__ = ["bootstrap"]


def bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    lib_sizes: np.ndarray,
    *,
    predict_func: PredictFunc,
    n_samples: int = 20,
    replace: bool = True,
    seed: int | None = 42,
    mask: np.ndarray | None = None,
    batch_size: int | None = 20,
) -> Iterator[tuple[tuple[int, slice], Callable[[], np.ndarray]]]:
    """
    Iterate `predict_func(X[library], Y[library], Q)` for `n_samples` times for each library size in `lib_sizes`.

    By gathering the results of the returned tasks, you can get `(len(lib_sizes), n_samples, M, E')` array of predictions.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (N,) or (N, E)
    Y : np.ndarray
        The target data of shape (N,) or (N, E'), sampled together with `X`.
    Q : np.ndarray
        The query points of shape (M,) or (M, E), shared by all sampled libraries as a read-only view.
    lib_sizes : np.ndarray
        The 1D array of library sizes, yielded in the given order. Sizes may exceed N only if `replace` is True.
    predict_func : PredictFunc
        The batched prediction function, such as `simplex_projection` or ``functools.partial(smap, theta=...)``.
    n_samples : int, default 20
        The number of libraries to sample for each size in `lib_sizes`.
    replace : bool, default True
        Whether to sample the library points with replacement.
    seed : int or None, default 42
        The seed of the random generator. If None, uses fresh entropy.
    mask : np.ndarray or None
        Boolean mask of shape (M, N) indicating, for each query in `Q`, which library points to include.
        Each task selects the columns of its sampled points. Use `theiler_window` for temporal exclusion.
    batch_size : int or None, default 20
        The maximum number of libraries per task. If None, uses `n_samples`.

    Yields
    ------
    index : tuple[int, slice]
        The position of the task's libraries in an array of shape (len(`lib_sizes`), `n_samples`), such as the scores.
    samples : Callable[[], np.ndarray]
        The task that returns the predictions on B sampled libraries, of shape (B, M, E'), where B <= `batch_size` and E' = 1 if `Y` is 1D.

    Raises
    ------
    ValueError
        - If `X`, `Y`, or `Q` is empty or not 1D or 2D.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If the dimension of `X` and `Q` do not match.
        - If `lib_sizes` is not a nonempty 1D array of positive integers.
        - If some size in `lib_sizes` exceeds N when `replace` is False.
        - If `n_samples` or `batch_size` is not positive.
        - If `mask` does not have shape (M, N).
        - If `predict_func` raises when a task is executed.

    Examples
    --------
    ```python
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from edmkit.ccm import bootstrap
    from edmkit.embedding import embed
    from edmkit.metrics import pearson_correlation
    from edmkit.simplex_projection import simplex_projection

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    coordinates = np.array([[0, tau * j] for j in range(E)])
    embedding, _ = embed(coordinates, x)
    shift = tau * (E - 1)

    pool_size = 200
    Tp = 1
    X = embedding[:pool_size - shift]
    Y = x[shift + Tp : pool_size + Tp]
    Q = embedding[pool_size - shift : -Tp]
    actual = x[pool_size + Tp :, None]

    lib_sizes = np.array([10, 50, 190])
    n_samples = 100


    def score(samples):
        predictions = samples()  # (B, M, 1)
        return pearson_correlation(predictions, np.broadcast_to(actual, predictions.shape))


    scores = np.empty((len(lib_sizes), n_samples))
    tasks = bootstrap(X, Y, Q, lib_sizes, predict_func=simplex_projection, n_samples=n_samples)
    indices, samples = zip(*tasks)
    with ThreadPoolExecutor() as executor:
        for index, result in zip(indices, executor.map(score, samples)):
            scores[index] = result

    for lib_size, correlation in zip(lib_sizes, scores.mean(axis=1)):
        print(f"Correlation (L={lib_size}): {correlation:.3f}")
    ```
    """
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if Q.ndim == 1:
        Q = Q[:, None]

    if X.ndim != 2 or Y.ndim != 2 or Q.ndim != 2:
        raise ValueError(f"X, Y, and Q must be 1D or 2D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.size == 0 or Y.size == 0 or Q.size == 0:
        raise ValueError("X, Y, and Q must be nonempty")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got X.shape={X.shape}, Y.shape={Y.shape}")
    if X.shape[1] != Q.shape[1]:
        raise ValueError(f"X and Q must have the same dimension, got X.shape={X.shape}, Q.shape={Q.shape}")
    if lib_sizes.ndim != 1 or lib_sizes.size == 0 or not np.issubdtype(lib_sizes.dtype, np.integer) or np.any(lib_sizes <= 0):
        raise ValueError("lib_sizes must be a nonempty 1D integer array of positive sizes")
    if not replace and np.any(lib_sizes > len(X)):
        raise ValueError(f"lib_sizes cannot exceed N={len(X)} when replace is False")
    batch_size = n_samples if batch_size is None else batch_size
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError(f"n_samples and batch_size must be positive, got n_samples={n_samples}, batch_size={batch_size}")
    if mask is not None and (mask.shape != (len(Q), len(X)) or mask.dtype != np.bool_):
        raise ValueError(f"mask must be boolean with shape {(len(Q), len(X))}, got shape={mask.shape}, dtype={mask.dtype}")

    rng = np.random.default_rng(seed)
    for i, lib_size in enumerate(lib_sizes):
        for start in range(0, n_samples, batch_size):
            batch = min(batch_size, n_samples - start)
            if replace:
                indices = rng.choice(len(X), size=(batch, lib_size))
            else:
                indices = np.stack([rng.choice(len(X), size=lib_size, replace=False) for _ in range(batch)])
            yield (i, slice(start, start + batch)), partial(samples, X, Y, Q, indices, predict_func=predict_func, mask=mask)


def samples(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    indices: np.ndarray,
    *,
    predict_func: PredictFunc,
    mask: np.ndarray | None,
) -> np.ndarray:
    queries = np.broadcast_to(Q, (len(indices), *Q.shape))
    sampled_mask = None if mask is None else mask[:, indices].transpose(1, 0, 2)
    return predict_func(X[indices], Y[indices], queries, mask=sampled_mask)
