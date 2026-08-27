from collections.abc import Callable

import numpy as np

from edmkit.metrics import MetricFunc
from edmkit.types import PredictFunc

__all__ = [
    "SampleFunc",
    "AggregateFunc",
    "make_sample_func",
    "bootstrap",
    "ccm",
]

type SampleFunc = Callable[[np.ndarray, int], np.ndarray]
"""SampleFunc is a function that takes (pool, size) and returns a sampled array."""
type AggregateFunc = Callable[[np.ndarray], float]
"""AggregateFunc is a function that takes an array of values and returns a single value."""


def make_sample_func(seed: int | None = 42) -> SampleFunc:
    """Create a sample function with its own independent RNG."""
    rng = np.random.default_rng(seed)

    def sample_func(pool: np.ndarray, size: int) -> np.ndarray:
        return rng.choice(pool, size=size, replace=True)

    return sample_func


def bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    metric_func: MetricFunc,
    n_samples: int = 20,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    batch_size: int | None = 20,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping and return per-sample scores.

    Same as :func:`ccm` but returns the raw per-sample scores instead of
    aggregating them.

    Parameters
    ----------
    X : np.ndarray
        Library delay vectors of shape ``(T,)`` or ``(T, E)`` (potential response).
    Y : np.ndarray
        Target values of shape ``(T,)`` or ``(T, E')`` (potential driver).
        A 2D `Y` cross-maps to a delay vector; the metric scores all columns jointly.
    lib_sizes : np.ndarray
        Array of library sizes to test convergence.
    predict_func : PredictFunc
        Prediction function with signature (X, Y, Q) -> predictions.
    metric_func : MetricFunc
        Metric function with signature (predictions, observations) -> metric value.
    n_samples : int, default 20
        Number of random samples per library size for bootstrapping.
    library_pool : np.ndarray
        1-D array of integer indices from which library members are sampled.
    prediction_pool : np.ndarray
        1-D array of integer indices that are predicted.
    sample_func : SampleFunc or None, default None
        Function responsible for drawing a library sample of a given size.
        When None, a fresh RNG-backed sampler is created per call.
    batch_size : int or None, default 20
        If specified, predictions are made in batches to limit memory usage.

    Returns
    -------
    samples : np.ndarray
        Per-sample skill scores of shape ``(n_samples, len(lib_sizes))``.
    """
    if sample_func is None:
        sample_func = make_sample_func()

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same length, got {X.shape[0]} and {Y.shape[0]}")
    if not callable(predict_func):
        raise ValueError(f"predict_func must be callable, got {type(predict_func)}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if batch_size is None:
        batch_size = n_samples
    else:
        batch_size = min(batch_size, n_samples)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    prediction_indices = np.tile(prediction_pool, (batch_size, 1))
    Q = X[prediction_indices]
    actual = Y[prediction_indices]

    samples = np.zeros((n_samples, len(lib_sizes)))

    for i, lib_size in enumerate(lib_sizes):
        remaining = n_samples
        while remaining > 0:
            batch = min(batch_size, remaining)

            library_indices = np.vstack([sample_func(library_pool, lib_size) for _ in range(batch)])

            lib_X = X[library_indices]
            lib_Y = Y[library_indices]

            predictions = predict_func(lib_X, lib_Y, Q[:batch])

            offset = n_samples - remaining
            samples[offset : offset + batch, i] = metric_func(predictions, actual[:batch])
            remaining -= batch

    return samples


def ccm(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    metric_func: MetricFunc,
    n_samples: int = 20,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    aggregate_func: AggregateFunc = np.mean,
    batch_size: int | None = 20,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping using a custom prediction function.

    CCM tests for causality from X to Y by using the attractor reconstructed from Y
    to predict values of X. If X causes Y, then Y's attractor contains information
    about X, allowing cross-mapping from Y to X.

    Parameters
    ----------
    X : np.ndarray
        Library delay vectors of shape ``(T,)`` or ``(T, E)`` (potential response).
    Y : np.ndarray
        Target values of shape ``(T,)`` or ``(T, E')`` (potential driver).
        A 2D `Y` cross-maps to a delay vector; the metric scores all columns jointly.
    lib_sizes : np.ndarray
        Array of library sizes to test convergence.
    predict_func : PredictFunc
        Prediction function with signature (X, Y, Q) -> predictions.
        Can be `simplex_projection`, `smap` with partial application, or a custom function.
    metric_func : MetricFunc
        Metric function with signature (predictions, observations) -> metric value.
    n_samples : int, default 100
        Number of random samples per library size for bootstrapping.
    library_pool : np.ndarray
        1-D array of integer indices from which library members are sampled.
    prediction_pool : np.ndarray
        1-D array of integer indices that are predicted.
    sample_func : SampleFunc or None, default None
        Function responsible for drawing a library sample of a given size.
        It receives ``(pool, size)`` and returns an array of indices.
        When None, a fresh RNG-backed sampler is created per call.
    aggregate_func : AggregateFunc, default np.mean
        Reducer applied to the skill samples for each library size.
    batch_size : int or None, default 20
        If not specified, batch_size == n_samples.
        If specified, predictions are made in batches to limit memory usage.
    Returns
    -------
    correlations : np.ndarray
        Mean correlation coefficient for each library size.

    Raises
    ------
    ValueError
        - If `X` and `Y` have different lengths.
        - If `lib_sizes` contains non-positive values.
        - If `predict_func` is not callable.
        - If `n_samples` is not positive.
        - If `aggregate_func` is not callable.
        - If `library_pool` or `prediction_pool` is invalid.

    Notes
    -----
    - Higher correlation at larger library sizes indicates convergence and suggests X influences Y (X -> Y causality)
    - Convergence is the key signature of causality in CCM
    - The method uses Y's attractor to predict X (cross-mapping)

    Examples
    --------
    ```python
    from functools import partial

    import numpy as np

    from edmkit.ccm import ccm
    from edmkit.embedding import lagged_embed
    from edmkit.metrics import pearson_correlation
    from edmkit.simplex_projection import simplex_projection
    from edmkit.smap import smap

    # Generate coupled logistic maps (X drives Y)
    N = 1000
    rx, ry, Bxy = 3.8, 3.5, 0.02
    X = np.zeros(N)
    Y = np.zeros(N)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N):
        X[i] = X[i - 1] * (rx - rx * X[i - 1])
        Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]

    tau = 1
    E = 2

    # To test X -> Y causality, cross-map from Y's attractor to X
    Y_embedding = lagged_embed(Y, tau=tau, e=E)
    shift = tau * (E - 1)
    X_aligned = X[shift:]

    library_pool = np.arange(Y_embedding.shape[0] // 2)
    prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

    # logarithmic within range 10 to max library size
    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=5, dtype=int)

    # Using simplex projection
    correlations = ccm(
        Y_embedding,
        X_aligned,
        lib_sizes=lib_sizes,
        predict_func=simplex_projection,
        metric_func=pearson_correlation,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )

    # Using S-Map with partial application
    correlations = ccm(
        Y_embedding,
        X_aligned,
        lib_sizes=lib_sizes,
        predict_func=partial(smap, theta=2.0, alpha=1e-10),
        metric_func=pearson_correlation,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )
    ```
    """
    if aggregate_func is None or not callable(aggregate_func):
        raise ValueError("aggregate_func must be a callable")

    samples = bootstrap(
        X,
        Y,
        lib_sizes,
        predict_func,
        metric_func,
        n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sample_func,
        batch_size=batch_size,
    )

    return np.array([aggregate_func(samples[:, i]) for i in range(samples.shape[1])])
