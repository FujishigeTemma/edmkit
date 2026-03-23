from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import numpy as np

from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap
from edmkit.types import PredictFunc

SampleFunc: TypeAlias = Callable[[np.ndarray, int], np.ndarray]
"""SampleFunc is a function that takes (pool, size) and returns a sampled array."""
AggregateFunc: TypeAlias = Callable[[np.ndarray], float]
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
    n_samples: int = 20,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    batch_size: int | None = 10,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping and return per-sample correlations.

    Same as :func:`ccm` but returns the raw per-sample scores instead of
    aggregating them.

    Parameters
    ----------
    X : np.ndarray
        Library time series (potential response)
    Y : np.ndarray
        Target time series (potential driver)
    lib_sizes : np.ndarray
        Array of library sizes to test convergence.
    predict_func : :type: `PredictFunc`
        Prediction function with signature (X, Y, Q) -> predictions.
    n_samples : int, default 20
        Number of random samples per library size for bootstrapping.
    library_pool : np.ndarray
        1-D array of integer indices from which library members are sampled.
    prediction_pool : np.ndarray
        1-D array of integer indices that are predicted.
    sample_func : :type: `SampleFunc` | None, default None
        Function responsible for drawing a library sample of a given size.
        When None, a fresh RNG-backed sampler is created per call.
    batch_size : int | None, default 10
        If specified, predictions are made in batches to limit memory usage.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, len(lib_sizes))
        Per-sample correlation coefficients.
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
            samples[offset : offset + batch, i] = pearson_correlation(predictions, actual[:batch])
            remaining -= batch

    return samples


def ccm(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    n_samples: int = 20,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    aggregate_func: AggregateFunc = np.mean,
    batch_size: int | None = 10,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping using a custom prediction function.

    CCM tests for causality from X to Y by using the attractor reconstructed from Y
    to predict values of X. If X causes Y, then Y's attractor contains information
    about X, allowing cross-mapping from Y to X.

    Parameters
    ----------
    X : np.ndarray
        Library time series (potential response)
    Y : np.ndarray
        Target time series (potential driver)
    lib_sizes : np.ndarray
        Array of library sizes to test convergence.
    predict_func : :type: `PredictFunc`
        Prediction function with signature (X, Y, Q) -> predictions.
        Can be `simplex_projection`, `smap` with partial application, or a custom function.
    n_samples : int, default 100
        Number of random samples per library size for bootstrapping.
    library_pool : np.ndarray
        1-D array of integer indices from which library members are sampled.
    prediction_pool : np.ndarray
        1-D array of integer indices that are predicted.
    sample_func : :type: `SampleFunc` | None, default None
        Function responsible for drawing a library sample of a given size.
        It receives `(pool, size)` and returns an array of indices.
        When None, a fresh RNG-backed sampler is created per call.
    aggregate_func : :type: `AggregateFunc`, default `np.mean`
        Reducer applied to the correlation samples for each library size.
    batch_size : int | None, default None
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

    from edmkit.ccm import ccm, simplex_projection, smap
    from edmkit.embedding import lagged_embed

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
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )

    # Using S-Map with partial application
    correlations = ccm(
        Y_embedding,
        X_aligned,
        lib_sizes=lib_sizes,
        predict_func=partial(smap, theta=2.0, alpha=1e-10),
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
        n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sample_func,
        batch_size=batch_size,
    )

    return np.array([aggregate_func(samples[:, i]) for i in range(samples.shape[1])])


def pearson_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute vectorized Pearson correlation between X and Y.

    Parameters
    ----------
    X : np.ndarray
        1D or 2D array of shape (L,) or (B, L)
    Y : np.ndarray
        1D or 2D array of shape (L,) or (B, L)

    Returns
    -------
    correlation : np.ndarray
        Pearson correlation coefficient(s) between X and Y. Shape (B,) if inputs are 2D, else scalar.
    """
    if X.ndim == 1:
        X = X[None, :]
    if Y.ndim == 1:
        Y = Y[None, :]

    mean_X = X.mean(axis=1, keepdims=True)
    mean_Y = Y.mean(axis=1, keepdims=True)
    cov = ((X - mean_X) * (Y - mean_Y)).mean(axis=1)
    std_X = X.std(axis=1)
    std_Y = Y.std(axis=1)
    denom = std_X * std_Y
    safe_denom = np.where(denom > 0, denom, 1.0)
    correlation = np.where(denom > 0, cov / safe_denom, 0.0)

    return correlation.squeeze()


def with_simplex_projection(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    n_samples: int = 100,
    use_tensor: bool = False,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    aggregate_func: AggregateFunc = np.mean,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping using simplex projection.

    This is a convenience wrapper around the general ccm function using
    simplex_projection as the prediction method.

    Parameters
    ----------
    X : np.ndarray
        Library time series (potential response)
    Y : np.ndarray
        Target time series (potential driver)
    lib_sizes : np.ndarray
        Array of library sizes to test convergence
    n_samples : int, default 100
        Number of random samples per library size for bootstrapping
    use_tensor : bool, default False
        Whether to use tinygrad tensors for computation
    library_pool : np.ndarray, optional
        Indices that can be used to draw library samples. Defaults to the full range.
    prediction_pool : np.ndarray, optional
        Indices that should be predicted (leave-one-out over this set). Defaults to the full range.
    sample_func : callable, optional
        Function responsible for drawing a library sample of a given size.
        When omitted, a fresh RNG-backed sampler is created per call.
    aggregate_func : callable, optional
        Reducer applied to the correlation samples for each library size.
        Falls back to `np.mean` when omitted.
    Returns
    -------
    correlations : np.ndarray
        Mean correlation coefficient for each library size

    Raises
    ------
    ValueError
        - If the underlying ccm call detects invalid arguments

    Examples
    --------
    ```python
    import numpy as np

    from edmkit import ccm
    from edmkit.embedding import lagged_embed

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

    correlations = ccm.with_simplex_projection(
        Y_embedding,
        X_aligned,
        lib_sizes=lib_sizes,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )
    ```
    """
    predict_func = partial(simplex_projection, use_tensor=use_tensor)

    return ccm(
        X=X,
        Y=Y,
        lib_sizes=lib_sizes,
        predict_func=predict_func,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sample_func,
        aggregate_func=aggregate_func,
    )


def with_smap(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    theta: float,
    alpha: float = 1e-10,
    n_samples: int = 100,
    use_tensor: bool = False,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc | None = None,
    aggregate_func: AggregateFunc = np.mean,
) -> np.ndarray:
    """
    Perform Convergent Cross Mapping using S-Map (local linear regression).

    This is a convenience wrapper around the general ccm function using
    smap as the prediction method.

    Parameters
    ----------
    X : np.ndarray
        Library time series (potential response)
    Y : np.ndarray
        Target time series (potential driver)
    lib_sizes : np.ndarray
        Array of library sizes to test convergence
    theta : float
        Nonlinearity parameter for S-Map
    alpha : float, default 1e-10
        Regularization parameter for S-Map
    n_samples : int, default 100
        Number of random samples per library size for bootstrapping
    use_tensor : bool, default False
        Whether to use tinygrad tensors for computation
    library_pool : np.ndarray, optional
        Indices that can be used to draw library samples. Defaults to the full range.
    prediction_pool : np.ndarray, optional
        Indices that should be predicted (leave-one-out over this set). Defaults to the full range.
    sample_func : callable, optional
        Function responsible for drawing a library sample of a given size.
        When omitted, a fresh RNG-backed sampler is created per call.
    aggregate_func : callable, optional
        Reducer applied to the correlation samples for each library size.
        Falls back to `np.mean` when omitted.
    Returns
    -------
    correlations : np.ndarray
        Mean correlation coefficient for each library size

    Raises
    ------
    ValueError
        - If the underlying ccm call detects invalid arguments

    Examples
    --------
    ```python
    import numpy as np

    from edmkit import ccm
    from edmkit.embedding import lagged_embed

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

    correlations = ccm.with_smap(
        Y_embedding,
        X_aligned,
        lib_sizes=lib_sizes,
        theta=2.0,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )
    ```
    """
    predict_func = partial(smap, theta=theta, alpha=alpha, use_tensor=use_tensor)

    return ccm(
        X=X,
        Y=Y,
        lib_sizes=lib_sizes,
        predict_func=predict_func,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sample_func,
        aggregate_func=aggregate_func,
    )
