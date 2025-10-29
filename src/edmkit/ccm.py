"""Convergent Cross Mapping (CCM) for causality detection in time series."""

from collections.abc import Callable
from functools import partial

import numpy as np

from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap

rng = np.random.default_rng(42)


def default_sampler(pool: np.ndarray, size: int) -> np.ndarray:
    return rng.choice(pool, size=size, replace=True)


PredictFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
Sampler = Callable[[np.ndarray, int], np.ndarray]
Aggregator = Callable[[np.ndarray], float]


def ccm(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    n_samples: int = 20,
    *,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sampler: Sampler = default_sampler,
    aggregator: Aggregator = np.mean,
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
        Prediction function with signature (lib_X, lib_Y, pred_X) -> predictions.
        Can be `simplex_projection`, `smap` with partial application, or a custom function.
    n_samples : int, default 100
        Number of random samples per library size for bootstrapping.
    library_pool : np.ndarray
        1-D array of integer indices from which library members are sampled.
    prediction_pool : np.ndarray
        1-D array of integer indices that are predicted.
    sampler : :type: `Sampler`, default `default_sampler`
        Function responsible for drawing a library sample of a given size.
        It receives `(pool, size)` and returns an array of indices.
    aggregator : :type: `Aggregator`, default `np.mean`
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
        - If `aggregator` is not callable.
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

    # Generate coupled data
    X = np.random.randn(500)
    Y = np.zeros(500)
    Y[0] = np.random.randn()
    for i in range(1, 500):
        Y[i] = 0.7 * Y[i - 1] + 0.3 * X[i - 1] + 0.1 * np.random.randn()

    tau = 1
    E = 3

    Y_embedding = lagged_embed(Y, tau=tau, e=E)
    shift = tau * (E - 1)
    X_aligned = X[shift:]

    library_pool = np.arange(Y_embedding.shape[0] // 2)
    prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

    # logarithmic within range 10 to max library size
    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=4, dtype=int)

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
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same length, got {X.shape[0]} and {Y.shape[0]}")
    if not callable(predict_func):
        raise ValueError(f"predict_func must be callable, got {type(predict_func)}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if aggregator is None or not callable(aggregator):
        raise ValueError("aggregator must be a callable")

    if batch_size is None:
        batch_size = n_samples
    else:
        batch_size = min(batch_size, n_samples)

    correlations = np.zeros(len(lib_sizes))

    for i, lib_size in enumerate(lib_sizes):
        samples = np.zeros(n_samples)
        remaining = n_samples
        while remaining > 0:
            batch = min(batch_size, remaining)

            library_indices = np.vstack([sampler(library_pool, lib_size) for _ in range(batch)])  # (batch, lib_size)
            prediction_indices = np.tile(prediction_pool, (batch, 1))  # (batch, len(prediction_pool))

            lib_X = X[library_indices]
            lib_X = lib_X.reshape(lib_X.shape[0], lib_X.shape[1], -1)  # if (N,) or (N, 1) then (batch, lib_size) not (batch, lib_size, 1), so reshape
            lib_Y = Y[library_indices]
            lib_Y = lib_Y.reshape(lib_Y.shape[0], lib_Y.shape[1], -1)  # if (N,) or (N, 1) then (batch, lib_size) not (batch, lib_size, 1), so reshape
            query_points = X[prediction_indices]

            predictions = predict_func(lib_X, lib_Y, query_points)
            actual = Y[prediction_indices]

            samples[n_samples - remaining : n_samples - remaining + batch] = pearson_correlation(predictions, actual)
            remaining -= batch

        correlations[i] = aggregator(samples)

    return correlations


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
    X = np.expand_dims(X, axis=-1) if X.ndim == 2 else X
    Y = np.expand_dims(Y, axis=-1) if Y.ndim == 2 else Y

    mean_X = X.mean(axis=1, keepdims=True)
    mean_Y = Y.mean(axis=1, keepdims=True)
    cov = ((X - mean_X) * (Y - mean_Y)).mean(axis=1)
    std_X = X.std(axis=1, keepdims=True)
    std_Y = Y.std(axis=1, keepdims=True)
    correlation = (cov / (std_X * std_Y)).mean(axis=1)

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
    sampler: Sampler = default_sampler,
    aggregator: Aggregator = np.mean,
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
    sampler : callable, optional
        Function responsible for drawing a library sample of a given size.
        Falls back to `default_sampler` (bootstrap with replacement) when omitted.
    aggregator : callable, optional
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

    # Generate coupled data
    X = np.random.randn(500)
    Y = np.zeros(500)
    Y[0] = np.random.randn()
    for i in range(1, 500):
        Y[i] = 0.7 * Y[i - 1] + 0.3 * X[i - 1] + 0.1 * np.random.randn()

    tau = 1
    E = 3

    Y_embedding = lagged_embed(Y, tau=tau, e=E)
    shift = tau * (E - 1)
    X_aligned = X[shift:]

    library_pool = np.arange(Y_embedding.shape[0] // 2)
    prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

    # logarithmic within range 10 to max library size
    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=4, dtype=int)

    correlations = ccm.with_simplex_projection(
        X,
        Y,
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
        sampler=sampler,
        aggregator=aggregator,
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
    sampler: Sampler = default_sampler,
    aggregator: Aggregator = np.mean,
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
    sampler : callable, optional
        Function responsible for drawing a library sample of a given size.
        Falls back to `default_sampler` (bootstrap with replacement) when omitted.
    aggregator : callable, optional
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

    # Generate coupled data
    X = np.random.randn(500)
    Y = np.zeros(500)
    Y[0] = np.random.randn()
    for i in range(1, 500):
        Y[i] = 0.7 * Y[i - 1] + 0.3 * X[i - 1] + 0.1 * np.random.randn()

    tau = 1
    E = 3

    Y_embedding = lagged_embed(Y, tau=tau, e=E)
    shift = tau * (E - 1)
    X_aligned = X[shift:]

    library_pool = np.arange(Y_embedding.shape[0] // 2)
    prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

    # logarithmic within range 10 to max library size
    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=4, dtype=int)

    correlations = ccm.with_smap(
        X,
        Y,
        lib_sizes=lib_sizes,
        theta=1.0,
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
        sampler=sampler,
        aggregator=aggregator,
    )
