from functools import partial
from itertools import product

import numpy as np

from edmkit.metrics import MetricFunc, mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import SplitFunc, sliding_folds
from edmkit.types import PredictFunc


def lagged_embed(x: np.ndarray, tau: int, e: int):
    """Lagged embedding of a time series `x`.

    Parameters
    ----------
    x : np.ndarray
        1D time series of shape ``(N,)``.
    tau : int
        Time delay.
    e : int
        Embedding dimension.

    Returns
    -------
    np.ndarray
        Embedded array of shape ``(N - (e - 1) * tau, e)``.

    Raises
    ------
    ValueError
        - If `x` is not a 1D array.
        - If `tau` or `e` is not positive.
        - If `e * tau >= len(x)`.

    Notes
    -----
    - While open to interpretation, it's generally more intuitive to consider the embedding as starting from the `(e - 1) * tau`th element of the original time series and ending at the `len(x) - 1`th element (the last value), rather than starting from the 0th element and ending at `len(x) - 1 - (e - 1) * tau`.
    - This distinction reflects whether we think of "attaching past values to the present" or "attaching future values to the present". The information content of the result is the same either way.
    - The use of `reversed` in the implementation emphasizes this perspective.

    Examples
    --------
    ```
    import numpy as np
    from edm.embedding import lagged_embed

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tau = 2
    e = 3

    E = lagged_embed(x, tau, e)
    print(E)
    print(E.shape)
    # [[4 2 0]
    #  [5 3 1]
    #  [6 4 2]
    #  [7 5 3]
    #  [8 6 4]
    #  [9 7 5]]
    # (6, 3)
    ```
    """
    if not len(x.shape) == 1:
        raise ValueError(f"X must be a 1D array, got x.shape={x.shape}")
    if tau <= 0 or e <= 0:
        raise ValueError(f"tau and e must be positive, got tau={tau}, e={e}")
    if (e - 1) * tau >= x.shape[0]:
        raise ValueError(f"e and tau must satisfy `(e - 1) * tau < len(X)`, got e={e}, tau={tau}")

    return np.array([x[tau * (e - 1) :]] + [x[tau * i : -tau * ((e - 1) - i)] for i in reversed(range(e - 1))]).transpose()


def scan(
    x: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    E: list[int],
    tau: list[int],
    n_ahead: int = 1,
    split: SplitFunc | None = None,
    predict: PredictFunc | None = None,
    metric: MetricFunc | None = None,
) -> np.ndarray:
    """Grid search over (E, tau) with cross-validation.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Time series to embed.
    Y : np.ndarray or None, shape (N,) or (N, M)
        Prediction target. If None, self-prediction (Y = x).
    E : list[int]
        Embedding dimension candidates.
    tau : list[int]
        Time delay candidates.
    n_ahead : int
        Prediction horizon (steps ahead).
    split : SplitFunc or None
        Callable ``(n: int) -> list[Fold]``. Defaults to sliding_folds.
    predict : PredictFunc or None
        Prediction function. Defaults to ``simplex_projection``.
    metric : MetricFunc or None
        Evaluation metric. Defaults to ``mean_rho``.

    Returns
    -------
    scores : np.ndarray, shape (len(E), len(tau), K_max)
        Per-fold CV metric for each (E, tau) combination.
        K_max is the maximum number of folds across all E values.
        Entries where the fold does not exist are NaN.
    """
    N = len(x)

    if Y is None:
        Y = x
    if predict is None:
        predict = simplex_projection
    if metric is None:
        metric = mean_rho
    if split is None:
        split = partial(
            sliding_folds,
            train_size=max(N // 5, 2),
            validation_size=max(N // 10, 1),
        )

    if Y.ndim == 1:
        Y = Y[:, None]

    n_tau = len(tau)
    tau_max = max(tau)
    n_targets = Y.shape[1]

    # collect ndarrays of shape (n_tau, n_folds) for each E, then pack into a single ndarray at the end
    results: list[np.ndarray | None] = []

    for e in E:
        k = e + 1
        max_lag = (e - 1) * tau_max
        n_usable = N - max_lag - n_ahead

        if n_usable < 2:
            results.append(None)
            continue

        embeddings = [lagged_embed(x, t, e)[-(n_usable + n_ahead) : -n_ahead] for t in tau]

        Y_aligned = Y[max_lag + n_ahead : N]

        folds = split(n_usable)
        folds = [fold for fold in folds if len(fold.train) >= k]  # ensure at least k points
        n_folds = len(folds)

        if n_folds == 0:
            results.append(None)
            continue

        validation_size = len(folds[0].validation)  # now only support fixed validation size across folds, which simplifies batching
        max_train_size = max(len(fold.train) for fold in folds)
        batch_size = n_tau * n_folds

        X_batch = np.zeros((batch_size, max_train_size, e))
        Y_batch = np.zeros((batch_size, max_train_size, n_targets))
        mask = np.zeros((batch_size, max_train_size), dtype=bool)
        Q = np.empty((batch_size, validation_size, e))
        Y_validation = np.empty((batch_size, validation_size, n_targets))

        for batch_idx, (tau_idx, fold_idx) in enumerate(product(range(n_tau), range(n_folds))):
            X = embeddings[tau_idx]
            fold = folds[fold_idx]

            n_train = len(fold.train)

            X_batch[batch_idx, :n_train] = X[fold.train]
            Y_batch[batch_idx, :n_train] = Y_aligned[fold.train]
            Q[batch_idx] = X[fold.validation]
            mask[batch_idx, :n_train] = True
            Y_validation[batch_idx] = Y_aligned[fold.validation]

        predictions = predict(X_batch, Y_batch, Q, mask=None if mask.all() else mask)
        batch_result = metric(predictions, Y_validation)
        results.append(batch_result.reshape(n_tau, n_folds))

    K_max = max((r.shape[1] for r in results if r is not None), default=0)  # max(len(folds)) for all E values, or 0 if no valid folds
    scores = np.full((len(E), n_tau, K_max), np.nan)
    for batch_idx, batch_result in enumerate(results):
        if batch_result is not None:
            scores[batch_idx, :, : batch_result.shape[1]] = batch_result

    return scores


def select(
    scores: np.ndarray,
    *,
    E: list[int],
    tau: list[int],
) -> tuple[int, int, float]:
    """Select best (E, tau) from scan results.

    Ranks each (E, tau) by ``mean - SE`` where SE is the standard error
    of the mean across folds.  This penalises combinations whose scores
    vary widely across folds (unstable predictions) and those with fewer
    valid folds (less certainty), favouring parameters we are *confident*
    perform well.

    Parameters
    ----------
    scores : np.ndarray, shape (len(E), len(tau), K_max)
        Output of ``scan``.
    E : list[int]
        Embedding dimension candidates (same as passed to ``scan``).
    tau : list[int]
        Time delay candidates (same as passed to ``scan``).

    Returns
    -------
    (best_E, best_tau, best_score)
        ``best_score`` is the mean over folds (not the adjusted value)
        so that it remains directly interpretable.
    """
    K = np.sum(~np.isnan(scores), axis=2)
    nan_out = np.full(scores.shape[:2], np.nan)

    mean_scores = np.divide(
        np.nansum(scores, axis=2),
        K,
        out=nan_out.copy(),
        where=K > 0,
    )

    # SE = sqrt(var / K) = sqrt(sum_sq / (K * (K - 1)))
    sum_sq = np.nansum((scores - mean_scores[:, :, None]) ** 2, axis=2)
    se = np.sqrt(
        np.divide(
            sum_sq,
            K * np.maximum(K - 1, 1),
            out=np.zeros_like(nan_out),
            where=K > 1,
        )
    )

    adjusted = mean_scores - se
    flat_idx = int(np.nanargmax(adjusted))
    e_idx, t_idx = np.unravel_index(flat_idx, adjusted.shape)
    return E[e_idx], tau[t_idx], float(mean_scores[e_idx, t_idx])
