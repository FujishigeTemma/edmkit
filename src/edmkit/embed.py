"""Embedding parameter grid search and selection.

``scan`` performs cross-validated grid search over (E, tau) combinations.
``select`` picks the best (E, tau) from scan results.
"""

from collections.abc import Callable
from functools import partial
from itertools import product
from typing import Protocol

import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.metrics import mean_rho
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import Fold, sliding_folds


class MetricFunc(Protocol):
    """Metric function protocol: (predictions, observations) -> score."""

    def __call__(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
    ) -> float | np.ndarray: ...


SplitFunc = Callable[[int], list[Fold]]


def scan(
    x: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    E: list[int],
    tau: list[int],
    n_ahead: int = 1,
    split: SplitFunc | None = None,
    predict: "Callable | None" = None,
    metric: "MetricFunc | None" = None,
) -> np.ndarray:
    """Grid search over (E, tau) with cross-validation.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Time series to embed.
    Y : np.ndarray or None, shape (T,) or (T, M)
        Prediction target. If None, self-prediction (Y = x).
    E : list[int]
        Embedding dimension candidates.
    tau : list[int]
        Time delay candidates.
    n_ahead : int
        Prediction horizon (steps ahead).
    split : SplitFunc or None
        Callable ``(n: int) -> list[Fold]``. Defaults to sliding_folds.
    predict : callable or None
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
    T = len(x)

    if Y is None:
        Y = x

    if predict is None:
        predict = simplex_projection
    if metric is None:
        metric = mean_rho
    if split is None:
        split = partial(
            sliding_folds,
            train_size=max(T // 5, 2),
            validation_size=max(T // 10, 1),
        )

    # (T,) → (T, 1)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y_target = Y
    M = Y_target.shape[1]

    tau_max = max(tau)

    # --- K_max pre-computation ---
    K_per_e: list[int] = []
    for e in E:
        max_shift = (e - 1) * tau_max
        N = T - max_shift - n_ahead
        K_per_e.append(len(split(N)) if N >= 2 else 0)
    K_max = max(K_per_e) if K_per_e else 0

    scores = np.full((len(E), len(tau), K_max), np.nan)

    for e_idx, e in enumerate(E):
        # --- 1. Embedding construction and tau alignment ---
        max_shift = (e - 1) * tau_max
        usable_len = T - max_shift - n_ahead

        if usable_len < 2:
            continue

        embeddings: list[np.ndarray] = []
        for t in tau:
            emb = lagged_embed(x, t, e)
            emb_common_tail = emb[-(usable_len + n_ahead) :]
            emb_aligned = emb_common_tail[:usable_len]
            embeddings.append(emb_aligned)

        # --- 2. Target alignment ---
        Y_aligned = Y_target[max_shift + n_ahead : T]

        # --- 3. Split → K folds ---
        folds = split(usable_len)
        K = len(folds)
        if K == 0:
            continue

        # --- 4. Batch construction (tau × fold Cartesian product) ---
        B = len(tau) * K
        N_train_max = max(len(fold.train) for fold in folds)
        val_size = len(folds[0].validation)

        # k = e + 1 neighbors required; skip if insufficient training points
        k = e + 1
        if N_train_max < k:
            continue

        X_batch = np.zeros((B, N_train_max, e))
        Y_batch = np.zeros((B, N_train_max, M))
        mask = np.zeros((B, N_train_max), dtype=bool)
        Q = np.empty((B, val_size, e))
        Y_val = np.empty((B, val_size, M))

        for i, (tau_idx, fold_idx) in enumerate(
            product(range(len(tau)), range(K))
        ):
            emb = embeddings[tau_idx]
            fold = folds[fold_idx]
            n_train = len(fold.train)
            X_batch[i, :n_train] = emb[fold.train]
            Y_batch[i, :n_train] = Y_aligned[fold.train]
            mask[i, :n_train] = True
            Q[i] = emb[fold.validation]
            Y_val[i] = Y_aligned[fold.validation]

        # Skip mask if all elements valid
        if mask.all():
            mask_arg = None
        else:
            mask_arg = mask

        # --- 5. Predict: single 3D batch call ---
        predictions = predict(X_batch, Y_batch, Q, mask=mask_arg)

        # --- 6. Metric: 3D batch → per-fold scores ---
        fold_scores = metric(predictions, Y_val)
        scores[e_idx, :, :K] = np.asarray(fold_scores).reshape(len(tau), K)

    return scores


def select(
    scores: np.ndarray,
    *,
    E: list[int],
    tau: list[int],
) -> tuple[int, int, float]:
    """Select best (E, tau) from scan results.

    Aggregates over the fold axis (axis=2) with nanmean, then
    finds the (E, tau) combination with the highest mean score.

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
    """
    mean_scores = np.nanmean(scores, axis=2)
    flat_idx = int(np.nanargmax(mean_scores))
    e_idx, t_idx = np.unravel_index(flat_idx, mean_scores.shape)
    return E[e_idx], tau[t_idx], float(mean_scores[e_idx, t_idx])
