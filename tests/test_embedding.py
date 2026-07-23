from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.embedding import lagged_embed, scan, select
from edmkit.metrics import MetricFunc, mean_rho, rmse
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import SplitFunc, expanding_folds, sliding_folds
from edmkit.types import PredictFunc


class EmbeddingProblem(NamedTuple):
    x: np.ndarray
    tau: int
    e: int


class ScanProblem(NamedTuple):
    x: np.ndarray
    Y: np.ndarray | None
    E: list[int]
    tau: list[int]
    split: SplitFunc | None = None
    predict: PredictFunc | None = None
    metric: MetricFunc | None = None


class SelectProblem(NamedTuple):
    scores: np.ndarray
    E: list[int]
    tau: list[int]
    expected: tuple[int, int, float]


def embed_reference(problem: EmbeddingProblem) -> np.ndarray:
    present = np.arange((problem.e - 1) * problem.tau, len(problem.x))[:, None]
    lags = problem.tau * np.arange(problem.e)[None, :]
    return problem.x[present - lags]


@st.composite
def embedding_problems(draw):
    tau = draw(st.integers(1, 6))
    e = draw(st.integers(1, 7))
    minimum = (e - 1) * tau + 1
    n = draw(st.integers(minimum, minimum + 30))
    x = draw(hnp.arrays(np.int64, n, elements=st.integers(-10_000, 10_000)))
    return EmbeddingProblem(x, tau, e)


def scan_reference(problem: ScanProblem) -> np.ndarray:
    n = len(problem.x)
    target = problem.x if problem.Y is None else problem.Y
    target = target[:, None] if target.ndim == 1 else target
    split = problem.split or partial(sliding_folds, train_size=max(n // 5, 2), validation_size=max(n // 10, 1))
    predict = problem.predict or simplex_projection
    metric = problem.metric or mean_rho
    tau_max = max(problem.tau)
    rows: list[np.ndarray | None] = []

    for e in problem.E:
        max_lag = (e - 1) * tau_max
        n_usable = n - max_lag
        if n_usable < 2:
            rows.append(None)
            continue

        folds = [fold for fold in split(n_usable) if len(fold.train) >= e + 1]
        if not folds:
            rows.append(None)
            continue

        aligned_target = target[max_lag:n]
        scores = np.empty((len(problem.tau), len(folds)))
        for tau_index, delay in enumerate(problem.tau):
            embedded = embed_reference(EmbeddingProblem(problem.x, delay, e))[-n_usable:]
            for fold_index, fold in enumerate(folds):
                predictions = predict(embedded[fold.train], aligned_target[fold.train], embedded[fold.validation])
                predictions = np.asarray(predictions).reshape(len(fold.validation), target.shape[1])
                scores[tau_index, fold_index] = metric(predictions, aligned_target[fold.validation])
        rows.append(scores)

    n_folds = max((row.shape[1] for row in rows if row is not None), default=0)
    expected = np.full((len(problem.E), len(problem.tau), n_folds), np.nan)
    for e_index, row in enumerate(rows):
        if row is not None:
            expected[e_index, :, : row.shape[1]] = row
    return expected


def check_scan(problem: ScanProblem) -> None:
    actual = scan(problem.x, problem.Y, E=problem.E, tau=problem.tau, split=problem.split, predict=problem.predict, metric=problem.metric)
    np.testing.assert_allclose(actual, scan_reference(problem), atol=1e-12, rtol=1e-12, equal_nan=True, strict=True)


def check_select(problem: SelectProblem) -> None:
    actual = select(problem.scores, E=problem.E, tau=problem.tau)
    assert actual[:2] == problem.expected[:2]
    assert actual[2] == pytest.approx(problem.expected[2])


def mean_predictor(X, Y, Q, *, mask=None):
    if mask is None:
        means = Y.mean(axis=-2)
    else:
        valid = mask[..., None]
        means = (Y * valid).sum(axis=-2) / valid.sum(axis=-2)
    return np.broadcast_to(means[..., None, :], (*Q.shape[:-1], Y.shape[-1]))


t = np.linspace(0.0, 8.0 * np.pi, 80)
signal = np.sin(t) + 0.01 * t
multivariate = np.column_stack([signal, np.cos(t) - 0.02 * t])
shifted = np.sin(np.linspace(0.0, 10.0, 73)) + np.linspace(0.0, 0.2, 73)
padding_signal = np.sin(np.linspace(0.0, 6.0, 30))
unusable_signal = np.linspace(0.0, 1.0, 30)

SCANS = {
    "default-self-target": ScanProblem(signal, None, [1, 2, 4], [1, 2]),
    "expanding-mask-and-custom-predictor": ScanProblem(
        signal,
        multivariate,
        [1, 3],
        [1, 2],
        split=partial(expanding_folds, initial_train_size=18, validation_size=5, stride=7),
        predict=mean_predictor,
        metric=rmse,
    ),
    "caller-shifted-target": ScanProblem(shifted[:-3], shifted[3:], [2, 3], [1, 2]),
    "nan-fold-padding": ScanProblem(
        padding_signal,
        None,
        [2, 6],
        [1, 2],
        partial(sliding_folds, train_size=10, validation_size=4, stride=4),
    ),
    "partially-unusable-candidates": ScanProblem(unusable_signal, None, [2, 8], [1, 2]),
    "all-candidates-unusable": ScanProblem(unusable_signal[:2], unusable_signal[28:], [2], [1]),
}

SELECTIONS = {
    "highest-risk-adjusted-score": SelectProblem(
        np.array([[[0.6, 0.8, np.nan], [0.4, 0.5, 0.6]], [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]]),
        [2, 3],
        [1, 2],
        (3, 1, 0.8),
    ),
    "variance-penalty": SelectProblem(
        np.array([[[0.5, 0.5, 0.5, 0.5]], [[0.9, 0.1, 0.9, 0.1]]]),
        [1, 2],
        [1],
        (1, 1, 0.5),
    ),
    "missing-folds": SelectProblem(
        np.array([[[np.nan, 0.2, 0.4]], [[0.5, np.nan, np.nan]]]),
        [2, 4],
        [3],
        (4, 3, 0.5),
    ),
}

INVALID: dict[str, Callable[[], object]] = {
    "embedding-matrix-input": partial(lagged_embed, np.zeros((2, 3)), tau=1, e=2),
    "embedding-zero-tau": partial(lagged_embed, np.arange(5), tau=0, e=2),
    "embedding-negative-tau": partial(lagged_embed, np.arange(5), tau=-1, e=2),
    "embedding-zero-dimension": partial(lagged_embed, np.arange(5), tau=1, e=0),
    "embedding-negative-dimension": partial(lagged_embed, np.arange(5), tau=1, e=-1),
    "embedding-insufficient-history": partial(lagged_embed, np.arange(5), tau=2, e=4),
    "select-no-scores": partial(select, np.full((2, 2, 3), np.nan), E=[1, 2], tau=[1, 2]),
}


@given(problem=embedding_problems())
def test_compatibility(problem: EmbeddingProblem) -> None:
    actual = lagged_embed(problem.x, tau=problem.tau, e=problem.e)
    np.testing.assert_array_equal(actual, embed_reference(problem), strict=True)


@pytest.mark.parametrize("problem", SCANS.values(), ids=SCANS.keys())
def test_scan(problem: ScanProblem) -> None:
    check_scan(problem)


@pytest.mark.parametrize("problem", SELECTIONS.values(), ids=SELECTIONS.keys())
def test_select(problem: SelectProblem) -> None:
    check_select(problem)


@pytest.mark.parametrize("call", INVALID.values(), ids=INVALID.keys())
def test_invalid(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()
