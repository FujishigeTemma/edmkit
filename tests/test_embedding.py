from __future__ import annotations

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


class LaggedEmbedProblem(NamedTuple):
    x: np.ndarray
    tau: int
    e: int


class LaggedEmbedCase(NamedTuple):
    x: np.ndarray
    tau: int
    e: int


class ScanCase(NamedTuple):
    x: np.ndarray
    Y: np.ndarray | None
    E: list[int]
    tau: list[int]
    split: SplitFunc | None = None
    predict: PredictFunc | None = None
    metric: MetricFunc | None = None


class SelectCase(NamedTuple):
    scores: np.ndarray
    E: list[int]
    tau: list[int]


def embed_reference(problem: LaggedEmbedProblem) -> np.ndarray:
    present = np.arange((problem.e - 1) * problem.tau, len(problem.x))[:, None]
    lags = problem.tau * np.arange(problem.e)[None, :]
    return problem.x[present - lags]


@st.composite
def lagged_embed_problems(draw):
    tau = draw(st.integers(1, 6))
    e = draw(st.integers(1, 7))
    minimum = (e - 1) * tau + 1
    n = draw(st.integers(minimum, minimum + 30))
    x = draw(hnp.arrays(np.int64, n, elements=st.integers(-10_000, 10_000)))
    return LaggedEmbedProblem(x, tau, e)


def scan_reference(
    x: np.ndarray,
    Y: np.ndarray | None,
    E: list[int],
    tau: list[int],
    split: SplitFunc | None,
    predict: PredictFunc | None,
    metric: MetricFunc | None,
) -> np.ndarray:
    n = len(x)
    target = x if Y is None else Y
    target = target[:, None] if target.ndim == 1 else target
    split = split or partial(sliding_folds, train_size=max(n // 5, 2), validation_size=max(n // 10, 1))
    predict = predict or simplex_projection
    metric = metric or mean_rho
    tau_max = max(tau)
    rows: list[np.ndarray | None] = []

    for e in E:
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
        scores = np.empty((len(tau), len(folds)))
        for tau_index, delay in enumerate(tau):
            embedded = embed_reference(LaggedEmbedProblem(x, delay, e))[-n_usable:]
            for fold_index, fold in enumerate(folds):
                predictions = predict(embedded[fold.train], aligned_target[fold.train], embedded[fold.validation])
                predictions = np.asarray(predictions).reshape(len(fold.validation), target.shape[1])
                scores[tau_index, fold_index] = metric(predictions, aligned_target[fold.validation])
        rows.append(scores)

    n_folds = max((row.shape[1] for row in rows if row is not None), default=0)
    expected = np.full((len(E), len(tau), n_folds), np.nan)
    for e_index, row in enumerate(rows):
        if row is not None:
            expected[e_index, :, : row.shape[1]] = row
    return expected


def check_scan(
    x: np.ndarray,
    Y: np.ndarray | None,
    E: list[int],
    tau: list[int],
    split: SplitFunc | None,
    predict: PredictFunc | None,
    metric: MetricFunc | None,
) -> None:
    actual = scan(x, Y, E=E, tau=tau, split=split, predict=predict, metric=metric)
    np.testing.assert_allclose(actual, scan_reference(x, Y, E, tau, split, predict, metric), atol=1e-12, rtol=1e-12, equal_nan=True, strict=True)


def select_reference(scores: np.ndarray, E: list[int], tau: list[int]) -> tuple[int, int, float]:
    counts = np.sum(~np.isnan(scores), axis=2)
    means = np.divide(np.nansum(scores, axis=2), counts, out=np.full(scores.shape[:2], np.nan), where=counts > 0)
    sum_squares = np.nansum((scores - means[:, :, None]) ** 2, axis=2)
    standard_errors = np.sqrt(np.divide(sum_squares, counts * np.maximum(counts - 1, 1), out=np.zeros(scores.shape[:2]), where=counts > 1))
    e_index, tau_index = np.unravel_index(int(np.nanargmax(means - standard_errors)), means.shape)
    return E[e_index], tau[tau_index], float(means[e_index, tau_index])


def check_select(scores: np.ndarray, E: list[int], tau: list[int]) -> None:
    actual = select(scores, E=E, tau=tau)
    expected = select_reference(scores, E, tau)
    assert actual[:2] == expected[:2]
    assert actual[2] == pytest.approx(expected[2])


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

SCAN_VALID = {
    "default-self-target": ScanCase(signal, None, [1, 2, 4], [1, 2]),
    "expanding-mask-and-custom-predictor": ScanCase(
        signal,
        multivariate,
        [1, 3],
        [1, 2],
        split=partial(expanding_folds, initial_train_size=18, validation_size=5, stride=7),
        predict=mean_predictor,
        metric=rmse,
    ),
    "caller-shifted-target": ScanCase(shifted[:-3], shifted[3:], [2, 3], [1, 2]),
    "nan-fold-padding": ScanCase(
        padding_signal,
        None,
        [2, 6],
        [1, 2],
        partial(sliding_folds, train_size=10, validation_size=4, stride=4),
    ),
    "partially-unusable-candidates": ScanCase(unusable_signal, None, [2, 8], [1, 2]),
    "all-candidates-unusable": ScanCase(unusable_signal[:2], unusable_signal[28:], [2], [1]),
}

SELECT_VALID = {
    "highest-risk-adjusted-score": SelectCase(
        np.array([[[0.6, 0.8, np.nan], [0.4, 0.5, 0.6]], [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]]),
        [2, 3],
        [1, 2],
    ),
    "variance-penalty": SelectCase(
        np.array([[[0.5, 0.5, 0.5, 0.5]], [[0.9, 0.1, 0.9, 0.1]]]),
        [1, 2],
        [1],
    ),
    "missing-folds": SelectCase(
        np.array([[[np.nan, 0.2, 0.4]], [[0.5, np.nan, np.nan]]]),
        [2, 4],
        [3],
    ),
}

SELECT_INVALID = {
    "no-scores": SelectCase(np.full((2, 2, 3), np.nan), [1, 2], [1, 2]),
}


@given(problem=lagged_embed_problems())
def test_lagged_embed_compatibility(problem: LaggedEmbedProblem) -> None:
    actual = lagged_embed(problem.x, tau=problem.tau, e=problem.e)
    np.testing.assert_array_equal(actual, embed_reference(problem), strict=True)


LAGGED_EMBED_INVALID = {
    "matrix-input": LaggedEmbedCase(np.zeros((2, 3)), 1, 2),
    "zero-tau": LaggedEmbedCase(np.arange(5), 0, 2),
    "negative-tau": LaggedEmbedCase(np.arange(5), -1, 2),
    "zero-dimension": LaggedEmbedCase(np.arange(5), 1, 0),
    "negative-dimension": LaggedEmbedCase(np.arange(5), 1, -1),
    "insufficient-history": LaggedEmbedCase(np.arange(5), 2, 4),
}


@pytest.mark.parametrize("case", LAGGED_EMBED_INVALID.values(), ids=LAGGED_EMBED_INVALID.keys())
def test_lagged_embed_invalid(case: LaggedEmbedCase) -> None:
    with pytest.raises(ValueError):
        lagged_embed(case.x, tau=case.tau, e=case.e)


@pytest.mark.parametrize("case", SCAN_VALID.values(), ids=SCAN_VALID.keys())
def test_scan_valid(case: ScanCase) -> None:
    check_scan(*case)


@pytest.mark.parametrize("case", SELECT_VALID.values(), ids=SELECT_VALID.keys())
def test_select_valid(case: SelectCase) -> None:
    check_select(*case)


@pytest.mark.parametrize("case", SELECT_INVALID.values(), ids=SELECT_INVALID.keys())
def test_select_invalid(case: SelectCase) -> None:
    with pytest.raises(ValueError):
        select(case.scores, E=case.E, tau=case.tau)
