from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest

from edmkit.metrics import mae, mean_rho, rhos, rmse


type Metric = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Problem(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class Expected(NamedTuple):
    metric: Metric
    problem: Problem
    value: np.ndarray | float


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(x @ y / denominator)


def rhos_reference(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]
    if predictions.ndim == 2:
        return np.array([correlation(predictions[:, d], observations[:, d]) for d in range(predictions.shape[1])])
    return np.array(
        [[correlation(predictions[b, :, d], observations[b, :, d]) for d in range(predictions.shape[2])] for b in range(predictions.shape[0])]
    )


def mean_rho_reference(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    return rhos_reference(predictions, observations).mean(axis=-1)


def rmse_reference(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    axes = tuple(range(max(predictions.ndim - 2, 0), predictions.ndim))
    return np.sqrt(np.mean((predictions - observations) ** 2, axis=axes))


def mae_reference(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    axes = tuple(range(max(predictions.ndim - 2, 0), predictions.ndim))
    return np.mean(np.abs(predictions - observations), axis=axes)


def check_metric(metric: Metric, problem: Problem, expected: np.ndarray | float) -> None:
    actual = np.asarray(metric(*problem))
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)


COMPATIBILITY = {
    "rhos": (rhos, rhos_reference),
    "mean-rho": (mean_rho, mean_rho_reference),
    "rmse": (rmse, rmse_reference),
    "mae": (mae, mae_reference),
}
METRICS = {name: metric for name, (metric, _) in COMPATIBILITY.items()}

predictions = np.array([[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [8.0, -2.0]])
observations = np.array([[2.0, -1.0], [1.0, 2.0], [5.0, 4.0], [7.0, 8.0]])
INPUTS = {
    "1d": Problem(predictions[:, 0], observations[:, 0]),
    "2d": Problem(predictions, observations),
    "3d": Problem(np.stack([predictions, -predictions]), np.stack([observations, observations[::-1]])),
}

CONSTANTS = {
    "1d": (Problem(np.ones(4), np.arange(4.0)), np.zeros(1), np.array(0.0)),
    "2d": (Problem(np.ones((4, 2)), np.arange(8.0).reshape(4, 2)), np.zeros(2), np.array(0.0)),
    "3d": (Problem(np.ones((2, 4, 2)), np.arange(16.0).reshape(2, 4, 2)), np.zeros((2, 2)), np.zeros(2)),
}
MIXED = Problem(np.column_stack([np.ones(4), np.arange(4.0)]), np.column_stack([np.arange(4.0), 2 * np.arange(4.0) + 1]))
EDGES = {
    **{f"rhos-constant-{rank}": Expected(rhos, problem, expected) for rank, (problem, expected, _) in CONSTANTS.items()},
    **{f"mean-rho-constant-{rank}": Expected(mean_rho, problem, expected) for rank, (problem, _, expected) in CONSTANTS.items()},
    "rhos-mixed-constant-and-varying": Expected(rhos, MIXED, np.array([0.0, 1.0])),
    "mean-rho-mixed-constant-and-varying": Expected(mean_rho, MIXED, np.array(0.5)),
}

INVALID = {
    "shape-mismatch": Problem(np.zeros((3, 2)), np.zeros((3, 1))),
    "scalar": Problem(np.array(1.0), np.array(1.0)),
    "four-dimensional": Problem(np.zeros((2, 3, 4, 5)), np.zeros((2, 3, 4, 5))),
}


@pytest.mark.parametrize(("metric", "reference"), COMPATIBILITY.values(), ids=COMPATIBILITY.keys())
@pytest.mark.parametrize("problem", INPUTS.values(), ids=INPUTS.keys())
def test_compatibility(metric: Metric, reference: Metric, problem: Problem) -> None:
    check_metric(metric, problem, reference(*problem))


@pytest.mark.parametrize("case", EDGES.values(), ids=EDGES.keys())
def test_valid(case: Expected) -> None:
    check_metric(case.metric, case.problem, case.value)


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
@pytest.mark.parametrize("problem", INVALID.values(), ids=INVALID.keys())
def test_invalid(metric: Metric, problem: Problem) -> None:
    with pytest.raises(ValueError):
        metric(*problem)
