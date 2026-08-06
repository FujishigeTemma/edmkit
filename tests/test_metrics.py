from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest

from edmkit.metrics import mae, mean_rho, rhos, rmse


type Metric = Callable[[np.ndarray, np.ndarray], np.ndarray]


class RhosCase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class MeanRhoCase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class RMSECase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class MAECase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


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


def check_metric(metric: Metric, predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    actual = np.asarray(metric(predictions, observations))
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)


predictions = np.array([[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [8.0, -2.0]])
observations = np.array([[2.0, -1.0], [1.0, 2.0], [5.0, 4.0], [7.0, 8.0]])
RMSE_VALID = {
    "1d": RMSECase(predictions[:, 0], observations[:, 0]),
    "2d": RMSECase(predictions, observations),
    "3d": RMSECase(np.stack([predictions, -predictions]), np.stack([observations, observations[::-1]])),
}

CONSTANTS = {
    "1d": (np.ones(4), np.arange(4.0)),
    "2d": (np.ones((4, 2)), np.arange(8.0).reshape(4, 2)),
    "3d": (np.ones((2, 4, 2)), np.arange(16.0).reshape(2, 4, 2)),
}
MIXED = (np.column_stack([np.ones(4), np.arange(4.0)]), np.column_stack([np.arange(4.0), 2 * np.arange(4.0) + 1]))

RHOS_VALID = {
    **{name: RhosCase(*case) for name, case in RMSE_VALID.items()},
    **{f"constant-{rank}": RhosCase(*case) for rank, case in CONSTANTS.items()},
    "mixed-constant-and-varying": RhosCase(*MIXED),
}
RHOS_INVALID = {
    "shape-mismatch": RhosCase(np.zeros((3, 2)), np.zeros((3, 1))),
    "scalar": RhosCase(np.array(1.0), np.array(1.0)),
    "four-dimensional": RhosCase(np.zeros((2, 3, 4, 5)), np.zeros((2, 3, 4, 5))),
}

MEAN_RHO_VALID = {name: MeanRhoCase(*case) for name, case in RHOS_VALID.items()}
MEAN_RHO_INVALID = {name: MeanRhoCase(*case) for name, case in RHOS_INVALID.items()}
RMSE_INVALID = {name: RMSECase(*case) for name, case in RHOS_INVALID.items()}
MAE_VALID = {name: MAECase(*case) for name, case in RMSE_VALID.items()}
MAE_INVALID = {name: MAECase(*case) for name, case in RHOS_INVALID.items()}


@pytest.mark.parametrize("case", RHOS_VALID.values(), ids=RHOS_VALID.keys())
def test_rhos_valid(case: RhosCase) -> None:
    check_metric(rhos, *case, rhos_reference(*case))


@pytest.mark.parametrize("case", RHOS_INVALID.values(), ids=RHOS_INVALID.keys())
def test_rhos_invalid(case: RhosCase) -> None:
    with pytest.raises(ValueError):
        rhos(*case)


@pytest.mark.parametrize("case", MEAN_RHO_VALID.values(), ids=MEAN_RHO_VALID.keys())
def test_mean_rho_valid(case: MeanRhoCase) -> None:
    check_metric(mean_rho, *case, mean_rho_reference(*case))


@pytest.mark.parametrize("case", MEAN_RHO_INVALID.values(), ids=MEAN_RHO_INVALID.keys())
def test_mean_rho_invalid(case: MeanRhoCase) -> None:
    with pytest.raises(ValueError):
        mean_rho(*case)


@pytest.mark.parametrize("case", RMSE_VALID.values(), ids=RMSE_VALID.keys())
def test_rmse_valid(case: RMSECase) -> None:
    check_metric(rmse, *case, rmse_reference(*case))


@pytest.mark.parametrize("case", RMSE_INVALID.values(), ids=RMSE_INVALID.keys())
def test_rmse_invalid(case: RMSECase) -> None:
    with pytest.raises(ValueError):
        rmse(*case)


@pytest.mark.parametrize("case", MAE_VALID.values(), ids=MAE_VALID.keys())
def test_mae_valid(case: MAECase) -> None:
    check_metric(mae, *case, mae_reference(*case))


@pytest.mark.parametrize("case", MAE_INVALID.values(), ids=MAE_INVALID.keys())
def test_mae_invalid(case: MAECase) -> None:
    with pytest.raises(ValueError):
        mae(*case)
