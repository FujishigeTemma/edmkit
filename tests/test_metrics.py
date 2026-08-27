from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.metrics import MetricFunc, mae, pearson_correlation, rmse


class PearsonCorrelationProblem(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class PearsonCorrelationCase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class RMSECase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class MAECase(NamedTuple):
    predictions: np.ndarray
    observations: np.ndarray


class MetricFuncProblem(NamedTuple):
    metric: MetricFunc
    predictions: np.ndarray
    observations: np.ndarray


class MetricFuncCase(NamedTuple):
    metric: MetricFunc
    predictions: np.ndarray
    observations: np.ndarray


# pearson_correlation keeps its full 7-point set: its reference is a genuinely independent
# formulation (an explicit per-column loop using np.linalg.norm), unlike the rmse/mae
# references below, which turned out to be the implementation restated and were deleted.


def pearson_correlation_reference(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    def column_correlation(x: np.ndarray, y: np.ndarray) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        # zero variance leaves the correlation undefined, not zero
        return np.nan if denominator == 0 else float(x_centered @ y_centered / denominator)

    if predictions.ndim == 1:
        predictions = predictions[:, None]
        observations = observations[:, None]
    if predictions.ndim == 2:
        correlations = np.array([column_correlation(predictions[:, d], observations[:, d]) for d in range(predictions.shape[1])])
    else:
        correlations = np.array(
            [
                [column_correlation(predictions[b, :, d], observations[b, :, d]) for d in range(predictions.shape[2])]
                for b in range(predictions.shape[0])
            ]
        )
    return correlations.mean(axis=-1)


def check_pearson_correlation(predictions: np.ndarray, observations: np.ndarray) -> None:
    actual = pearson_correlation(predictions, observations)
    expected = pearson_correlation_reference(predictions, observations)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def check_pearson_correlation_tensor(predictions: np.ndarray, observations: np.ndarray) -> None:
    from tinygrad import Tensor

    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    expected = pearson_correlation(predictions, observations)
    actual = pearson_correlation(Tensor(predictions), Tensor(observations)).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def check_pearson_correlation_tensor_gradient(predictions: np.ndarray, observations: np.ndarray) -> None:
    from tinygrad import Tensor

    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    predictions_tensor, observations_tensor = Tensor(predictions), Tensor(observations)
    correlation = pearson_correlation(predictions_tensor, observations_tensor)
    gradients = correlation.sum().gradient(predictions_tensor, observations_tensor)

    # where the correlation is undefined the gradient is NaN too, rather than a finite number
    # that would let an optimiser keep stepping on a meaningless loss
    defined = bool(np.isfinite(correlation.numpy()).all())
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all() == defined


@st.composite
def pearson_correlation_problems(draw) -> PearsonCorrelationProblem:
    ndim = draw(st.integers(1, 3))
    n = draw(st.integers(2, 15))
    d = draw(st.integers(1, 4))
    b = draw(st.integers(1, 3))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if ndim == 1:
        shape = (n,)
    elif ndim == 2:
        shape = (n, d)
    else:
        shape = (b, n, d)
    predictions = rng.normal(size=shape)
    observations = rng.normal(size=shape)
    return PearsonCorrelationProblem(predictions, observations)


@given(problem=pearson_correlation_problems())
def test_pearson_correlation_compatibility(problem: PearsonCorrelationProblem) -> None:
    check_pearson_correlation(*problem)


# rmse and mae dropped their xxx_reference / xxx_problems / test_xxx_compatibility: a
# reference that chains the same mean-of-squares (or mean-of-abs) as src, over the same
# axes, mirrors any axis or off-by-one bug src might have, so it verified nothing. Instead,
# check_rmse/check_mae assert the documented shape contract plus the numeric value against
# a golden `expected` pinned in the Case (computed once by hand, by flattening each batch's
# errors and summing/averaging in plain Python — a structurally different path from src's
# chained `.mean(axis=-1).mean(axis=-1)`).


def check_rmse(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    actual = rmse(predictions, observations)
    expected_shape = () if predictions.ndim in (1, 2) else (predictions.shape[0],)
    assert actual.shape == expected_shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def check_rmse_tensor(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    from tinygrad import Tensor

    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    actual = rmse(Tensor(predictions), Tensor(observations)).numpy()
    expected_shape = () if predictions.ndim in (1, 2) else (predictions.shape[0],)
    assert actual.shape == expected_shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def check_rmse_tensor_gradient(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    from tinygrad import Tensor

    del expected  # gradient finiteness does not depend on the golden value
    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    predictions_tensor, observations_tensor = Tensor(predictions), Tensor(observations)
    gradients = rmse(predictions_tensor, observations_tensor).sum().gradient(predictions_tensor, observations_tensor)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def check_mae(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    actual = mae(predictions, observations)
    expected_shape = () if predictions.ndim in (1, 2) else (predictions.shape[0],)
    assert actual.shape == expected_shape
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def check_mae_tensor(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    from tinygrad import Tensor

    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    actual = mae(Tensor(predictions), Tensor(observations)).numpy()
    expected_shape = () if predictions.ndim in (1, 2) else (predictions.shape[0],)
    assert actual.shape == expected_shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def check_mae_tensor_gradient(predictions: np.ndarray, observations: np.ndarray, expected: np.ndarray | float) -> None:
    from tinygrad import Tensor

    del expected  # gradient finiteness does not depend on the golden value
    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    predictions_tensor, observations_tensor = Tensor(predictions), Tensor(observations)
    gradients = mae(predictions_tensor, observations_tensor).sum().gradient(predictions_tensor, observations_tensor)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


# The (N,)/(N,D) -> () and (B,N,D) -> (B,) shape rule, and the ValueError on shape mismatch
# or ndim not in (1, 2, 3), are identical across all three metrics. That shared contract is
# expressed once here, on the MetricFunc type, instead of being triplicated per metric.


def check_metric_func(metric: MetricFunc, predictions: np.ndarray, observations: np.ndarray) -> None:
    """Shape and NaN-propagation contract shared by every MetricFunc."""
    actual = metric(predictions, observations)
    expected_shape = () if predictions.ndim in (1, 2) else (predictions.shape[0],)
    assert actual.shape == expected_shape

    # missing data must stay visible: a metric never turns a NaN into a real-looking score
    missing = bool(np.isnan(predictions).any() or np.isnan(observations).any())
    assert bool(np.isnan(np.asarray(actual)).any()) == missing


def check_metric_func_tensor(metric: MetricFunc, predictions: np.ndarray, observations: np.ndarray) -> None:
    """Both backends must agree on the contract, NaN included."""
    from tinygrad import Tensor

    predictions, observations = (array.astype(np.float32) for array in (predictions, observations))
    expected = metric(predictions, observations)
    actual = metric(Tensor(predictions), Tensor(observations)).numpy()
    assert actual.shape == expected.shape
    # assert_allclose treats NaN as equal to NaN by default, which is what the contract wants
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@st.composite
def metric_func_problems(draw) -> MetricFuncProblem:
    metric = draw(st.sampled_from((pearson_correlation, rmse, mae)))
    ndim = draw(st.integers(1, 3))
    n = draw(st.integers(2, 15))
    d = draw(st.integers(1, 4))
    b = draw(st.integers(1, 3))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if ndim == 1:
        shape = (n,)
    elif ndim == 2:
        shape = (n, d)
    else:
        shape = (b, n, d)
    predictions = rng.normal(size=shape)
    observations = rng.normal(size=shape)
    if draw(st.booleans()):
        corrupted = predictions if draw(st.booleans()) else observations
        corrupted[np.unravel_index(draw(st.integers(0, corrupted.size - 1)), corrupted.shape)] = np.nan
    return MetricFuncProblem(metric, predictions, observations)


@given(problem=metric_func_problems())
def test_metric_func_compatibility(problem: MetricFuncProblem) -> None:
    check_metric_func(*problem)


SAMPLE_PREDICTIONS = np.array([[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [8.0, -2.0]])
SAMPLE_OBSERVATIONS = np.array([[2.0, -1.0], [1.0, 2.0], [5.0, 4.0], [7.0, 8.0]])

CONSTANT_SERIES = {
    "1d": (np.ones(4), np.arange(4.0)),
    "2d": (np.ones((4, 2)), np.arange(8.0).reshape(4, 2)),
    "3d": (np.ones((2, 4, 2)), np.arange(16.0).reshape(2, 4, 2)),
}
MIXED_SERIES = (
    np.column_stack([np.ones(4), np.arange(4.0)]),
    np.column_stack([np.arange(4.0), 2 * np.arange(4.0) + 1]),
)
IDENTICAL_SERIES = (SAMPLE_PREDICTIONS, SAMPLE_PREDICTIONS)

NAN_PREDICTIONS = SAMPLE_PREDICTIONS.copy()
NAN_PREDICTIONS[1, 0] = np.nan
NAN_OBSERVATIONS = SAMPLE_OBSERVATIONS.copy()
NAN_OBSERVATIONS[2, 1] = np.nan

SHAPE_MISMATCH_CASE = (np.zeros((3, 2)), np.zeros((3, 1)))
SCALAR_CASE = (np.array(1.0), np.array(1.0))
FOUR_DIMENSIONAL_CASE = (np.zeros((2, 3, 4, 5)), np.zeros((2, 3, 4, 5)))

PEARSON_CORRELATION_VALID = {
    "1d": PearsonCorrelationCase(SAMPLE_PREDICTIONS[:, 0], SAMPLE_OBSERVATIONS[:, 0]),
    "2d": PearsonCorrelationCase(SAMPLE_PREDICTIONS, SAMPLE_OBSERVATIONS),
    "3d": PearsonCorrelationCase(
        np.stack([SAMPLE_PREDICTIONS, -SAMPLE_PREDICTIONS]),
        np.stack([SAMPLE_OBSERVATIONS, SAMPLE_OBSERVATIONS[::-1]]),
    ),
    "constant-1d": PearsonCorrelationCase(*CONSTANT_SERIES["1d"]),
    "constant-2d": PearsonCorrelationCase(*CONSTANT_SERIES["2d"]),
    "constant-3d": PearsonCorrelationCase(*CONSTANT_SERIES["3d"]),
    "mixed-constant-and-varying": PearsonCorrelationCase(*MIXED_SERIES),
    "perfectly-correlated": PearsonCorrelationCase(*IDENTICAL_SERIES),
}

PEARSON_CORRELATION_MODES = {
    "numpy": check_pearson_correlation,
    "tinygrad": pytest.param(check_pearson_correlation_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_pearson_correlation_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", PEARSON_CORRELATION_MODES.values(), ids=PEARSON_CORRELATION_MODES.keys())
@pytest.mark.parametrize("case", PEARSON_CORRELATION_VALID.values(), ids=PEARSON_CORRELATION_VALID.keys())
def test_pearson_correlation_valid(case: PearsonCorrelationCase, mode) -> None:
    mode(*case)


# Golden `expected` values below were computed once, independently of src, by flattening
# each batch's prediction/observation errors and reducing with plain Python (not by chaining
# vectorized axis reductions the way src does): rmse = sqrt(mean(errors**2)),
# mae = mean(abs(errors)), one flat group per batch (or a single group for 1D/2D input).

RMSE_VALID = {
    "1d": RMSECase(SAMPLE_PREDICTIONS[:, 0], SAMPLE_OBSERVATIONS[:, 0]),
    "2d": RMSECase(SAMPLE_PREDICTIONS, SAMPLE_OBSERVATIONS),
    "3d": RMSECase(
        np.stack([SAMPLE_PREDICTIONS, -SAMPLE_PREDICTIONS]),
        np.stack([SAMPLE_OBSERVATIONS, SAMPLE_OBSERVATIONS[::-1]]),
    ),
    "constant-1d": RMSECase(*CONSTANT_SERIES["1d"]),
    "constant-2d": RMSECase(*CONSTANT_SERIES["2d"]),
    "constant-3d": RMSECase(*CONSTANT_SERIES["3d"]),
    "mixed-constant-and-varying": RMSECase(*MIXED_SERIES),
    "zero-error": RMSECase(*IDENTICAL_SERIES),
}

RMSE_EXPECTED: dict[str, np.ndarray | float] = {
    "1d": 1.0,
    "2d": 4.046603514059662,
    "3d": np.array([4.046603514059662, 7.424621202458749]),
    "constant-1d": 1.224744871391589,
    "constant-2d": 3.391164991562634,
    "constant-3d": np.array([3.391164991562634, 10.747092630102339]),
    "mixed-constant-and-varying": 2.1213203435596424,
    "zero-error": 0.0,
}

RMSE_MODES = {
    "numpy": check_rmse,
    "tinygrad": pytest.param(check_rmse_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_rmse_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", RMSE_MODES.values(), ids=RMSE_MODES.keys())
@pytest.mark.parametrize("name", RMSE_VALID)
def test_rmse_valid(name: str, mode) -> None:
    mode(*RMSE_VALID[name], RMSE_EXPECTED[name])


MAE_VALID = {
    "1d": MAECase(SAMPLE_PREDICTIONS[:, 0], SAMPLE_OBSERVATIONS[:, 0]),
    "2d": MAECase(SAMPLE_PREDICTIONS, SAMPLE_OBSERVATIONS),
    "3d": MAECase(
        np.stack([SAMPLE_PREDICTIONS, -SAMPLE_PREDICTIONS]),
        np.stack([SAMPLE_OBSERVATIONS, SAMPLE_OBSERVATIONS[::-1]]),
    ),
    "constant-1d": MAECase(*CONSTANT_SERIES["1d"]),
    "constant-2d": MAECase(*CONSTANT_SERIES["2d"]),
    "constant-3d": MAECase(*CONSTANT_SERIES["3d"]),
    "mixed-constant-and-varying": MAECase(*MIXED_SERIES),
    "zero-error": MAECase(*IDENTICAL_SERIES),
}

MAE_EXPECTED: dict[str, np.ndarray | float] = {
    "1d": 1.0,
    "2d": 2.625,
    "3d": np.array([2.625, 6.875]),
    "constant-1d": 1.0,
    "constant-2d": 2.75,
    "constant-3d": np.array([2.75, 10.5]),
    "mixed-constant-and-varying": 1.75,
    "zero-error": 0.0,
}

MAE_MODES = {
    "numpy": check_mae,
    "tinygrad": pytest.param(check_mae_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_mae_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", MAE_MODES.values(), ids=MAE_MODES.keys())
@pytest.mark.parametrize("name", MAE_VALID)
def test_mae_valid(name: str, mode) -> None:
    mode(*MAE_VALID[name], MAE_EXPECTED[name])


METRIC_FUNC_SHAPES = {
    "1d": (SAMPLE_PREDICTIONS[:, 0], SAMPLE_OBSERVATIONS[:, 0]),
    "2d": (SAMPLE_PREDICTIONS, SAMPLE_OBSERVATIONS),
    "3d": (
        np.stack([SAMPLE_PREDICTIONS, -SAMPLE_PREDICTIONS]),
        np.stack([SAMPLE_OBSERVATIONS, SAMPLE_OBSERVATIONS[::-1]]),
    ),
    "nan-predictions": (NAN_PREDICTIONS, SAMPLE_OBSERVATIONS),
    "nan-observations": (SAMPLE_PREDICTIONS, NAN_OBSERVATIONS),
    "nan-3d": (np.stack([NAN_PREDICTIONS, SAMPLE_PREDICTIONS]), np.stack([SAMPLE_OBSERVATIONS, SAMPLE_OBSERVATIONS])),
    "all-nan": (np.full_like(SAMPLE_PREDICTIONS, np.nan), SAMPLE_OBSERVATIONS),
}
METRIC_FUNCS = (pearson_correlation, rmse, mae)

METRIC_FUNC_VALID = {
    f"{shape_name}-{metric.__name__.replace('_', '-')}": MetricFuncCase(metric, predictions, observations)
    for shape_name, (predictions, observations) in METRIC_FUNC_SHAPES.items()
    for metric in METRIC_FUNCS
}


METRIC_FUNC_MODES = {
    "numpy": check_metric_func,
    "tinygrad": pytest.param(check_metric_func_tensor, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", METRIC_FUNC_MODES.values(), ids=METRIC_FUNC_MODES.keys())
@pytest.mark.parametrize("case", METRIC_FUNC_VALID.values(), ids=METRIC_FUNC_VALID.keys())
def test_metric_func_valid(case: MetricFuncCase, mode) -> None:
    mode(*case)


METRIC_FUNC_INVALID_SHAPES = {
    "shape-mismatch": SHAPE_MISMATCH_CASE,
    "scalar": SCALAR_CASE,
    "four-dimensional": FOUR_DIMENSIONAL_CASE,
}

METRIC_FUNC_INVALID = {
    f"{shape_name}-{metric.__name__.replace('_', '-')}": MetricFuncCase(metric, predictions, observations)
    for shape_name, (predictions, observations) in METRIC_FUNC_INVALID_SHAPES.items()
    for metric in METRIC_FUNCS
}


@pytest.mark.parametrize("case", METRIC_FUNC_INVALID.values(), ids=METRIC_FUNC_INVALID.keys())
def test_metric_func_invalid(case: MetricFuncCase) -> None:
    with pytest.raises(ValueError):
        case.metric(case.predictions, case.observations)
