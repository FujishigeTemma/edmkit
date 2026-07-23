from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.util import autocorrelation, dtw, pad, pairwise_distance, pairwise_distance_np


class Distance(NamedTuple):
    a: np.ndarray
    b: np.ndarray | None


class DistanceShapes(NamedTuple):
    a: tuple[int, ...]
    b: tuple[int, ...] | None


class SequencePair(NamedTuple):
    a: np.ndarray
    b: np.ndarray


class Series(NamedTuple):
    x: np.ndarray
    max_lag: int
    step: int


class Pad(NamedTuple):
    arrays: list[np.ndarray]
    expected: np.ndarray


def distance_reference(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.sum((a[..., :, None, :] - b[..., None, :, :]) ** 2, axis=-1)


def dtw_reference(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.full((len(a) + 1, len(b) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1])


def autocorrelation_reference(x: np.ndarray, max_lag: int, step: int) -> np.ndarray:
    centered = x - x.mean()
    scale = len(x) * np.var(centered)
    return np.asarray([np.dot(centered[: len(x) - lag], centered[lag:]) / scale for lag in range(0, min(max_lag, len(x)), step)])


def check_numpy_distance(case: Distance) -> None:
    actual = pairwise_distance_np(case.a, case.b)
    expected = distance_reference(case.a, case.b)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)


def check_tensor_distance(case: DistanceShapes) -> None:
    from tinygrad import Tensor

    rng = np.random.default_rng(0)
    a = rng.normal(size=case.a).astype(np.float32)
    b = None if case.b is None else rng.normal(size=case.b).astype(np.float32)
    actual = pairwise_distance(Tensor(a), None if b is None else Tensor(b)).numpy()
    expected = distance_reference(a, b)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def check_dtw(case: SequencePair) -> None:
    np.testing.assert_allclose(dtw(case.a, case.b), dtw_reference(case.a, case.b), atol=1e-12, rtol=1e-12)


def check_autocorrelation(case: Series) -> None:
    actual = autocorrelation(case.x, case.max_lag, case.step)
    expected = autocorrelation_reference(case.x, case.max_lag, case.step)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def check_pad(case: Pad) -> None:
    actual = pad(case.arrays)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, case.expected)


FLOATS = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


@st.composite
def distance_cases(draw):
    batch = draw(st.none() | st.integers(1, 3))
    n, m, d = draw(st.integers(1, 8)), draw(st.integers(1, 8)), draw(st.integers(1, 4))
    a_shape = (n, d) if batch is None else (batch, n, d)
    b_shape = (m, d) if batch is None else (batch, m, d)
    a = draw(hnp.arrays(np.float64, a_shape, elements=FLOATS))
    b = None if draw(st.booleans()) else draw(hnp.arrays(np.float64, b_shape, elements=FLOATS))
    return Distance(a, b)


@st.composite
def sequence_cases(draw):
    n, m, d = draw(st.integers(1, 10)), draw(st.integers(1, 10)), draw(st.integers(1, 3))
    a = draw(hnp.arrays(np.float64, (n, d), elements=FLOATS))
    b = draw(hnp.arrays(np.float64, (m, d), elements=FLOATS))
    return SequencePair(a, b)


@st.composite
def series_cases(draw):
    n = draw(st.integers(2, 32))
    values = draw(st.lists(st.integers(-20, 20), min_size=n, max_size=n, unique=True))
    return Series(np.asarray(values, dtype=np.float64), draw(st.integers(1, n + 5)), draw(st.integers(1, 4)))


@given(case=distance_cases())
def test_numpy_distance(case: Distance) -> None:
    check_numpy_distance(case)


@given(case=sequence_cases())
def test_dtw(case: SequencePair) -> None:
    check_dtw(case)


@given(case=series_cases())
def test_autocorrelation(case: Series) -> None:
    check_autocorrelation(case)


VALID = [
    pytest.param(partial(check_tensor_distance, DistanceShapes((6, 3), (4, 3))), id="tensor-2d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor_distance, DistanceShapes((6, 3), None)), id="tensor-2d-self", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor_distance, DistanceShapes((2, 5, 3), (2, 4, 3))), id="tensor-3d", marks=pytest.mark.gpu),
    pytest.param(
        partial(
            check_pad,
            Pad(
                [np.array([[1, 2], [3, 4]]), np.array([[5], [6]])],
                np.array([[[1, 2], [3, 4]], [[5, 0], [6, 0]]]),
            ),
        ),
        id="pad-widths",
    ),
]


@pytest.mark.parametrize("check", VALID)
def test_valid(check: Callable[[], None]) -> None:
    check()


def call_invalid_numpy_distance(case: Distance) -> None:
    pairwise_distance_np(case.a, case.b)


def call_invalid_tensor_distance(case: Distance) -> None:
    from tinygrad import Tensor

    pairwise_distance(Tensor(case.a), None if case.b is None else Tensor(case.b))


def call_invalid_pad(arrays: list[np.ndarray]) -> None:
    pad(arrays)


INVALID = [
    pytest.param(partial(call_invalid_numpy_distance, Distance(np.zeros(3), None)), id="numpy-distance-rank"),
    pytest.param(
        partial(call_invalid_numpy_distance, Distance(np.zeros((3, 2)), np.zeros((2, 3, 2)))),
        id="numpy-distance-rank-mismatch",
    ),
    pytest.param(partial(call_invalid_tensor_distance, Distance(np.zeros(3), None)), id="tensor-distance-rank", marks=pytest.mark.gpu),
    pytest.param(
        partial(call_invalid_tensor_distance, Distance(np.zeros((3, 2)), np.zeros((2, 3, 2)))),
        id="tensor-distance-rank-mismatch",
        marks=pytest.mark.gpu,
    ),
    pytest.param(partial(call_invalid_pad, [np.zeros(3)]), id="pad-rank"),
    pytest.param(partial(call_invalid_pad, [np.zeros((2, 1)), np.zeros((3, 2))]), id="pad-length"),
]


@pytest.mark.parametrize("call", INVALID)
def test_invalid(call: Callable[[], None]) -> None:
    with pytest.raises(ValueError):
        call()
