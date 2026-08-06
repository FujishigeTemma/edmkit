from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.util import autocorrelation, dtw, pad, pairwise_distance, pairwise_distance_np


class PairwiseDistanceNPProblem(NamedTuple):
    a: np.ndarray
    b: np.ndarray | None


class PairwiseDistanceNPCase(NamedTuple):
    a: np.ndarray
    b: np.ndarray | None


class PairwiseDistanceCase(NamedTuple):
    a: np.ndarray
    b: np.ndarray | None


class DTWProblem(NamedTuple):
    a: np.ndarray
    b: np.ndarray


class AutocorrelationProblem(NamedTuple):
    x: np.ndarray
    max_lag: int
    step: int


class PadCase(NamedTuple):
    arrays: list[np.ndarray]


def pairwise_distance_reference(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
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


def check_pairwise_distance_np(a: np.ndarray, b: np.ndarray | None) -> None:
    actual = pairwise_distance_np(a, b)
    expected = pairwise_distance_reference(a, b)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)


def check_pairwise_distance(a: np.ndarray, b: np.ndarray | None) -> None:
    from tinygrad import Tensor

    actual = pairwise_distance(Tensor(a), None if b is None else Tensor(b)).numpy()
    expected = pairwise_distance_reference(a, b)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def check_dtw(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_allclose(dtw(a, b), dtw_reference(a, b), atol=1e-12, rtol=1e-12)


def check_autocorrelation(x: np.ndarray, max_lag: int, step: int) -> None:
    actual = autocorrelation(x, max_lag, step)
    expected = autocorrelation_reference(x, max_lag, step)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def pad_reference(arrays: list[np.ndarray]) -> np.ndarray:
    width = max(array.shape[1] for array in arrays)
    return np.stack([np.pad(array, ((0, 0), (0, width - array.shape[1]))) for array in arrays])


def check_pad(arrays: list[np.ndarray]) -> None:
    actual = pad(arrays)
    expected = pad_reference(arrays)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


FLOATS = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


@st.composite
def pairwise_distance_np_problems(draw):
    batch = draw(st.none() | st.integers(1, 3))
    n, m, d = draw(st.integers(1, 8)), draw(st.integers(1, 8)), draw(st.integers(1, 4))
    a_shape = (n, d) if batch is None else (batch, n, d)
    b_shape = (m, d) if batch is None else (batch, m, d)
    a = draw(hnp.arrays(np.float64, a_shape, elements=FLOATS))
    b = None if draw(st.booleans()) else draw(hnp.arrays(np.float64, b_shape, elements=FLOATS))
    return PairwiseDistanceNPProblem(a, b)


@st.composite
def dtw_problems(draw):
    n, m, d = draw(st.integers(1, 10)), draw(st.integers(1, 10)), draw(st.integers(1, 3))
    a = draw(hnp.arrays(np.float64, (n, d), elements=FLOATS))
    b = draw(hnp.arrays(np.float64, (m, d), elements=FLOATS))
    return DTWProblem(a, b)


@st.composite
def autocorrelation_problems(draw):
    n = draw(st.integers(2, 32))
    values = draw(st.lists(st.integers(-20, 20), min_size=n, max_size=n, unique=True))
    return AutocorrelationProblem(np.asarray(values, dtype=np.float64), draw(st.integers(1, n + 5)), draw(st.integers(1, 4)))


@given(problem=pairwise_distance_np_problems())
def test_pairwise_distance_np_compatibility(problem: PairwiseDistanceNPProblem) -> None:
    check_pairwise_distance_np(*problem)


PAIRWISE_DISTANCE_NP_INVALID = {
    "rank": PairwiseDistanceNPCase(np.zeros(3), None),
    "rank-mismatch": PairwiseDistanceNPCase(np.zeros((3, 2)), np.zeros((2, 3, 2))),
}


@pytest.mark.parametrize("case", PAIRWISE_DISTANCE_NP_INVALID.values(), ids=PAIRWISE_DISTANCE_NP_INVALID.keys())
def test_pairwise_distance_np_invalid(case: PairwiseDistanceNPCase) -> None:
    with pytest.raises(ValueError):
        pairwise_distance_np(case.a, case.b)


PAIRWISE_DISTANCE_VALID = {
    "2d": PairwiseDistanceCase(
        np.random.default_rng(0).normal(size=(6, 3)).astype(np.float32), np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    ),
    "2d-self": PairwiseDistanceCase(np.random.default_rng(2).normal(size=(6, 3)).astype(np.float32), None),
    "3d": PairwiseDistanceCase(
        np.random.default_rng(3).normal(size=(2, 5, 3)).astype(np.float32),
        np.random.default_rng(4).normal(size=(2, 4, 3)).astype(np.float32),
    ),
}

PAIRWISE_DISTANCE_INVALID = {
    "rank": PairwiseDistanceCase(np.zeros(3), None),
    "rank-mismatch": PairwiseDistanceCase(np.zeros((3, 2)), np.zeros((2, 3, 2))),
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", PAIRWISE_DISTANCE_VALID.values(), ids=PAIRWISE_DISTANCE_VALID.keys())
def test_pairwise_distance_valid(case: PairwiseDistanceCase) -> None:
    check_pairwise_distance(*case)


@pytest.mark.gpu
@pytest.mark.parametrize("case", PAIRWISE_DISTANCE_INVALID.values(), ids=PAIRWISE_DISTANCE_INVALID.keys())
def test_pairwise_distance_invalid(case: PairwiseDistanceCase) -> None:
    from tinygrad import Tensor

    with pytest.raises(ValueError):
        pairwise_distance(Tensor(case.a), None if case.b is None else Tensor(case.b))


@given(problem=dtw_problems())
def test_dtw_compatibility(problem: DTWProblem) -> None:
    check_dtw(*problem)


@given(problem=autocorrelation_problems())
def test_autocorrelation_compatibility(problem: AutocorrelationProblem) -> None:
    check_autocorrelation(*problem)


PAD_VALID = {
    "different-widths": PadCase(
        [np.array([[1, 2], [3, 4]]), np.array([[5], [6]])],
    ),
}

PAD_INVALID = {
    "rank": PadCase([np.zeros(3)]),
    "length": PadCase([np.zeros((2, 1)), np.zeros((3, 2))]),
}


@pytest.mark.parametrize("case", PAD_VALID.values(), ids=PAD_VALID.keys())
def test_pad_valid(case: PadCase) -> None:
    check_pad(*case)


@pytest.mark.parametrize("case", PAD_INVALID.values(), ids=PAD_INVALID.keys())
def test_pad_invalid(case: PadCase) -> None:
    with pytest.raises(ValueError):
        pad(case.arrays)
