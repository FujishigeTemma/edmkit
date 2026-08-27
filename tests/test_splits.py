from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from edmkit.splits import Fold, expanding_folds, sliding_folds, temporal_fold


class TemporalFoldProblem(NamedTuple):
    n: int
    train_ratio: float
    gap: int


class TemporalFoldCase(NamedTuple):
    n: int
    train_ratio: float
    gap: int


class ExpandingFoldsProblem(NamedTuple):
    n: int
    initial_train_size: int
    validation_size: int
    stride: int | None
    gap: int


class ExpandingFoldsCase(NamedTuple):
    n: int
    initial_train_size: int
    validation_size: int
    stride: int | None
    gap: int


class SlidingFoldsProblem(NamedTuple):
    n: int
    train_size: int
    validation_size: int
    stride: int | None
    gap: int


class SlidingFoldsCase(NamedTuple):
    n: int
    train_size: int
    validation_size: int
    stride: int | None
    gap: int


def check_fold(fold: Fold, n: int) -> None:
    """Contract every Fold must satisfy, regardless of which splitter produced it."""
    assert fold.train.min() >= 0 and fold.train.max() < n
    assert fold.validation.min() >= 0 and fold.validation.max() < n
    assert set(fold.train.tolist()).isdisjoint(fold.validation.tolist())
    assert fold.train.max() < fold.validation.min()


def check_temporal_fold(n: int, train_ratio: float, gap: int, expected: Fold) -> None:
    actual = temporal_fold(n, train_ratio, gap=gap)
    np.testing.assert_array_equal(actual.train, expected.train, strict=True)
    np.testing.assert_array_equal(actual.validation, expected.validation, strict=True)
    check_fold(actual, n)

    repeated = temporal_fold(n, train_ratio, gap=gap)
    np.testing.assert_array_equal(repeated.train, actual.train, strict=True)
    np.testing.assert_array_equal(repeated.validation, actual.validation, strict=True)


def check_expanding_folds(n: int, initial_train_size: int, validation_size: int, stride: int | None, gap: int, expected: list[Fold]) -> None:
    actual = expanding_folds(n, initial_train_size=initial_train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(actual) == len(expected)

    previous_train: np.ndarray | None = None
    previous_validation_start = -1
    for fold, expected_fold in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(fold.train, expected_fold.train, strict=True)
        np.testing.assert_array_equal(fold.validation, expected_fold.validation, strict=True)
        check_fold(fold, n)
        assert int(fold.validation.min()) >= previous_validation_start
        if previous_train is not None:
            assert set(previous_train.tolist()) <= set(fold.train.tolist())
        previous_train = fold.train
        previous_validation_start = int(fold.validation.min())

    repeated = expanding_folds(n, initial_train_size=initial_train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(repeated) == len(actual)
    for fold, repeated_fold in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(repeated_fold.train, fold.train, strict=True)
        np.testing.assert_array_equal(repeated_fold.validation, fold.validation, strict=True)


def check_sliding_folds(n: int, train_size: int, validation_size: int, stride: int | None, gap: int, expected: list[Fold]) -> None:
    actual = sliding_folds(n, train_size=train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(actual) == len(expected)

    previous_validation_start = -1
    for fold, expected_fold in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(fold.train, expected_fold.train, strict=True)
        np.testing.assert_array_equal(fold.validation, expected_fold.validation, strict=True)
        check_fold(fold, n)
        assert fold.train.size == train_size
        assert int(fold.validation.min()) >= previous_validation_start
        previous_validation_start = int(fold.validation.min())

    repeated = sliding_folds(n, train_size=train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(repeated) == len(actual)
    for fold, repeated_fold in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(repeated_fold.train, fold.train, strict=True)
        np.testing.assert_array_equal(repeated_fold.validation, fold.validation, strict=True)


@st.composite
def temporal_fold_problems(draw):
    n = draw(st.integers(3, 40))
    train_ratio = draw(st.floats(0.05, 0.95, allow_nan=False))
    gap = draw(st.integers(0, n))
    train_end = int(n * train_ratio)
    assume(train_end >= 1)
    assume(train_end + gap < n)
    return TemporalFoldProblem(n, train_ratio, gap)


@given(problem=temporal_fold_problems())
def test_temporal_fold_compatibility(problem: TemporalFoldProblem) -> None:
    n, train_ratio, gap = problem
    actual = temporal_fold(n, train_ratio, gap=gap)
    check_fold(actual, n)
    # Docstring contract: train starts at 0, validation runs to the end, and the
    # boundary between them is separated by exactly `gap` skipped indices.
    assert int(actual.train.min()) == 0
    assert int(actual.validation.max()) == n - 1
    assert int(actual.validation.min()) - int(actual.train.max()) - 1 == gap

    repeated = temporal_fold(n, train_ratio, gap=gap)
    np.testing.assert_array_equal(repeated.train, actual.train, strict=True)
    np.testing.assert_array_equal(repeated.validation, actual.validation, strict=True)


@st.composite
def expanding_folds_problems(draw):
    n = draw(st.integers(5, 60))
    initial_train_size = draw(st.integers(1, n - 1))
    validation_size = draw(st.integers(1, max(1, n - initial_train_size)))
    gap = draw(st.integers(0, 5))
    stride = draw(st.one_of(st.none(), st.integers(1, 10)))
    return ExpandingFoldsProblem(n, initial_train_size, validation_size, stride, gap)


@given(problem=expanding_folds_problems())
def test_expanding_folds_compatibility(problem: ExpandingFoldsProblem) -> None:
    n, initial_train_size, validation_size, stride, gap = problem
    actual = expanding_folds(n, initial_train_size=initial_train_size, validation_size=validation_size, stride=stride, gap=gap)

    # Fold count and validation-window spacing follow directly from the documented
    # contract (first validation window starts at initial_train_size + gap, then
    # slides by stride), independent of how the source counts them with its loop.
    resolved_stride = validation_size if stride is None else stride
    start = initial_train_size + gap
    expected_count = 0 if n < start + validation_size else (n - start - validation_size) // resolved_stride + 1
    assert len(actual) == expected_count

    previous_train: np.ndarray | None = None
    previous_validation_start = -1
    for fold in actual:
        check_fold(fold, n)
        assert fold.validation.size == validation_size
        assert int(fold.validation.min()) >= previous_validation_start
        if previous_validation_start >= 0:
            assert int(fold.validation.min()) - previous_validation_start == resolved_stride
        if previous_train is not None:
            assert set(previous_train.tolist()) <= set(fold.train.tolist())
        previous_train = fold.train
        previous_validation_start = int(fold.validation.min())

    repeated = expanding_folds(n, initial_train_size=initial_train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(repeated) == len(actual)
    for fold, repeated_fold in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(repeated_fold.train, fold.train, strict=True)
        np.testing.assert_array_equal(repeated_fold.validation, fold.validation, strict=True)


@st.composite
def sliding_folds_problems(draw):
    n = draw(st.integers(5, 60))
    train_size = draw(st.integers(1, n - 1))
    validation_size = draw(st.integers(1, max(1, n - train_size)))
    gap = draw(st.integers(0, 5))
    stride = draw(st.one_of(st.none(), st.integers(1, 10)))
    return SlidingFoldsProblem(n, train_size, validation_size, stride, gap)


@given(problem=sliding_folds_problems())
def test_sliding_folds_compatibility(problem: SlidingFoldsProblem) -> None:
    n, train_size, validation_size, stride, gap = problem
    actual = sliding_folds(n, train_size=train_size, validation_size=validation_size, stride=stride, gap=gap)

    resolved_stride = validation_size if stride is None else stride
    start = train_size + gap
    expected_count = 0 if n < start + validation_size else (n - start - validation_size) // resolved_stride + 1
    assert len(actual) == expected_count

    previous_validation_start = -1
    for fold in actual:
        check_fold(fold, n)
        assert fold.train.size == train_size
        assert fold.validation.size == validation_size
        assert int(fold.validation.min()) >= previous_validation_start
        if previous_validation_start >= 0:
            assert int(fold.validation.min()) - previous_validation_start == resolved_stride
        previous_validation_start = int(fold.validation.min())

    repeated = sliding_folds(n, train_size=train_size, validation_size=validation_size, stride=stride, gap=gap)
    assert len(repeated) == len(actual)
    for fold, repeated_fold in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(repeated_fold.train, fold.train, strict=True)
        np.testing.assert_array_equal(repeated_fold.validation, fold.validation, strict=True)


TEMPORAL_FOLD_VALID = {
    "gap": TemporalFoldCase(12, 0.5, 2),
    "fraction-rounding": TemporalFoldCase(7, 0.5, 0),
}

TEMPORAL_FOLD_EXPECTED: dict[str, Fold] = {
    "gap": Fold(np.arange(6), np.arange(8, 12)),
    "fraction-rounding": Fold(np.arange(3), np.arange(3, 7)),
}

EXPANDING_FOLDS_VALID = {
    "gap-and-overlap": ExpandingFoldsCase(20, 6, 4, 3, 1),
    "default-stride": ExpandingFoldsCase(18, 4, 3, None, 0),
    "no-complete-validation": ExpandingFoldsCase(5, 4, 3, None, 0),
}

EXPANDING_FOLDS_EXPECTED: dict[str, list[Fold]] = {
    "gap-and-overlap": [
        Fold(np.arange(6), np.arange(7, 11)),
        Fold(np.arange(9), np.arange(10, 14)),
        Fold(np.arange(12), np.arange(13, 17)),
        Fold(np.arange(15), np.arange(16, 20)),
    ],
    "default-stride": [
        Fold(np.arange(4), np.arange(4, 7)),
        Fold(np.arange(7), np.arange(7, 10)),
        Fold(np.arange(10), np.arange(10, 13)),
        Fold(np.arange(13), np.arange(13, 16)),
    ],
    "no-complete-validation": [],
}

SLIDING_FOLDS_VALID = {
    "gap-and-overlap": SlidingFoldsCase(20, 6, 4, 3, 1),
    "default-stride": SlidingFoldsCase(18, 4, 3, None, 0),
    "no-complete-validation": SlidingFoldsCase(5, 4, 3, None, 0),
}

SLIDING_FOLDS_EXPECTED: dict[str, list[Fold]] = {
    "gap-and-overlap": [
        Fold(np.arange(6), np.arange(7, 11)),
        Fold(np.arange(3, 9), np.arange(10, 14)),
        Fold(np.arange(6, 12), np.arange(13, 17)),
        Fold(np.arange(9, 15), np.arange(16, 20)),
    ],
    "default-stride": [
        Fold(np.arange(4), np.arange(4, 7)),
        Fold(np.arange(3, 7), np.arange(7, 10)),
        Fold(np.arange(6, 10), np.arange(10, 13)),
        Fold(np.arange(9, 13), np.arange(13, 16)),
    ],
    "no-complete-validation": [],
}

TEMPORAL_FOLD_INVALID = {
    "zero-ratio": TemporalFoldCase(10, 0.0, 0),
    "full-ratio": TemporalFoldCase(10, 1.0, 0),
    "negative-gap": TemporalFoldCase(10, 0.5, -1),
    "empty-train": TemporalFoldCase(1, 0.5, 0),
    "empty-validation": TemporalFoldCase(10, 0.9, 5),
}

EXPANDING_FOLDS_INVALID = {
    "zero-length": ExpandingFoldsCase(0, 4, 2, None, 0),
    "zero-train": ExpandingFoldsCase(10, 0, 2, None, 0),
    "zero-validation": ExpandingFoldsCase(10, 4, 0, None, 0),
    "negative-gap": ExpandingFoldsCase(10, 4, 2, None, -1),
    "zero-stride": ExpandingFoldsCase(10, 4, 2, 0, 0),
    "negative-stride": ExpandingFoldsCase(10, 4, 2, -1, 0),
}

SLIDING_FOLDS_INVALID = {
    "zero-length": SlidingFoldsCase(0, 4, 2, None, 0),
    "zero-train": SlidingFoldsCase(10, 0, 2, None, 0),
    "zero-validation": SlidingFoldsCase(10, 4, 0, None, 0),
    "negative-gap": SlidingFoldsCase(10, 4, 2, None, -1),
    "zero-stride": SlidingFoldsCase(10, 4, 2, 0, 0),
    "negative-stride": SlidingFoldsCase(10, 4, 2, -1, 0),
}


@pytest.mark.parametrize("name", TEMPORAL_FOLD_VALID)
def test_temporal_fold_valid(name: str) -> None:
    check_temporal_fold(*TEMPORAL_FOLD_VALID[name], TEMPORAL_FOLD_EXPECTED[name])


@pytest.mark.parametrize("case", TEMPORAL_FOLD_INVALID.values(), ids=TEMPORAL_FOLD_INVALID.keys())
def test_temporal_fold_invalid(case: TemporalFoldCase) -> None:
    with pytest.raises(ValueError):
        temporal_fold(case.n, case.train_ratio, gap=case.gap)


@pytest.mark.parametrize("name", EXPANDING_FOLDS_VALID)
def test_expanding_folds_valid(name: str) -> None:
    check_expanding_folds(*EXPANDING_FOLDS_VALID[name], EXPANDING_FOLDS_EXPECTED[name])


@pytest.mark.parametrize("case", EXPANDING_FOLDS_INVALID.values(), ids=EXPANDING_FOLDS_INVALID.keys())
def test_expanding_folds_invalid(case: ExpandingFoldsCase) -> None:
    with pytest.raises(ValueError):
        expanding_folds(
            case.n,
            initial_train_size=case.initial_train_size,
            validation_size=case.validation_size,
            stride=case.stride,
            gap=case.gap,
        )


@pytest.mark.parametrize("name", SLIDING_FOLDS_VALID)
def test_sliding_folds_valid(name: str) -> None:
    check_sliding_folds(*SLIDING_FOLDS_VALID[name], SLIDING_FOLDS_EXPECTED[name])


@pytest.mark.parametrize("case", SLIDING_FOLDS_INVALID.values(), ids=SLIDING_FOLDS_INVALID.keys())
def test_sliding_folds_invalid(case: SlidingFoldsCase) -> None:
    with pytest.raises(ValueError):
        sliding_folds(case.n, train_size=case.train_size, validation_size=case.validation_size, stride=case.stride, gap=case.gap)
