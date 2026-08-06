from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from edmkit.splits import Fold, expanding_folds, sliding_folds, temporal_fold


type ExpectedFolds = tuple[tuple[range, range], ...]


class TemporalFoldCase(NamedTuple):
    length: int
    train_ratio: float
    gap: int
    expected: ExpectedFolds


class ExpandingFoldsCase(NamedTuple):
    length: int
    initial_train_size: int
    validation_size: int
    stride: int | None
    gap: int
    expected: ExpectedFolds


class SlidingFoldsCase(NamedTuple):
    length: int
    train_size: int
    validation_size: int
    stride: int | None
    gap: int
    expected: ExpectedFolds


TEMPORAL_FOLD_VALID = {
    "gap": TemporalFoldCase(
        12,
        0.5,
        2,
        ((range(6), range(8, 12)),),
    ),
    "fraction-rounding": TemporalFoldCase(
        7,
        0.5,
        0,
        ((range(3), range(3, 7)),),
    ),
}

EXPANDING_FOLDS_VALID = {
    "gap-and-overlap": ExpandingFoldsCase(
        20,
        6,
        4,
        3,
        1,
        (
            (range(6), range(7, 11)),
            (range(9), range(10, 14)),
            (range(12), range(13, 17)),
            (range(15), range(16, 20)),
        ),
    ),
    "default-stride": ExpandingFoldsCase(
        18,
        4,
        3,
        None,
        0,
        (
            (range(4), range(4, 7)),
            (range(7), range(7, 10)),
            (range(10), range(10, 13)),
            (range(13), range(13, 16)),
        ),
    ),
    "no-complete-validation": ExpandingFoldsCase(
        5,
        4,
        3,
        None,
        0,
        (),
    ),
}

SLIDING_FOLDS_VALID = {
    "gap-and-overlap": SlidingFoldsCase(
        20,
        6,
        4,
        3,
        1,
        (
            (range(6), range(7, 11)),
            (range(3, 9), range(10, 14)),
            (range(6, 12), range(13, 17)),
            (range(9, 15), range(16, 20)),
        ),
    ),
    "default-stride": SlidingFoldsCase(
        18,
        4,
        3,
        None,
        0,
        (
            (range(4), range(4, 7)),
            (range(3, 7), range(7, 10)),
            (range(6, 10), range(10, 13)),
            (range(9, 13), range(13, 16)),
        ),
    ),
    "no-complete-validation": SlidingFoldsCase(
        5,
        4,
        3,
        None,
        0,
        (),
    ),
}

TEMPORAL_FOLD_INVALID = {
    "zero-ratio": TemporalFoldCase(10, 0.0, 0, ()),
    "full-ratio": TemporalFoldCase(10, 1.0, 0, ()),
    "negative-gap": TemporalFoldCase(10, 0.5, -1, ()),
    "empty-train": TemporalFoldCase(1, 0.5, 0, ()),
    "empty-validation": TemporalFoldCase(10, 0.9, 5, ()),
}

EXPANDING_FOLDS_INVALID = {
    "zero-length": ExpandingFoldsCase(0, 4, 2, None, 0, ()),
    "zero-train": ExpandingFoldsCase(10, 0, 2, None, 0, ()),
    "zero-validation": ExpandingFoldsCase(10, 4, 0, None, 0, ()),
    "negative-gap": ExpandingFoldsCase(10, 4, 2, None, -1, ()),
    "zero-stride": ExpandingFoldsCase(10, 4, 2, 0, 0, ()),
    "negative-stride": ExpandingFoldsCase(10, 4, 2, -1, 0, ()),
}

SLIDING_FOLDS_INVALID = {
    "zero-length": SlidingFoldsCase(0, 4, 2, None, 0, ()),
    "zero-train": SlidingFoldsCase(10, 0, 2, None, 0, ()),
    "zero-validation": SlidingFoldsCase(10, 4, 0, None, 0, ()),
    "negative-gap": SlidingFoldsCase(10, 4, 2, None, -1, ()),
    "zero-stride": SlidingFoldsCase(10, 4, 2, 0, 0, ()),
    "negative-stride": SlidingFoldsCase(10, 4, 2, -1, 0, ()),
}


def check_folds(folds: list[Fold], expected: ExpectedFolds) -> None:
    assert len(folds) == len(expected)
    for fold, (expected_train, expected_validation) in zip(folds, expected, strict=True):
        np.testing.assert_array_equal(fold.train, expected_train, strict=True)
        np.testing.assert_array_equal(fold.validation, expected_validation, strict=True)


@pytest.mark.parametrize("case", TEMPORAL_FOLD_VALID.values(), ids=TEMPORAL_FOLD_VALID.keys())
def test_temporal_fold_valid(case: TemporalFoldCase) -> None:
    fold = temporal_fold(case.length, train_ratio=case.train_ratio, gap=case.gap)
    check_folds([fold], case.expected)


@pytest.mark.parametrize("case", TEMPORAL_FOLD_INVALID.values(), ids=TEMPORAL_FOLD_INVALID.keys())
def test_temporal_fold_invalid(case: TemporalFoldCase) -> None:
    with pytest.raises(ValueError):
        temporal_fold(case.length, train_ratio=case.train_ratio, gap=case.gap)


@pytest.mark.parametrize("case", EXPANDING_FOLDS_VALID.values(), ids=EXPANDING_FOLDS_VALID.keys())
def test_expanding_folds_valid(case: ExpandingFoldsCase) -> None:
    folds = expanding_folds(
        case.length,
        initial_train_size=case.initial_train_size,
        validation_size=case.validation_size,
        stride=case.stride,
        gap=case.gap,
    )
    check_folds(folds, case.expected)


@pytest.mark.parametrize("case", EXPANDING_FOLDS_INVALID.values(), ids=EXPANDING_FOLDS_INVALID.keys())
def test_expanding_folds_invalid(case: ExpandingFoldsCase) -> None:
    with pytest.raises(ValueError):
        expanding_folds(
            case.length,
            initial_train_size=case.initial_train_size,
            validation_size=case.validation_size,
            stride=case.stride,
            gap=case.gap,
        )


@pytest.mark.parametrize("case", SLIDING_FOLDS_VALID.values(), ids=SLIDING_FOLDS_VALID.keys())
def test_sliding_folds_valid(case: SlidingFoldsCase) -> None:
    folds = sliding_folds(
        case.length,
        train_size=case.train_size,
        validation_size=case.validation_size,
        stride=case.stride,
        gap=case.gap,
    )
    check_folds(folds, case.expected)


@pytest.mark.parametrize("case", SLIDING_FOLDS_INVALID.values(), ids=SLIDING_FOLDS_INVALID.keys())
def test_sliding_folds_invalid(case: SlidingFoldsCase) -> None:
    with pytest.raises(ValueError):
        sliding_folds(case.length, train_size=case.train_size, validation_size=case.validation_size, stride=case.stride, gap=case.gap)
