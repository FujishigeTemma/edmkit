from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest

from edmkit.splits import Fold, expanding_folds, sliding_folds, temporal_fold


type ExpectedFolds = tuple[tuple[range, range], ...]


class SplitCase(NamedTuple):
    run: Callable[[], Fold | list[Fold]]
    expected: ExpectedFolds


VALID: dict[str, SplitCase] = {
    "temporal-gap": SplitCase(
        partial(temporal_fold, 12, train_ratio=0.5, gap=2),
        ((range(6), range(8, 12)),),
    ),
    "temporal-fraction-rounding": SplitCase(
        partial(temporal_fold, 7, train_ratio=0.5),
        ((range(3), range(3, 7)),),
    ),
    "expanding-gap-and-overlap": SplitCase(
        partial(expanding_folds, 20, initial_train_size=6, validation_size=4, stride=3, gap=1),
        (
            (range(6), range(7, 11)),
            (range(9), range(10, 14)),
            (range(12), range(13, 17)),
            (range(15), range(16, 20)),
        ),
    ),
    "expanding-default-stride": SplitCase(
        partial(expanding_folds, 18, initial_train_size=4, validation_size=3),
        (
            (range(4), range(4, 7)),
            (range(7), range(7, 10)),
            (range(10), range(10, 13)),
            (range(13), range(13, 16)),
        ),
    ),
    "expanding-no-complete-validation": SplitCase(
        partial(expanding_folds, 5, initial_train_size=4, validation_size=3),
        (),
    ),
    "sliding-gap-and-overlap": SplitCase(
        partial(sliding_folds, 20, train_size=6, validation_size=4, stride=3, gap=1),
        (
            (range(6), range(7, 11)),
            (range(3, 9), range(10, 14)),
            (range(6, 12), range(13, 17)),
            (range(9, 15), range(16, 20)),
        ),
    ),
    "sliding-default-stride": SplitCase(
        partial(sliding_folds, 18, train_size=4, validation_size=3),
        (
            (range(4), range(4, 7)),
            (range(3, 7), range(7, 10)),
            (range(6, 10), range(10, 13)),
            (range(9, 13), range(13, 16)),
        ),
    ),
    "sliding-no-complete-validation": SplitCase(
        partial(sliding_folds, 5, train_size=4, validation_size=3),
        (),
    ),
}


@pytest.mark.parametrize("case", VALID.values(), ids=VALID.keys())
def test_valid(case: SplitCase) -> None:
    result = case.run()
    folds = [result] if isinstance(result, Fold) else result

    assert len(folds) == len(case.expected)
    for fold, (expected_train, expected_validation) in zip(folds, case.expected, strict=True):
        np.testing.assert_array_equal(fold.train, expected_train, strict=True)
        np.testing.assert_array_equal(fold.validation, expected_validation, strict=True)


INVALID: dict[str, Callable[[], object]] = {
    "temporal-zero-ratio": partial(temporal_fold, 10, train_ratio=0.0),
    "temporal-full-ratio": partial(temporal_fold, 10, train_ratio=1.0),
    "temporal-negative-gap": partial(temporal_fold, 10, train_ratio=0.5, gap=-1),
    "temporal-empty-train": partial(temporal_fold, 1, train_ratio=0.5),
    "temporal-empty-validation": partial(temporal_fold, 10, train_ratio=0.9, gap=5),
    "expanding-zero-length": partial(expanding_folds, 0, initial_train_size=4, validation_size=2),
    "expanding-zero-train": partial(expanding_folds, 10, initial_train_size=0, validation_size=2),
    "expanding-zero-validation": partial(expanding_folds, 10, initial_train_size=4, validation_size=0),
    "expanding-negative-gap": partial(expanding_folds, 10, initial_train_size=4, validation_size=2, gap=-1),
    "expanding-zero-stride": partial(expanding_folds, 10, initial_train_size=4, validation_size=2, stride=0),
    "expanding-negative-stride": partial(expanding_folds, 10, initial_train_size=4, validation_size=2, stride=-1),
    "sliding-zero-length": partial(sliding_folds, 0, train_size=4, validation_size=2),
    "sliding-zero-train": partial(sliding_folds, 10, train_size=0, validation_size=2),
    "sliding-zero-validation": partial(sliding_folds, 10, train_size=4, validation_size=0),
    "sliding-negative-gap": partial(sliding_folds, 10, train_size=4, validation_size=2, gap=-1),
    "sliding-zero-stride": partial(sliding_folds, 10, train_size=4, validation_size=2, stride=0),
    "sliding-negative-stride": partial(sliding_folds, 10, train_size=4, validation_size=2, stride=-1),
}


@pytest.mark.parametrize("call", INVALID.values(), ids=INVALID.keys())
def test_invalid(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()
