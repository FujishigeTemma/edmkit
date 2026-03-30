import numpy as np
import pytest

from edmkit.splits import expanding_folds, sliding_folds, temporal_fold


def assert_valid_folds(folds: list) -> None:
    for fold in folds:
        assert np.all(fold.train >= 0)
        assert np.all(fold.validation >= 0)
        assert set(fold.train).isdisjoint(set(fold.validation))


class TestTemporalFold:
    def test_temporal_fold_respects_gap_and_partition(self):
        fold = temporal_fold(12, train_ratio=0.5, gap=2)
        np.testing.assert_array_equal(fold.train, np.arange(6))
        np.testing.assert_array_equal(fold.validation, np.arange(8, 12))

    @pytest.mark.parametrize(
        ("train_ratio", "gap"),
        [
            (0.0, 0),
            (1.0, 0),
            (0.5, -1),
        ],
    )
    def test_temporal_fold_rejects_invalid_parameters(self, train_ratio: float, gap: int):
        with pytest.raises(ValueError):
            temporal_fold(10, train_ratio=train_ratio, gap=gap)

    def test_temporal_fold_rejects_empty_validation(self):
        with pytest.raises(ValueError, match="Validation"):
            temporal_fold(10, train_ratio=0.9, gap=5)


class TestExpandingFolds:
    def test_expanding_folds_grow_train_window(self):
        folds = expanding_folds(20, initial_train_size=6, validation_size=4, stride=3, gap=1)
        np.testing.assert_array_equal(folds[0].train, np.arange(6))
        np.testing.assert_array_equal(folds[0].validation, np.arange(7, 11))
        np.testing.assert_array_equal(folds[1].train, np.arange(9))
        np.testing.assert_array_equal(folds[1].validation, np.arange(10, 14))
        assert len(folds[-1].train) > len(folds[0].train)
        assert_valid_folds(folds)

    def test_expanding_folds_return_empty_when_validation_never_fits(self):
        assert expanding_folds(5, initial_train_size=4, validation_size=3) == []

    @pytest.mark.parametrize(
        ("n", "initial_train_size", "validation_size", "gap"),
        [
            (0, 4, 2, 0),
            (10, 0, 2, 0),
            (10, 4, 0, 0),
            (10, 4, 2, -1),
        ],
    )
    def test_expanding_folds_reject_invalid_parameters(self, n: int, initial_train_size: int, validation_size: int, gap: int):
        with pytest.raises(ValueError):
            expanding_folds(n, initial_train_size=initial_train_size, validation_size=validation_size, gap=gap)

    @pytest.mark.parametrize("stride", [0, -1])
    def test_expanding_folds_reject_non_positive_stride(self, stride: int):
        with pytest.raises(ValueError, match="stride"):
            expanding_folds(20, initial_train_size=6, validation_size=4, stride=stride)


class TestSlidingFolds:
    def test_sliding_folds_keep_fixed_train_size(self):
        folds = sliding_folds(20, train_size=6, validation_size=4, stride=3, gap=1)
        np.testing.assert_array_equal(folds[0].train, np.arange(0, 6))
        np.testing.assert_array_equal(folds[0].validation, np.arange(7, 11))
        np.testing.assert_array_equal(folds[1].train, np.arange(3, 9))
        np.testing.assert_array_equal(folds[1].validation, np.arange(10, 14))
        assert all(len(fold.train) == 6 for fold in folds)
        assert_valid_folds(folds)

    def test_smaller_stride_produces_more_folds(self):
        default = sliding_folds(40, train_size=10, validation_size=5)
        overlapping = sliding_folds(40, train_size=10, validation_size=5, stride=2)
        assert len(overlapping) > len(default)

    @pytest.mark.parametrize(
        ("n", "train_size", "validation_size", "gap"),
        [
            (0, 4, 2, 0),
            (10, 0, 2, 0),
            (10, 4, 0, 0),
            (10, 4, 2, -1),
        ],
    )
    def test_sliding_folds_reject_invalid_parameters(self, n: int, train_size: int, validation_size: int, gap: int):
        with pytest.raises(ValueError):
            sliding_folds(n, train_size=train_size, validation_size=validation_size, gap=gap)

    @pytest.mark.parametrize("stride", [0, -1])
    def test_sliding_folds_reject_non_positive_stride(self, stride: int):
        with pytest.raises(ValueError, match="stride"):
            sliding_folds(20, train_size=6, validation_size=4, stride=stride)
