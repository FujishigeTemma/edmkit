"""Tests for edmkit.splits — time-series cross-validation utilities.

Covers temporal_fold, expanding_folds, sliding_folds with gap, stride,
boundary conditions, and hypothesis property-based index validity.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.splits import Fold, expanding_folds, sliding_folds, temporal_fold


# ---------------------------------------------------------------------------
# temporal_fold
# ---------------------------------------------------------------------------

class TestTemporalFold:
    def test_basic_split(self):
        fold = temporal_fold(100, 0.8)
        assert len(fold.train) == 80
        assert len(fold.validation) == 20
        assert fold.train[-1] < fold.validation[0]

    def test_train_val_cover_all(self):
        fold = temporal_fold(100, 0.8)
        assert len(fold.train) + len(fold.validation) == 100

    def test_gap(self):
        fold = temporal_fold(100, 0.8, gap=5)
        assert fold.train[-1] + 5 + 1 == fold.validation[0]
        assert len(fold.train) + 5 + len(fold.validation) == 100

    def test_no_overlap(self):
        fold = temporal_fold(100, 0.5, gap=3)
        train_set = set(fold.train)
        val_set = set(fold.validation)
        assert train_set.isdisjoint(val_set)

    def test_invalid_ratio_low(self):
        with pytest.raises(ValueError):
            temporal_fold(100, 0.0)

    def test_invalid_ratio_high(self):
        with pytest.raises(ValueError):
            temporal_fold(100, 1.0)

    def test_gap_eats_validation(self):
        with pytest.raises(ValueError, match="Validation"):
            temporal_fold(10, 0.9, gap=5)


# ---------------------------------------------------------------------------
# expanding_folds
# ---------------------------------------------------------------------------

class TestExpandingFolds:
    def test_basic(self):
        folds = expanding_folds(20, initial_train_size=10, validation_size=5)
        assert len(folds) == 2
        # First fold: train grows from initial_train_size
        assert len(folds[0].train) == 10
        assert len(folds[0].validation) == 5
        # Second fold: train is larger
        assert len(folds[1].train) > len(folds[0].train)
        assert len(folds[1].validation) == 5

    def test_train_expands(self):
        folds = expanding_folds(50, initial_train_size=10, validation_size=5)
        for i in range(1, len(folds)):
            assert len(folds[i].train) > len(folds[i - 1].train)

    def test_validation_fixed_size(self):
        folds = expanding_folds(50, initial_train_size=10, validation_size=5)
        for fold in folds:
            assert len(fold.validation) == 5

    def test_gap(self):
        folds = expanding_folds(30, initial_train_size=10, validation_size=5, gap=2)
        for fold in folds:
            assert fold.train[-1] + 2 + 1 == fold.validation[0]

    def test_stride(self):
        folds_default = expanding_folds(50, initial_train_size=10, validation_size=5)
        folds_stride = expanding_folds(
            50, initial_train_size=10, validation_size=5, stride=3,
        )
        # With smaller stride, we get more folds
        assert len(folds_stride) >= len(folds_default)

    def test_n_too_small(self):
        """When n is too small, returns empty list."""
        folds = expanding_folds(5, initial_train_size=10, validation_size=5)
        assert folds == []

    def test_no_overlap(self):
        folds = expanding_folds(50, initial_train_size=10, validation_size=5)
        for fold in folds:
            assert set(fold.train).isdisjoint(set(fold.validation))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            expanding_folds(0, initial_train_size=10, validation_size=5)
        with pytest.raises(ValueError):
            expanding_folds(50, initial_train_size=0, validation_size=5)
        with pytest.raises(ValueError):
            expanding_folds(50, initial_train_size=10, validation_size=0)
        with pytest.raises(ValueError):
            expanding_folds(50, initial_train_size=10, validation_size=5, gap=-1)


# ---------------------------------------------------------------------------
# sliding_folds
# ---------------------------------------------------------------------------

class TestSlidingFolds:
    def test_basic(self):
        folds = sliding_folds(20, train_size=10, validation_size=5)
        assert len(folds) == 2
        for fold in folds:
            assert len(fold.train) == 10
            assert len(fold.validation) == 5

    def test_train_fixed_size(self):
        folds = sliding_folds(50, train_size=10, validation_size=5)
        for fold in folds:
            assert len(fold.train) == 10

    def test_train_slides(self):
        folds = sliding_folds(50, train_size=10, validation_size=5)
        for i in range(1, len(folds)):
            assert folds[i].train[0] > folds[i - 1].train[0]

    def test_gap(self):
        folds = sliding_folds(30, train_size=10, validation_size=5, gap=2)
        for fold in folds:
            assert fold.train[-1] + 2 + 1 == fold.validation[0]

    def test_stride(self):
        folds_default = sliding_folds(50, train_size=10, validation_size=5)
        folds_stride = sliding_folds(
            50, train_size=10, validation_size=5, stride=3,
        )
        assert len(folds_stride) >= len(folds_default)

    def test_n_too_small(self):
        folds = sliding_folds(5, train_size=10, validation_size=5)
        assert folds == []

    def test_no_overlap(self):
        folds = sliding_folds(50, train_size=10, validation_size=5)
        for fold in folds:
            assert set(fold.train).isdisjoint(set(fold.validation))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            sliding_folds(0, train_size=10, validation_size=5)
        with pytest.raises(ValueError):
            sliding_folds(50, train_size=0, validation_size=5)
        with pytest.raises(ValueError):
            sliding_folds(50, train_size=10, validation_size=0)
        with pytest.raises(ValueError):
            sliding_folds(50, train_size=10, validation_size=5, gap=-1)


# ---------------------------------------------------------------------------
# Hypothesis: fold indices are valid
# ---------------------------------------------------------------------------

class TestHypothesisFoldIndices:
    @given(
        n=st.integers(min_value=10, max_value=200),
        ratio=st.floats(min_value=0.1, max_value=0.9),
        gap=st.integers(min_value=0, max_value=5),
    )
    def test_temporal_fold_valid_indices(self, n, ratio, gap):
        try:
            fold = temporal_fold(n, ratio, gap=gap)
        except ValueError:
            return  # Invalid params are fine
        assert np.all(fold.train >= 0)
        assert np.all(fold.train < n)
        assert np.all(fold.validation >= 0)
        assert np.all(fold.validation < n)
        assert set(fold.train.tolist()).isdisjoint(set(fold.validation.tolist()))

    @given(
        n=st.integers(min_value=5, max_value=200),
        initial=st.integers(min_value=2, max_value=50),
        val_size=st.integers(min_value=1, max_value=20),
        gap=st.integers(min_value=0, max_value=5),
    )
    def test_expanding_folds_valid_indices(self, n, initial, val_size, gap):
        folds = expanding_folds(n, initial_train_size=initial, validation_size=val_size, gap=gap)
        for fold in folds:
            assert np.all(fold.train >= 0)
            assert np.all(fold.train < n)
            assert np.all(fold.validation >= 0)
            assert np.all(fold.validation < n)
            assert set(fold.train.tolist()).isdisjoint(set(fold.validation.tolist()))

    @given(
        n=st.integers(min_value=5, max_value=200),
        train=st.integers(min_value=2, max_value=50),
        val_size=st.integers(min_value=1, max_value=20),
        gap=st.integers(min_value=0, max_value=5),
    )
    def test_sliding_folds_valid_indices(self, n, train, val_size, gap):
        folds = sliding_folds(n, train_size=train, validation_size=val_size, gap=gap)
        for fold in folds:
            assert np.all(fold.train >= 0)
            assert np.all(fold.train < n)
            assert np.all(fold.validation >= 0)
            assert np.all(fold.validation < n)
            assert set(fold.train.tolist()).isdisjoint(set(fold.validation.tolist()))
