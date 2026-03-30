from functools import partial

import numpy as np
import pytest

from edmkit.embedding import scan, select
from edmkit.metrics import rmse
from edmkit.splits import sliding_folds


class TestScanExamples:
    def test_scan_returns_expected_grid_shape(self, logistic_map: np.ndarray):
        scores = scan(logistic_map, E=[1, 2, 3], tau=[1, 2])
        assert scores.shape[:2] == (3, 2)
        assert scores.shape[2] >= 1
        assert np.isfinite(scores[0]).all()

    def test_scan_supports_multivariate_target(self, logistic_map: np.ndarray):
        target = np.column_stack([logistic_map, np.roll(logistic_map, -1)])
        target[-1, 1] = target[-2, 1]
        scores = scan(logistic_map, target, E=[2], tau=[1])
        assert scores.shape[0] == 1
        assert np.isfinite(scores).any()

    def test_scan_uses_custom_split_and_metric(self, logistic_map: np.ndarray):
        split = partial(sliding_folds, train_size=80, validation_size=20, stride=20)
        scores = scan(logistic_map, E=[2], tau=[1], split=split, metric=rmse)
        assert scores.shape == (1, 1, 20)
        assert np.isfinite(scores).all()

    def test_scan_marks_unusable_configurations_as_nan(self):
        x = np.linspace(0.0, 1.0, 30)
        scores = scan(x, E=[2, 8], tau=[1, 2])
        assert np.isfinite(scores[0]).any()
        assert np.isnan(scores[1]).all()

    def test_scan_supports_n_ahead_greater_than_one(self, logistic_map: np.ndarray):
        scores = scan(logistic_map, E=[2, 3], tau=[1], n_ahead=3)
        assert scores.shape[:2] == (2, 1)
        assert scores.shape[2] >= 1
        assert np.isfinite(scores).any()

    def test_larger_n_ahead_does_not_create_more_folds(self, logistic_map: np.ndarray):
        one_step = scan(logistic_map, E=[2], tau=[1], n_ahead=1)
        five_step = scan(logistic_map, E=[2], tau=[1], n_ahead=5)
        assert one_step.shape[2] >= five_step.shape[2]
        assert np.isfinite(one_step).any()

    def test_scan_returns_all_nan_when_n_ahead_leaves_too_few_usable_points(self):
        x = np.linspace(0.0, 1.0, 30)
        scores = scan(x, E=[2], tau=[1], n_ahead=28)
        assert np.isnan(scores).all()


class TestSelectExamples:
    def test_select_picks_highest_risk_adjusted_score(self):
        scores = np.array(
            [
                [[0.6, 0.8, np.nan], [0.4, 0.5, 0.6]],
                [[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]],
            ]
        )
        best_e, best_tau, best_score = select(scores, E=[2, 3], tau=[1, 2])
        assert (best_e, best_tau) == (3, 1)
        assert best_score == pytest.approx(0.8)

    def test_select_penalizes_high_variance(self):
        # Two (E, tau) with identical mean but different variance.
        # E=1: stable folds  → low SE  → high adjusted
        # E=2: volatile folds → high SE → low adjusted
        scores = np.full((2, 1, 4), np.nan)
        scores[0, 0, :] = [0.5, 0.5, 0.5, 0.5]  # mean=0.5, std=0
        scores[1, 0, :] = [0.9, 0.1, 0.9, 0.1]  # mean=0.5, std≈0.46
        best_e, _, best_score = select(scores, E=[1, 2], tau=[1])
        assert best_e == 1
        assert best_score == pytest.approx(0.5)

    def test_select_raises_on_all_nan_scores(self):
        with pytest.raises(ValueError):
            select(np.full((2, 2, 3), np.nan), E=[1, 2], tau=[1, 2])
