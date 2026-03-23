"""Tests for edmkit.embed — scan / select grid search.

Covers scores shape, select picks best, self-prediction on known signals,
2D/3D consistency, and edge cases.
"""

import numpy as np
import pytest
from functools import partial

from edmkit.embed import scan, select
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import sliding_folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic_map(n: int, r: float = 3.8, x0: float = 0.4, transient: int = 50) -> np.ndarray:
    x = np.zeros(n + transient)
    x[0] = x0
    for i in range(1, n + transient):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[transient:]


# ---------------------------------------------------------------------------
# scan: scores shape
# ---------------------------------------------------------------------------

class TestScanShape:
    def test_scores_shape_basic(self):
        x = _logistic_map(200)
        E_list = [1, 2, 3]
        tau_list = [1, 2]
        scores = scan(x, E=E_list, tau=tau_list)
        assert scores.ndim == 3
        assert scores.shape[0] == len(E_list)
        assert scores.shape[1] == len(tau_list)
        # K_max >= 1
        assert scores.shape[2] >= 1

    def test_scores_shape_single_E_tau(self):
        x = _logistic_map(200)
        scores = scan(x, E=[2], tau=[1])
        assert scores.shape[0] == 1
        assert scores.shape[1] == 1

    def test_scores_not_all_nan(self):
        x = _logistic_map(200)
        scores = scan(x, E=[2, 3], tau=[1, 2])
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: self-prediction quality
# ---------------------------------------------------------------------------

class TestScanSelfPrediction:
    def test_logistic_self_prediction(self):
        """Self-prediction on logistic map should produce positive rho."""
        x = _logistic_map(300)
        scores = scan(x, E=[1, 2, 3], tau=[1])
        # At least one (E, tau) combo should have positive mean score
        mean_scores = np.nanmean(scores, axis=2)
        assert np.nanmax(mean_scores) > 0.0

    def test_sine_self_prediction(self):
        """Sine wave: E=2, tau=5 should yield near-perfect prediction."""
        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 20)
        scores = scan(x, E=[2], tau=[5])
        mean_score = np.nanmean(scores)
        assert mean_score > 0.9


# ---------------------------------------------------------------------------
# scan: custom split/predict/metric
# ---------------------------------------------------------------------------

class TestScanCustom:
    def test_custom_split(self):
        x = _logistic_map(200)
        custom_split = partial(sliding_folds, train_size=50, validation_size=20)
        scores = scan(x, E=[2], tau=[1], split=custom_split)
        assert scores.shape[2] >= 1
        assert not np.all(np.isnan(scores))

    def test_custom_predict(self):
        x = _logistic_map(200)
        scores = scan(x, E=[2], tau=[1], predict=simplex_projection)
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: Y target
# ---------------------------------------------------------------------------

class TestScanWithTarget:
    def test_cross_prediction(self):
        """Y != x: cross-prediction mode."""
        x = _logistic_map(300)
        # Y is just shifted x
        Y = np.roll(x, -1)
        Y[-1] = x[-1]
        scores = scan(x, Y, E=[2], tau=[1])
        assert scores.ndim == 3
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: edge cases
# ---------------------------------------------------------------------------

class TestScanEdgeCases:
    def test_large_E_skipped(self):
        """E too large for available data → NaN rows but no crash."""
        x = _logistic_map(50)
        scores = scan(x, E=[2, 20], tau=[1])
        # E=20 with T=50 leaves very few points → likely NaN
        assert scores.shape[0] == 2
        # E=2 should still have valid scores
        assert not np.all(np.isnan(scores[0]))

    def test_y_2d_target(self):
        """Y with shape (T, M) where M > 1."""
        x = _logistic_map(200)
        Y = np.column_stack([x, np.roll(x, -1)])
        scores = scan(x, Y, E=[2], tau=[1])
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# select: basic
# ---------------------------------------------------------------------------

class TestSelect:
    def test_select_returns_best(self):
        x = _logistic_map(300)
        E_list = [1, 2, 3, 4]
        tau_list = [1, 2]
        scores = scan(x, E=E_list, tau=tau_list)
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E in E_list
        assert best_tau in tau_list
        assert isinstance(best_score, float)

    def test_select_consistent_with_manual(self):
        """select result matches manual nanmean + argmax."""
        E_list = [1, 2, 3]
        tau_list = [1, 2]
        rng = np.random.default_rng(42)
        scores = rng.standard_normal((3, 2, 5))
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)

        mean_scores = np.nanmean(scores, axis=2)
        flat_idx = np.argmax(mean_scores)
        e_idx, t_idx = np.unravel_index(flat_idx, mean_scores.shape)

        assert best_E == E_list[e_idx]
        assert best_tau == tau_list[t_idx]
        assert best_score == pytest.approx(mean_scores[e_idx, t_idx])

    def test_select_logistic_reasonable(self):
        """For logistic map, select should return a valid (E, tau) with positive score."""
        x = _logistic_map(300)
        E_list = [1, 2, 3, 4]
        tau_list = [1]
        scores = scan(x, E=E_list, tau=tau_list)
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E in E_list
        assert best_score > 0.5
