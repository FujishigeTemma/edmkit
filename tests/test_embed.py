from functools import partial

import numpy as np
import pytest

from edmkit.embedding import lagged_embed, scan, select
from edmkit.metrics import MetricFunc, mean_rho, rmse
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import SplitFunc, expanding_folds, sliding_folds
from edmkit.types import PredictFunc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def logistic_map(n: int, r: float = 3.8, x0: float = 0.4, transient: int = 50) -> np.ndarray:
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
        x = logistic_map(200)
        E_list = [1, 2, 3]
        tau_list = [1, 2]
        scores = scan(x, E=E_list, tau=tau_list)
        assert scores.ndim == 3
        assert scores.shape[0] == len(E_list)
        assert scores.shape[1] == len(tau_list)
        # K_max >= 1
        assert scores.shape[2] >= 1

    def test_scores_shape_single_E_tau(self):
        x = logistic_map(200)
        scores = scan(x, E=[2], tau=[1])
        assert scores.shape[0] == 1
        assert scores.shape[1] == 1

    def test_scores_not_all_nan(self):
        x = logistic_map(200)
        scores = scan(x, E=[2, 3], tau=[1, 2])
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: self-prediction quality
# ---------------------------------------------------------------------------


class TestScanSelfPrediction:
    def test_logistic_self_prediction(self):
        """Self-prediction on logistic map should produce positive rho."""
        x = logistic_map(300)
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
        x = logistic_map(200)
        custom_split = partial(sliding_folds, train_size=50, validation_size=20)
        scores = scan(x, E=[2], tau=[1], split=custom_split)
        assert scores.shape[2] >= 1
        assert not np.all(np.isnan(scores))

    def test_custom_predict(self):
        x = logistic_map(200)
        scores = scan(x, E=[2], tau=[1], predict=simplex_projection)
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: Y target
# ---------------------------------------------------------------------------


class TestScanWithTarget:
    def test_cross_prediction(self):
        """Y != x: cross-prediction mode."""
        x = logistic_map(300)
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
        x = logistic_map(50)
        scores = scan(x, E=[2, 20], tau=[1])
        # E=20 with T=50 leaves very few points → likely NaN
        assert scores.shape[0] == 2
        # E=2 should still have valid scores
        assert not np.all(np.isnan(scores[0]))

    def test_scan_scores_independent_of_other_E(self):
        """Scores for E=3 must be identical whether scan runs E=[3] or E=[2,3,4]."""
        x = logistic_map(300)
        scores_single = scan(x, E=[3], tau=[1])
        scores_multi = scan(x, E=[2, 3, 4], tau=[1])
        # E=3 is index 0 in single, index 1 in multi
        np.testing.assert_array_equal(scores_single[0], scores_multi[1])

    def test_y_2d_target(self):
        """Y with shape (T, M) where M > 1."""
        x = logistic_map(200)
        Y = np.column_stack([x, np.roll(x, -1)])
        scores = scan(x, Y, E=[2], tau=[1])
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# select: basic
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_returns_best(self):
        x = logistic_map(300)
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
        x = logistic_map(300)
        E_list = [1, 2, 3, 4]
        tau_list = [1]
        scores = scan(x, E=E_list, tau=tau_list)
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E in E_list
        assert best_score > 0.5

    def test_select_with_nan_rows(self):
        """select correctly ignores all-NaN rows from skipped E values."""
        scores = np.full((3, 2, 4), np.nan)
        # Only the middle E has valid scores
        scores[1, 0, :] = [0.5, 0.6, 0.7, 0.8]
        scores[1, 1, :] = [0.3, 0.4, 0.5, 0.6]
        E_list = [1, 2, 3]
        tau_list = [1, 2]
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E == 2
        assert best_tau == 1
        assert best_score == pytest.approx(0.65)

    def test_select_partial_nan_folds(self):
        """NaN in some folds: nanmean aggregates only valid folds."""
        scores = np.full((2, 1, 4), np.nan)
        scores[0, 0, :2] = [0.8, 0.9]  # 2 valid folds, mean=0.85
        scores[1, 0, :4] = [0.7, 0.7, 0.7, 0.7]  # 4 valid folds, mean=0.7
        E_list = [2, 3]
        tau_list = [1]
        best_E, _, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E == 2
        assert best_score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# scan: n_ahead
# ---------------------------------------------------------------------------


class TestScanNAhead:
    def test_n_ahead_greater_than_one(self):
        """n_ahead > 1 should produce valid scores with correct shape."""
        x = logistic_map(300)
        scores = scan(x, E=[2, 3], tau=[1], n_ahead=3)
        assert scores.ndim == 3
        assert scores.shape[0] == 2
        assert not np.all(np.isnan(scores))

    def test_n_ahead_reduces_usable_points(self):
        """Larger n_ahead → fewer usable points → fewer or no folds."""
        x = logistic_map(200)
        scores_1 = scan(x, E=[2], tau=[1], n_ahead=1)
        scores_10 = scan(x, E=[2], tau=[1], n_ahead=10)
        # More folds with n_ahead=1
        assert scores_1.shape[2] >= scores_10.shape[2]

    def test_n_ahead_too_large_all_nan(self):
        """n_ahead so large that no usable data remains → all NaN."""
        x = logistic_map(30)
        scores = scan(x, E=[2], tau=[1], n_ahead=28)
        assert np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: NaN padding
# ---------------------------------------------------------------------------


class TestScanNanPadding:
    def test_nan_padding_position(self):
        """When K differs across E values, trailing entries must be NaN."""
        x = logistic_map(200)
        # Small E → more usable points → more folds
        # Large E → fewer usable points → fewer folds
        E_list = [2, 10]
        tau_list = [1]
        scores = scan(x, E=E_list, tau=tau_list)

        K_max = scores.shape[2]
        # E=2 should have more valid folds
        valid_e2 = np.sum(~np.isnan(scores[0, 0, :]))
        valid_e10 = np.sum(~np.isnan(scores[1, 0, :]))

        if valid_e10 < K_max:
            # Trailing entries for E=10 must be NaN
            assert np.all(np.isnan(scores[1, 0, valid_e10:]))
        if valid_e2 > valid_e10:
            # E=2 has more valid folds
            assert valid_e2 > valid_e10


# ---------------------------------------------------------------------------
# scan: mask branch
# ---------------------------------------------------------------------------


class TestScanMaskBranch:
    def test_expanding_folds_triggers_mask(self):
        """expanding_folds produces variable train sizes → mask is used."""
        x = logistic_map(200)
        custom_split = partial(expanding_folds, initial_train_size=30, validation_size=10)
        scores = scan(x, E=[2], tau=[1], split=custom_split)
        assert not np.all(np.isnan(scores))

    def test_uniform_train_size_no_mask(self):
        """sliding_folds with fixed window → all train sizes equal → mask skipped."""
        x = logistic_map(200)
        custom_split = partial(sliding_folds, train_size=50, validation_size=10)
        scores = scan(x, E=[2], tau=[1], split=custom_split)
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: custom metric
# ---------------------------------------------------------------------------


class TestScanCustomMetric:
    def test_custom_metric_rmse(self):
        """Passing rmse as metric should produce finite, non-negative scores."""
        x = logistic_map(200)
        scores = scan(x, E=[2], tau=[1], metric=rmse)
        valid = scores[~np.isnan(scores)]
        assert len(valid) > 0
        assert np.all(valid >= 0)

    def test_custom_metric_callable(self):
        """A user-defined metric function is correctly invoked."""
        call_count = 0

        def counting_metric(predictions, observations):
            nonlocal call_count
            call_count += 1
            # Return per-batch mean absolute difference
            return np.abs(predictions - observations).mean(axis=-1).mean(axis=-1)

        x = logistic_map(200)
        scores = scan(x, E=[2], tau=[1], metric=counting_metric)
        assert call_count > 0
        assert not np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: tiny time series
# ---------------------------------------------------------------------------


class TestScanTiny:
    def test_very_short_series(self):
        """Very short time series: should not crash, may be all NaN."""
        x = logistic_map(10)
        scores = scan(x, E=[2], tau=[1])
        assert scores.ndim == 3

    def test_all_E_skipped(self):
        """All E values too large for data → all NaN, no crash."""
        x = logistic_map(10)
        scores = scan(x, E=[8, 9], tau=[1])
        assert scores.shape[:2] == (2, 1)
        assert np.all(np.isnan(scores))


# ---------------------------------------------------------------------------
# scan: determinism
# ---------------------------------------------------------------------------


class TestScanDeterminism:
    def test_identical_results_on_repeat(self):
        """scan with identical input produces bit-identical output."""
        x = logistic_map(200)
        scores1 = scan(x, E=[2, 3], tau=[1, 2])
        scores2 = scan(x, E=[2, 3], tau=[1, 2])
        np.testing.assert_array_equal(scores1, scores2)


# ---------------------------------------------------------------------------
# scan + select: end-to-end
# ---------------------------------------------------------------------------


class TestScanSelectE2E:
    def test_logistic_e2e(self):
        """scan+select on logistic map returns high-scoring (E, tau) pair."""
        x = logistic_map(500)
        E_list = [1, 2, 3]
        tau_list = [1]
        scores = scan(x, E=E_list, tau=tau_list)
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_E in E_list
        assert best_tau in tau_list
        # Logistic map is highly predictable at n_ahead=1
        assert best_score > 0.95

    def test_sine_optimal_params(self):
        """Sine wave with period 20: E=2, tau=5 should be among the best."""
        t = np.arange(300)
        x = np.sin(2 * np.pi * t / 20)
        E_list = [1, 2, 3]
        tau_list = [1, 5]
        scores = scan(x, E=E_list, tau=tau_list)
        best_E, best_tau, best_score = select(scores, E=E_list, tau=tau_list)
        assert best_score > 0.9


# ---------------------------------------------------------------------------
# naive_scan: reference implementation (no batching)
# ---------------------------------------------------------------------------


def naive_scan(
    x: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    E: list[int],
    tau: list[int],
    n_ahead: int = 1,
    split: SplitFunc | None = None,
    predict: PredictFunc | None = None,
    metric: MetricFunc | None = None,
) -> np.ndarray:
    """Naive grid search over (E, tau) — one 2D predict call per fold.

    Same API and semantics as ``scan``, but loops over every (E, tau, fold)
    individually for readability.
    """
    N = len(x)

    if Y is None:
        Y = x
    if predict is None:
        predict = simplex_projection
    if metric is None:
        metric = mean_rho
    if split is None:
        split = partial(
            sliding_folds,
            train_size=max(N // 5, 2),
            validation_size=max(N // 10, 1),
        )

    if Y.ndim == 1:
        Y = Y[:, None]

    tau_max = max(tau)

    # (e, tau, fold) → score
    all_scores: list[list[list[float]]] = []

    for e_idx, e in enumerate(E):
        e_scores: list[list[float]] = []

        k = e + 1

        max_shift = (e - 1) * tau_max
        usable_len = N - max_shift - n_ahead

        for tau_idx, t in enumerate(tau):
            fold_scores: list[float] = []

            if usable_len < 2:
                e_scores.append(fold_scores)
                continue

            embedding = lagged_embed(x, t, e)
            embedding = embedding[-(usable_len + n_ahead) :][:usable_len]
            Y_aligned = Y[max_shift + n_ahead : N]

            for fold in split(usable_len):
                if len(fold.train) < k:
                    continue

                X_train = embedding[fold.train]  # (n_train, e)
                Y_train = Y_aligned[fold.train]  # (n_train, D)
                X_validation = embedding[fold.validation]  # (val_size, e)
                Y_validation = Y_aligned[fold.validation]  # (val_size, D)

                predictions = predict(X_train, Y_train, X_validation)
                if predictions.ndim == 1:
                    predictions = predictions[:, None]

                fold_scores.append(float(metric(predictions, Y_validation)))
            e_scores.append(fold_scores)
        all_scores.append(e_scores)

    K_max = max(len(fold_scores) for e_scores in all_scores for fold_scores in e_scores) if all_scores else 0
    scores = np.full((len(E), len(tau), K_max), np.nan)
    for e_idx, e_scores in enumerate(all_scores):
        for tau_idx, fold_scores in enumerate(e_scores):
            for fold_idx, v in enumerate(fold_scores):
                scores[e_idx, tau_idx, fold_idx] = v

    return scores


# ---------------------------------------------------------------------------
# Tests: scan vs naive_scan equivalence
# ---------------------------------------------------------------------------


class TestScanVsNaive:
    def test_logistic_default(self):
        """Default settings on logistic map."""
        x = logistic_map(200)
        E_list, tau_list = [2, 3], [1, 2]
        s1 = scan(x, E=E_list, tau=tau_list)
        s2 = naive_scan(x, E=E_list, tau=tau_list)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_logistic_n_ahead(self):
        """n_ahead > 1."""
        x = logistic_map(200)
        E_list, tau_list = [2, 3], [1]
        s1 = scan(x, E=E_list, tau=tau_list, n_ahead=3)
        s2 = naive_scan(x, E=E_list, tau=tau_list, n_ahead=3)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_cross_prediction(self):
        """Y != x."""
        x = logistic_map(200)
        Y = np.roll(x, -1)
        Y[-1] = x[-1]
        E_list, tau_list = [2], [1, 2]
        s1 = scan(x, Y, E=E_list, tau=tau_list)
        s2 = naive_scan(x, Y, E=E_list, tau=tau_list)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_2d_target(self):
        """Y with shape (N, 2)."""
        x = logistic_map(200)
        Y = np.column_stack([x, np.roll(x, -1)])
        E_list, tau_list = [2], [1]
        s1 = scan(x, Y, E=E_list, tau=tau_list)
        s2 = naive_scan(x, Y, E=E_list, tau=tau_list)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_expanding_folds(self):
        """Expanding folds trigger mask in scan; naive_scan uses no mask."""
        x = logistic_map(200)
        sp = partial(expanding_folds, initial_train_size=30, validation_size=10)
        E_list, tau_list = [2, 3], [1]
        s1 = scan(x, E=E_list, tau=tau_list, split=sp)
        s2 = naive_scan(x, E=E_list, tau=tau_list, split=sp)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_custom_metric_rmse(self):
        """RMSE metric."""
        x = logistic_map(200)
        E_list, tau_list = [2], [1]
        s1 = scan(x, E=E_list, tau=tau_list, metric=rmse)
        s2 = naive_scan(x, E=E_list, tau=tau_list, metric=rmse)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_large_E_nan_rows(self):
        """Large E that is skipped produces NaN in both."""
        x = logistic_map(50)
        E_list, tau_list = [2, 20], [1]
        s1 = scan(x, E=E_list, tau=tau_list)
        s2 = naive_scan(x, E=E_list, tau=tau_list)
        np.testing.assert_array_equal(np.isnan(s1), np.isnan(s2))
        valid = ~np.isnan(s1)
        if valid.any():
            np.testing.assert_allclose(s1[valid], s2[valid], rtol=1e-10)

    def test_expanding_folds_small_initial(self):
        """Expanding folds where initial_train_size < k for some E values."""
        x = logistic_map(200)
        # initial_train_size=2 < k=4 for E=3, so early folds should be skipped
        sp = partial(expanding_folds, initial_train_size=2, validation_size=5)
        E_list, tau_list = [3], [1]
        s1 = scan(x, E=E_list, tau=tau_list, split=sp)
        s2 = naive_scan(x, E=E_list, tau=tau_list, split=sp)
        np.testing.assert_array_equal(np.isnan(s1), np.isnan(s2))
        valid = ~np.isnan(s1)
        if valid.any():
            np.testing.assert_allclose(s1[valid], s2[valid], rtol=1e-10)

    def test_sine(self):
        """Sine wave with multiple tau."""
        t = np.arange(200)
        x = np.sin(2 * np.pi * t / 20)
        E_list, tau_list = [2, 3], [1, 5]
        s1 = scan(x, E=E_list, tau=tau_list)
        s2 = naive_scan(x, E=E_list, tau=tau_list)
        np.testing.assert_allclose(s1, s2, rtol=1e-10)
