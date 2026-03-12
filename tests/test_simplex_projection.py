import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stn
from scipy.stats import spearmanr

from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection


# ---------------------------------------------------------------------------
# hypothesis strategies
# ---------------------------------------------------------------------------
@st.composite
def simplex_inputs(draw, min_n=10, max_n=50, min_e=1, max_e=5, min_m=1, max_m=10):
    """有効な simplex_projection 入力を生成"""
    E = draw(st.integers(min_e, max_e))
    N = draw(st.integers(max(E + 2, min_n), max_n))
    M = draw(st.integers(min_m, max_m))
    X = draw(stn.arrays(np.float64, (N, E), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    Y = draw(stn.arrays(np.float64, (N,), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    query = draw(stn.arrays(np.float64, (M, E), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    return X, Y, query


# ===========================================================================
# 3.3.1 解析解テスト
# ===========================================================================
class TestAnalyticalSolutions:
    def test_identity_prediction(self):
        """(A) クエリ点がライブラリ点と一致 → 最近傍の Y 値を予測"""
        rng = np.random.default_rng(42)
        N, E = 20, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)

        # ライブラリ点をクエリとして使用
        query_idx = [3, 7, 15]
        query = X[query_idx]
        predictions = simplex_projection(X, Y, query)

        np.testing.assert_allclose(predictions, Y[query_idx], atol=1e-15)

    def test_linear_system_high_accuracy(self, bounded_linear_series):
        """(B) 有界な線形力学系で高精度な予測"""
        x = bounded_linear_series
        E, tau = 1, 1
        embedded = lagged_embed(x, tau=tau, e=E)
        shift = (E - 1) * tau

        lib_size = 400
        X = embedded[:lib_size]
        Y = x[shift + 1 : shift + 1 + lib_size]

        query = embedded[lib_size:-1]
        actual = x[shift + 1 + lib_size :]

        predictions = simplex_projection(X, Y, query)

        rho = np.corrcoef(predictions, actual)[0, 1]
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        assert rho > 0.99
        assert rmse < 0.05

    def test_constant_target(self):
        """(A) Y が定数 c → 予測値 = c"""
        rng = np.random.default_rng(0)
        N, E = 30, 3
        X = rng.standard_normal((N, E))
        c = 42.0
        Y = np.full(N, c)
        query = rng.standard_normal((5, E))

        predictions = simplex_projection(X, Y, query)
        np.testing.assert_allclose(predictions, c, atol=1e-14, rtol=1e-14)


# ===========================================================================
# 3.3.2 数学的性質テスト (example-based)
# ===========================================================================
class TestMathematicalProperties:
    def test_neighbor_count_equals_e_plus_1(self):
        """(A) k = E + 1 近傍を使用: equidistant neighbors yield unweighted mean"""
        # Place k = E+1 = 3 library points equidistant from the query point.
        # All other library points are far away so they are not selected as neighbors.
        # When all k neighbors are equidistant, weights are equal, so prediction = mean(Y_neighbors).
        E = 2
        k = E + 1  # = 3

        # Query at origin
        query = np.array([[0.0, 0.0]])

        # 3 points on a unit circle (equidistant from origin)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        X_near = np.column_stack([np.cos(angles), np.sin(angles)])
        Y_near = np.array([1.0, 4.0, 7.0])

        # Add distant points so N > k (required for valid library)
        X_far = np.full((5, E), 100.0) + np.eye(5, E)
        Y_far = np.array([999.0, 999.0, 999.0, 999.0, 999.0])

        X = np.vstack([X_near, X_far])
        Y = np.concatenate([Y_near, Y_far])

        actual = simplex_projection(X, Y, query)
        expected = np.mean(Y_near)  # = 4.0, unweighted mean of equidistant neighbors
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_prediction_improves_with_optimal_e(self, lorenz_series):
        """(C) 高次元カオス系では最適な E が E=1 より高精度"""
        # Lorenz x-component (dimension ≈ 2.06) では E>1 が必要
        # サブサンプリングで時系列を粗くし、埋め込み次元の差を明確にする
        x = lorenz_series[::5, 0]  # 5 点に 1 点（dt_eff = 0.05）
        tau = 1
        Tp = 1
        correlations = {}
        for E in range(1, 6):
            embedded = lagged_embed(x, tau=tau, e=E)
            shift = (E - 1) * tau
            N_embed = len(embedded)
            lib_size = 50
            if lib_size + Tp > N_embed:
                continue
            X = embedded[:lib_size]
            Y = x[shift + Tp : shift + Tp + lib_size]
            pred_end = N_embed - Tp
            if pred_end <= lib_size + 5:
                continue
            query = embedded[lib_size:pred_end]
            actual = x[shift + lib_size + Tp : shift + pred_end + Tp]
            preds = simplex_projection(X, Y, query)
            correlations[E] = np.corrcoef(preds, actual)[0, 1]

        best_e = max(correlations, key=lambda e: correlations[e])
        assert best_e > 1, f"Best E={best_e}, expected > 1 for Lorenz (dim ≈ 2.06)"
        assert correlations[best_e] > correlations[1], f"Best E={best_e}, corr={correlations[best_e]:.4f}, E=1 corr={correlations[1]:.4f}"


# ===========================================================================
# 3.3.3 Property-Based Tests (hypothesis)
# ===========================================================================
class TestSimplexProperties:
    @given(inputs=simplex_inputs())
    def test_prediction_in_target_range_property(self, inputs):
        """(A) 予測値は Y の値域に収まる（凸結合性）"""
        X, Y, query = inputs
        predictions = simplex_projection(X, Y, query)
        assert np.all(predictions >= Y.min() - 1e-14)
        assert np.all(predictions <= Y.max() + 1e-14)

    @given(inputs=simplex_inputs())
    def test_constant_target_property(self, inputs):
        """(A) 任意の定数 Y に対して予測値 = その定数"""
        X, _, query = inputs
        c = np.float64(42.0)
        Y_const = np.full(X.shape[0], c)
        predictions = simplex_projection(X, Y_const, query)
        np.testing.assert_allclose(predictions, c, atol=1e-14, rtol=1e-14)

    @given(inputs=simplex_inputs(), noise_seed=st.integers(0, 2**32 - 1), perm_seed=st.integers(0, 2**32 - 1))
    def test_permutation_invariance_property(self, inputs, noise_seed, perm_seed):
        """(A) ライブラリ点の順序入れ替えで結果不変"""
        X, Y, query = inputs
        # 各点に一意なノイズを加えて KDTree タイブレークを防ぐ
        rng = np.random.default_rng(noise_seed)
        X = X + rng.normal(0, 1e-8, X.shape)
        pred_original = simplex_projection(X, Y, query)
        perm = np.random.default_rng(perm_seed).permutation(X.shape[0])
        pred_permuted = simplex_projection(X[perm], Y[perm], query)
        np.testing.assert_allclose(pred_original, pred_permuted, atol=1e-12)


# ===========================================================================
# 3.3.4 収束性・傾向テスト
# ===========================================================================
class TestConvergenceTrends:
    def test_noise_sensitivity(self, logistic_map):
        """(C) ノイズ増加で精度低下"""
        x = logistic_map
        E, tau = 2, 1
        rng = np.random.default_rng(42)

        noise_levels = [0.0, 0.05, 0.15, 0.3]
        correlations = []
        for sigma in noise_levels:
            x_noisy = x + rng.normal(0, sigma, len(x))
            embedded = lagged_embed(x_noisy, tau=tau, e=E)
            shift = (E - 1) * tau
            lib_size = 300
            X = embedded[:lib_size]
            Y = x_noisy[shift + 1 : shift + 1 + lib_size]
            query = embedded[lib_size:-1]
            actual = x_noisy[shift + 1 + lib_size :]
            preds = simplex_projection(X, Y, query)
            correlations.append(np.corrcoef(preds, actual)[0, 1])

        # Spearman 順位相関 < 0 (精度が単調減少の傾向)
        rho, _ = spearmanr(noise_levels, correlations)
        assert rho < 0

    def test_simplex_forecast_decay(self, lorenz_series):
        """(C) 予測ホライズン増加で精度低下（Lorenz x-component）"""
        x = lorenz_series[:, 0]
        E, tau = 3, 1

        correlations = {}
        for Tp in [1, 2, 5, 10]:
            embedded = lagged_embed(x, tau=tau, e=E)
            shift = (E - 1) * tau
            N_embed = len(embedded)
            lib_size = 300
            if lib_size + Tp > N_embed:
                continue
            X = embedded[:lib_size]
            Y = x[shift + Tp : shift + Tp + lib_size]
            pred_end = N_embed - Tp
            if pred_end <= lib_size + 10:
                continue
            query = embedded[lib_size:pred_end]
            actual = x[shift + lib_size + Tp : shift + pred_end + Tp]
            preds = simplex_projection(X, Y, query)
            correlations[Tp] = np.corrcoef(preds, actual)[0, 1]

        Tp_values = sorted(correlations.keys())
        corr_values = [correlations[tp] for tp in Tp_values]
        rho, _ = spearmanr(Tp_values, corr_values)
        assert rho < 0, f"Expected negative Spearman correlation (forecast decay), got rho={rho:.4f}"


# ===========================================================================
# 3.3.5 自己無撞着性テスト
# ===========================================================================
class TestSelfConsistency:
    def test_batch_vs_individual(self):
        """(A) バッチ入力と個別入力の結果一致"""
        rng = np.random.default_rng(42)
        B, N, E, M = 3, 30, 2, 10
        X_2d = rng.standard_normal((N, E))
        Y_2d = rng.standard_normal(N)
        query_2d = rng.standard_normal((M, E))

        indiv = simplex_projection(X_2d, Y_2d, query_2d)

        X_3d = np.tile(X_2d[None], (B, 1, 1))
        Y_3d = np.tile(Y_2d[:, None][None], (B, 1, 1))
        query_3d = np.tile(query_2d[None], (B, 1, 1))
        batch = simplex_projection(X_3d, Y_3d, query_3d).squeeze(-1)

        for b in range(B):
            np.testing.assert_allclose(batch[b], indiv, atol=1e-14)

    @pytest.mark.gpu
    def test_numpy_vs_tensor_2d(self):
        """(A) 2D入力で use_tensor=True/False が同一結果"""
        rng = np.random.default_rng(42)
        N, E, M = 50, 2, 10
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        query = rng.standard_normal((M, E))

        pred_np = simplex_projection(X, Y, query, use_tensor=False)
        pred_tensor = simplex_projection(X, Y, query, use_tensor=True)

        np.testing.assert_allclose(pred_np, pred_tensor, atol=1e-5, rtol=1e-5)

    @pytest.mark.gpu
    def test_numpy_vs_tensor_3d_batch(self):
        """(A) 3Dバッチ入力で use_tensor=True/False が同一結果"""
        rng = np.random.default_rng(42)
        B, N, E, M = 3, 30, 2, 10
        X = rng.standard_normal((B, N, E))
        Y = rng.standard_normal((B, N, 1))
        query = rng.standard_normal((B, M, E))

        pred_np = simplex_projection(X, Y, query, use_tensor=False)
        pred_tensor = simplex_projection(X, Y, query, use_tensor=True)

        np.testing.assert_allclose(pred_np, pred_tensor, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 3.3.6 退化ケース
# ===========================================================================
class TestDegenerateCases:
    def test_single_query_point(self):
        """クエリ点1つ"""
        rng = np.random.default_rng(0)
        N, E = 20, 3
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        query = rng.standard_normal((1, E))

        pred = simplex_projection(X, Y, query)
        assert np.isfinite(pred).all()
        assert pred.shape == (1,) or pred.ndim == 0  # single prediction
        # Prediction must be within the convex hull of Y values (convex combination property)
        assert pred.item() >= Y.min() - 1e-14
        assert pred.item() <= Y.max() + 1e-14

    def test_minimum_library_size(self):
        """ライブラリサイズ = E + 1（最小近傍数）→ 全点が近傍として使われ解析解と一致"""
        # Place all N library points equidistant from the query point.
        # When N == k and all distances are equal, prediction = unweighted mean of all Y.
        # Use vertices of a regular simplex in E=3 dimensions, centered at origin.
        X = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        # Verify all points are equidistant from origin
        dists = np.linalg.norm(X, axis=1)
        assert np.allclose(dists, dists[0]), "Library points must be equidistant from query"

        Y = np.array([2.0, 4.0, 6.0, 8.0])
        query = np.array([[0.0, 0.0, 0.0]])  # origin

        pred = simplex_projection(X, Y, query)
        assert np.isfinite(pred).all()

        # All neighbors equidistant → equal weights → prediction = mean(Y)
        expected = np.mean(Y)  # = 5.0
        np.testing.assert_allclose(pred, expected, atol=1e-14)
