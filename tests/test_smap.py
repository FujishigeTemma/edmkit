import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stn
from scipy.stats import spearmanr

from edmkit.embedding import lagged_embed
from edmkit.smap import smap


# ---------------------------------------------------------------------------
# hypothesis strategies
# ---------------------------------------------------------------------------
@st.composite
def smap_inputs(draw, min_e=1, max_e=3, min_n=8, max_n=30, min_m=1, max_m=5):
    """有効な smap 入力を生成（well-conditioned に限定）"""
    E = draw(st.integers(min_e, max_e))
    N = draw(st.integers(max(E + 3, min_n), max_n))
    M = draw(st.integers(min_m, max_m))
    X = draw(stn.arrays(np.float64, (N, E), elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)))
    Y = draw(stn.arrays(np.float64, (N,), elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)))
    Q = draw(stn.arrays(np.float64, (M, E), elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)))
    theta = draw(st.floats(0, 5))
    alpha = draw(st.floats(1e-8, 1e-2))
    return X, Y, Q, theta, alpha


# ---------------------------------------------------------------------------
# Helper: 手動 OLS 予測
# ---------------------------------------------------------------------------
def ols(X, Y, Q):
    """Pure OLS predictions (theta=0, alpha=0)"""
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    Q_aug = np.hstack([np.ones((Q.shape[0], 1)), Q])
    Y_col = Y[:, None] if Y.ndim == 1 else Y
    XTX = X_aug.T @ X_aug
    XTY = X_aug.T @ Y_col
    C = np.linalg.solve(XTX, XTY)
    return (Q_aug @ C).squeeze()


# ===========================================================================
# 3.4.1 解析解テスト
# ===========================================================================
class TestAnalyticalSolutions:
    def test_theta_zero_equals_ols_alpha_zero(self):
        """(A) theta=0, alpha=0 で OLS と厳密一致"""
        rng = np.random.default_rng(42)
        N, E = 50, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((10, E))

        expected = ols(X, Y, Q)
        actual = smap(X, Y, Q, theta=0.0, alpha=0.0)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_theta_zero_approx_ols_alpha_default(self):
        """(B) theta=0, alpha=1e-10 で OLS に近似"""
        rng = np.random.default_rng(42)
        N, E = 50, 1
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((10, E))

        ols_predictions = ols(X, Y, Q)
        smap_predictions = smap(X, Y, Q, theta=0.0, alpha=1e-10)
        np.testing.assert_allclose(smap_predictions, ols_predictions, atol=1e-9)

    def test_linear_system_recovery_alpha_zero(self):
        """(A) Y = a*X + b, theta=0, alpha=0 で完全復元"""
        rng = np.random.default_rng(42)
        N, E = 50, 1
        X = rng.uniform(0, 1, (N, E))
        a, b = 2.0, 3.0
        Y = (a * X + b).squeeze()
        Q = rng.uniform(0, 1, (10, E))
        expected = (a * Q + b).squeeze()

        actual = smap(X, Y, Q, theta=0.0, alpha=0.0)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_linear_system_recovery_alpha_default(self):
        """(B) 同上、alpha=1e-10 で近似復元"""
        rng = np.random.default_rng(42)
        N, E = 50, 1
        X = rng.uniform(0, 1, (N, E))
        a, b = 2.0, 3.0
        Y = (a * X + b).squeeze()
        Q = rng.uniform(0, 1, (10, E))
        expected = (a * Q + b).squeeze()

        actual = smap(X, Y, Q, theta=0.0, alpha=1e-10)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_constant_target(self):
        """(A) Y が定数 c → 予測値 = c"""
        rng = np.random.default_rng(0)
        N, E = 30, 2
        X = rng.standard_normal((N, E))
        c = 7.0
        Y = np.full(N, c)
        Q = rng.standard_normal((5, E))

        predictions = smap(X, Y, Q, theta=2.0, alpha=0.0)
        np.testing.assert_allclose(predictions, c, atol=1e-12, rtol=1e-12)


# ===========================================================================
# 3.4.2 数学的性質テスト
# ===========================================================================
class TestMathematicalProperties:
    def test_theta_increases_locality(self):
        """(A) theta 増加に伴い遠方点の影響が減少"""
        rng = np.random.default_rng(42)
        E = 1
        # 近傍に Y=0、遠方に Y=100 を配置
        X = np.vstack([rng.uniform(-0.1, 0.1, (15, E)), rng.uniform(5, 6, (15, E))])
        Y = np.concatenate([np.zeros(15), np.full(15, 100.0)])
        Q = np.array([[0.0]])

        pred_theta_0 = smap(X, Y, Q, theta=0.0, alpha=0.0)
        pred_theta_4 = smap(X, Y, Q, theta=4.0, alpha=0.0)

        # theta=4 では近傍（Y≈0）に偏るので予測値は theta=0 より小さい
        assert pred_theta_4 < pred_theta_0

    def test_regularization_effect(self):
        """(A) alpha 増加に伴い非切片係数のノルムが縮小"""
        rng = np.random.default_rng(42)
        N, E = 30, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((5, E))

        # alpha 増加で予測値のばらつき（係数のノルム効果）が縮小
        pred_small_alpha = smap(X, Y, Q, theta=0.0, alpha=1e-10)
        pred_large_alpha = smap(X, Y, Q, theta=0.0, alpha=1.0)

        # 大きい alpha → 係数が縮小 → 予測値が平均に近づく → 分散が小さい
        assert np.var(pred_large_alpha) < np.var(pred_small_alpha)

    def test_intercept_not_regularized(self):
        """(A) 切片項は正則化されない"""
        rng = np.random.default_rng(42)
        N, E = 50, 1
        X = rng.uniform(0, 1, (N, E))
        c = 10.0
        Y = np.full(N, c)  # 定数: 切片 = c, 傾き = 0
        Q = np.array([[0.5]])

        # 非常に大きい alpha でも定数ターゲットなら予測は正確
        predictions = smap(X, Y, Q, theta=0.0, alpha=100.0)
        np.testing.assert_allclose(predictions, c, atol=1e-14, rtol=1e-14)

    def test_permutation_invariance_of_library(self):
        """(A) ライブラリ順序に依存しない"""
        rng = np.random.default_rng(42)
        N, E = 20, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((5, E))

        pred_orig = smap(X, Y, Q, theta=2.0, alpha=1e-10)
        perm = rng.permutation(N)
        pred_perm = smap(X[perm], Y[perm], Q, theta=2.0, alpha=1e-10)
        np.testing.assert_allclose(pred_orig, pred_perm, atol=1e-14)

    def test_coefficients_state_dependent(self, logistic_map):
        """(A) theta > 0 で回帰係数がクエリ点ごとに異なる"""
        x = logistic_map
        E = 2
        embedded = lagged_embed(x, tau=1, e=E)
        shift = E - 1
        X = embedded[:300]
        Y = x[shift + 1 : shift + 1 + 300]

        theta = 4.0
        # 2つの異なるクエリ点で数値微分による局所傾き推定
        q1, q2 = X[10:11], X[100:101]
        delta = 1e-5
        q1p = q1.copy()
        q1p[0, 0] += delta
        q2p = q2.copy()
        q2p[0, 0] += delta

        slope1 = (smap(X, Y, q1p, theta=theta, alpha=1e-10) - smap(X, Y, q1, theta=theta, alpha=1e-10)) / delta
        slope2 = (smap(X, Y, q2p, theta=theta, alpha=1e-10) - smap(X, Y, q2, theta=theta, alpha=1e-10)) / delta

        assert abs(slope1 - slope2) > 0.01, f"Slopes should differ: {slope1:.4f} vs {slope2:.4f}"

    def test_weight_formula_hand_computed(self):
        """(A) Hand-computed WLS for a small N=3, E=1 example with known theta"""
        # Small example where we can verify the weighted least squares by hand
        X = np.array([[1.0], [2.0], [4.0]])
        Y = np.array([1.0, 3.0, 2.0])
        Q = np.array([[3.0]])
        theta = 2.0

        # Hand-compute: distances from query [3] to X points
        # d = |3-1|=2, |3-2|=1, |3-4|=1  => d_mean = (2+1+1)/3 = 4/3
        d_mean = 4.0 / 3.0
        w = np.exp(-theta * np.array([2.0, 1.0, 1.0]) / d_mean)
        # w = [exp(-3), exp(-1.5), exp(-1.5)]

        # Augmented X: [[1,1],[1,2],[1,4]], query_aug: [[1,3]]
        X_aug = np.array([[1, 1], [1, 2], [1, 4]], dtype=float)
        q_aug = np.array([[1, 3]], dtype=float)
        W = np.diag(w)
        C = np.linalg.solve(X_aug.T @ W @ X_aug, X_aug.T @ W @ Y)
        expected = q_aug @ C

        actual = smap(X, Y, Q, theta=theta, alpha=0.0)
        np.testing.assert_allclose(actual, expected.squeeze(), atol=1e-12)


# ===========================================================================
# 3.4.3 Property-Based Tests (hypothesis)
# ===========================================================================
class TestSmapProperties:
    @settings(deadline=None)
    @given(inputs=smap_inputs())  # ty: ignore[missing-argument]
    def test_constant_target_property(self, inputs):
        """(A) 任意の有効入力で、定数 Y に対して予測値 = その定数"""
        X, _, Q, theta, alpha = inputs
        c = 7.0
        Y_const = np.full(X.shape[0], c)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        assume(np.linalg.cond(X_aug) < 1e12)
        predictions = smap(X, Y_const, Q, theta=theta, alpha=alpha)
        np.testing.assert_allclose(predictions, c, atol=1e-6)

    @settings(deadline=None)
    @given(inputs=smap_inputs(), seed=st.integers(0, 2**32 - 1))  # ty: ignore[missing-argument]
    def test_permutation_invariance_property(self, inputs, seed):
        """(A) ライブラリ行の並び替えで結果不変"""
        X, Y, Q, theta, alpha = inputs
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        assume(np.linalg.cond(X_aug) < 1e8)
        pred_orig = smap(X, Y, Q, theta=theta, alpha=alpha)
        perm = np.random.default_rng(seed).permutation(X.shape[0])
        pred_perm = smap(X[perm], Y[perm], Q, theta=theta, alpha=alpha)
        np.testing.assert_allclose(pred_orig, pred_perm, atol=1e-12)


# ===========================================================================
# 3.4.4 収束性・傾向テスト
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
            Q = embedded[lib_size:-1]
            actual = x_noisy[shift + 1 + lib_size :]
            predictions = smap(X, Y, Q, theta=2.0, alpha=1e-10)
            correlations.append(np.corrcoef(predictions, actual)[0, 1])

        rho, _ = spearmanr(noise_levels, correlations)
        assert rho < 0

    def test_forecast_horizon_decay(self, lorenz_series):
        """(C) 予測ホライズン増加で精度低下（Lorenz x-component）"""
        # サブサンプリングで粗くし、予測困難度を上げる
        x = lorenz_series[::5, 0]
        E, tau = 3, 1

        correlations = {}
        for Tp in [1, 2, 5, 10]:
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
            Q = embedded[lib_size:pred_end]
            actual = x[shift + lib_size + Tp : shift + pred_end + Tp]
            predictions = smap(X, Y, Q, theta=2.0, alpha=1e-10)
            correlations[Tp] = np.corrcoef(predictions, actual)[0, 1]

        Tp_values = sorted(correlations.keys())
        corr_values = [correlations[tp] for tp in Tp_values]
        rho, _ = spearmanr(Tp_values, corr_values)
        assert rho < 0, f"Expected negative rank correlation, got rho={rho:.4f}"


# ===========================================================================
# 3.4.5 theta に関する比較テスト
# ===========================================================================
class TestThetaComparison:
    def test_theta_nonzero_better_on_nonlinear(self, logistic_map):
        """(C) 非線形系では theta>0 が theta=0 より高精度"""
        x = logistic_map
        E, tau = 2, 1
        embedded = lagged_embed(x, tau=tau, e=E)
        shift = (E - 1) * tau
        lib_size = 300
        X = embedded[:lib_size]
        Y = x[shift + 1 : shift + 1 + lib_size]
        query = embedded[lib_size:-1]
        actual = x[shift + 1 + lib_size :]

        rmse_0 = np.sqrt(np.mean((smap(X, Y, query, theta=0.0, alpha=1e-10) - actual) ** 2))
        best_rmse = rmse_0
        for theta in [2.0, 4.0]:
            rmse_t = np.sqrt(np.mean((smap(X, Y, query, theta=theta, alpha=1e-10) - actual) ** 2))
            best_rmse = min(best_rmse, rmse_t)

        improvement = (rmse_0 - best_rmse) / rmse_0 if rmse_0 > 0 else 0
        assert improvement > 0, f"RMSE improvement {improvement:.4f} <= 0"


# ===========================================================================
# 3.4.6 自己無撞着性テスト
# ===========================================================================
class TestSmapSelfConsistency:
    def test_batch_vs_individual(self):
        """(A) バッチ処理と個別処理の一致"""
        rng = np.random.default_rng(42)
        B, N, M, E = 3, 30, 5, 2
        X_2d = rng.standard_normal((N, E))
        Y_2d = rng.standard_normal(N)
        Q_2d = rng.standard_normal((M, E))

        indiv = smap(X_2d, Y_2d, Q_2d, theta=2.0, alpha=1e-10)

        X_3d = np.tile(X_2d[None], (B, 1, 1))
        Y_3d = np.tile(Y_2d[:, None][None], (B, 1, 1))
        Q_3d = np.tile(Q_2d[None], (B, 1, 1))
        batch = smap(X_3d, Y_3d, Q_3d, theta=2.0, alpha=1e-10).squeeze(-1)

        for b in range(B):
            np.testing.assert_allclose(batch[b], indiv, atol=1e-12)

    # TODO: smap の tensor 実装が完了したら NotImplementedError テストを
    #       numpy vs tensor 一致テストに置き換える
    @pytest.mark.gpu
    def test_tensor_not_implemented(self):
        """use_tensor=True は未実装で NotImplementedError を送出"""
        rng = np.random.default_rng(42)
        N, E = 20, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((3, E))

        with pytest.raises(NotImplementedError):
            smap(X, Y, Q, theta=2.0, use_tensor=True)


# ===========================================================================
# 3.4.7 退化ケース
# ===========================================================================
class TestDegenerateCases:
    def test_single_query_point(self):
        """クエリ点1つ: scalar output within reasonable range"""
        rng = np.random.default_rng(0)
        N, E = 20, 3
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((1, E))

        predictions = smap(X, Y, Q, theta=2.0, alpha=1e-10)
        assert np.isfinite(predictions).all()
        assert predictions.ndim == 0 or predictions.shape == (1,), f"Expected scalar or (1,), got shape {predictions.shape}"
        # Prediction should be in a plausible range (within library Y range, with margin)
        y_range = Y.max() - Y.min()
        assert Y.min() - y_range <= float(predictions) <= Y.max() + y_range

    def test_minimum_library_size(self):
        """ライブラリサイズ = E + 2（回帰の切片含む最小点数）"""
        E = 3
        N = E + 2  # = 5, 切片含む E+1 パラメータに対して最小限の自由度
        rng = np.random.default_rng(0)
        X = rng.standard_normal((N, E))
        # Use a linear relationship so we can verify correctness
        coeffs = np.array([1.0, -0.5, 2.0])
        intercept = 3.0
        Y = X @ coeffs + intercept
        Q = rng.standard_normal((2, E))
        expected = Q @ coeffs + intercept

        predictions = smap(X, Y, Q, theta=0.0, alpha=0.0)
        assert np.isfinite(predictions).all()
        assert predictions.shape == (2,)
        # With a perfect linear system and theta=0, predictions should match
        np.testing.assert_allclose(predictions, expected, atol=1e-10)


# ===========================================================================
# 3.4.8 数値安定性テスト
# ===========================================================================
class TestNumericalStability:
    def test_ill_conditioned_library(self):
        """ほぼ共線的なライブラリ点でも正則化により安定"""
        N, E = 20, 2
        rng = np.random.default_rng(42)
        # ほぼ共線的: X[:,1] ≈ X[:,0]
        x0 = rng.standard_normal(N)
        X = np.column_stack([x0, x0 + rng.normal(0, 1e-8, N)])
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((3, E))

        predictions = smap(X, Y, Q, theta=2.0, alpha=1e-10)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.parametrize("theta", [10.0, 20.0, 50.0])
    def test_large_theta_stability(self, theta):
        """theta が大きくても NaN/Inf が発生しない"""
        rng = np.random.default_rng(42)
        N, E = 30, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = rng.standard_normal((5, E))

        predictions = smap(X, Y, Q, theta=theta, alpha=1e-10)
        assert np.all(np.isfinite(predictions))
