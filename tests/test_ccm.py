import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stn
from scipy.stats import spearmanr

from edmkit.ccm import bootstrap, ccm, pearson_correlation, with_simplex_projection, with_smap
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from tests.helpers import make_seeded_sampler


# ---------------------------------------------------------------------------
# hypothesis strategies
# ---------------------------------------------------------------------------
@st.composite
def nonconst_array(draw, B, L):
    """Generate a (B, L) float array where every row has std > 0."""
    elems = st.floats(-100, 100, allow_nan=False, allow_infinity=False)
    arr = draw(stn.arrays(np.float64, (B, L), elements=elems))
    # Ensure every row is non-constant by adding a small ramp where needed
    for row in range(B):
        if np.std(arr[row]) <= 1e-10:
            arr[row] += np.linspace(0, 1, L)
    return arr


@st.composite
def paired_nonconst_arrays(draw, min_len=5, max_len=50):
    """Generate paired arrays with matching shape and std > 0 for every row."""
    B = draw(st.integers(1, 5))
    L = draw(st.integers(min_len, max_len))
    X = draw(nonconst_array(B, L))
    Y = draw(nonconst_array(B, L))
    return X, Y


@st.composite
def float_arrays(draw, min_len=5, max_len=50):
    """Generate float arrays (non-constant rows guaranteed)."""
    B = draw(st.integers(1, 5))
    L = draw(st.integers(min_len, max_len))
    return draw(nonconst_array(B, L))


# ---------------------------------------------------------------------------
# shared test data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ccm_test_data():
    """簡易テスト用データ（CCM 構成要素・自己無撞着性テスト共通）"""
    rng = np.random.default_rng(42)
    N = 200
    X = rng.standard_normal(N)
    Y = np.zeros(N)
    Y[0] = rng.standard_normal()
    for i in range(1, N):
        Y[i] = 0.7 * Y[i - 1] + 0.3 * X[i - 1] + 0.1 * rng.standard_normal()
    E, tau = 2, 1
    Y_embed = lagged_embed(Y, tau=tau, e=E)
    shift = (E - 1) * tau
    X_aligned = X[shift:]
    N_embed = Y_embed.shape[0]
    return Y_embed, X_aligned, N_embed


# ===========================================================================
# 3.5.1 pearson_correlation
# ===========================================================================
class TestPearsonCorrelationExamples:
    def test_perfect_positive(self):
        """Y = aX + b (a > 0) → corr = 1.0"""
        X = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        Y = 2.0 * X + 3.0
        corr = pearson_correlation(X, Y)
        np.testing.assert_allclose(corr, 1.0, atol=1e-14)

    def test_perfect_negative(self):
        """Y = -aX + b → corr = -1.0"""
        X = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        Y = -2.0 * X + 10.0
        corr = pearson_correlation(X, Y)
        np.testing.assert_allclose(corr, -1.0, atol=1e-14)

    def test_uncorrelated(self):
        """直交する正弦波 → corr ≈ 0"""
        t = np.linspace(0, 2 * np.pi, 1000)
        X = np.sin(t)[None, :]  # (1, 1000)
        Y = np.cos(t)[None, :]
        corr = pearson_correlation(X, Y)
        np.testing.assert_allclose(corr, 0.0, atol=0.01)

    def test_self_correlation(self):
        """corr(X, X) = 1.0"""
        X = np.array([[1, 3, 5, 7, 9]], dtype=np.float64)
        corr = pearson_correlation(X, X)
        np.testing.assert_allclose(corr, 1.0, atol=1e-14)

    def test_1d_input(self):
        """1D 入力でも正しく動作する"""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = 2.0 * X + 3.0
        corr = pearson_correlation(X, Y)
        np.testing.assert_allclose(corr, 1.0, atol=1e-14)


class TestPearsonCorrelationProperties:
    @given(pair=paired_nonconst_arrays())
    def test_pearson_range_property(self, pair):
        """-1 <= corr(X, Y) <= 1"""
        X, Y = pair
        corr = pearson_correlation(X, Y)
        assert np.all(corr >= -1 - 1e-14)
        assert np.all(corr <= 1 + 1e-14)

    @given(pair=paired_nonconst_arrays())
    def test_pearson_symmetry_property(self, pair):
        """corr(X, Y) = corr(Y, X)"""
        X, Y = pair
        np.testing.assert_allclose(pearson_correlation(X, Y), pearson_correlation(Y, X), atol=1e-14)

    @given(X=float_arrays())
    def test_pearson_self_correlation_property(self, X):
        """corr(X, X) = 1"""
        corr = pearson_correlation(X, X)
        np.testing.assert_allclose(corr, 1.0, atol=1e-14)

    @given(X=float_arrays(), a=st.floats(0.1, 10), b=st.floats(-100, 100))
    def test_pearson_linear_invariance_property(self, X, a, b):
        """corr(X, a*X + b) = 1 for a > 0"""
        Y = a * X + b
        assume(np.all(np.std(Y, axis=1) > 1e-10))
        corr = pearson_correlation(X, Y)
        np.testing.assert_allclose(corr, 1.0, atol=1e-12)


# ===========================================================================
# 3.5.2 CCM 構成要素テスト
# ===========================================================================
class TestCCMComponents:
    def test_custom_sampler(self, ccm_test_data):
        """カスタムサンプラーが正しく呼ばれる"""
        Y_embed, X_aligned, _ = ccm_test_data
        N = Y_embed.shape[0]
        lib_pool = np.arange(N // 2)
        pred_pool = np.arange(N // 2, N)
        lib_sizes = np.array([20])

        call_count = [0]

        def counting_sampler(pool, size):
            call_count[0] += 1
            # Intentionally uses a fixed seed on every call, so all samples
            # are identical.  This is acceptable because the test only checks
            # the call count, not the sampled values.
            return np.random.default_rng(42).choice(pool, size=size, replace=True)

        ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=5,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=counting_sampler,
        )
        assert call_count[0] == 5  # n_samples 回呼ばれる

    def test_custom_aggregator(self, ccm_test_data):
        """カスタムアグリゲータ（中央値）が適用される"""
        Y_embed, X_aligned, _ = ccm_test_data
        N = Y_embed.shape[0]
        lib_pool = np.arange(N // 2)
        pred_pool = np.arange(N // 2, N)
        lib_sizes = np.array([30])

        sampler = make_seeded_sampler(42)

        samples = bootstrap(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=10,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=sampler,
        )

        sampler2 = make_seeded_sampler(42)
        result = ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=10,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=sampler2,
            aggregator=np.median,
        )
        np.testing.assert_allclose(result[0], np.median(samples[:, 0]), atol=1e-14)

    def test_reproducibility_with_fixed_seed(self, ccm_test_data):
        """同一シードで同一結果"""
        Y_embed, X_aligned, _ = ccm_test_data
        N = Y_embed.shape[0]
        lib_pool = np.arange(N // 2)
        pred_pool = np.arange(N // 2, N)
        lib_sizes = np.array([20, 50])

        result1 = ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=10,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )
        result2 = ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=10,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )
        np.testing.assert_array_equal(result1, result2)


# ===========================================================================
# 3.5.3 自己無撞着性テスト
# ===========================================================================
class TestCCMSelfConsistency:
    def test_identical_series_gives_high_correlation(self):
        """When X embedding equals Y (i.e. predicting itself), ccm() returns high correlation."""
        # Create a deterministic series where the embedding perfectly predicts the target
        N = 300
        # Use logistic map for rich dynamics
        x = np.zeros(N)
        x[0] = 0.4
        for i in range(1, N):
            x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

        E, tau = 2, 1
        X_embed = lagged_embed(x, tau=tau, e=E)
        shift = (E - 1) * tau
        # Target is the same series, so cross-mapping should be very accurate
        Y_target = x[shift:]

        N_embed = X_embed.shape[0]
        half = N_embed // 2
        lib_pool = np.arange(half)
        pred_pool = np.arange(half, N_embed)
        lib_sizes = np.array([50, half])

        correlations = ccm(
            X_embed,
            Y_target,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=20,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )

        # Self-prediction via simplex on a deterministic system should yield high correlation
        assert correlations[-1] > 0.7, f"Expected high correlation, got {correlations[-1]:.4f}"

    def test_with_simplex_on_causal_system(self, causal_pair):
        """with_simplex_projection() produces meaningful results on a known causal system."""
        X, Y = causal_pair
        E, tau = 2, 1

        Y_embed = lagged_embed(Y, tau=tau, e=E)
        shift = (E - 1) * tau
        X_aligned = X[shift:]

        N = Y_embed.shape[0]
        half = N // 2
        lib_pool = np.arange(half)
        pred_pool = np.arange(half, N)
        lib_sizes = np.array([50, half])

        correlations = with_simplex_projection(
            Y_embed,
            X_aligned,
            lib_sizes,
            n_samples=20,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )

        # On a causal system, correlation at the largest library size should be positive
        assert correlations[-1] > 0, f"Expected positive correlation, got {correlations[-1]:.4f}"

    def test_with_smap_on_causal_system(self, causal_pair):
        """with_smap() produces meaningful results on a known causal system."""
        X, Y = causal_pair
        E, tau = 2, 1

        Y_embed = lagged_embed(Y, tau=tau, e=E)
        shift = (E - 1) * tau
        X_aligned = X[shift:]

        N = Y_embed.shape[0]
        half = N // 2
        lib_pool = np.arange(half)
        pred_pool = np.arange(half, N)
        lib_sizes = np.array([50, half])

        correlations = with_smap(
            Y_embed,
            X_aligned,
            lib_sizes,
            theta=2.0,
            alpha=1e-10,
            n_samples=20,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )

        # On a causal system, correlation at the largest library size should be positive
        assert correlations[-1] > 0, f"Expected positive correlation, got {correlations[-1]:.4f}"


# ===========================================================================
# 3.5.4 CCM 収束性テスト
# ===========================================================================
class TestCCMConvergence:
    @pytest.mark.slow
    def test_convergence_with_known_causality(self, causal_pair):
        """(C) X→Y の因果がある系で cross-map 精度が収束"""
        X, Y = causal_pair
        E, tau = 2, 1

        # Y のアトラクタから X を cross-map (Y xmap X)
        Y_embed = lagged_embed(Y, tau=tau, e=E)
        shift = (E - 1) * tau
        X_aligned = X[shift:]

        N = Y_embed.shape[0]
        half = N // 2
        lib_pool = np.arange(half)
        pred_pool = np.arange(half, N)
        lib_sizes = np.array([10, 20, 50, 100, 200, half])

        correlations = ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=50,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )

        # Spearman rank correlation > 0 verifies convergence trend
        rho, _ = spearmanr(lib_sizes, correlations)
        assert rho > 0

    def test_no_convergence_without_causality(self, independent_pair):
        """(C) 独立な系列では収束しない"""
        X, Y = independent_pair
        E, tau = 2, 1

        Y_embed = lagged_embed(Y, tau=tau, e=E)
        shift = (E - 1) * tau
        X_aligned = X[shift:]

        N = Y_embed.shape[0]
        half = N // 2
        lib_pool = np.arange(half)
        pred_pool = np.arange(half, N)
        lib_sizes = np.array([10, 50, 100, 200, half])

        correlations = ccm(
            Y_embed,
            X_aligned,
            lib_sizes,
            predict_func=simplex_projection,
            n_samples=50,
            library_pool=lib_pool,
            prediction_pool=pred_pool,
            sampler=make_seeded_sampler(42),
        )

        assert correlations[-1] < 0.15, f"corr(max_lib)={correlations[-1]:.4f}"
        assert correlations[-1] - correlations[0] < 0.1, f"Unexpected convergence: delta={correlations[-1] - correlations[0]:.4f}"
