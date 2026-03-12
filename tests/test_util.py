import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stn
from tinygrad import Tensor

from edmkit.util import autocorrelation, dtw, pad, pairwise_distance, pairwise_distance_np

# ---------------------------------------------------------------------------
# hypothesis strategies
# ---------------------------------------------------------------------------
finite_floats = st.floats(-100, 100, allow_nan=False, allow_infinity=False)


def point_clouds(min_n=1, max_n=20, min_d=1, max_d=10):
    return stn.arrays(
        np.float64,
        st.tuples(st.integers(min_n, max_n), st.integers(min_d, max_d)),
        elements=finite_floats,
    )


@st.composite
def paired_point_clouds(draw, min_n=1, max_n=20, min_d=1, max_d=10):
    """次元数が一致する2つの点群を生成"""
    D = draw(st.integers(min_d, max_d))
    N_a = draw(st.integers(min_n, max_n))
    N_b = draw(st.integers(min_n, max_n))
    A = draw(stn.arrays(np.float64, (N_a, D), elements=finite_floats))
    B = draw(stn.arrays(np.float64, (N_b, D), elements=finite_floats))
    return A, B


@st.composite
def cloud_with_translation(draw):
    A = draw(point_clouds(min_n=3))
    c = draw(stn.arrays(np.float64, (1, A.shape[1]), elements=finite_floats))
    return A, c


def time_series(min_len=1, max_len=20, min_d=1, max_d=5):
    return stn.arrays(
        np.float64,
        st.tuples(st.integers(min_len, max_len), st.integers(min_d, max_d)),
        elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    )


# ===========================================================================
# 3.2.1 pairwise_distance / pairwise_distance_np
# ===========================================================================
class TestPairwiseDistanceExamples:
    def test_known_2d_distances(self):
        """手計算可能な2-3点の二乗距離"""
        A = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distance_np(A, A)
        expected = np.array([[0.0, 25.0], [25.0, 0.0]])
        np.testing.assert_allclose(D, expected, atol=1e-14)

    def test_identity_distance_zero_diagonal(self):
        """A=B のとき対角要素が 0"""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 3))
        D = pairwise_distance_np(A, A)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-14)

    def test_single_point(self):
        """1点同士の距離は差の二乗和"""
        A = np.array([[1.0, 2.0, 3.0]])
        B = np.array([[4.0, 5.0, 6.0]])
        D = pairwise_distance_np(A, B)
        expected = np.sum((A - B) ** 2)
        np.testing.assert_allclose(D[0, 0], expected, atol=1e-14)

    def test_batch_consistency(self):
        """バッチ次元を手動でループした結果と一致"""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 5, 4))
        B = rng.standard_normal((3, 7, 4))
        D_batch = pairwise_distance_np(A, B)
        for b in range(3):
            D_single = pairwise_distance_np(A[b], B[b])
            np.testing.assert_allclose(D_batch[b], D_single, atol=1e-12)


class TestPairwiseDistanceProperties:
    @given(A=point_clouds())
    def test_non_negativity_property(self, A):
        """二乗距離は常に非負"""
        D = pairwise_distance_np(A, A)
        assert np.all(D >= 0)

    @given(A=point_clouds())
    def test_self_distance_diagonal_zero_property(self, A):
        """自己距離の対角は 0（A^2+A^2-2A@A^T の桁落ちで O(1e-10) まで）"""
        D = pairwise_distance_np(A, A)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)

    @given(pair=paired_point_clouds())
    def test_symmetry_property(self, pair):
        """D(A, B) = D(B, A)^T"""
        A, B = pair
        D_ab = pairwise_distance_np(A, B)
        D_ba = pairwise_distance_np(B, A)
        np.testing.assert_allclose(D_ab, D_ba.T, atol=1e-10, rtol=1e-6)

    @given(data=cloud_with_translation())
    def test_translation_invariance_property(self, data):
        """D(A+c, A+c) = D(A, A)"""
        A, c = data
        D_orig = pairwise_distance_np(A, A)
        D_shifted = pairwise_distance_np(A + c, A + c)
        np.testing.assert_allclose(D_orig, D_shifted, atol=1e-9)

    @given(A=point_clouds(min_n=3))
    def test_triangle_inequality_property(self, A):
        """sqrt(D[i,k]) <= sqrt(D[i,j]) + sqrt(D[j,k]) for all i,j,k"""
        D = pairwise_distance_np(A, A)
        dist = np.sqrt(np.maximum(D, 0))
        n = A.shape[0]
        for i in range(min(n, 5)):
            for j in range(min(n, 5)):
                for k in range(min(n, 5)):
                    assert dist[i, k] <= dist[i, j] + dist[j, k] + 1e-10

    @pytest.mark.gpu
    @given(pair=paired_point_clouds(max_n=10, max_d=5))
    @settings(deadline=None)
    def test_numpy_tensor_agreement_property(self, pair):
        """NumPy と Tensor パスが一致（float32 精度内）"""
        A, B = pair
        D_np = pairwise_distance_np(A, B)
        D_tensor = pairwise_distance(Tensor(A.astype(np.float32)), Tensor(B.astype(np.float32))).numpy()
        np.testing.assert_allclose(D_np, D_tensor, atol=1e-3, rtol=1e-3)


# ===========================================================================
# 3.2.2 dtw
# ===========================================================================
class TestDTWExamples:
    def test_identical_sequences(self):
        """同一系列の DTW 距離は 0"""
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(dtw(A, A), 0.0, atol=1e-14)

    def test_single_element(self):
        """1要素同士はユークリッド距離"""
        A = np.array([[1.0, 0.0]])
        B = np.array([[0.0, 1.0]])
        expected = np.sqrt(2.0)
        np.testing.assert_allclose(dtw(A, B), expected, atol=1e-14)

    def test_known_small_case(self):
        """2-3要素で手計算"""
        A = np.array([[0.0], [1.0], [2.0]])
        B = np.array([[0.0], [2.0]])
        # DP: dp[1,1]=0, dp[2,1]=0+1=1, dp[3,1]=1+2=3
        #     dp[1,2]=0+2=2, dp[2,2]=min(1,2,0)+1=1, dp[3,2]=min(3,1,1)+0=1
        expected = 1.0
        np.testing.assert_allclose(dtw(A, B), expected, atol=1e-14)


class TestDTWProperties:
    @given(pair=paired_point_clouds(max_n=20, max_d=5))
    def test_dtw_non_negativity_property(self, pair):
        """DTW 距離は非負"""
        A, B = pair
        assert dtw(A, B) >= -1e-14

    @given(A=time_series())
    def test_dtw_identity_property(self, A):
        """自己距離は 0"""
        np.testing.assert_allclose(dtw(A, A), 0.0, atol=1e-14)

    @given(pair=paired_point_clouds(max_n=20, max_d=5))
    def test_dtw_symmetry_property(self, pair):
        """DTW(A, B) = DTW(B, A)"""
        A, B = pair
        np.testing.assert_allclose(dtw(A, B), dtw(B, A), atol=1e-14)


# ===========================================================================
# 3.2.3 autocorrelation
# ===========================================================================
class TestAutocorrelationExamples:
    def test_lag_zero_is_one(self):
        """ラグ0の自己相関は 1.0"""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = autocorrelation(x, max_lag=5)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-14)

    def test_white_noise(self):
        """長い白色ノイズのラグ>0 は approx 0"""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10000)
        result = autocorrelation(x, max_lag=20)
        np.testing.assert_allclose(result[1:], 0.0, atol=0.03)

    def test_sine_wave(self):
        """正弦波の半周期ラグで approx -1"""
        period = 20
        t = np.arange(1000)
        x = np.sin(2 * np.pi * t / period)
        result = autocorrelation(x, max_lag=period + 1)
        np.testing.assert_allclose(result[period // 2], -1.0, atol=0.02)

    def test_constant_signal(self):
        """定数信号は NaN or Inf（分散 0 で除算）"""
        x = np.ones(50)
        with np.errstate(invalid="ignore"):
            result = autocorrelation(x, max_lag=5)
        assert np.all(~np.isfinite(result))


class TestAutocorrelationProperties:
    @given(
        x=stn.arrays(
            np.float64,
            st.integers(10, 200),
            elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        )
    )
    def test_lag_zero_is_one_property(self, x):
        """任意の非定数信号でラグ0の自己相関は 1.0"""
        assume(np.var(x) > 1e-10)
        result = autocorrelation(x, max_lag=1)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-14)


# ===========================================================================
# 3.2.4 pad
# ===========================================================================
class TestPadExamples:
    def test_same_dimensions_no_padding(self):
        """同一サイズの配列はパディング不要"""
        arrays = [np.ones((5, 3)), np.ones((5, 3)) * 2]
        result = pad(arrays)
        assert result.shape == (2, 5, 3)
        np.testing.assert_array_equal(result[0], arrays[0])
        np.testing.assert_array_equal(result[1], arrays[1])

    def test_mixed_dimensions_zero_fill(self):
        """異なるサイズの配列がゼロ埋めされる"""
        a1 = np.ones((4, 2))
        a2 = np.ones((4, 5)) * 3
        result = pad([a1, a2])
        assert result.shape == (2, 4, 5)
        np.testing.assert_array_equal(result[0, :, :2], a1)
        np.testing.assert_array_equal(result[0, :, 2:], 0)
        np.testing.assert_array_equal(result[1], a2)


class TestPadProperties:
    @given(data=st.data())
    def test_pad_shape_and_values_property(self, data):
        """任意の配列リストに対して形状とパディング値が正しい"""
        B = data.draw(st.integers(1, 5))
        L = data.draw(st.integers(1, 20))
        dims = data.draw(st.lists(st.integers(1, 10), min_size=B, max_size=B))
        arrays = [data.draw(stn.arrays(np.float64, (L, d), elements=finite_floats)) for d in dims]
        result = pad(arrays)
        max_d = max(dims)
        assert result.shape == (B, L, max_d)
        for i, arr in enumerate(arrays):
            np.testing.assert_array_equal(result[i, :, : arr.shape[1]], arr)
        for i, d in enumerate(dims):
            if d < max_d:
                assert np.all(result[i, :, d:] == 0)
