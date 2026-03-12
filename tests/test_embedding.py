import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as stn

from edmkit.embedding import lagged_embed


# ===========================================================================
# hypothesis strategies
# ===========================================================================
@st.composite
def valid_embed_inputs(draw):
    tau = draw(st.integers(1, 10))
    e = draw(st.integers(1, 10))
    min_len = (e - 1) * tau + 1
    x = draw(
        stn.arrays(
            np.float64,
            st.integers(min_len, max(min_len, 200)),
            elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        )
    )
    return x, tau, e


# ===========================================================================
# 3.1.1 解析解テスト (example-based)
# ===========================================================================
class TestAnalyticalSolutions:
    def test_identity_embedding(self):
        """e=1, tau=1 ではリシェイプのみ"""
        x = np.arange(10, dtype=float)
        result = lagged_embed(x, tau=1, e=1)
        np.testing.assert_array_equal(result, x.reshape(-1, 1))

    def test_known_values(self):
        """小さな配列で手計算と照合"""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        result = lagged_embed(x, tau=2, e=3)
        expected = np.array(
            [
                [4, 2, 0],
                [5, 3, 1],
                [6, 4, 2],
                [7, 5, 3],
                [8, 6, 4],
                [9, 7, 5],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(result, expected)

    def test_linear_sequence(self):
        """等差数列の入力で各行の差分が tau 刻み"""
        x = np.arange(20, dtype=float) * 3  # step=3
        tau, e = 2, 4
        result = lagged_embed(x, tau=tau, e=e)
        # 列間の差（col 0 が最新, col -1 が最古）: 各行で隣接列の差 = -tau * step
        for row in range(result.shape[0]):
            diffs = np.diff(result[row])
            np.testing.assert_array_equal(diffs, np.full(e - 1, -tau * 3))


# ===========================================================================
# 3.1.3 Property-Based Tests (hypothesis)
# ===========================================================================
class TestProperties:
    @given(data=valid_embed_inputs())  # ty: ignore[missing-argument]
    def test_output_shape_property(self, data):
        """任意の有効な (x, tau, e) に対して出力形状が正しい"""
        x, tau, e = data
        result = lagged_embed(x, tau=tau, e=e)
        assert result.shape == (len(x) - (e - 1) * tau, e)

    @given(data=valid_embed_inputs())  # ty: ignore[missing-argument]
    def test_sliding_window_property(self, data):
        """連続する行が1ステップずつスライドし、全要素が正しい位置の入力値"""
        x, tau, e = data
        result = lagged_embed(x, tau=tau, e=e)
        for col in range(e):
            for row in range(result.shape[0]):
                assert result[row, col] == x[tau * (e - 1 - col) + row]


# ===========================================================================
# 3.1.4 エラーハンドリング
# ===========================================================================
class TestErrorHandling:
    def test_rejects_2d_input(self):
        with pytest.raises(ValueError, match="1D"):
            lagged_embed(np.zeros((5, 2)), tau=1, e=1)

    def test_rejects_zero_tau(self):
        with pytest.raises(ValueError, match="positive"):
            lagged_embed(np.arange(10, dtype=float), tau=0, e=1)

    def test_rejects_zero_e(self):
        with pytest.raises(ValueError, match="positive"):
            lagged_embed(np.arange(10, dtype=float), tau=1, e=0)

    def test_rejects_insufficient_length(self):
        with pytest.raises(ValueError, match=r"\(e - 1\) \* tau < len"):
            lagged_embed(np.arange(4, dtype=float), tau=2, e=3)  # (3-1)*2=4 >= 4

    def test_minimum_valid_length(self):
        """len(x) == (e-1)*tau + 1 の境界で成功"""
        e, tau = 3, 2
        min_len = (e - 1) * tau + 1  # = 5
        x = np.arange(min_len, dtype=float)
        result = lagged_embed(x, tau=tau, e=e)
        assert result.shape == (1, e)
