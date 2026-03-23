"""Tests for edmkit.metrics — prediction evaluation metrics.

Covers 1D auto-promote, 2D standard path, 3D batch path,
shape mismatch validation, and hypothesis property-based tests.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from edmkit.metrics import mae, mae_per_dim, mean_rho, rho_per_dim, rmse, rmse_per_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_arrays_2d(N: int, D: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, D)), rng.standard_normal((N, D))


def _random_arrays_3d(B: int, N: int, D: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, N, D)), rng.standard_normal((B, N, D))


# ---------------------------------------------------------------------------
# Perfect prediction
# ---------------------------------------------------------------------------

class TestPerfectPrediction:
    def test_mean_rho_perfect(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert mean_rho(a, a) == pytest.approx(1.0)

    def test_rmse_perfect(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_mae_perfect(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert mae(a, a) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Anti-correlated
# ---------------------------------------------------------------------------

class TestAntiCorrelated:
    def test_mean_rho_anti(self):
        a = np.array([[1.0], [2.0], [3.0]])
        b = np.array([[3.0], [2.0], [1.0]])
        assert mean_rho(a, b) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Constant prediction → rho = 0
# ---------------------------------------------------------------------------

class TestConstantPrediction:
    def test_rho_constant(self):
        pred = np.array([[5.0], [5.0], [5.0]])
        obs = np.array([[1.0], [2.0], [3.0]])
        assert mean_rho(pred, obs) == pytest.approx(0.0)

    def test_rho_per_dim_constant(self):
        pred = np.array([[5.0], [5.0], [5.0]])
        obs = np.array([[1.0], [2.0], [3.0]])
        result = rho_per_dim(pred, obs)
        np.testing.assert_allclose(result, [0.0])


# ---------------------------------------------------------------------------
# 1D / 2D compatibility
# ---------------------------------------------------------------------------

class TestOneDimPromote:
    def test_mean_rho_1d_2d_equal(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        assert mean_rho(a, b) == pytest.approx(mean_rho(a[:, None], b[:, None]))

    def test_rmse_1d_2d_equal(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        assert rmse(a, b) == pytest.approx(rmse(a[:, None], b[:, None]))

    def test_mae_1d_2d_equal(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        assert mae(a, b) == pytest.approx(mae(a[:, None], b[:, None]))

    def test_rho_per_dim_1d_shape(self):
        """rho_per_dim(1d, 1d) should return shape (1,)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = rho_per_dim(a, b)
        assert result.shape == (1,)

    def test_rmse_per_dim_1d_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = rmse_per_dim(a, b)
        assert result.shape == (1,)

    def test_mae_per_dim_1d_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = mae_per_dim(a, b)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# 3D batch
# ---------------------------------------------------------------------------

class TestBatch3D:
    def test_mean_rho_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = mean_rho(pred, obs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_rmse_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = rmse(pred, obs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_mae_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = mae(pred, obs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_rho_per_dim_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = rho_per_dim(pred, obs)
        assert result.shape == (4, 3)

    def test_rmse_per_dim_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = rmse_per_dim(pred, obs)
        assert result.shape == (4, 3)

    def test_mae_per_dim_3d_shape(self):
        pred, obs = _random_arrays_3d(4, 20, 3)
        result = mae_per_dim(pred, obs)
        assert result.shape == (4, 3)


# ---------------------------------------------------------------------------
# 3D / 2D consistency
# ---------------------------------------------------------------------------

class TestBatch3DConsistency:
    """mean_rho(a_3d, b_3d)[i] == mean_rho(a_3d[i], b_3d[i])"""

    def test_mean_rho_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = mean_rho(pred, obs)
        for i in range(3):
            assert batch_result[i] == pytest.approx(mean_rho(pred[i], obs[i]))

    def test_rmse_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = rmse(pred, obs)
        for i in range(3):
            assert batch_result[i] == pytest.approx(rmse(pred[i], obs[i]))

    def test_mae_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = mae(pred, obs)
        for i in range(3):
            assert batch_result[i] == pytest.approx(mae(pred[i], obs[i]))

    def test_rho_per_dim_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = rho_per_dim(pred, obs)
        for i in range(3):
            np.testing.assert_allclose(batch_result[i], rho_per_dim(pred[i], obs[i]))

    def test_rmse_per_dim_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = rmse_per_dim(pred, obs)
        for i in range(3):
            np.testing.assert_allclose(batch_result[i], rmse_per_dim(pred[i], obs[i]))

    def test_mae_per_dim_3d_2d(self):
        pred, obs = _random_arrays_3d(3, 20, 2)
        batch_result = mae_per_dim(pred, obs)
        for i in range(3):
            np.testing.assert_allclose(batch_result[i], mae_per_dim(pred[i], obs[i]))


# ---------------------------------------------------------------------------
# Shape mismatch → ValueError
# ---------------------------------------------------------------------------

class TestShapeMismatch:
    def test_mean_rho_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_rho(np.zeros((5, 2)), np.zeros((5, 3)))

    def test_rmse_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            rmse(np.zeros((5, 2)), np.zeros((4, 2)))

    def test_mae_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            mae(np.zeros((5,)), np.zeros((4,)))

    def test_4d_rejected(self):
        with pytest.raises(ValueError, match="1D, 2D, or 3D"):
            mean_rho(np.zeros((2, 3, 4, 5)), np.zeros((2, 3, 4, 5)))


# ---------------------------------------------------------------------------
# Hypothesis: shape handling properties
# ---------------------------------------------------------------------------

reasonable_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def matched_arrays_nd(draw, *, ndim: int):
    """Draw matched arrays of a given ndim (1, 2, or 3)."""
    if ndim == 1:
        n = draw(st.integers(min_value=3, max_value=30))
        shape = (n,)
    elif ndim == 2:
        n = draw(st.integers(min_value=3, max_value=30))
        d = draw(st.integers(min_value=1, max_value=5))
        shape = (n, d)
    else:
        b = draw(st.integers(min_value=1, max_value=4))
        n = draw(st.integers(min_value=3, max_value=20))
        d = draw(st.integers(min_value=1, max_value=5))
        shape = (b, n, d)
    pred = draw(arrays(dtype=np.float64, shape=shape, elements=reasonable_floats))
    obs = draw(arrays(dtype=np.float64, shape=shape, elements=reasonable_floats))
    return pred, obs


class TestHypothesisShapes:
    @given(matched_arrays_nd(ndim=1))
    def test_1d_mean_rho_returns_float(self, pair):
        pred, obs = pair
        result = mean_rho(pred, obs)
        assert isinstance(result, float)

    @given(matched_arrays_nd(ndim=2))
    def test_2d_mean_rho_returns_float(self, pair):
        pred, obs = pair
        result = mean_rho(pred, obs)
        assert isinstance(result, float)

    @given(matched_arrays_nd(ndim=3))
    def test_3d_mean_rho_returns_batch(self, pair):
        pred, obs = pair
        result = mean_rho(pred, obs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (pred.shape[0],)

    @given(matched_arrays_nd(ndim=1))
    def test_1d_rho_per_dim_shape(self, pair):
        pred, obs = pair
        result = rho_per_dim(pred, obs)
        assert result.shape == (1,)

    @given(matched_arrays_nd(ndim=2))
    def test_2d_rho_per_dim_shape(self, pair):
        pred, obs = pair
        result = rho_per_dim(pred, obs)
        assert result.shape == (pred.shape[1],)

    @given(matched_arrays_nd(ndim=3))
    def test_3d_rho_per_dim_shape(self, pair):
        pred, obs = pair
        result = rho_per_dim(pred, obs)
        assert result.shape == (pred.shape[0], pred.shape[2])

    @given(matched_arrays_nd(ndim=2))
    def test_rmse_non_negative(self, pair):
        pred, obs = pair
        assert rmse(pred, obs) >= 0.0

    @given(matched_arrays_nd(ndim=2))
    def test_mae_non_negative(self, pair):
        pred, obs = pair
        assert mae(pred, obs) >= 0.0

    @given(matched_arrays_nd(ndim=3))
    def test_3d_rmse_non_negative(self, pair):
        pred, obs = pair
        result = rmse(pred, obs)
        assert np.all(result >= 0.0)
