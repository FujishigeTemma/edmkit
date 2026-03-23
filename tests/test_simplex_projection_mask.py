"""Tests for simplex_projection mask support.

Verifies backward compatibility (mask=None), 2D/3D mask behavior,
all-True mask equivalence, insufficient valid points error, and
hypothesis shape properties.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edmkit.simplex_projection import simplex_projection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_2d(N: int, E: int, M: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, E))
    Y = rng.standard_normal((N, 1))
    Q = rng.standard_normal((M, E))
    return X, Y, Q


# ---------------------------------------------------------------------------
# Backward compatibility: mask=None
# ---------------------------------------------------------------------------

class TestMaskNoneBackwardCompat:
    def test_identity_prediction_no_mask(self):
        """Existing identity test still passes with mask=None (default)."""
        rng = np.random.default_rng(42)
        N, E = 20, 2
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal(N)
        Q = X[[3, 7, 15]]
        predictions = simplex_projection(X, Y, Q, mask=None)
        np.testing.assert_allclose(predictions, Y[[3, 7, 15]], atol=1e-15)


# ---------------------------------------------------------------------------
# 2D mask: padding rows + mask → same as real-only
# ---------------------------------------------------------------------------

class TestMask2D:
    def test_padding_rows_masked_out(self):
        """Append random padding rows with mask=False → same result as unmasked real-only."""
        X_real, Y_real, Q = _make_data_2d(30, 2, 5)
        pred_no_mask = simplex_projection(X_real, Y_real, Q)

        # Append 10 padding rows
        rng = np.random.default_rng(99)
        pad_X = rng.standard_normal((10, 2)) * 100  # far away
        pad_Y = rng.standard_normal((10, 1)) * 100
        X_padded = np.vstack([X_real, pad_X])
        Y_padded = np.vstack([Y_real, pad_Y])
        mask = np.array([True] * 30 + [False] * 10)

        pred_masked = simplex_projection(X_padded, Y_padded, Q, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_no_mask, atol=1e-12)

    def test_all_true_mask_equals_no_mask(self):
        """mask=all True → identical to mask=None."""
        X, Y, Q = _make_data_2d(25, 3, 5)
        pred_none = simplex_projection(X, Y, Q)
        mask = np.ones(25, dtype=bool)
        pred_masked = simplex_projection(X, Y, Q, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_none, atol=1e-14)

    def test_too_few_valid_points_raises(self):
        """mask valid count < k → ValueError."""
        E = 3
        N = 10
        X, Y, Q = _make_data_2d(N, E, 2)
        mask = np.zeros(N, dtype=bool)
        mask[:3] = True  # only 3 valid, need k=4
        with pytest.raises(ValueError, match="valid"):
            simplex_projection(X, Y, Q, mask=mask)


# ---------------------------------------------------------------------------
# 3D mask: batch elements with different valid counts
# ---------------------------------------------------------------------------

class TestMask3D:
    def test_batch_mask_matches_2d(self):
        """Each 3D batch element with mask matches independent 2D masked call."""
        rng = np.random.default_rng(42)
        B, N, E, M = 3, 20, 2, 5
        X_3d = rng.standard_normal((B, N, E))
        Y_3d = rng.standard_normal((B, N, 1))
        Q_3d = rng.standard_normal((B, M, E))
        # Different valid counts per batch element
        mask = np.ones((B, N), dtype=bool)
        mask[0, 15:] = False  # 15 valid
        mask[1, 18:] = False  # 18 valid
        # mask[2] all True → 20 valid

        pred_3d = simplex_projection(X_3d, Y_3d, Q_3d, mask=mask)
        assert pred_3d.shape == (B, M, 1)

        for b in range(B):
            valid = mask[b]
            pred_2d = simplex_projection(X_3d[b][valid], Y_3d[b][valid], Q_3d[b])
            np.testing.assert_allclose(pred_3d[b].squeeze(), pred_2d.squeeze(), atol=1e-12)

    def test_3d_all_true_equals_no_mask(self):
        rng = np.random.default_rng(42)
        B, N, E, M = 2, 15, 2, 4
        X = rng.standard_normal((B, N, E))
        Y = rng.standard_normal((B, N, 1))
        Q = rng.standard_normal((B, M, E))

        pred_none = simplex_projection(X, Y, Q)
        mask = np.ones((B, N), dtype=bool)
        pred_masked = simplex_projection(X, Y, Q, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_none, atol=1e-14)

    def test_3d_too_few_valid_raises(self):
        E = 2
        rng = np.random.default_rng(42)
        B, N, M = 2, 10, 3
        X = rng.standard_normal((B, N, E))
        Y = rng.standard_normal((B, N, 1))
        Q = rng.standard_normal((B, M, E))
        mask = np.ones((B, N), dtype=bool)
        mask[1, 3:] = False  # batch 1: only 3 valid, need k=3 → ok, but let's make it 2
        mask[1, 2:] = False  # batch 1: only 2 valid, need k=3
        with pytest.raises(ValueError, match="valid"):
            simplex_projection(X, Y, Q, mask=mask)


# ---------------------------------------------------------------------------
# 2D tensor path: mask support
# ---------------------------------------------------------------------------

class TestMaskTensor2D:
    @pytest.mark.gpu
    def test_tensor_2d_mask_padding(self):
        """Tensor 2D: padding + mask → same as real-only."""
        X_real, Y_real, Q = _make_data_2d(30, 2, 5)
        pred_no_mask = simplex_projection(X_real, Y_real, Q, use_tensor=True)

        rng = np.random.default_rng(99)
        pad_X = rng.standard_normal((10, 2)) * 100
        pad_Y = rng.standard_normal((10, 1)) * 100
        X_padded = np.vstack([X_real, pad_X])
        Y_padded = np.vstack([Y_real, pad_Y])
        mask = np.array([True] * 30 + [False] * 10)

        pred_masked = simplex_projection(X_padded, Y_padded, Q, mask=mask, use_tensor=True)
        np.testing.assert_allclose(pred_masked, pred_no_mask, atol=1e-4)

    @pytest.mark.gpu
    def test_tensor_3d_mask_not_implemented(self):
        """Tensor 3D with mask raises NotImplementedError."""
        rng = np.random.default_rng(42)
        B, N, E, M = 2, 10, 2, 3
        X = rng.standard_normal((B, N, E))
        Y = rng.standard_normal((B, N, 1))
        Q = rng.standard_normal((B, M, E))
        mask = np.ones((B, N), dtype=bool)
        mask[0, 8:] = False
        with pytest.raises(NotImplementedError):
            simplex_projection(X, Y, Q, mask=mask, use_tensor=True)


# ---------------------------------------------------------------------------
# Hypothesis: random mask → correct shape
# ---------------------------------------------------------------------------

class TestHypothesisMask:
    @settings(deadline=None)
    @given(
        E=st.integers(1, 4),
        data=st.data(),
    )
    def test_random_mask_shape_2d(self, E, data):
        k = E + 1
        N = data.draw(st.integers(k + 1, 30))
        M = data.draw(st.integers(1, 5))
        n_valid = data.draw(st.integers(k, N))

        rng = np.random.default_rng(0)
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal((N, 1))
        Q = rng.standard_normal((M, E))
        mask = np.zeros(N, dtype=bool)
        mask[:n_valid] = True

        pred = simplex_projection(X, Y, Q, mask=mask)
        # 2D path squeezes trailing dim when E'=1
        pred = np.atleast_1d(pred)
        assert pred.shape[0] == M
        assert np.all(np.isfinite(pred))
