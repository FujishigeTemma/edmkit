"""Tests for smap mask support.

Verifies backward compatibility (mask=None), 2D/3D mask behavior,
all-True mask equivalence, and hypothesis shape properties.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from edmkit.smap import smap


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
# Backward compat: mask=None
# ---------------------------------------------------------------------------

class TestMaskNoneBackwardCompat:
    def test_theta_zero_no_mask(self):
        X, Y, Q = _make_data_2d(30, 2, 5)
        pred = smap(X, Y, Q, theta=0.0, mask=None)
        assert np.all(np.isfinite(pred))

    def test_theta_positive_no_mask(self):
        X, Y, Q = _make_data_2d(30, 2, 5)
        pred = smap(X, Y, Q, theta=2.0, mask=None)
        assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# 2D mask
# ---------------------------------------------------------------------------

class TestMask2D:
    def test_padding_rows_masked_out(self):
        """Append padding rows masked out → same as real-only."""
        X_real, Y_real, Q = _make_data_2d(30, 2, 5)
        pred_no_mask = smap(X_real, Y_real, Q, theta=2.0)

        rng = np.random.default_rng(99)
        pad_X = rng.standard_normal((10, 2)) * 100
        pad_Y = rng.standard_normal((10, 1)) * 100
        X_padded = np.vstack([X_real, pad_X])
        Y_padded = np.vstack([Y_real, pad_Y])
        mask = np.array([True] * 30 + [False] * 10)

        pred_masked = smap(X_padded, Y_padded, Q, theta=2.0, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_no_mask, atol=1e-10)

    def test_padding_theta_zero(self):
        """theta=0 with mask → same as real-only."""
        X_real, Y_real, Q = _make_data_2d(30, 2, 5)
        pred_no_mask = smap(X_real, Y_real, Q, theta=0.0)

        rng = np.random.default_rng(99)
        pad_X = rng.standard_normal((10, 2)) * 100
        pad_Y = rng.standard_normal((10, 1)) * 100
        X_padded = np.vstack([X_real, pad_X])
        Y_padded = np.vstack([Y_real, pad_Y])
        mask = np.array([True] * 30 + [False] * 10)

        pred_masked = smap(X_padded, Y_padded, Q, theta=0.0, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_no_mask, atol=1e-10)

    def test_all_true_mask_equals_no_mask(self):
        X, Y, Q = _make_data_2d(25, 3, 5)
        pred_none = smap(X, Y, Q, theta=2.0)
        mask = np.ones(25, dtype=bool)
        pred_masked = smap(X, Y, Q, theta=2.0, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_none, atol=1e-14)


# ---------------------------------------------------------------------------
# 3D mask
# ---------------------------------------------------------------------------

class TestMask3D:
    def test_batch_mask_matches_2d(self):
        """Each 3D batch element with mask matches independent 2D masked call."""
        rng = np.random.default_rng(42)
        B, N, E, M = 3, 20, 2, 5
        X_3d = rng.standard_normal((B, N, E))
        Y_3d = rng.standard_normal((B, N, 1))
        Q_3d = rng.standard_normal((B, M, E))
        mask = np.ones((B, N), dtype=bool)
        mask[0, 15:] = False
        mask[1, 18:] = False

        pred_3d = smap(X_3d, Y_3d, Q_3d, theta=2.0, mask=mask)
        assert pred_3d.shape == (B, M, 1)

        for b in range(B):
            valid = mask[b]
            pred_2d = smap(X_3d[b][valid], Y_3d[b][valid], Q_3d[b], theta=2.0)
            np.testing.assert_allclose(pred_3d[b].squeeze(), pred_2d.squeeze(), atol=1e-10)

    def test_3d_all_true_equals_no_mask(self):
        rng = np.random.default_rng(42)
        B, N, E, M = 2, 15, 2, 4
        X = rng.standard_normal((B, N, E))
        Y = rng.standard_normal((B, N, 1))
        Q = rng.standard_normal((B, M, E))

        pred_none = smap(X, Y, Q, theta=2.0)
        mask = np.ones((B, N), dtype=bool)
        pred_masked = smap(X, Y, Q, theta=2.0, mask=mask)
        np.testing.assert_allclose(pred_masked, pred_none, atol=1e-14)

    def test_batch_mask_theta_zero(self):
        """theta=0 with 3D mask matches 2D element-wise."""
        rng = np.random.default_rng(42)
        B, N, E, M = 2, 20, 2, 5
        X_3d = rng.standard_normal((B, N, E))
        Y_3d = rng.standard_normal((B, N, 1))
        Q_3d = rng.standard_normal((B, M, E))
        mask = np.ones((B, N), dtype=bool)
        mask[0, 16:] = False

        pred_3d = smap(X_3d, Y_3d, Q_3d, theta=0.0, mask=mask)

        for b in range(B):
            valid = mask[b]
            pred_2d = smap(X_3d[b][valid], Y_3d[b][valid], Q_3d[b], theta=0.0)
            np.testing.assert_allclose(pred_3d[b].squeeze(), pred_2d.squeeze(), atol=1e-10)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

class TestHypothesisMask:
    @settings(deadline=None)
    @given(
        E=st.integers(1, 3),
        data=st.data(),
    )
    def test_random_mask_shape_2d(self, E, data):
        N = data.draw(st.integers(E + 3, 25))
        M = data.draw(st.integers(1, 5))
        n_valid = data.draw(st.integers(E + 3, N))

        rng = np.random.default_rng(0)
        X = rng.standard_normal((N, E))
        Y = rng.standard_normal((N, 1))
        Q = rng.standard_normal((M, E))
        mask = np.zeros(N, dtype=bool)
        mask[:n_valid] = True

        pred = smap(X, Y, Q, theta=2.0, mask=mask)
        pred = np.atleast_1d(pred)
        assert pred.shape[0] == M
        assert np.all(np.isfinite(pred))
