import numpy as np
import pytest

from edmkit.simplex_projection import simplex_projection


class TestSimplexProjectionMaskEquivalence:
    def test_masked_2d_call_matches_explicit_subset(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(14, 2))
        y = rng.normal(size=(14, 1))
        q = rng.normal(size=(5, 2))
        mask = np.array([True] * 10 + [False] * 4)

        expected = simplex_projection(x[mask], y[mask], q)
        actual = simplex_projection(x, y, q, mask=mask)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_masked_3d_call_matches_per_batch_subset(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 15, 2))
        y = rng.normal(size=(3, 15, 1))
        q = rng.normal(size=(3, 4, 2))
        mask = np.ones((3, 15), dtype=bool)
        mask[0, 12:] = False
        mask[1, 10:] = False

        actual = simplex_projection(x, y, q, mask=mask)
        for batch in range(3):
            expected = simplex_projection(x[batch][mask[batch]], y[batch][mask[batch]], q[batch])
            np.testing.assert_allclose(actual[batch].squeeze(-1), expected.squeeze(), atol=1e-12, rtol=1e-12)

    def test_all_true_mask_matches_no_mask(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(12, 3))
        y = rng.normal(size=(12,))
        q = rng.normal(size=(3, 3))
        masked = simplex_projection(x, y, q, mask=np.ones(len(x), dtype=bool))
        unmasked = simplex_projection(x, y, q)
        np.testing.assert_allclose(masked, unmasked, atol=1e-12, rtol=1e-12)

    def test_rejects_mask_with_too_few_points(self):
        x = np.arange(10.0).reshape(5, 2)
        y = np.arange(5.0)
        q = np.array([[0.0, 0.0]])
        mask = np.array([True, True, False, False, False])
        with pytest.raises(ValueError, match="Not enough points"):
            simplex_projection(x, y, q, mask=mask)
