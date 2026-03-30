import numpy as np
import pytest

from edmkit.smap import smap


class TestSMapMaskEquivalence:
    def test_masked_2d_call_matches_explicit_subset(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(16, 2))
        y = rng.normal(size=(16, 1))
        q = rng.normal(size=(4, 2))
        mask = np.array([True] * 11 + [False] * 5)

        expected = smap(x[mask], y[mask], q, theta=2.0)
        actual = smap(x, y, q, theta=2.0, mask=mask)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_masked_3d_call_matches_per_batch_subset(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 18, 2))
        y = rng.normal(size=(2, 18, 1))
        q = rng.normal(size=(2, 5, 2))
        mask = np.ones((2, 18), dtype=bool)
        mask[0, 14:] = False
        mask[1, 11:] = False

        actual = smap(x, y, q, theta=2.0, mask=mask)
        for batch in range(2):
            expected = smap(x[batch][mask[batch]], y[batch][mask[batch]], q[batch], theta=2.0)
            np.testing.assert_allclose(actual[batch].squeeze(-1), expected.squeeze(), atol=1e-10, rtol=1e-10)

    def test_all_true_mask_matches_no_mask(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(12, 2))
        y = rng.normal(size=(12, 1))
        q = rng.normal(size=(3, 2))
        masked = smap(x, y, q, theta=0.0, mask=np.ones(len(x), dtype=bool))
        unmasked = smap(x, y, q, theta=0.0)
        np.testing.assert_allclose(masked, unmasked, atol=1e-12, rtol=1e-12)

    def test_rejects_mask_with_too_few_valid_points(self):
        x = np.arange(10.0).reshape(5, 2)
        y = np.arange(5.0)[:, None]
        q = np.array([[0.0, 0.0]])
        mask = np.array([True, True, False, False, False])
        with pytest.raises(ValueError, match="Not enough valid"):
            smap(x, y, q, theta=1.0, mask=mask)
