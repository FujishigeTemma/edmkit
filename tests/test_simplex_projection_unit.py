import numpy as np
import pytest

from edmkit.simplex_projection import knn, simplex_projection


class TestSimplexProjectionExamples:
    def test_identity_query_recovers_library_targets(self):
        x = np.array([[0.0], [2.0], [5.0], [9.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        predictions = simplex_projection(x, y, x)
        np.testing.assert_allclose(predictions, y, atol=1e-12, rtol=1e-12)

    def test_knn_returns_expected_neighbors(self):
        x = np.array([[0.0], [2.0], [5.0], [9.0]])
        q = np.array([[1.0], [8.0]])
        distances, indices = knn(x, q, k=2)
        np.testing.assert_allclose(distances, np.array([[1.0, 1.0], [1.0, 3.0]]))
        np.testing.assert_array_equal(indices, np.array([[1, 0], [3, 2]]))

    def test_constant_target_is_recovered_exactly(self):
        x = np.array([[0.0], [1.0], [3.0], [6.0]])
        y = np.full(len(x), 7.5)
        q = np.array([[2.0], [4.0]])
        predictions = simplex_projection(x, y, q)
        np.testing.assert_allclose(predictions, 7.5, atol=1e-12, rtol=1e-12)

    def test_predictions_stay_within_scalar_target_range(self):
        x = np.array([[0.0], [1.0], [2.0], [4.0], [7.0]])
        y = np.array([-2.0, 1.0, 3.0, 5.0, 8.0])
        q = np.array([[1.5], [3.5]])
        predictions = simplex_projection(x, y, q)
        assert np.all(predictions >= y.min() - 1e-12)
        assert np.all(predictions <= y.max() + 1e-12)

    def test_batch_path_matches_loop(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 12, 2))
        y = rng.normal(size=(3, 12, 1))
        q = rng.normal(size=(3, 5, 2))
        batched = simplex_projection(x, y, q)
        for batch in range(x.shape[0]):
            expected = simplex_projection(x[batch], y[batch], q[batch])
            np.testing.assert_allclose(batched[batch].squeeze(-1), expected.squeeze(), atol=1e-12, rtol=1e-12)

    def test_rejects_shape_mismatch(self):
        x = np.zeros((5, 2))
        y = np.zeros((4, 1))
        q = np.zeros((2, 2))
        with pytest.raises(ValueError, match="same length"):
            simplex_projection(x, y, q)

    @pytest.mark.gpu
    def test_tensor_path_matches_numpy_path(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(16, 3)).astype(np.float32)
        y = rng.normal(size=(16, 2)).astype(np.float32)
        q = rng.normal(size=(4, 3)).astype(np.float32)
        expected = simplex_projection(x, y, q, use_tensor=False)
        actual = simplex_projection(x, y, q, use_tensor=True)
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
