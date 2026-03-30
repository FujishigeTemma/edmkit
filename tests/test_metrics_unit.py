import numpy as np
import pytest

from edmkit.metrics import mae, mean_rho, rhos, rmse, validate_and_promote


class TestMetricValidation:
    def test_validate_and_promote_turns_1d_into_column_vectors(self):
        predictions = np.array([1.0, 2.0, 3.0])
        observations = np.array([1.0, 2.0, 3.0])
        promoted_predictions, promoted_observations = validate_and_promote(predictions, observations)
        assert promoted_predictions.shape == (3, 1)
        assert promoted_observations.shape == (3, 1)

    def test_validate_and_promote_preserves_3d_shape(self):
        predictions = np.zeros((2, 3, 4))
        observations = np.ones((2, 3, 4))
        promoted_predictions, promoted_observations = validate_and_promote(predictions, observations)
        assert promoted_predictions.shape == (2, 3, 4)
        assert promoted_observations.shape == (2, 3, 4)

    def test_validate_and_promote_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_and_promote(np.zeros((3, 2)), np.zeros((3, 1)))

    def test_validate_and_promote_rejects_4d_input(self):
        with pytest.raises(ValueError, match="1D, 2D, or 3D"):
            validate_and_promote(np.zeros((2, 3, 4, 5)), np.zeros((2, 3, 4, 5)))


class TestMetricExamples:
    def test_rhos_returns_per_dimension_correlation(self):
        predictions = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        observations = np.array([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]])
        actual = rhos(predictions, observations)
        np.testing.assert_allclose(actual, np.array([1.0, -1.0]), atol=1e-12, rtol=1e-12)

    def test_mean_rho_averages_across_dimensions(self):
        predictions = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        observations = np.array([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]])
        assert mean_rho(predictions, observations) == pytest.approx(0.0)

    def test_rmse_matches_hand_calculation(self):
        predictions = np.array([1.0, 3.0, 5.0])
        observations = np.array([1.0, 2.0, 7.0])
        expected = np.sqrt((0.0**2 + 1.0**2 + 2.0**2) / 3.0)
        assert rmse(predictions, observations) == pytest.approx(expected)

    def test_mae_matches_hand_calculation(self):
        predictions = np.array([1.0, 3.0, 5.0])
        observations = np.array([1.0, 2.0, 7.0])
        expected = (0.0 + 1.0 + 2.0) / 3.0
        assert mae(predictions, observations) == pytest.approx(expected)

    def test_constant_inputs_return_zero_correlation(self):
        predictions = np.array([2.0, 2.0, 2.0])
        observations = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(rhos(predictions, observations), np.array([0.0]), atol=1e-12, rtol=1e-12)


class TestMetricBatchConsistency:
    def test_rhos_3d_matches_per_batch_2d_calls(self):
        predictions = np.array(
            [
                [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],
                [[5.0, 1.0], [7.0, 2.0], [9.0, 3.0], [11.0, 4.0]],
            ]
        )
        observations = np.array(
            [
                [[2.0, 1.0], [4.0, 2.0], [6.0, 3.0], [8.0, 4.0]],
                [[4.0, 8.0], [6.0, 6.0], [8.0, 4.0], [10.0, 2.0]],
            ]
        )
        batched = rhos(predictions, observations)
        assert batched.shape == (2, 2)
        for batch in range(predictions.shape[0]):
            np.testing.assert_allclose(batched[batch], rhos(predictions[batch], observations[batch]), atol=1e-12, rtol=1e-12)

    def test_mean_rho_3d_matches_per_batch_2d_calls(self):
        predictions = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[4.0], [3.0], [2.0], [1.0]],
            ]
        )
        observations = np.array(
            [
                [[2.0], [4.0], [6.0], [8.0]],
                [[1.0], [2.0], [3.0], [4.0]],
            ]
        )
        batched = mean_rho(predictions, observations)
        assert batched.shape == (2,)
        for batch in range(predictions.shape[0]):
            assert batched[batch] == pytest.approx(mean_rho(predictions[batch], observations[batch]))

    def test_rmse_3d_matches_per_batch_2d_calls(self):
        predictions = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [0.0, -1.0]],
            ]
        )
        observations = np.array(
            [
                [[2.0, 0.0], [1.0, 2.0]],
                [[1.0, 1.0], [1.0, -2.0]],
            ]
        )
        batched = rmse(predictions, observations)
        assert batched.shape == (2,)
        for batch in range(predictions.shape[0]):
            assert batched[batch] == pytest.approx(rmse(predictions[batch], observations[batch]))

    def test_mae_3d_matches_per_batch_2d_calls(self):
        predictions = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [0.0, -1.0]],
            ]
        )
        observations = np.array(
            [
                [[2.0, 0.0], [1.0, 2.0]],
                [[1.0, 1.0], [1.0, -2.0]],
            ]
        )
        batched = mae(predictions, observations)
        assert batched.shape == (2,)
        for batch in range(predictions.shape[0]):
            assert batched[batch] == pytest.approx(mae(predictions[batch], observations[batch]))
