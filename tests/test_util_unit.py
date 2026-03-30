import numpy as np
import pytest
from tinygrad import Tensor

from edmkit.util import autocorrelation, dtw, pad, pairwise_distance, pairwise_distance_np


class TestPadExamples:
    def test_pad_merges_2d_arrays_and_zero_pads_tail(self):
        arrays = [np.array([[1, 2], [3, 4]]), np.array([[5], [6]])]
        actual = pad(arrays)
        expected = np.array([[[1, 2], [3, 4]], [[5, 0], [6, 0]]])
        np.testing.assert_array_equal(actual, expected)

    def test_pad_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            pad([np.zeros((2, 1)), np.zeros((3, 1))])

    def test_pad_rejects_non_2d_arrays(self):
        with pytest.raises(ValueError, match="2D"):
            pad([np.zeros(5)])


class TestDistanceExamples:
    def test_pairwise_distance_np_matches_known_squared_distances(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        actual = pairwise_distance_np(a)
        expected = np.array([[0.0, 25.0], [25.0, 0.0]])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_pairwise_distance_batch_matches_loop(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 4, 3))
        b = rng.normal(size=(2, 5, 3))
        batched = pairwise_distance_np(a, b)
        for batch in range(2):
            expected = pairwise_distance_np(a[batch], b[batch])
            np.testing.assert_allclose(batched[batch], expected, atol=1e-12, rtol=1e-12)

    @pytest.mark.gpu
    def test_tensor_distance_matches_numpy(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(6, 2)).astype(np.float32)
        b = rng.normal(size=(5, 2)).astype(np.float32)
        expected = pairwise_distance_np(a, b)
        actual = pairwise_distance(Tensor(a), Tensor(b)).numpy()
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)

    def test_pairwise_distance_np_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            pairwise_distance_np(np.zeros(5))

    def test_pairwise_distance_np_rejects_mismatched_ndim(self):
        with pytest.raises(ValueError, match="same number of dimensions"):
            pairwise_distance_np(np.zeros((3, 2)), np.zeros((2, 3, 2)))


class TestDtwExamples:
    def test_dtw_is_zero_for_identical_sequences(self):
        sequence = np.array([[0.0], [1.0], [2.0]])
        assert dtw(sequence, sequence) == pytest.approx(0.0)

    def test_dtw_matches_known_small_example(self):
        a = np.array([[0.0], [1.0], [2.0]])
        b = np.array([[0.0], [2.0]])
        assert dtw(a, b) == pytest.approx(1.0)


class TestAutocorrelationExamples:
    def test_lag_zero_is_one_for_nonconstant_signal(self):
        x = np.sin(np.linspace(0.0, 4.0 * np.pi, 128))
        actual = autocorrelation(x, max_lag=8)
        assert actual[0] == pytest.approx(1.0)

    def test_half_period_of_sine_is_negative(self):
        period = 20
        t = np.arange(200)
        x = np.sin(2.0 * np.pi * t / period)
        actual = autocorrelation(x, max_lag=period + 1)
        assert actual[period // 2] == pytest.approx(-1.0, abs=0.06)
