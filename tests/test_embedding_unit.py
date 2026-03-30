import numpy as np
import pytest

from edmkit.embedding import lagged_embed


class TestLaggedEmbedExamples:
    def test_known_example_matches_expected_indices(self):
        x = np.arange(10)
        embedded = lagged_embed(x, tau=2, e=3)
        expected = np.array(
            [
                [4, 2, 0],
                [5, 3, 1],
                [6, 4, 2],
                [7, 5, 3],
                [8, 6, 4],
                [9, 7, 5],
            ]
        )
        np.testing.assert_array_equal(embedded, expected)

    def test_minimum_valid_length_produces_single_row(self):
        x = np.array([10, 20, 30, 40, 50])
        embedded = lagged_embed(x, tau=2, e=3)
        np.testing.assert_array_equal(embedded, np.array([[50, 30, 10]]))

    def test_output_shape_matches_contract(self):
        x = np.arange(20)
        tau = 3
        e = 4
        embedded = lagged_embed(x, tau=tau, e=e)
        assert embedded.shape == (20 - (e - 1) * tau, e)


class TestLaggedEmbedValidation:
    def test_rejects_non_1d_input(self):
        with pytest.raises(ValueError, match="1D"):
            lagged_embed(np.arange(9).reshape(3, 3), tau=1, e=2)

    @pytest.mark.parametrize(
        ("tau", "e"),
        [
            (0, 2),
            (-1, 2),
            (1, 0),
            (1, -2),
        ],
    )
    def test_rejects_non_positive_tau_or_e(self, tau: int, e: int):
        with pytest.raises(ValueError, match="positive"):
            lagged_embed(np.arange(10), tau=tau, e=e)

    def test_rejects_insufficient_length(self):
        with pytest.raises(ValueError, match="tau"):
            lagged_embed(np.arange(5), tau=2, e=4)
