import numpy as np
import pytest

from edmkit.ccm import with_simplex_projection, make_sample_func
from edmkit.embedding import lagged_embed


def prepare_directional_inputs(x: np.ndarray, y: np.ndarray):
    e = 2
    tau = 1
    y_embedding = lagged_embed(y, tau=tau, e=e)
    x_embedding = lagged_embed(x, tau=tau, e=e)
    shift = (e - 1) * tau
    x_aligned = x[shift:]
    y_aligned = y[shift:]
    n = len(y_embedding)
    library_pool = np.arange(n // 2)
    prediction_pool = np.arange(n // 2, n)
    lib_sizes = np.array([20, 40, 80, 160])
    return y_embedding, x_embedding, x_aligned, y_aligned, library_pool, prediction_pool, lib_sizes


class TestCCMTrend:
    @pytest.mark.slow
    def test_causal_direction_is_stronger_than_reverse(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_embedding, x_aligned, y_aligned, library_pool, prediction_pool, lib_sizes = prepare_directional_inputs(x, y)

        forward = with_simplex_projection(
            y_embedding,
            x_aligned,
            lib_sizes,
            n_samples=20,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=42),
        )
        reverse = with_simplex_projection(
            x_embedding,
            y_aligned,
            lib_sizes,
            n_samples=20,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=42),
        )

        # The coupled fixture yields a clear directional gap by the largest library size.
        assert forward[-1] > reverse[-1] + 0.25
        # CCM should strengthen as the library grows on the truly causal direction.
        assert forward[-1] > forward[0] + 0.25

    @pytest.mark.slow
    def test_independent_series_do_not_show_strong_ccm_signal(self, independent_pair: tuple[np.ndarray, np.ndarray]):
        x, y = independent_pair
        y_embedding, _, x_aligned, _, library_pool, prediction_pool, lib_sizes = prepare_directional_inputs(x, y)

        correlations = with_simplex_projection(
            y_embedding,
            x_aligned,
            lib_sizes,
            n_samples=20,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=7),
        )

        # Independent logistic maps should stay near zero cross-map skill.
        assert float(np.max(correlations)) < 0.2
