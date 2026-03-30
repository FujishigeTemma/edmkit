from functools import partial

import numpy as np
import pytest

from edmkit.ccm import bootstrap, ccm, make_sample_func, pearson_correlation, with_simplex_projection, with_smap
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


def prepare_ccm_inputs(x: np.ndarray, y: np.ndarray, *, e: int = 2, tau: int = 1):
    y_embedding = lagged_embed(y, tau=tau, e=e)
    shift = (e - 1) * tau
    x_aligned = x[shift:]
    n = len(y_embedding)
    library_pool = np.arange(n // 2)
    prediction_pool = np.arange(n // 2, n)
    return y_embedding, x_aligned, library_pool, prediction_pool


class TestPearsonCorrelationExamples:
    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 3.0 * x + 5.0
        np.testing.assert_allclose(pearson_correlation(x, y), 1.0, atol=1e-12)

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = -2.0 * x + 1.0
        np.testing.assert_allclose(pearson_correlation(x, y), -1.0, atol=1e-12)

    def test_constant_input_returns_zero(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(pearson_correlation(x, y), 0.0, atol=1e-12)


class TestCCMHelpers:
    def test_make_sample_func_is_reproducible(self):
        pool = np.arange(10)
        first = make_sample_func(seed=7)(pool, 5)
        second = make_sample_func(seed=7)(pool, 5)
        np.testing.assert_array_equal(first, second)

    def test_bootstrap_batching_does_not_change_results(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_aligned, library_pool, prediction_pool = prepare_ccm_inputs(x, y)
        lib_sizes = np.array([20, 40, 80])

        unbatched = bootstrap(
            y_embedding,
            x_aligned,
            lib_sizes,
            simplex_projection,
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=3),
            batch_size=None,
        )
        batched = bootstrap(
            y_embedding,
            x_aligned,
            lib_sizes,
            simplex_projection,
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=3),
            batch_size=2,
        )

        np.testing.assert_allclose(batched, unbatched, atol=1e-12, rtol=1e-12)

    def test_ccm_uses_requested_aggregator(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_aligned, library_pool, prediction_pool = prepare_ccm_inputs(x, y)
        lib_sizes = np.array([20, 40])

        samples = bootstrap(
            y_embedding,
            x_aligned,
            lib_sizes,
            simplex_projection,
            n_samples=8,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=11),
            batch_size=4,
        )
        actual = ccm(
            y_embedding,
            x_aligned,
            lib_sizes,
            simplex_projection,
            n_samples=8,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=11),
            aggregate_func=np.median,
            batch_size=4,
        )
        np.testing.assert_allclose(actual, np.median(samples, axis=0), atol=1e-12, rtol=1e-12)

    def test_with_simplex_projection_matches_direct_ccm_call(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_aligned, library_pool, prediction_pool = prepare_ccm_inputs(x, y)
        lib_sizes = np.array([20, 40])

        expected = ccm(
            y_embedding,
            x_aligned,
            lib_sizes,
            partial(simplex_projection, use_tensor=False),
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=5),
        )
        actual = with_simplex_projection(
            y_embedding,
            x_aligned,
            lib_sizes,
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=5),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_with_smap_matches_direct_ccm_call(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_aligned, library_pool, prediction_pool = prepare_ccm_inputs(x, y)
        lib_sizes = np.array([20, 40])

        expected = ccm(
            y_embedding,
            x_aligned,
            lib_sizes,
            partial(smap, theta=2.0, alpha=1e-6, use_tensor=False),
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=13),
        )
        actual = with_smap(
            y_embedding,
            x_aligned,
            lib_sizes,
            theta=2.0,
            alpha=1e-6,
            n_samples=6,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed=13),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_rejects_invalid_aggregate_func(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding, x_aligned, library_pool, prediction_pool = prepare_ccm_inputs(x, y)
        with pytest.raises(ValueError, match="aggregate_func"):
            ccm(
                y_embedding,
                x_aligned,
                np.array([20]),
                simplex_projection,
                library_pool=library_pool,
                prediction_pool=prediction_pool,
                aggregate_func=None,  # ty: ignore[invalid-argument-type]
            )
