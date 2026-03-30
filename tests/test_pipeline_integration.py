import numpy as np

from edmkit.ccm import make_sample_func, with_simplex_projection
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


class TestForecastPipelines:
    def test_lagged_embed_plus_simplex_projection_tracks_sine_wave(self):
        x = np.sin(np.linspace(0.0, 8.0 * np.pi, 160))
        embedded = lagged_embed(x, tau=1, e=3)
        library_size = 90
        lib_x = embedded[:library_size]
        lib_y = x[3 : 3 + library_size]
        q = embedded[library_size:-1]
        actual = x[3 + library_size : 3 + library_size + len(q)]

        predictions = simplex_projection(lib_x, lib_y, q)
        rho = np.corrcoef(predictions, actual)[0, 1]
        # A clean periodic signal should remain almost perfectly forecastable end-to-end.
        assert rho > 0.99

    def test_lagged_embed_plus_smap_tracks_bounded_linear_series(self, bounded_linear_series: np.ndarray):
        embedded = lagged_embed(bounded_linear_series, tau=1, e=2)
        library_size = 300
        lib_x = embedded[:library_size]
        lib_y = bounded_linear_series[2 : 2 + library_size]
        q = embedded[library_size:-1]
        actual = bounded_linear_series[2 + library_size : 2 + library_size + len(q)]

        predictions = smap(lib_x, lib_y, q, theta=0.0)
        rho = np.corrcoef(predictions, actual)[0, 1]
        # This fixture is near-linear, so the global linear baseline should be almost exact.
        assert rho > 0.999


class TestCcmPipeline:
    def test_with_simplex_projection_returns_finite_scores(self, causal_pair: tuple[np.ndarray, np.ndarray]):
        x, y = causal_pair
        y_embedding = lagged_embed(y, tau=1, e=2)
        x_aligned = x[1:]
        n = len(y_embedding)

        correlations = with_simplex_projection(
            y_embedding,
            x_aligned,
            lib_sizes=np.array([20, 40]),
            n_samples=6,
            library_pool=np.arange(n // 2),
            prediction_pool=np.arange(n // 2, n),
            sample_func=make_sample_func(seed=9),
        )

        assert correlations.shape == (2,)
        assert np.isfinite(correlations).all()
