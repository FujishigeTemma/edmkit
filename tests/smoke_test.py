"""Smoke tests for importability and one minimal success path per public surface."""

import numpy as np

from edmkit import generate
from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


def test_minimal_forecast_calls():
    x = np.sin(np.linspace(0.0, 4.0 * np.pi, 60))
    embedded = lagged_embed(x, tau=1, e=3)

    simplex_predictions = simplex_projection(embedded[:30], x[3:33], embedded[30:-1])
    smap_predictions = smap(embedded[:30], x[3:33], embedded[30:-1], theta=1.0)

    assert np.isfinite(simplex_predictions).all()
    assert np.isfinite(smap_predictions).all()


def test_minimal_ccm_call():
    x = np.sin(np.linspace(0.0, 6.0 * np.pi, 100))
    embedded = lagged_embed(x, tau=1, e=2)
    n = len(embedded)

    correlations = with_simplex_projection(
        embedded,
        x[1:],
        lib_sizes=np.array([10, 20]),
        n_samples=3,
        library_pool=np.arange(n // 2),
        prediction_pool=np.arange(n // 2, n),
    )

    assert correlations.shape == (2,)
    assert np.isfinite(correlations).all()


def test_minimal_generator_calls():
    _, lorenz_x = generate.lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=1,
    )
    _, mg_x = generate.mackey_glass(tau=17.0, n=10, beta=0.2, gamma=0.1, x0=0.9, dt=1.0, t_max=50)
    _, dp_x = generate.double_pendulum(
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=9.81,
        X0=np.array([0.5, 0.3, 0.0, 0.0]),
        dt=0.01,
        t_max=1,
    )

    assert np.isfinite(lorenz_x).all()
    assert np.isfinite(mg_x).all()
    assert np.isfinite(dp_x).all()
