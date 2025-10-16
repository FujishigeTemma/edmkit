import numpy as np

from edmkit import generate, lagged_embed, simplex_projection, smap


def data() -> np.ndarray:
    return np.sin(np.linspace(0, 4 * np.pi, 120))


def test_simplex_and_smap_end_to_end():
    x = data()
    tau = 1
    embedding_dim = 3
    horizon = 1
    lib_size = 80

    embedding = lagged_embed(x, tau, embedding_dim)
    shift = tau * (embedding_dim - 1)

    X = embedding[: lib_size - shift]
    Y = embedding[horizon : lib_size - shift + horizon, 0]
    query_points = embedding[lib_size - shift :]

    simplex_predictions = simplex_projection(X, Y, query_points)
    assert simplex_predictions.shape == (len(query_points),)
    assert np.all(np.isfinite(simplex_predictions))

    smap_predictions = smap(X, Y, query_points, theta=1.0)
    assert smap_predictions.shape == (len(query_points),)
    assert np.all(np.isfinite(smap_predictions))


def test_generate_lorenz():
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    X0 = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    t_max = 30

    t, X = generate.lorenz(sigma, rho, beta, X0, dt, t_max)

    assert t.ndim == 1 and X.ndim == 2
    assert len(t) == len(X)
    assert X.shape[1] == 3
    assert np.all(np.isfinite(X))
