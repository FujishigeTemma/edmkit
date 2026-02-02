import numpy as np
import pytest

from edmkit.ccm import with_simplex_projection, with_smap
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


@pytest.fixture
def data():
    X = np.random.randn(500)
    Y = np.zeros(500)
    Y[0] = np.random.randn()
    for i in range(1, 500):
        Y[i] = 0.7 * Y[i - 1] + 0.3 * X[i - 1] + 0.1 * np.random.randn()

    tau = 1
    E = 3

    Y_embedding = lagged_embed(Y, tau=tau, e=E)
    shift = tau * (E - 1)
    X_aligned = X[shift:]
    return X_aligned, Y_embedding


def manual_ccm_with_simplex(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    *,
    n_samples: int,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sampler,
    aggregator,
    use_tensor: bool,
) -> np.ndarray:
    correlations = np.zeros(len(lib_sizes))
    for i, lib_size in enumerate(lib_sizes):
        samples = np.zeros(n_samples)
        for j in range(n_samples):
            lib_indices = sampler(library_pool, lib_size)

            lib_X = X[lib_indices]
            lib_Y = Y[lib_indices]
            query_points = X[prediction_pool]

            predictions = simplex_projection(lib_X, lib_Y, query_points, use_tensor=use_tensor)
            actual = Y[prediction_pool]

            samples[j] = np.corrcoef(predictions, actual)[0, 1]

        correlations[i] = aggregator(samples)

    return np.asarray(correlations)


@pytest.mark.parametrize("use_tensor", [False, True])
def test_with_simplex(data, use_tensor):
    X, Y = data

    library_pool = np.arange(X.shape[0] // 2)
    prediction_pool = np.arange(X.shape[0] // 2, X.shape[0])

    # logarithmic within range 10 to max library size
    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=4, dtype=int)
    n_samples = 10

    manual_result = manual_ccm_with_simplex(
        Y,
        X,
        lib_sizes,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sampler=lambda pool, size: np.random.default_rng(42).choice(pool, size=size, replace=False),
        aggregator=np.mean,
        use_tensor=use_tensor,
    )

    library_result = with_simplex_projection(
        Y,
        X,
        lib_sizes=lib_sizes,
        n_samples=n_samples,
        use_tensor=use_tensor,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sampler=lambda pool, size: np.random.default_rng(42).choice(pool, size=size, replace=False),
    )

    assert np.allclose(library_result, manual_result, atol=1e-12), f"with_simplex_projection result {library_result}, manual {manual_result}"


def manual_ccm_with_smap(
    X: np.ndarray,
    Y: np.ndarray,
    lib_sizes: np.ndarray,
    *,
    theta: float,
    alpha: float,
    n_samples: int,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sampler,
    aggregator,
    use_tensor: bool,
) -> np.ndarray:
    correlations = np.zeros(len(lib_sizes))
    for i, lib_size in enumerate(lib_sizes):
        samples = np.zeros(n_samples)
        for j in range(n_samples):
            lib_indices = sampler(library_pool, lib_size)

            lib_X = X[lib_indices]
            lib_Y = Y[lib_indices]
            query_points = X[prediction_pool]

            predictions = smap(lib_X, lib_Y, query_points, theta=theta, alpha=alpha, use_tensor=use_tensor)
            actual = Y[prediction_pool]

            samples[j] = np.corrcoef(predictions, actual)[0, 1]

        correlations[i] = aggregator(samples)

    return np.asarray(correlations)


def test_with_smap(data):
    X, Y = data

    library_pool = np.arange(X.shape[0] // 2)
    prediction_pool = np.arange(X.shape[0] // 2, X.shape[0])

    lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=4, dtype=int)
    n_samples = 10

    manual_result = manual_ccm_with_smap(
        Y,
        X,
        lib_sizes,
        theta=2.0,
        alpha=1e-10,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sampler=lambda pool, size: np.random.default_rng(42).choice(pool, size=size, replace=False),
        aggregator=np.mean,
        use_tensor=False,
    )

    library_result = with_smap(
        Y,
        X,
        lib_sizes=lib_sizes,
        theta=2.0,
        alpha=1e-10,
        n_samples=n_samples,
        use_tensor=False,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sampler=lambda pool, size: np.random.default_rng(42).choice(pool, size=size, replace=False),
    )

    assert np.allclose(library_result, manual_result, atol=1e-12), f"with_smap result {library_result}, manual {manual_result}"
