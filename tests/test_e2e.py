import numpy as np

from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


def test_simplex_and_smap_end_to_end():
    x = np.sin(np.linspace(0, 4 * np.pi, 120))
    tau, E, Tp = 1, 3, 1
    lib_size = 80

    embedding = lagged_embed(x, tau, E)
    shift = tau * (E - 1)

    X = embedding[: lib_size - shift]
    Y = embedding[Tp : lib_size - shift + Tp, 0]
    Q = embedding[lib_size - shift :]

    simplex_predictions = simplex_projection(X, Y, Q)
    assert simplex_predictions.shape == (len(Q),)
    assert np.all(np.isfinite(simplex_predictions))

    smap_predictions = smap(X, Y, Q, theta=1.0)
    assert smap_predictions.shape == (len(Q),)
    assert np.all(np.isfinite(smap_predictions))

    # For a sine wave, simplex prediction should correlate positively with actual values
    actual = x[lib_size + Tp - 1 : lib_size + Tp - 1 + len(Q)]
    simplex_corr = np.corrcoef(simplex_predictions, actual[: len(simplex_predictions)])[0, 1]
    assert simplex_corr > 0, f"Expected positive correlation for sine wave, got {simplex_corr}"


def test_ccm_pipeline():
    N = 300
    rx, ry, Bxy = 3.8, 3.5, 0.02
    X = np.zeros(N)
    Y = np.zeros(N)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N):
        X[i] = X[i - 1] * (rx - rx * X[i - 1])
        Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]

    E, tau = 2, 1
    Y_embed = lagged_embed(Y, tau=tau, e=E)
    shift = (E - 1) * tau
    X_aligned = X[shift:]

    N_embed = Y_embed.shape[0]
    library_pool = np.arange(N_embed // 2)
    prediction_pool = np.arange(N_embed // 2, N_embed)
    lib_sizes = np.array([10, 50, N_embed // 2])

    correlations = with_simplex_projection(
        Y_embed,
        X_aligned,
        lib_sizes,
        n_samples=5,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
    )

    assert correlations.shape == (len(lib_sizes),)
    assert np.all(np.isfinite(correlations))
    # Correlations should not be all negative for a causal system
    assert np.any(correlations > 0), f"Expected at least some positive correlations, got {correlations}"
