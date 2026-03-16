"""End-to-end integration tests — 精度・振る舞いまで検証する結合テスト。"""

import numpy as np

from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap
from tests.helpers import make_seeded_sampler


def test_simplex_and_smap_end_to_end():
    x = np.sin(np.linspace(0, 4 * np.pi, 120))
    tau, E, Tp = 1, 3, 1
    lib_size = 80

    embedding = lagged_embed(x, tau, E)
    shift = tau * (E - 1)

    X = embedding[: lib_size - shift]
    Y = embedding[Tp : lib_size - shift + Tp, 0]
    query = embedding[lib_size - shift :]

    simplex_pred = simplex_projection(X, Y, query)
    assert simplex_pred.shape == (len(query),)
    assert np.all(np.isfinite(simplex_pred))

    smap_pred = smap(X, Y, query, theta=1.0)
    assert smap_pred.shape == (len(query),)
    assert np.all(np.isfinite(smap_pred))

    # For a sine wave, simplex prediction should correlate positively with actual values
    actual = x[lib_size + Tp - 1 : lib_size + Tp - 1 + len(query)]
    simplex_corr = np.corrcoef(simplex_pred, actual[: len(simplex_pred)])[0, 1]
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
    lib_pool = np.arange(N_embed // 2)
    pred_pool = np.arange(N_embed // 2, N_embed)
    lib_sizes = np.array([10, 50, N_embed // 2])

    correlations = with_simplex_projection(
        Y_embed,
        X_aligned,
        lib_sizes,
        n_samples=5,
        library_pool=lib_pool,
        prediction_pool=pred_pool,
        sampler=make_seeded_sampler(42),
    )

    assert correlations.shape == (len(lib_sizes),)
    assert np.all(np.isfinite(correlations))
    # Correlations should not be all negative for a causal system
    assert np.any(correlations > 0), f"Expected at least some positive correlations, got {correlations}"
