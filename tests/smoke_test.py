import numpy as np

from edmkit import generate, lagged_embed, simplex_projection, smap
from edmkit.ccm import with_simplex_projection

from tests.helpers import make_seeded_sampler


def test_simplex_and_smap_end_to_end():
    """Simplex と S-Map のエンドツーエンド疎通テスト"""
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
    simplex_corr = np.corrcoef(simplex_pred, actual[:len(simplex_pred)])[0, 1]
    assert simplex_corr > 0, f"Expected positive correlation for sine wave, got {simplex_corr}"


def test_ccm_pipeline():
    """CCM パイプラインの疎通テスト"""
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


def test_generate_lorenz():
    """Lorenz 生成器の疎通テスト"""
    t, X = generate.lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=30,
    )
    assert t.ndim == 1 and X.ndim == 2
    assert len(t) == len(X)
    assert X.shape[1] == 3
    assert np.all(np.isfinite(X))


def test_generate_mackey_glass():
    """Mackey-Glass 生成器の疎通テスト"""
    t, x = generate.mackey_glass(
        tau=17.0,
        n=10,
        beta=0.2,
        gamma=0.1,
        x0=0.9,
        dt=1.0,
        t_max=200,
    )
    assert t.ndim == 1 and x.ndim == 1
    assert len(t) == len(x)
    assert np.all(np.isfinite(x))


def test_generate_double_pendulum():
    """Double Pendulum 生成器の疎通テスト"""
    t, X = generate.double_pendulum(
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=9.81,
        X0=np.array([np.pi / 4, np.pi / 6, 0.0, 0.0]),
        dt=0.01,
        t_max=10,
    )
    assert t.ndim == 1 and X.ndim == 2
    assert X.shape[1] == 4
    assert np.all(np.isfinite(X))
