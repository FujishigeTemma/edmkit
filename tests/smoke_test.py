"""Smoke test — パッケージのインポートと最小限のAPI呼び出しのみ確認。

CI の release ワークフローで `uv run --isolated --no-project` で実行されるため、
tests パッケージや dev 依存には一切依存してはならない。
"""

import numpy as np

from edmkit import generate
from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap


def test_basic_call():
    x = np.sin(np.linspace(0, 4 * np.pi, 60))
    emb = lagged_embed(x, tau=1, e=3)
    assert emb.ndim == 2

    pred = simplex_projection(emb[:30], emb[1:31, 0], emb[30:])
    assert np.all(np.isfinite(pred))

    pred = smap(emb[:30], emb[1:31, 0], emb[30:], theta=1.0)
    assert np.all(np.isfinite(pred))


def test_ccm_basic_call():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 4 * np.pi, 100))
    emb = lagged_embed(x, tau=1, e=2)
    target = x[1:]

    n = emb.shape[0]
    corrs = with_simplex_projection(
        emb,
        target,
        lib_sizes=np.array([10, 30]),
        n_samples=3,
        library_pool=np.arange(n // 2),
        prediction_pool=np.arange(n // 2, n),
        sampler=lambda pool, size: rng.choice(pool, size=size, replace=True),
    )
    assert corrs.shape == (2,)
    assert np.all(np.isfinite(corrs))


def test_generators():
    _, X = generate.lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8 / 3,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=1,
    )
    assert X.ndim == 2 and np.all(np.isfinite(X))

    _, x = generate.mackey_glass(tau=17.0, n=10, beta=0.2, gamma=0.1, x0=0.9, dt=1.0, t_max=50)
    assert x.ndim == 1 and np.all(np.isfinite(x))

    _, X = generate.double_pendulum(
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=9.81,
        X0=np.array([0.5, 0.3, 0.0, 0.0]),
        dt=0.01,
        t_max=1,
    )
    assert X.ndim == 2 and np.all(np.isfinite(X))


if __name__ == "__main__":
    test_basic_call()
    test_ccm_basic_call()
    test_generators()
    print("smoke tests passed")
