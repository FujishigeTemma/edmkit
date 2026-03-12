import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

# --- hypothesis profiles ---
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=500,
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# --- Shared fixtures ---


@pytest.fixture(scope="module")
def bounded_linear_series():
    """有界な線形系: x[t+1] = 0.9*x[t] + 0.05*sin(0.1*t), N=500"""
    N = 500
    x = np.zeros(N)
    x[0] = 0.5
    for i in range(1, N):
        x[i] = 0.9 * x[i - 1] + 0.05 * np.sin(0.1 * i)
    return x


@pytest.fixture(scope="module")
def sine_wave():
    """x[t] = sin(2pi * t / 20), N=100"""
    t = np.arange(100)
    return np.sin(2 * np.pi * t / 20)


@pytest.fixture(scope="module")
def logistic_map():
    """x[t+1] = 3.8 * x[t] * (1 - x[t]), x[0] = 0.4, N=500
    先頭50点を過渡状態として破棄。"""
    N_total = 550
    x = np.zeros(N_total)
    x[0] = 0.4
    for i in range(1, N_total):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    return x[50:]


@pytest.fixture(scope="module")
def lorenz_series():
    """Lorenz system sigma=10, rho=28, beta=8/3, dt=0.01, N=500
    先頭 2000 steps (20 time units) を破棄。"""
    from edmkit.generate import lorenz

    _, X = lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=50,
    )
    return X[2000 : 2000 + 500]  # 過渡除去後 500 points


@pytest.fixture(scope="module")
def causal_pair():
    """X->Y の一方向因果がある結合 Logistic map, N=1000
    rx=3.8, ry=3.5, Bxy=0.02, Byx=0"""
    N_total = 1050
    rx, ry, Bxy = 3.8, 3.5, 0.02
    X = np.zeros(N_total)
    Y = np.zeros(N_total)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N_total):
        X[i] = X[i - 1] * (rx - rx * X[i - 1])
        Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]
    return X[50:], Y[50:]


@pytest.fixture(scope="module")
def independent_pair():
    """独立な2つの Logistic map (r1=3.8, r2=3.7), N=1000"""
    N_total = 1050
    r1, r2 = 3.8, 3.7
    X = np.zeros(N_total)
    Y = np.zeros(N_total)
    X[0] = 0.4
    Y[0] = 0.2
    for i in range(1, N_total):
        X[i] = X[i - 1] * (r1 - r1 * X[i - 1])
        Y[i] = Y[i - 1] * (r2 - r2 * Y[i - 1])
    return X[50:], Y[50:]


@pytest.fixture(scope="module")
def bidirectional_pair():
    """双方向因果 Logistic map, N=1000
    rx=3.8, ry=3.5, Bxy=0.02, Byx=0.02"""
    N_total = 1050
    rx, ry = 3.8, 3.5
    Bxy, Byx = 0.02, 0.02
    X = np.zeros(N_total)
    Y = np.zeros(N_total)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N_total):
        X[i] = X[i - 1] * (rx - rx * X[i - 1]) + Byx * Y[i - 1]
        Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]
    return X[50:], Y[50:]
