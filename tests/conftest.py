import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

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


@pytest.fixture(scope="module")
def bounded_linear_series() -> np.ndarray:
    """Bounded linear system used to test near-linear forecasting behavior."""
    n = 500
    x = np.zeros(n)
    x[0] = 0.5
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + 0.05 * np.sin(0.1 * i)
    return x


@pytest.fixture(scope="module")
def logistic_map() -> np.ndarray:
    """Chaotic logistic map used where nonlinear locality should help."""
    n_total = 550
    x = np.zeros(n_total)
    x[0] = 0.4
    for i in range(1, n_total):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    return x[50:]


@pytest.fixture(scope="module")
def lorenz_series() -> np.ndarray:
    """Lorenz attractor after a transient, for state-space tests."""
    from edmkit.generate import lorenz

    _, x = lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=50,
    )
    return x[2000 : 2000 + 500]


@pytest.fixture(scope="module")
def causal_pair() -> tuple[np.ndarray, np.ndarray]:
    """Coupled logistic maps with one-way coupling X -> Y."""
    n_total = 1050
    rx, ry, bxy = 3.8, 3.5, 0.02
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    x[0], y[0] = 0.4, 0.2
    for i in range(1, n_total):
        x[i] = x[i - 1] * (rx - rx * x[i - 1])
        y[i] = y[i - 1] * (ry - ry * y[i - 1]) + bxy * x[i - 1]
    return x[50:], y[50:]


@pytest.fixture(scope="module")
def independent_pair() -> tuple[np.ndarray, np.ndarray]:
    """Independent logistic maps used as a null-causality baseline."""
    n_total = 1050
    r1, r2 = 3.8, 3.7
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    x[0] = 0.4
    y[0] = 0.2
    for i in range(1, n_total):
        x[i] = x[i - 1] * (r1 - r1 * x[i - 1])
        y[i] = y[i - 1] * (r2 - r2 * y[i - 1])
    return x[50:], y[50:]
