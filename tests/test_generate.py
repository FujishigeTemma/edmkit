import numpy as np
import pytest

from edmkit.generate import double_pendulum, lorenz, mackey_glass, to_xy
from edmkit.util import autocorrelation


# ===========================================================================
# Module-scope fixtures
# ===========================================================================
@pytest.fixture(scope="module")
def lorenz_result():
    return lorenz(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        X0=np.array([1.0, 1.0, 1.0]),
        dt=0.01,
        t_max=30,
    )


@pytest.fixture(scope="module")
def mackey_glass_result():
    return mackey_glass(
        tau=17.0,
        n=10,
        beta=0.2,
        gamma=0.1,
        x0=0.9,
        dt=1.0,
        t_max=500,
    )


@pytest.fixture(scope="module")
def double_pendulum_result():
    return double_pendulum(
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=9.81,
        X0=np.array([np.pi / 2, np.pi / 4, 0.0, 0.0]),
        dt=0.01,
        t_max=10,
    )


# ===========================================================================
# 3.6 Common tests
# ===========================================================================
@pytest.fixture(
    params=["lorenz", "mackey_glass", "double_pendulum"],
)
def generator_result(request, lorenz_result, mackey_glass_result, double_pendulum_result):
    results = {
        "lorenz": lorenz_result,
        "mackey_glass": mackey_glass_result,
        "double_pendulum": double_pendulum_result,
    }
    return request.param, results[request.param]


class TestGeneratorCommon:
    def test_output_shape(self, generator_result):
        """t is 1D, state array has matching length"""
        name, (t, X) = generator_result
        assert t.ndim == 1
        if name == "mackey_glass":
            assert X.ndim == 1
        else:
            assert X.ndim == 2
        assert len(t) == len(X)

    def test_time_array(self, generator_result):
        """Time array starts at 0 with uniform spacing dt"""
        name, (t, _) = generator_result
        dt_map = {"lorenz": 0.01, "mackey_glass": 1.0, "double_pendulum": 0.01}
        np.testing.assert_allclose(np.diff(t), dt_map[name], atol=1e-14)
        np.testing.assert_allclose(t[0], 0.0, atol=1e-14)

    def test_deterministic(self, generator_result):
        """Same parameters yield identical results"""
        name, (t1, X1) = generator_result
        if name == "lorenz":
            t2, X2 = lorenz(
                sigma=10.0,
                rho=28.0,
                beta=8.0 / 3.0,
                X0=np.array([1.0, 1.0, 1.0]),
                dt=0.01,
                t_max=30,
            )
        elif name == "mackey_glass":
            t2, X2 = mackey_glass(
                tau=17.0,
                n=10,
                beta=0.2,
                gamma=0.1,
                x0=0.9,
                dt=1.0,
                t_max=500,
            )
        else:
            t2, X2 = double_pendulum(
                m1=1.0,
                m2=1.0,
                L1=1.0,
                L2=1.0,
                g=9.81,
                X0=np.array([np.pi / 2, np.pi / 4, 0.0, 0.0]),
                dt=0.01,
                t_max=10,
            )
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(X1, X2)

    def test_no_nan_or_inf(self, generator_result):
        """Output contains no NaN or Inf"""
        _, (_, X) = generator_result
        assert np.all(np.isfinite(X))


# ===========================================================================
# Lorenz-specific tests
# ===========================================================================
class TestLorenzSpecific:
    def test_attractor_boundedness(self):
        """Lorenz attractor is bounded (dissipative system) -- 2-12"""
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        _, X = lorenz(
            sigma=sigma,
            rho=rho,
            beta=beta,
            X0=np.array([1.0, 1.0, 1.0]),
            dt=0.01,
            t_max=50,
        )
        # Settled trajectory after 20 time units
        X_settled = X[2000:]
        assert np.all(np.abs(X_settled[:, 0]) < 50)
        assert np.all(np.abs(X_settled[:, 1]) < 50)
        # Mathematical upper bound for z: rho + sigma^2 / beta ~ 65.5
        z_upper = rho + sigma**2 / beta  # ~65.5
        assert np.all(X_settled[:, 2] < z_upper + 1)  # ~66.5 with small margin
        assert np.all(X_settled[:, 2] > 0)

    def test_sensitive_dependence(self):
        """Small perturbation in initial conditions leads to divergence -- 2-13"""
        kwargs = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0, dt=0.01)
        _, X1 = lorenz(X0=np.array([1.0, 1.0, 1.0]), t_max=30, **kwargs)
        _, X2 = lorenz(X0=np.array([1.0 + 1e-10, 1.0, 1.0]), t_max=30, **kwargs)
        # Early time: trajectories are very close (tightened from 1e-5 to 1e-8)
        assert np.max(np.abs(X1[:100] - X2[:100])) < 1e-8
        assert np.max(np.abs(X1[-500:] - X2[-500:])) > 1.0


# ===========================================================================
# Mackey-Glass-specific tests
# ===========================================================================
class TestMackeyGlassSpecific:
    def test_positivity(self):
        """Positive initial value yields always-positive trajectory"""
        _, x = mackey_glass(
            tau=17.0,
            n=10,
            beta=0.2,
            gamma=0.1,
            x0=0.9,
            dt=1.0,
            t_max=500,
        )
        assert np.all(x > 0)

    def test_chaos_for_large_tau(self):
        """tau > 17 produces irregular behavior -- 2-11"""
        _, x = mackey_glass(
            tau=30.0,
            n=10,
            beta=0.2,
            gamma=0.1,
            x0=0.9,
            dt=1.0,
            t_max=2000,
        )
        x_settled = x[500:]
        acf = autocorrelation(x_settled, max_lag=200)
        # Statistical threshold for significance: 2 / sqrt(N)
        # A periodic signal would never dip below this; a chaotic one will
        threshold = 2 / np.sqrt(len(x_settled))
        assert np.min(np.abs(acf[50:])) < threshold


# ===========================================================================
# Double Pendulum-specific tests
# ===========================================================================
class TestDoublePendulumSpecific:
    def test_state_dimension(self):
        """Output has 4 state dimensions (theta1, theta2, omega1, omega2)"""
        _, X = double_pendulum(
            m1=1.0,
            m2=1.0,
            L1=1.0,
            L2=1.0,
            g=9.81,
            X0=np.array([np.pi / 2, np.pi / 4, 0.0, 0.0]),
            dt=0.01,
            t_max=10,
        )
        assert X.shape[1] == 4

    def test_to_xy_conversion(self):
        """Angle-to-Cartesian conversion is geometrically correct"""
        L1, L2 = 1.5, 1.0
        theta1 = np.array([0.0, np.pi / 2, np.pi])
        theta2 = np.array([0.0, np.pi / 2, np.pi])

        x1, y1, x2, y2 = to_xy(L1, L2, theta1, theta2)

        # theta1=0: x1=0, y1=-L1
        np.testing.assert_allclose(x1[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(y1[0], -L1, atol=1e-14)
        # theta1=pi/2: x1=L1, y1~0
        np.testing.assert_allclose(x1[1], L1, atol=1e-14)
        np.testing.assert_allclose(y1[1], 0.0, atol=1e-14)
        # x2 = x1 + L2*sin(theta2), y2 = y1 - L2*cos(theta2)
        np.testing.assert_allclose(x2, x1 + L2 * np.sin(theta2), atol=1e-14)
        np.testing.assert_allclose(y2, y1 - L2 * np.cos(theta2), atol=1e-14)
