import numpy as np

from edmkit.generate import double_pendulum, lorenz, mackey_glass, to_xy


class TestGeneratorContracts:
    def test_lorenz_returns_expected_shape_and_initial_state(self):
        x0 = np.array([1.0, 1.0, 1.0])
        t, x = lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0, X0=x0, dt=0.01, t_max=1)
        assert t.shape == (100,)
        assert x.shape == (100, 3)
        np.testing.assert_array_equal(x[0], x0)
        assert np.isfinite(x).all()

    def test_mackey_glass_keeps_history_at_initial_value_before_delay(self):
        t, x = mackey_glass(tau=4.0, n=10, beta=0.2, gamma=0.1, x0=0.9, dt=1.0, t_max=20)
        assert t.shape == (20,)
        np.testing.assert_allclose(x[:4], 0.9, atol=1e-12, rtol=1e-12)
        assert np.isfinite(x).all()

    def test_double_pendulum_returns_expected_shape_and_initial_state(self):
        x0 = np.array([0.5, 0.3, 0.0, 0.0])
        t, x = double_pendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81, X0=x0, dt=0.01, t_max=1)
        assert t.shape == (100,)
        assert x.shape == (100, 4)
        np.testing.assert_array_equal(x[0], x0)
        assert np.isfinite(x).all()

    def test_to_xy_preserves_link_lengths(self):
        theta1 = np.array([0.0, np.pi / 3])
        theta2 = np.array([np.pi / 4, np.pi / 6])
        x1, y1, x2, y2 = to_xy(2.0, 3.0, theta1, theta2)
        np.testing.assert_allclose(np.sqrt(x1**2 + y1**2), 2.0, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3.0, atol=1e-12, rtol=1e-12)
