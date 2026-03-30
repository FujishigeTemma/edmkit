import numpy as np

from edmkit.generate import lorenz, mackey_glass


class TestGeneratorTrends:
    def test_lorenz_shows_sensitivity_to_initial_conditions(self):
        _, x1 = lorenz(
            sigma=10.0,
            rho=28.0,
            beta=8.0 / 3.0,
            X0=np.array([1.0, 1.0, 1.0]),
            dt=0.01,
            t_max=20,
        )
        _, x2 = lorenz(
            sigma=10.0,
            rho=28.0,
            beta=8.0 / 3.0,
            X0=np.array([1.0, 1.0, 1.0001]),
            dt=0.01,
            t_max=20,
        )
        initial_gap = np.linalg.norm(x1[0] - x2[0])
        final_gap = np.linalg.norm(x1[-1] - x2[-1])
        # Lorenz trajectories should diverge substantially from a tiny perturbation.
        assert final_gap > initial_gap * 100

    def test_mackey_glass_stays_positive_for_standard_parameters(self):
        _, x = mackey_glass(tau=17.0, n=10, beta=0.2, gamma=0.1, x0=1.2, dt=1.0, t_max=200)
        assert np.min(x) >= 0.0
        # Standard parameters stay in a bounded positive regime in this implementation.
        assert np.max(x) < 2.5
