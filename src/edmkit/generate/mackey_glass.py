import numpy as np


def mackey_glass(tau: float, n: int, beta: float, gamma: float, x0: float, dt: float, t_max: int):
    """Generate a Mackey-Glass chaotic time series via forward Euler integration.

    Parameters
    ----------
    tau : float
        Delay parameter (typical: 17 for chaos).
    n : int
        Nonlinearity exponent (typical: 10).
    beta : float
        Feedback strength (typical: 0.2).
    gamma : float
        Decay rate (typical: 0.1).
    x0 : float
        Initial condition.
    dt : float
        Integration time step.
    t_max : int
        Maximum time.

    Returns
    -------
    t : np.ndarray
        Time array.
    x : np.ndarray
        1D time series.
    """

    def f(x, x_tau):
        return beta * x_tau / (1 + x_tau**n) - gamma * x

    t = np.arange(0, t_max, dt)
    x = np.zeros_like(t)

    tau_idx = int(tau / dt)
    x[:tau_idx] = x0

    for i in range(tau_idx, len(t)):
        x[i] = x[i - 1] + dt * f(x[i - 1], x[i - tau_idx])

    return t, x
