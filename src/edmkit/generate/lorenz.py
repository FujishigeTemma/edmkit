import numpy as np


def lorenz(
    sigma: float,
    rho: float,
    beta: float,
    X0: np.ndarray,
    dt: float,
    t_max: int,
):
    """Generate a Lorenz system trajectory via forward Euler integration.

    Parameters
    ----------
    sigma : float
        Prandtl number (typical: 10).
    rho : float
        Rayleigh number (typical: 28).
    beta : float
        Geometric factor (typical: 8/3).
    X0 : np.ndarray
        Initial condition of shape ``(3,)``.
    dt : float
        Integration time step.
    t_max : int
        Maximum time.

    Returns
    -------
    t : np.ndarray
        Time array.
    X : np.ndarray
        Trajectory of shape ``(N, 3)`` for ``(x, y, z)``.
    """

    def f(x: np.ndarray):
        return np.array([[-sigma, sigma, 0], [rho, -1, -x[0]], [0, x[0], -beta]]) @ x

    t = np.arange(0, t_max, dt)
    X = np.zeros((len(t), 3))
    X[0] = X0

    for i in range(1, len(t)):
        X[i] = X[i - 1] + dt * f(X[i - 1])

    return t, X
