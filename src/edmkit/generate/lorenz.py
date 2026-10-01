import numpy as np

__all__ = ["lorenz"]


def lorenz(sigma: float, rho: float, beta: float):
    """Vector field of the Lorenz system.

    Parameters
    ----------
    sigma : float
        Prandtl number (typical: 10).
    rho : float
        Rayleigh number (typical: 28).
    beta : float
        Geometric factor (typical: 8/3).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``.
    """

    def f(t: float, x: np.ndarray):
        return np.array([[-sigma, sigma, 0], [rho, -1, -x[0]], [0, x[0], -beta]]) @ x

    return f
