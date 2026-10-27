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
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.
    """

    def f(t: float, s: np.ndarray):
        x, y, z = np.moveaxis(s, -1, 0)
        return np.stack([sigma * (y - x), rho * x - y - x * z, x * y - beta * z], axis=-1)

    return f
