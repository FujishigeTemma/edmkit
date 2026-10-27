import numpy as np

__all__ = ["duffing"]


def duffing(alpha: float, beta: float, delta: float, gamma: float, omega: float):
    """Vector field of the forced Duffing oscillator.

    Parameters
    ----------
    alpha : float
        Linear stiffness (typical: -1).
    beta : float
        Cubic stiffness (typical: 1).
    delta : float
        Damping coefficient (typical: 0.3).
    gamma : float
        Forcing amplitude (typical: 0.5 for chaos).
    omega : float
        Forcing angular frequency (typical: 1.2).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, v)`` of shape ``(2,)``, or ``(B, 2)`` for a batch.
    """

    def f(t: float, s: np.ndarray):
        x, v = np.moveaxis(s, -1, 0)
        return np.stack([v, (-alpha - beta * x**2) * x - delta * v + gamma * np.cos(omega * t)], axis=-1)

    return f
