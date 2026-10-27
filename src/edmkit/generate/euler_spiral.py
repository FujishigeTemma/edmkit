import numpy as np

__all__ = ["euler_spiral"]


def euler_spiral(a: float):
    """Vector field of the Euler spiral (clothoid).

    Parameters
    ----------
    a : float
        Rate at which the curvature grows (typical: 1).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, y, theta)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.
    """

    def f(t: float, x: np.ndarray):
        theta = x[..., 2]
        return np.stack([np.cos(theta), np.sin(theta), np.full_like(theta, a * t)], axis=-1)

    return f
