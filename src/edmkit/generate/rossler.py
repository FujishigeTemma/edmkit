import numpy as np

__all__ = ["rossler"]


def rossler(a: float, b: float, c: float):
    """Vector field of the Rössler system.

    Parameters
    ----------
    a : float
        Spiral growth rate (typical: 0.2).
    b : float
        Constant injection into z (typical: 0.2).
    c : float
        Folding threshold (typical: 5.7).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.
    """

    def f(t: float, s: np.ndarray):
        x, y, z = np.moveaxis(s, -1, 0)
        return np.stack([-y - z, x + a * y, b + (x - c) * z], axis=-1)

    return f
