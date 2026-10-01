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
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``.
    """

    def f(t: float, x: np.ndarray):
        return np.array([[0, -1, -1], [1, a, 0], [0, 0, x[0] - c]]) @ x + np.array([0, 0, b])

    return f
