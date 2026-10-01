import numpy as np

__all__ = ["lorenz96"]


def lorenz96(F: float):
    """Vector field of the Lorenz 96 system.

    Parameters
    ----------
    F : float
        Forcing constant (typical: 8 for chaos).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state of shape ``(D,)`` with ``D >= 4``.
    """

    def f(t: float, x: np.ndarray):
        # p[i + 2] is x[i] on the ring, so this is (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + F
        p = np.concatenate([x[-2:], x, x[:1]])
        return (p[3:] - p[:-3]) * p[1:-2] - x + F

    return f
