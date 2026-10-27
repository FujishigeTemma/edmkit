import numpy as np

__all__ = ["chua"]


def chua(alpha: float, beta: float, m0: float, m1: float):
    """Vector field of Chua's circuit, with the piecewise-linear diode ``h(x) = m1 x + (m0 - m1) (|x + 1| - |x - 1|) / 2``.

    Parameters
    ----------
    alpha : float
        Ratio of capacitances (typical: 15.6).
    beta : float
        Ratio of inductance to resistance (typical: 28).
    m0 : float
        Diode slope for ``|x| < 1`` (typical: -8/7).
    m1 : float
        Diode slope for ``|x| > 1`` (typical: -5/7).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.
    """

    def f(t: float, s: np.ndarray):
        x, y, z = np.moveaxis(s, -1, 0)
        h = m1 * x + (m0 - m1) * np.clip(x, -1, 1)  # (|x + 1| - |x - 1|) / 2 is clip(x, -1, 1)
        return np.stack([alpha * (y - x - h), x - y + z, -beta * y], axis=-1)

    return f
