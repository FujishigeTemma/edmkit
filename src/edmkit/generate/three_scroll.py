import numpy as np

__all__ = ["three_scroll"]


def three_scroll(a: float, k: float, c: float, d: float, e: float, f: float):
    """Vector field of Li's three-scroll system.

    Parameters
    ----------
    a : float
        Coupling of ``x`` towards ``y`` (typical: 40).
    k : float
        Gain of ``x`` in ``dy/dt`` (typical: 55).
    c : float
        Gain of ``z`` in ``dz/dt`` (typical: 1.9 for chaos; Li's 11/6 gives a limit cycle).
    d : float
        Coefficient of ``x z`` in ``dx/dt`` (typical: 0.16).
    e : float
        Coefficient of ``x^2`` in ``dz/dt`` (typical: 0.65).
    f : float
        Gain of ``y`` in ``dy/dt`` (typical: 20).

    Returns
    -------
    rhs : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``rhs(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.
    """

    def rhs(t: float, s: np.ndarray):
        x, y, z = np.moveaxis(s, -1, 0)
        return np.stack([a * (y - x) + d * x * z, k * x + f * y - x * z, -e * x**2 + x * y + c * z], axis=-1)

    return rhs
