import numpy as np

__all__ = ["fitzhugh_nagumo"]


def fitzhugh_nagumo(a: float, b: float, epsilon: float, I: float):
    """Vector field of the FitzHugh-Nagumo neuron model.

    Parameters
    ----------
    a : float
        Offset of the recovery variable (typical: 0.7).
    b : float
        Damping of the recovery variable (typical: 0.8).
    epsilon : float
        Time scale of the recovery variable (typical: 0.08).
    I : float
        Applied current (typical: 0.5 for spiking).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(v, w)`` of shape ``(2,)``, or ``(B, 2)`` for a batch.
    """

    def f(t: float, x: np.ndarray):
        v, w = np.moveaxis(x, -1, 0)
        return np.stack([v - v**3 / 3 - w + I, epsilon * (v + a - b * w)], axis=-1)

    return f
