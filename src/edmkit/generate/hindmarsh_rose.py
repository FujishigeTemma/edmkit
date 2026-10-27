import numpy as np

__all__ = ["hindmarsh_rose"]


def hindmarsh_rose(a: float, b: float, c: float, d: float, r: float, s: float, x_R: float, I: float):
    """Vector field of the Hindmarsh-Rose neuron model.

    Parameters
    ----------
    a : float
        Cubic coefficient of the membrane potential (typical: 1).
    b : float
        Quadratic coefficient of the membrane potential (typical: 3).
    c : float
        Constant drive of the recovery variable (typical: 1).
    d : float
        Quadratic coupling of the potential into the recovery variable (typical: 5).
    r : float
        Time scale of the slow adaptation current (typical: 0.006).
    s : float
        Adaptation strength (typical: 4).
    x_R : float
        Resting potential (typical: -1.6).
    I : float
        Applied current (typical: 3.25 for chaotic bursting).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch: membrane potential, recovery variable and adaptation current.
    """

    def f(t: float, x: np.ndarray):
        v, w, u = np.moveaxis(x, -1, 0)
        return np.stack([w - a * v**3 + b * v**2 - u + I, c - d * v**2 - w, r * (s * (v - x_R) - u)], axis=-1)

    return f
