import numpy as np

__all__ = ["ctrnn"]


def ctrnn(g: float, J: np.ndarray):
    """Vector field of a continuous-time recurrent neural network ``dx/dt = -x + g J tanh(x)``.

    Parameters
    ----------
    g : float
        Gain (chaos for ``g > 1`` when ``J`` is large and random with variance ``1 / N``).
    J : np.ndarray
        Connectivity of shape ``(N, N)``; ``J[i, j]`` is the weight with which unit ``j`` drives unit ``i``.

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state of shape ``(N,)``.
    """

    def f(t: float, x: np.ndarray):
        return -x + g * J @ np.tanh(x)

    return f
