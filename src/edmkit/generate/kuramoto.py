import numpy as np

__all__ = ["kuramoto"]


def kuramoto(omega: np.ndarray, K: np.ndarray):
    """Vector field of the Kuramoto model ``dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)``.

    Parameters
    ----------
    omega : np.ndarray
        Natural frequencies of shape ``(N,)``.
    K : np.ndarray
        Coupling of shape ``(N, N)``; ``K[i, j]`` is the strength with which oscillator ``j`` drives oscillator ``i`` (typical: ``k / N`` everywhere).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for unwrapped phases of shape ``(N,)``, or ``(B, N)`` for a batch.
    """

    def f(t: float, x: np.ndarray):
        s, c = np.sin(x), np.cos(x)
        return omega + c * (s @ K.T) - s * (c @ K.T)

    return f
