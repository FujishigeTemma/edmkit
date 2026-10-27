import numpy as np

__all__ = ["lotka_volterra"]


def lotka_volterra(rho: np.ndarray, sigma: float | np.ndarray, S: float | np.ndarray):
    """Vector field of the generalized Lotka-Volterra system ``dx/dt = x (sigma - rho x) + S``.

    Parameters
    ----------
    rho : np.ndarray
        Interaction matrix of shape ``(N, N)``; ``rho[i, j]`` is the effect of species ``j`` on species ``i``.
    sigma : float | np.ndarray
        Growth rate, scalar or of shape ``(N,)`` (typical: 1).
    S : float | np.ndarray
        Constant input, scalar or of shape ``(N,)`` (typical: 0; small positive values keep winnerless competition switching).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state of shape ``(N,)``, or ``(B, N)`` for a batch.
    """

    def f(t: float, x: np.ndarray):
        return x * (sigma - x @ rho.T) + S

    return f
