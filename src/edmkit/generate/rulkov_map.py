import numpy as np

__all__ = ["rulkov_map"]


def rulkov_map(alpha: float, mu: float, sigma: float, epsilon: float, X0: np.ndarray, n_max: int):
    """Generate a time series of globally coupled Rulkov map neurons.

    Parameters
    ----------
    alpha : float
        Nonlinearity of the fast variable (typical: 4.1 for chaotic bursting).
    mu : float
        Time scale of the slow variable (typical: 0.001).
    sigma : float
        Drive of the slow variable (typical: -1.5).
    epsilon : float
        Coupling through the mean of the fast variables (typical: 0).
    X0 : np.ndarray
        Initial condition of shape ``(N, 2)``, one ``(x, y)`` per unit.
    n_max : int
        Number of steps.

    Returns
    -------
    n : np.ndarray
        Step array.
    X : np.ndarray
        Time series of shape ``(n_max, N, 2)``.
    """
    X = np.zeros((n_max, *np.shape(X0)))
    X[0] = X0

    for i in range(n_max - 1):
        x, y = X[i].T
        X[i + 1, :, 0] = alpha / (1 + x**2) + y + epsilon * x.mean()
        X[i + 1, :, 1] = y - mu * (x - sigma)

    return np.arange(n_max), X
