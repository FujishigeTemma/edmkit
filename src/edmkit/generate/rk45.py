from collections.abc import Callable

import numpy as np

__all__ = ["rk45"]

# Dormand-Prince 5(4)
A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
)
C = A.sum(axis=1)
E = np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


def rk45(
    f: Callable[[float, np.ndarray], np.ndarray],
    X0: np.ndarray,
    dt: float,
    t_max: int,
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.

    The step size adapts between samples, while the output stays on the uniform grid of spacing ``dt``.

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side of the system.
    X0 : np.ndarray
        Initial condition of shape ``(D,)``.
    dt : float
        Sampling time step.
    t_max : int
        Maximum time.
    rtol : float
        Relative tolerance of the local error.
    atol : float
        Absolute tolerance of the local error.

    Returns
    -------
    t : np.ndarray
        Time array.
    X : np.ndarray
        Trajectory of shape ``(N, D)``.
    """
    t = np.arange(0, t_max, dt)
    X = np.zeros((len(t), len(X0)))
    X[0] = X0

    K = np.zeros((7, len(X0)))
    h = dt

    for i in range(1, len(t)):
        s, x = t[i - 1], X[i - 1]

        while s < t[i]:
            last = h >= t[i] - s
            step = t[i] - s if last else h

            for j in range(7):
                K[j] = f(s + C[j] * step, x + step * (A[j, :j] @ K[:j]))

            x_new = x + step * (A[6] @ K[:6])
            error = np.max(np.abs(step * (E @ K)) / (atol + rtol * np.maximum(np.abs(x), np.abs(x_new))))
            accepted = error <= 1

            if accepted:
                s, x = t[i] if last else s + step, x_new
            if not (accepted and last):
                h = step * min(10, max(0.2, 0.9 * error**-0.2)) if error > 0 else step * 10

        X[i] = x

    return t, X
