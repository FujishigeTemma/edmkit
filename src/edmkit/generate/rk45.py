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
# quartic dense output (Shampine 1986): x(s + theta h) = x + h sum_j K_j sum_k P[j, k] theta^(k + 1)
P = np.array(
    [
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)
THETA_POWERS = np.arange(1, 5)


def rk45(
    f: Callable[[float, np.ndarray], np.ndarray],
    X0: np.ndarray,
    dt: float,
    t_max: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.

    Steps are chosen by the error alone, and the uniform grid of spacing ``dt`` is filled in by the method's dense output.
    A batch of initial conditions is integrated together, with steps that keep every member within the tolerance.

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side of the system, called with states of the shape of ``X0``.
    X0 : np.ndarray
        Initial condition of shape ``(D,)``, or ``(B, D)`` for a batch.
    dt : float
        Sampling time step.
    t_max : float
        End time, excluded from the grid.
    rtol : float
        Relative tolerance of the local error.
    atol : float
        Absolute tolerance of the local error.

    Returns
    -------
    t : np.ndarray
        Time array of shape ``(N,)``.
    X : np.ndarray
        Trajectory of shape ``(N, D)``, or ``(B, N, D)`` for a batch.

    Raises
    ------
    ValueError
        If ``X0`` is not of shape ``(D,)`` or ``(B, D)``.
    """
    X0 = np.asarray(X0, dtype=float)
    if X0.ndim not in (1, 2):
        raise ValueError(f"X0 must be of shape (D,) or (B, D), got X0.shape={X0.shape}")

    t = np.arange(0, t_max, dt)
    X = np.empty((len(t), *X0.shape))
    X[0] = X0

    K = np.empty((7, *X0.shape))
    Kf = K.reshape(7, -1)  # flat view of K, so the stage sums are plain matrix products
    K[0] = f(0.0, X0)
    s, x, h, i = 0.0, X0, dt, 1

    while i < len(t):
        step = min(h, t[-1] - s)
        for j in range(1, 7):
            # the last stage is evaluated at x_new, so an accepted step hands K[6] on as the next K[0]
            x_new = x + step * (A[j, :j] @ Kf[:j]).reshape(x.shape)
            K[j] = f(s + C[j] * step, x_new)
        error = np.max(np.abs(step * (E @ Kf).reshape(x.shape)) / (atol + rtol * np.maximum(np.abs(x), np.abs(x_new))))

        if error <= 1:
            n = np.searchsorted(t, s + step, side="right")
            if n > i:
                theta = (t[i:n] - s) / step
                X[i:n] = x + step * (theta[:, None] ** THETA_POWERS @ (P.T @ Kf)).reshape(n - i, *x.shape)
            s, x, i = s + step, x_new, n
            K[0] = K[6]
        h = step * (10 if error == 0 else min(10, max(0.2, 0.9 * error**-0.2)))

    return t, np.moveaxis(X, 0, -2)
