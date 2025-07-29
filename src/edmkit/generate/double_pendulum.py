import numpy as np


def double_pendulum(
    m1: float,
    m2: float,
    L1: float,
    L2: float,
    g: float,
    X0: np.ndarray,  # (theta1, theta2, omega1, omega2)
    dt: float,
    t_max: int,
):
    def f(x: np.ndarray):
        theta1, theta2, omega1, omega2 = x
        delta = theta1 - theta2

        denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        dtheta1_dt = omega1
        dtheta2_dt = omega2

        domega1_dt = (
            -g * (2 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2 * theta2)
            - 2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
        ) / (L1 * denom)

        domega2_dt = (
            2
            * np.sin(delta)
            * (omega1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * L2 * m2 * np.cos(delta))
        ) / (L2 * denom)

        return np.array([dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt])

    t = np.arange(0, t_max, dt)
    X = np.zeros((len(t), 4))
    X[0] = X0

    for i in range(1, len(t)):
        X[i] = X[i - 1] + dt * f(X[i - 1])

    return t, X


def to_xy(L1: float, L2: float, theta1: np.ndarray, theta2: np.ndarray):
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2
