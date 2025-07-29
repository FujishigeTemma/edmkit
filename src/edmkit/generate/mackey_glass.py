import numpy as np


def mackey_glass(
    tau: float, n: int, beta: float, gamma: float, x0: float, dt: float, t_max: int
):
    def f(x, x_tau):
        return beta * x_tau / (1 + x_tau**n) - gamma * x

    t = np.arange(0, t_max, dt)
    x = np.zeros_like(t)

    tau_idx = int(tau / dt)
    x[:tau_idx] = x0

    for i in range(tau_idx, len(t)):
        x[i] = x[i - 1] + dt * f(x[i - 1], x[i - tau_idx])

    return t, x
