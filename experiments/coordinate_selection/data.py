"""Generate the benchmark trajectories once, so both environments read identical arrays."""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from edmkit.util import autocorrelation


def lorenz(t, z):
    return [10 * (z[1] - z[0]), z[0] * (28 - z[2]) - z[1], z[0] * z[1] - 8 / 3 * z[2]]


def rossler(t, z):
    return [-z[1] - z[2], z[0] + 0.2 * z[1], 0.2 + z[2] * (z[0] - 5.7)]


SYSTEMS = {"lorenz": (lorenz, 0.02, [1.0, 1.0, 20.0]), "rossler": (rossler, 0.1, [1.0, 1.0, 0.0])}
# name: (system, observed variables, noise as a fraction of each variable's standard deviation)
CASES = {
    "lorenz-x": ("lorenz", [0], 0.0),
    "lorenz-x-noise": ("lorenz", [0], 0.05),
    "rossler-x": ("rossler", [0], 0.0),
    "rossler-x-noise": ("rossler", [0], 0.05),
    "lorenz-xyz": ("lorenz", [0, 1, 2], 0.0),
}
# PECUZAL in pecuzal-embedding costs O(TRAIN^2) interpreted operations per horizon in its first cycle, which bounds TRAIN.
TRAIN, TEST, SEEDS = 2000, 5000, 3

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, (system, observed, noise) in CASES.items():
        f, dt, z0 = SYSTEMS[system]
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            burn = 2000
            times = np.arange(burn + TRAIN + TEST) * dt
            sol = solve_ivp(f, (0, times[-1]), np.array(z0) + rng.normal(scale=0.1, size=3), t_eval=times, method="DOP853", rtol=1e-10, atol=1e-10)
            state = sol.y.T[burn:]
            obs = state[:, observed]
            obs = (obs - obs.mean(axis=0)) / obs.std(axis=0)
            clean = obs
            obs = clean + noise * rng.normal(size=obs.shape)
            acf = [autocorrelation(obs[:TRAIN, i], 200) for i in range(obs.shape[1])]
            width = max(int(np.argmax(a < 1 / np.e)) for a in acf)
            np.savez(
                out / f"{name}-{seed}.npz",
                train=obs[:TRAIN],
                test=obs[TRAIN:],
                train_clean=clean[:TRAIN],
                test_clean=clean[TRAIN:],
                train_state=state[:TRAIN],
                test_state=state[TRAIN:],
                width=width,
            )
            print(name, seed, "width", width)
