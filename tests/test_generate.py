from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest

from edmkit.generate import double_pendulum, lorenz, mackey_glass, to_xy


class Trajectory(NamedTuple):
    run: Callable[[], tuple[np.ndarray, np.ndarray]]
    time: np.ndarray
    shape: tuple[int, ...]
    initial: np.ndarray | float
    step_index: int
    step: np.ndarray | float
    history: np.ndarray | None = None


LORENZ_X0 = np.array([1.0, 2.0, 3.0])
LORENZ_DT = 0.01

MG_X0 = 0.9
MG_DT = 1.0

PENDULUM_THETA = 0.4
PENDULUM_DT = 0.01
PENDULUM_X0 = np.array([PENDULUM_THETA, PENDULUM_THETA, 0.0, 0.0])

TRAJECTORIES: dict[str, Trajectory] = {
    "lorenz-euler-step": Trajectory(
        partial(lorenz, sigma=10.0, rho=28.0, beta=8.0 / 3.0, X0=LORENZ_X0, dt=LORENZ_DT, t_max=1),
        np.arange(0, 1, LORENZ_DT),
        (100, 3),
        LORENZ_X0,
        1,
        LORENZ_X0 + LORENZ_DT * np.array([10.0, 23.0, -6.0]),
    ),
    "mackey-glass-delay-step": Trajectory(
        partial(mackey_glass, tau=2.0, n=10, beta=0.2, gamma=0.1, x0=MG_X0, dt=MG_DT, t_max=6),
        np.arange(0, 6, MG_DT),
        (6,),
        MG_X0,
        2,
        MG_X0 + MG_DT * (0.2 * MG_X0 / (1.0 + MG_X0**10) - 0.1 * MG_X0),
        np.full(2, MG_X0),
    ),
    "double-pendulum-euler-step": Trajectory(
        partial(double_pendulum, m1=1.2, m2=0.8, L1=2.0, L2=3.0, g=9.81, X0=PENDULUM_X0, dt=PENDULUM_DT, t_max=1),
        np.arange(0, 1, PENDULUM_DT),
        (100, 4),
        PENDULUM_X0,
        1,
        PENDULUM_X0 + PENDULUM_DT * np.array([0.0, 0.0, -9.81 * np.sin(PENDULUM_THETA) / 2.0, 0.0]),
    ),
}


@pytest.mark.parametrize("case", TRAJECTORIES.values(), ids=TRAJECTORIES.keys())
def test_trajectory(case: Trajectory) -> None:
    time, state = case.run()

    np.testing.assert_array_equal(time, case.time)
    assert state.shape == case.shape
    np.testing.assert_allclose(state[0], case.initial)
    if case.history is not None:
        np.testing.assert_allclose(state[: len(case.history)], case.history)
    np.testing.assert_allclose(state[case.step_index], case.step)
    assert np.isfinite(state).all()


class Geometry(NamedTuple):
    L1: float
    L2: float
    theta1: np.ndarray
    theta2: np.ndarray


GEOMETRIES: dict[str, Geometry] = {
    "pendulum-link-lengths": Geometry(
        2.0,
        3.0,
        np.array([0.0, np.pi / 3, -np.pi / 2]),
        np.array([np.pi / 4, np.pi / 6, np.pi]),
    )
}


@pytest.mark.parametrize("case", GEOMETRIES.values(), ids=GEOMETRIES.keys())
def test_to_xy(case: Geometry) -> None:
    x1, y1, x2, y2 = to_xy(case.L1, case.L2, case.theta1, case.theta2)
    np.testing.assert_allclose(np.hypot(x1, y1), case.L1)
    np.testing.assert_allclose(np.hypot(x2 - x1, y2 - y1), case.L2)
