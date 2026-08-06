from typing import NamedTuple

import numpy as np
import pytest

from edmkit.generate import double_pendulum, lorenz, mackey_glass, to_xy


class LorenzCase(NamedTuple):
    sigma: float
    rho: float
    beta: float
    X0: np.ndarray
    dt: float
    t_max: int


class MackeyGlassCase(NamedTuple):
    tau: float
    n: int
    beta: float
    gamma: float
    x0: float
    dt: float
    t_max: int


class DoublePendulumCase(NamedTuple):
    m1: float
    m2: float
    L1: float
    L2: float
    g: float
    X0: np.ndarray
    dt: float
    t_max: int


class ToXYCase(NamedTuple):
    L1: float
    L2: float
    theta1: np.ndarray
    theta2: np.ndarray


LORENZ_X0 = np.array([1.0, 2.0, 3.0])
LORENZ_DT = 0.01
MACKEY_GLASS_X0 = 0.9
MACKEY_GLASS_DT = 1.0
DOUBLE_PENDULUM_THETA = 0.4
DOUBLE_PENDULUM_DT = 0.01
DOUBLE_PENDULUM_X0 = np.array([DOUBLE_PENDULUM_THETA, DOUBLE_PENDULUM_THETA, 0.0, 0.0])


def check_trajectory(
    actual: tuple[np.ndarray, np.ndarray],
    *,
    time: np.ndarray,
    shape: tuple[int, ...],
    initial: np.ndarray | float,
    step_index: int,
    step: np.ndarray | float,
    history: np.ndarray | None = None,
) -> None:
    actual_time, state = actual
    np.testing.assert_array_equal(actual_time, time)
    assert state.shape == shape
    np.testing.assert_allclose(state[0], initial)
    if history is not None:
        np.testing.assert_allclose(state[: len(history)], history)
    np.testing.assert_allclose(state[step_index], step)
    assert np.isfinite(state).all()


LORENZ_VALID = {
    "default": LorenzCase(10.0, 28.0, 8.0 / 3.0, LORENZ_X0, LORENZ_DT, 1),
}

MACKEY_GLASS_VALID = {
    "delayed": MackeyGlassCase(2.0, 10, 0.2, 0.1, MACKEY_GLASS_X0, MACKEY_GLASS_DT, 6),
}

DOUBLE_PENDULUM_VALID = {
    "equal-angle": DoublePendulumCase(1.2, 0.8, 2.0, 3.0, 9.81, DOUBLE_PENDULUM_X0, DOUBLE_PENDULUM_DT, 1),
}

TO_XY_VALID = {
    "segment-lengths": ToXYCase(2.0, 3.0, np.array([0.0, np.pi / 3, -np.pi / 2]), np.array([np.pi / 4, np.pi / 6, np.pi])),
}


@pytest.mark.parametrize("case", LORENZ_VALID.values(), ids=LORENZ_VALID.keys())
def test_lorenz_valid(case: LorenzCase) -> None:
    actual = lorenz(*case)
    check_trajectory(
        actual,
        time=np.arange(0, case.t_max, case.dt),
        shape=(100, 3),
        initial=case.X0,
        step_index=1,
        step=case.X0 + case.dt * np.array([10.0, 23.0, -6.0]),
    )


@pytest.mark.parametrize("case", MACKEY_GLASS_VALID.values(), ids=MACKEY_GLASS_VALID.keys())
def test_mackey_glass_valid(case: MackeyGlassCase) -> None:
    actual = mackey_glass(*case)
    check_trajectory(
        actual,
        time=np.arange(0, case.t_max, case.dt),
        shape=(6,),
        initial=case.x0,
        step_index=2,
        step=case.x0 + case.dt * (case.beta * case.x0 / (1.0 + case.x0**case.n) - case.gamma * case.x0),
        history=np.full(2, case.x0),
    )


@pytest.mark.parametrize("case", DOUBLE_PENDULUM_VALID.values(), ids=DOUBLE_PENDULUM_VALID.keys())
def test_double_pendulum_valid(case: DoublePendulumCase) -> None:
    actual = double_pendulum(*case)
    check_trajectory(
        actual,
        time=np.arange(0, case.t_max, case.dt),
        shape=(100, 4),
        initial=case.X0,
        step_index=1,
        step=case.X0 + case.dt * np.array([0.0, 0.0, -case.g * np.sin(DOUBLE_PENDULUM_THETA) / case.L1, 0.0]),
    )


@pytest.mark.parametrize("case", TO_XY_VALID.values(), ids=TO_XY_VALID.keys())
def test_to_xy_valid(case: ToXYCase) -> None:
    x1, y1, x2, y2 = to_xy(*case)
    np.testing.assert_allclose(np.hypot(x1, y1), case.L1)
    np.testing.assert_allclose(np.hypot(x2 - x1, y2 - y1), case.L2)
