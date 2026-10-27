from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.integrate import solve_ivp

from edmkit.generate import rk45


class Rk45Problem(NamedTuple):
    """Forced linear system ``dx/dt = M x + b cos(omega t)`` from ``X0`` of shape ``(D,)`` or ``(B, D)``, kept as its coefficients so failing examples stay readable."""

    M: np.ndarray
    b: np.ndarray
    omega: float
    X0: np.ndarray
    dt: float
    t_max: int
    rtol: float


class Rk45Case(NamedTuple):
    M: np.ndarray
    b: np.ndarray
    omega: float
    X0: np.ndarray
    dt: float
    t_max: int
    rtol: float


def check_rk45(M: np.ndarray, b: np.ndarray, omega: float, X0: np.ndarray, dt: float, t_max: int, rtol: float) -> None:
    def f(t: float, x: np.ndarray) -> np.ndarray:
        return x @ M.T + b * np.cos(omega * t)

    t, X = rk45(f, X0, dt, t_max, rtol=rtol, atol=rtol)
    np.testing.assert_array_equal(t, np.arange(0, t_max, dt), strict=True)
    assert X.shape == X0.shape[:-1] + (len(t), X0.shape[-1])
    np.testing.assert_array_equal(X[..., 0, :], X0, strict=True)

    # the reference is solved far tighter than rtol, so the difference is rk45's own global error;
    # it must scale with rtol, which fails if the step-size control is broken; a batch shares its steps,
    # so each of its trajectories is held to the same bound
    for x0, x in zip(X0.reshape(-1, X0.shape[-1]), X.reshape(-1, len(t), X0.shape[-1])):
        expected = solve_ivp(f, (0, t[-1]), x0, method="DOP853", t_eval=t, rtol=1e-13, atol=1e-13).y.T
        np.testing.assert_allclose(x, expected, rtol=0, atol=100 * rtol * max(1.0, np.abs(expected).max()), strict=True)


@st.composite
def rk45_problems(draw) -> Rk45Problem:
    D = draw(st.integers(min_value=1, max_value=3))
    # a single initial condition (D,) or a batch of them (B, D)
    shape = draw(st.sampled_from([(D,), (draw(st.integers(min_value=1, max_value=3)), D)]))
    # values below 1e-100 underflow when squared, which breaks the step-size estimate of the scipy reference
    elements = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False).filter(lambda v: v == 0 or abs(v) > 1e-100)

    return Rk45Problem(
        M=draw(hnp.arrays(np.float64, (D, D), elements=elements)),
        b=draw(hnp.arrays(np.float64, D, elements=elements)),
        omega=draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        X0=draw(hnp.arrays(np.float64, shape, elements=elements)),
        # dt up to 1 forces several adaptive steps per sample
        dt=draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)),
        t_max=draw(st.integers(min_value=2, max_value=3)),
        rtol=10.0 ** draw(st.floats(min_value=-9.0, max_value=-3.0, allow_nan=False, allow_infinity=False)),
    )


@given(problem=rk45_problems())
def test_rk45_compatibility(problem: Rk45Problem) -> None:
    check_rk45(*problem)


RK45_VALID = {
    "decay": Rk45Case(np.array([[-1.0]]), np.array([0.0]), 0.0, np.array([1.0]), 0.1, 3, 1e-6),
    "rotation": Rk45Case(np.array([[0.0, 1.0], [-1.0, 0.0]]), np.zeros(2), 0.0, np.array([1.0, 0.0]), 0.5, 3, 1e-8),
    "stationary": Rk45Case(np.zeros((2, 2)), np.zeros(2), 0.0, np.array([1.0, -1.0]), 0.5, 2, 1e-6),
    "rotation_batch": Rk45Case(np.array([[0.0, 1.0], [-1.0, 0.0]]), np.zeros(2), 0.0, np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]), 0.5, 3, 1e-8),
}


@pytest.mark.parametrize("case", RK45_VALID.values(), ids=RK45_VALID.keys())
def test_rk45_valid(case: Rk45Case) -> None:
    check_rk45(*case)


RK45_INVALID = {
    "3d_initial_condition": Rk45Case(np.array([[-1.0]]), np.array([0.0]), 0.0, np.ones((2, 2, 1)), 0.1, 3, 1e-6),
}


@pytest.mark.parametrize("case", RK45_INVALID.values(), ids=RK45_INVALID.keys())
def test_rk45_invalid(case: Rk45Case) -> None:
    with pytest.raises(ValueError):
        check_rk45(*case)
