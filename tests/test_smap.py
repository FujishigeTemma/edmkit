from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.smap import smap, weights


class Problem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


class ProblemSpec(NamedTuple):
    seed: int
    batched: bool
    targets: int
    masked: bool
    theta: float
    alpha: float
    e: int = 2
    n: int = 16
    m: int = 4
    batches: int = 2


class WeightCase(NamedTuple):
    distances: np.ndarray
    theta: float
    mask: np.ndarray | None
    min_points: int
    expected: np.ndarray


class WeightError(NamedTuple):
    distances: np.ndarray
    theta: float
    mask: np.ndarray | None
    min_points: int


def make_problem(spec: ProblemSpec) -> Problem:
    rng = np.random.default_rng(spec.seed)
    if spec.batched:
        x = rng.normal(size=(spec.batches, spec.n, spec.e))
        y = rng.normal(size=(spec.batches, spec.n, spec.targets))
        q = rng.normal(size=(spec.batches, spec.m, spec.e))
        mask = np.ones((spec.batches, spec.n), dtype=bool) if spec.masked else None
        if mask is not None:
            for batch in range(spec.batches):
                n_remove = min(batch + 1, spec.n - (spec.e + 1))
                mask[batch, rng.permutation(spec.n)[:n_remove]] = False
    else:
        x = rng.normal(size=(spec.n, spec.e))
        y = rng.normal(size=(spec.n, spec.targets))
        q = rng.normal(size=(spec.m, spec.e))
        mask = np.ones(spec.n, dtype=bool) if spec.masked else None
        if mask is not None:
            mask[rng.permutation(spec.n)[:2]] = False
        if spec.targets == 1:
            y = y[:, 0]
    return Problem(x, y, q, spec.theta, spec.alpha, mask)


def weighted_lstsq_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    theta: float,
    alpha: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Independent S-Map reference returning the canonical ``(M, targets)`` shape."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        x, y = x[mask], y[mask]

    x_aug = np.column_stack([np.ones(len(x)), x])
    q_aug = np.column_stack([np.ones(len(q)), q])
    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    if theta == 0.0:
        query_weights = np.ones_like(distances)
    else:
        scale = np.maximum(distances.mean(axis=1, keepdims=True), 1e-6)
        query_weights = np.exp(-theta * distances / scale)

    penalty = np.eye(x_aug.shape[1])[1:]
    predictions = np.empty((len(q), y.shape[1]))
    for i, (q_row, weight) in enumerate(zip(q_aug, query_weights)):
        sqrt_weight = np.sqrt(weight)[:, None]
        design = sqrt_weight * x_aug
        target = sqrt_weight * y
        trace = max(float(np.sum(design**2)), 1e-12)
        regularizer = np.sqrt(alpha * trace) * penalty
        design = np.vstack([design, regularizer])
        target = np.vstack([target, np.zeros((len(regularizer), y.shape[1]))])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        predictions[i] = q_row @ coefficients
    return predictions


def check_problem(problem: Problem) -> None:
    actual = smap(problem.x, problem.y, problem.q, theta=problem.theta, alpha=problem.alpha, mask=problem.mask)
    if problem.x.ndim == 2:
        expected = weighted_lstsq_reference(
            problem.x,
            problem.y,
            problem.q,
            theta=problem.theta,
            alpha=problem.alpha,
            mask=problem.mask,
        ).squeeze()
    else:
        expected = np.stack(
            [
                weighted_lstsq_reference(
                    x,
                    y,
                    q,
                    theta=problem.theta,
                    alpha=problem.alpha,
                    mask=None if problem.mask is None else problem.mask[batch],
                )
                for batch, (x, y, q) in enumerate(zip(problem.x, problem.y, problem.q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def check_smap(spec: ProblemSpec) -> None:
    check_problem(make_problem(spec))


def check_weights(case: WeightCase) -> None:
    actual = weights(case.distances, case.theta, mask=case.mask, min_points=case.min_points)
    np.testing.assert_allclose(actual, case.expected, atol=1e-12, rtol=1e-12)


def check_tensor(spec: ProblemSpec) -> None:
    from tinygrad import Tensor

    problem = make_problem(spec)
    x, y, q = (array.astype(np.float32) for array in (problem.x, problem.y, problem.q))
    expected = smap(x, y, q, theta=problem.theta, alpha=problem.alpha, mask=problem.mask)
    actual = smap(
        Tensor(x),
        Tensor(y),
        Tensor(q),
        theta=problem.theta,
        alpha=problem.alpha,
        mask=None if problem.mask is None else Tensor(problem.mask),
    ).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def call_smap(problem: Problem) -> None:
    smap(problem.x, problem.y, problem.q, theta=problem.theta, alpha=problem.alpha, mask=problem.mask)


def call_weights(case: WeightError) -> None:
    weights(case.distances, case.theta, mask=case.mask, min_points=case.min_points)


def call_tensor(theta: float) -> None:
    from tinygrad import Tensor

    smap(Tensor.zeros(5, 2), Tensor.zeros(5), Tensor.zeros(2, 2), theta=theta)


@st.composite
def problems(draw):
    e = draw(st.integers(1, 3))
    return make_problem(
        ProblemSpec(
            seed=draw(st.integers(0, 2**32 - 1)),
            batched=draw(st.booleans()),
            targets=draw(st.integers(1, 3)),
            masked=draw(st.booleans()),
            theta=draw(st.sampled_from((0.0, 0.5, 2.5))),
            alpha=draw(st.sampled_from((0.0, 1e-4, 0.25))),
            e=e,
            n=draw(st.integers(e + 5, 18)),
            m=draw(st.integers(2, 5)),
            batches=draw(st.integers(1, 3)),
        )
    )


@given(problem=problems())
def test_compatibility(problem: Problem):
    check_problem(problem)


VALID_SPECS = {
    "global-scalar-2d": ProblemSpec(0, False, 1, False, 0.0, 0.0),
    "single-query-scalar-2d": ProblemSpec(5, False, 1, False, 1.0, 1e-4, m=1),
    "local-multitarget-2d": ProblemSpec(1, False, 2, False, 2.5, 1e-4),
    "regularized-masked-scalar-2d": ProblemSpec(2, False, 1, True, 1.0, 0.25),
    "local-scalar-batched-3d": ProblemSpec(3, True, 1, False, 1.5, 1e-4),
    "regularized-masked-multitarget-batched-3d": ProblemSpec(4, True, 2, True, 2.0, 0.1),
}

VALID = [
    *(pytest.param(partial(check_smap, spec), id=f"smap-{name}") for name, spec in VALID_SPECS.items()),
    pytest.param(
        partial(
            check_weights,
            WeightCase(
                np.array([[0.0, 1.0, np.inf]]),
                0.0,
                None,
                2,
                np.array([[1.0, 1.0, 0.0]]),
            ),
        ),
        id="weights-global-finite",
    ),
    pytest.param(
        partial(
            check_weights,
            WeightCase(
                np.array([[0.0, 1.0, 2.0]]),
                2.0,
                np.array([True, True, False]),
                2,
                np.array([[1.0, np.exp(-4.0), 0.0]]),
            ),
        ),
        id="weights-local-masked",
    ),
    *(pytest.param(partial(check_tensor, spec), id=f"tensor-{name}", marks=pytest.mark.gpu) for name, spec in VALID_SPECS.items()),
]


@pytest.mark.parametrize("check", VALID)
def test_valid(check: Callable[[], None]):
    check()


INVALID = [
    pytest.param(
        partial(call_smap, Problem(np.zeros((5, 2)), np.zeros(5), np.zeros((2, 2)), -1.0, 0.0, None)),
        id="smap-negative-theta",
    ),
    pytest.param(
        partial(call_smap, Problem(np.zeros((5, 2)), np.zeros(4), np.zeros((2, 2)), 1.0, 0.0, None)),
        id="smap-library-target-length",
    ),
    pytest.param(
        partial(
            call_smap,
            Problem(np.zeros((2, 5, 2)), np.zeros((2, 4, 1)), np.zeros((2, 2, 2)), 1.0, 0.0, None),
        ),
        id="smap-batched-library-target-length",
    ),
    pytest.param(
        partial(
            call_smap,
            Problem(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), 1.0, 0.0, None),
        ),
        id="smap-mixed-ranks",
    ),
    pytest.param(
        partial(
            call_smap,
            Problem(
                np.arange(10.0).reshape(5, 2),
                np.arange(5.0),
                np.array([[0.0, 0.0]]),
                1.0,
                0.0,
                np.array([True, True, False, False, False]),
            ),
        ),
        id="smap-too-few-unmasked-points",
    ),
    pytest.param(
        partial(call_weights, WeightError(np.array([[0.0, np.inf, np.inf]]), 2.0, None, 2)),
        id="weights-too-few-valid-points",
    ),
    pytest.param(
        partial(call_tensor, -1.0),
        id="tensor-negative-theta",
        marks=pytest.mark.gpu,
    ),
]


@pytest.mark.parametrize("call", INVALID)
def test_invalid(call: Callable[[], None]):
    with pytest.raises(ValueError):
        call()
