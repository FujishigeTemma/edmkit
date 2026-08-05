from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.special import expit

from edmkit.simplex_projection import knn, loo, simplex_projection, soft_simplex_projection


class Problem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    mask: np.ndarray | None


class ProblemSpec(NamedTuple):
    seed: int
    batched: bool
    targets: int
    masked: bool
    e: int = 2
    n: int = 12
    m: int = 4
    batches: int = 2


class KnnCase(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    k: int
    distances: np.ndarray
    indices: np.ndarray


class LooCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    theiler_window: int


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
    return Problem(x, y, q, mask)


def make_loo_case(spec: ProblemSpec, theiler_window: int) -> LooCase:
    problem = make_problem(spec)
    return LooCase(problem.x, problem.y, theiler_window)


def simplex_reference(x: np.ndarray, y: np.ndarray, q: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Brute-force simplex projection returning the canonical ``(M, targets)`` shape."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        x, y = x[mask], y[mask]

    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    indices = np.argsort(distances, axis=1, kind="stable")[:, : x.shape[1] + 1]
    nearest = np.take_along_axis(distances, indices, axis=1)
    scale = np.maximum(nearest[:, :1], 1e-6)
    weights = np.exp(-nearest / scale)
    return np.einsum("mk,mkt->mt", weights, y[indices]) / weights.sum(axis=1, keepdims=True)


def soft_simplex_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    softness: float = 0.02,
) -> np.ndarray:
    """Brute-force soft simplex projection returning the canonical ``(M, targets)`` shape."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        x, y = x[mask], y[mask]

    distances = np.maximum(np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1), 1e-6)
    k = x.shape[1] + 1
    neighbors = np.sort(distances, axis=1)[:, : k + 1]
    scale = neighbors[:, :1]
    radius = (neighbors[:, k - 1 : k] + neighbors[:, k : k + 1]) / 2
    weights = np.exp(-distances / scale) * expit((radius - distances) / (softness * radius))
    return weights @ y / weights.sum(axis=1, keepdims=True)


def check_problem(problem: Problem) -> None:
    actual = simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask)
    if problem.x.ndim == 2:
        expected = simplex_reference(problem.x, problem.y, problem.q, problem.mask).squeeze()
    else:
        expected = np.stack(
            [
                simplex_reference(x, y, q, None if problem.mask is None else problem.mask[batch])
                for batch, (x, y, q) in enumerate(zip(problem.x, problem.y, problem.q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_simplex(spec: ProblemSpec) -> None:
    check_problem(make_problem(spec))


def check_soft_simplex(spec: ProblemSpec, *, softness: float = 0.02) -> None:
    problem = make_problem(spec)
    actual = soft_simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask, softness=softness)
    if problem.x.ndim == 2:
        expected = soft_simplex_reference(problem.x, problem.y, problem.q, problem.mask, softness=softness).squeeze()
    else:
        expected = np.stack(
            [
                soft_simplex_reference(x, y, q, None if problem.mask is None else problem.mask[batch], softness=softness)
                for batch, (x, y, q) in enumerate(zip(problem.x, problem.y, problem.q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_soft_simplex_constant_target() -> None:
    rng = np.random.default_rng(30)
    x = rng.normal(size=(8, 2))
    q = rng.normal(size=(3, 2))
    y = np.tile([3.5, -2.0], (len(x), 1))
    actual = soft_simplex_projection(x, y, q, softness=0.5)
    np.testing.assert_allclose(actual, np.tile([3.5, -2.0], (len(q), 1)), atol=1e-12, rtol=1e-12)


def check_soft_simplex_hard_limit() -> None:
    problem = make_problem(ProblemSpec(31, False, 2, False))
    expected = simplex_projection(problem.x, problem.y, problem.q)
    actual = soft_simplex_projection(problem.x, problem.y, problem.q, softness=1e-8)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_soft_simplex_mask_filtering() -> None:
    problem = make_problem(ProblemSpec(32, False, 2, True))
    assert problem.mask is not None
    expected = soft_simplex_projection(problem.x[problem.mask], problem.y[problem.mask], problem.q, softness=0.1)
    actual = soft_simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask, softness=0.1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def check_expected(x: np.ndarray, y: np.ndarray, q: np.ndarray, expected: np.ndarray | float) -> None:
    actual = np.asarray(simplex_projection(x, y, q))
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)


def check_knn(case: KnnCase) -> None:
    distances, indices = knn(case.x, case.q, case.k)
    np.testing.assert_allclose(distances, case.distances)
    np.testing.assert_array_equal(indices, case.indices)


def loo_reference(x: np.ndarray, y: np.ndarray, theiler_window: int) -> np.ndarray:
    predictions = []
    for i in range(len(x)):
        mask = np.abs(np.arange(len(x)) - i) > theiler_window
        predictions.append(simplex_reference(x, y, x[i : i + 1], mask)[0])
    return np.asarray(predictions)


def check_loo(case: LooCase) -> None:
    actual = loo(case.x, case.y, theiler_window=case.theiler_window)
    if case.x.ndim == 2:
        expected = loo_reference(case.x, case.y, case.theiler_window).squeeze()
    else:
        expected = np.stack([loo_reference(x, y, case.theiler_window) for x, y in zip(case.x, case.y)])
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_tensor(spec: ProblemSpec) -> None:
    from tinygrad import Tensor

    problem = make_problem(spec)
    x, y, q = (array.astype(np.float32) for array in (problem.x, problem.y, problem.q))
    expected = simplex_projection(x, y, q, mask=problem.mask)
    mask = None if problem.mask is None else Tensor(problem.mask)
    actual = simplex_projection(Tensor(x), Tensor(y), Tensor(q), mask=mask).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_tensor_gradient(spec: ProblemSpec) -> None:
    from tinygrad import Tensor

    problem = make_problem(spec)
    x, y, q = (array.astype(np.float32) for array in (problem.x, problem.y, problem.q))
    q[..., :2, :] = x[..., :2, :]  # coincident query and library points produce zero distances
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = simplex_projection(X, Y, Q).sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def check_soft_tensor(spec: ProblemSpec) -> None:
    from tinygrad import Tensor

    problem = make_problem(spec)
    x, y, q = (array.astype(np.float32) for array in (problem.x, problem.y, problem.q))
    expected = soft_simplex_projection(x, y, q, mask=problem.mask, softness=0.1)
    mask = None if problem.mask is None else Tensor(problem.mask)
    actual = soft_simplex_projection(Tensor(x), Tensor(y), Tensor(q), mask=mask, softness=0.1).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


def check_soft_tensor_gradient(spec: ProblemSpec) -> None:
    from tinygrad import Tensor

    problem = make_problem(spec)
    x, y, q = (array.astype(np.float32) for array in (problem.x, problem.y, problem.q))
    q[..., :2, :] = x[..., :2, :]
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = soft_simplex_projection(X, Y, Q, softness=0.1).sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def call_simplex(problem: Problem) -> None:
    simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask)


def call_soft_simplex(problem: Problem) -> None:
    soft_simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask)


def call_soft_simplex_with_softness(problem: Problem, softness: float) -> None:
    soft_simplex_projection(problem.x, problem.y, problem.q, mask=problem.mask, softness=softness)


def call_loo(case: LooCase) -> None:
    loo(case.x, case.y, theiler_window=case.theiler_window)


@st.composite
def problems(draw):
    e = draw(st.integers(1, 3))
    return make_problem(
        ProblemSpec(
            seed=draw(st.integers(0, 2**32 - 1)),
            batched=draw(st.booleans()),
            targets=draw(st.integers(1, 3)),
            masked=draw(st.booleans()),
            e=e,
            n=draw(st.integers(e + 3, 16)),
            m=draw(st.integers(2, 6)),
            batches=draw(st.integers(1, 3)),
        )
    )


FINITE = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


@st.composite
def loo_cases(draw):
    e = draw(st.integers(1, 4))
    theiler_window = draw(st.integers(1, 10))
    minimum = 2 * theiler_window + e + 2
    n = draw(st.integers(minimum, max(minimum, 40)))
    x = draw(hnp.arrays(np.float64, (n, e), elements=FINITE))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    x = x + rng.uniform(-1e-6, 1e-6, x.shape)
    y = draw(hnp.arrays(np.float64, n, elements=FINITE))
    return LooCase(x, y, theiler_window)


@given(problem=problems())
def test_compatibility(problem: Problem):
    check_problem(problem)


@settings(deadline=5000)
@given(case=loo_cases())
def test_loo_compatibility(case: LooCase) -> None:
    check_loo(case)


IDENTITY_X = np.array([[0.0], [2.0], [5.0], [9.0]])
IDENTITY_Y = np.array([10.0, 20.0, 30.0, 40.0])

VALID = [
    pytest.param(partial(check_simplex, ProblemSpec(0, False, 1, False)), id="simplex-scalar-2d"),
    pytest.param(partial(check_simplex, ProblemSpec(1, False, 3, True)), id="simplex-masked-multitarget-2d"),
    pytest.param(partial(check_simplex, ProblemSpec(2, True, 1, False)), id="simplex-scalar-batched-3d"),
    pytest.param(partial(check_simplex, ProblemSpec(3, True, 2, True)), id="simplex-masked-multitarget-batched-3d"),
    pytest.param(partial(check_expected, IDENTITY_X, IDENTITY_Y, IDENTITY_X[:1], np.array(IDENTITY_Y[0])), id="simplex-self-query-single-output"),
    pytest.param(partial(check_soft_simplex, ProblemSpec(4, False, 1, False), softness=0.15), id="soft-simplex-scalar-2d"),
    pytest.param(partial(check_soft_simplex, ProblemSpec(5, True, 2, True), softness=0.15), id="soft-simplex-masked-multitarget-batched-3d"),
    pytest.param(check_soft_simplex_constant_target, id="soft-simplex-constant-target"),
    pytest.param(check_soft_simplex_hard_limit, id="soft-simplex-hard-limit"),
    pytest.param(check_soft_simplex_mask_filtering, id="soft-simplex-mask-matches-filtered-library"),
    pytest.param(
        partial(
            check_knn,
            KnnCase(
                np.array([[0.0], [2.0], [5.0], [9.0]]),
                np.array([[1.5], [8.0]]),
                2,
                np.array([[0.5, 1.5], [1.0, 3.0]]),
                np.array([[1, 0], [3, 2]]),
            ),
        ),
        id="knn-known-neighbors",
    ),
    pytest.param(partial(check_loo, make_loo_case(ProblemSpec(10, False, 1, False, n=20, m=20), 2)), id="loo-scalar-2d"),
    pytest.param(partial(check_loo, make_loo_case(ProblemSpec(11, True, 2, False, n=20, m=20), 2)), id="loo-multitarget-batched-3d"),
    pytest.param(partial(check_tensor, ProblemSpec(20, False, 1, False)), id="tensor-scalar-2d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor, ProblemSpec(21, False, 2, False)), id="tensor-multitarget-2d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor, ProblemSpec(22, True, 1, False)), id="tensor-scalar-batched-3d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor, ProblemSpec(23, True, 2, False)), id="tensor-multitarget-batched-3d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor, ProblemSpec(24, False, 2, True)), id="tensor-masked-2d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor, ProblemSpec(25, True, 2, True)), id="tensor-masked-batched-3d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor_gradient, ProblemSpec(26, False, 1, False)), id="tensor-gradient-coincident-2d", marks=pytest.mark.gpu),
    pytest.param(partial(check_tensor_gradient, ProblemSpec(27, True, 2, False)), id="tensor-gradient-coincident-batched-3d", marks=pytest.mark.gpu),
    pytest.param(partial(check_soft_tensor, ProblemSpec(28, True, 2, True)), id="soft-tensor-masked-multitarget-batched-3d", marks=pytest.mark.gpu),
    pytest.param(
        partial(check_soft_tensor_gradient, ProblemSpec(29, True, 2, False)), id="soft-tensor-gradient-coincident-batched-3d", marks=pytest.mark.gpu
    ),
]


@pytest.mark.parametrize("check", VALID)
def test_valid(check: Callable[[], None]):
    check()


ERRORS = [
    pytest.param(
        partial(call_simplex, Problem(np.zeros((5, 2)), np.zeros(4), np.zeros((2, 2)), None)),
        ValueError,
        None,
        id="simplex-library-target-length",
    ),
    pytest.param(
        partial(call_simplex, Problem(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), None)),
        ValueError,
        None,
        id="simplex-mixed-ranks",
    ),
    pytest.param(
        partial(call_simplex, Problem(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2, 3)), None)),
        ValueError,
        None,
        id="simplex-query-dimension",
    ),
    pytest.param(
        partial(
            call_simplex,
            Problem(
                np.arange(10.0).reshape(5, 2),
                np.arange(5.0),
                np.array([[0.0, 0.0]]),
                np.array([True, True, False, False, False]),
            ),
        ),
        ValueError,
        None,
        id="simplex-too-few-unmasked-neighbors",
    ),
    pytest.param(
        partial(call_soft_simplex, Problem(np.zeros((3, 2)), np.zeros(3), np.zeros((1, 2)), None)),
        ValueError,
        "Not enough points in X to find 4 neighbors, got N=3",
        id="soft-simplex-insufficient-library",
    ),
    pytest.param(
        partial(call_soft_simplex_with_softness, Problem(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None), 0.0),
        ValueError,
        "softness must be positive, got softness=0.0",
        id="soft-simplex-zero-softness",
    ),
    pytest.param(
        partial(call_soft_simplex_with_softness, Problem(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None), -0.1),
        ValueError,
        "softness must be positive, got softness=-0.1",
        id="soft-simplex-negative-softness",
    ),
    pytest.param(
        partial(call_loo, LooCase(np.zeros((10, 2)), np.zeros(9), 1)),
        ValueError,
        None,
        id="loo-library-target-length",
    ),
    pytest.param(
        partial(call_loo, LooCase(np.zeros((10, 2)), np.zeros(10), 100)),
        ValueError,
        None,
        id="loo-insufficient-library",
    ),
]


@pytest.mark.parametrize(("call", "error", "match"), ERRORS)
def test_invalid(call: Callable[[], None], error: type[Exception], match: str | None):
    with pytest.raises(error, match=match):
        call()
