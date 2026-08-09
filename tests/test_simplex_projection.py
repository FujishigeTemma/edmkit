from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.special import expit

from edmkit.simplex_projection import knn, loo, simplex_projection, soft_simplex_projection


class SimplexProjectionProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    mask: np.ndarray | None


class SimplexProjectionCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    mask: np.ndarray | None


class SoftSimplexProjectionProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    mask: np.ndarray | None
    softness: float


class SoftSimplexProjectionCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    mask: np.ndarray | None
    softness: float


class KNNCase(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    k: int
    distances: np.ndarray
    indices: np.ndarray


class LOOProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    theiler_window: int


class LOOCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    theiler_window: int


def simplex_projection_reference(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Brute-force simplex projection returning the canonical ``(M, targets)`` shape."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        x, y = x[mask], y[mask]

    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    k = x.shape[1] + 1 if k is None else k
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    nearest = np.take_along_axis(distances, indices, axis=1)
    scale = np.maximum(nearest[:, :1], 1e-6)
    weights = np.exp(-nearest / scale)
    return np.einsum("mk,mkt->mt", weights, y[indices]) / weights.sum(axis=1, keepdims=True)


def soft_simplex_projection_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> np.ndarray:
    """Brute-force soft simplex projection returning the canonical ``(M, targets)`` shape."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        x, y = x[mask], y[mask]

    distances = np.maximum(np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1), 1e-6)
    k = x.shape[1] + 1 if k is None else k
    neighbors = np.sort(distances, axis=1)[:, : k + 1]
    scale = neighbors[:, :1]
    radius = (neighbors[:, k - 1 : k] + neighbors[:, k : k + 1]) / 2
    weights = np.exp(-distances / scale) * expit((radius - distances) / (softness * radius))
    return weights @ y / weights.sum(axis=1, keepdims=True)


def check_simplex_projection(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> None:
    actual = simplex_projection(x, y, q, k=k, mask=mask)
    if x.ndim == 2:
        expected = simplex_projection_reference(x, y, q, k=k, mask=mask).squeeze()
    else:
        expected = np.stack(
            [
                simplex_projection_reference(xi, yi, qi, k=k, mask=None if mask is None else mask[batch])
                for batch, (xi, yi, qi) in enumerate(zip(x, y, q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_soft_simplex_projection(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> None:
    actual = soft_simplex_projection(x, y, q, k=k, mask=mask, softness=softness)
    if x.ndim == 2:
        expected = soft_simplex_projection_reference(x, y, q, k=k, mask=mask, softness=softness).squeeze()
    else:
        expected = np.stack(
            [
                soft_simplex_projection_reference(xi, yi, qi, k=k, mask=None if mask is None else mask[batch], softness=softness)
                for batch, (xi, yi, qi) in enumerate(zip(x, y, q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_knn(x: np.ndarray, q: np.ndarray, k: int, expected_distances: np.ndarray, expected_indices: np.ndarray) -> None:
    distances, indices = knn(x, q, k)
    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_array_equal(indices, expected_indices)


def loo_reference(x: np.ndarray, y: np.ndarray, theiler_window: int) -> np.ndarray:
    predictions = []
    for i in range(len(x)):
        mask = np.abs(np.arange(len(x)) - i) > theiler_window
        predictions.append(simplex_projection_reference(x, y, x[i : i + 1], mask=mask)[0])
    return np.asarray(predictions)


def check_loo(x: np.ndarray, y: np.ndarray, theiler_window: int) -> None:
    actual = loo(x, y, theiler_window=theiler_window)
    if x.ndim == 2:
        expected = loo_reference(x, y, theiler_window).squeeze()
    else:
        expected = np.stack([loo_reference(xi, yi, theiler_window) for xi, yi in zip(x, y)])
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def check_simplex_projection_tensor(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    expected = simplex_projection(x, y, q, k=k, mask=mask)
    tensor_mask = None if mask is None else Tensor(mask)
    actual = simplex_projection(Tensor(x), Tensor(y), Tensor(q), k=k, mask=tensor_mask).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_simplex_projection_tensor_gradient(
    x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None
) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    coincident = min(q.shape[-2], 2)
    q[..., :coincident, :] = x[..., :coincident, :]  # coincident query and library points produce zero distances
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = simplex_projection(X, Y, Q, k=k).sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def check_soft_simplex_projection_tensor(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    expected = soft_simplex_projection(x, y, q, k=k, mask=mask, softness=softness)
    tensor_mask = None if mask is None else Tensor(mask)
    actual = soft_simplex_projection(Tensor(x), Tensor(y), Tensor(q), k=k, mask=tensor_mask, softness=softness).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


def check_soft_simplex_projection_tensor_gradient(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    k: int | None = None,
    mask: np.ndarray | None = None,
    softness: float = 0.02,
) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    q[..., :2, :] = x[..., :2, :]
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = soft_simplex_projection(X, Y, Q, k=k, softness=softness).sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


@st.composite
def simplex_projection_problems(draw):
    e = draw(st.integers(1, 3))
    n = draw(st.integers(e + 3, 16))
    m = draw(st.integers(2, 6))
    targets = draw(st.integers(1, 3))
    batches = draw(st.integers(1, 3))
    batched = draw(st.booleans())
    masked = draw(st.booleans())
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if batched:
        x = rng.normal(size=(batches, n, e))
        y = rng.normal(size=(batches, n, targets))
        q = rng.normal(size=(batches, m, e))
        mask = np.ones((batches, n), dtype=bool) if masked else None
        if mask is not None:
            for batch in range(batches):
                n_remove = min(batch + 1, n - (e + 1))
                mask[batch, rng.permutation(n)[:n_remove]] = False
    else:
        x = rng.normal(size=(n, e))
        y = rng.normal(size=(n, targets))
        q = rng.normal(size=(m, e))
        mask = np.ones(n, dtype=bool) if masked else None
        if mask is not None:
            mask[rng.permutation(n)[:2]] = False
        if targets == 1:
            y = y[:, 0]
    return SimplexProjectionProblem(x, y, q, mask)


@st.composite
def soft_simplex_projection_problems(draw):
    e = draw(st.integers(1, 3))
    batches = draw(st.integers(1, 3))
    n = draw(st.integers(e + max(batches, 2) + 2, 18))
    m = draw(st.integers(2, 6))
    targets = draw(st.integers(1, 3))
    batched = draw(st.booleans())
    masked = draw(st.booleans())
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if batched:
        x = rng.normal(size=(batches, n, e))
        y = rng.normal(size=(batches, n, targets))
        q = rng.normal(size=(batches, m, e))
        mask = np.ones((batches, n), dtype=bool) if masked else None
        if mask is not None:
            for batch in range(batches):
                n_remove = min(batch + 1, n - (e + 2))
                mask[batch, rng.permutation(n)[:n_remove]] = False
    else:
        x = rng.normal(size=(n, e))
        y = rng.normal(size=(n, targets))
        q = rng.normal(size=(m, e))
        mask = np.ones(n, dtype=bool) if masked else None
        if mask is not None:
            mask[rng.permutation(n)[:2]] = False
        if targets == 1:
            y = y[:, 0]
    return SoftSimplexProjectionProblem(x, y, q, mask, 0.02)


FINITE = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)


@st.composite
def loo_problems(draw):
    e = draw(st.integers(1, 4))
    theiler_window = draw(st.integers(1, 10))
    minimum = 2 * theiler_window + e + 2
    n = draw(st.integers(minimum, max(minimum, 40)))
    x = draw(hnp.arrays(np.float64, (n, e), elements=FINITE))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    x = x + rng.uniform(-1e-6, 1e-6, x.shape)
    y = draw(hnp.arrays(np.float64, n, elements=FINITE))
    return LOOProblem(x, y, theiler_window)


IDENTITY_X = np.array([[0.0], [2.0], [5.0], [9.0]])
IDENTITY_Y = np.array([10.0, 20.0, 30.0, 40.0])

SIMPLEX_PROJECTION_VALID = {
    "scalar-2d": SimplexProjectionCase(
        np.random.default_rng(0).normal(size=(12, 2)),
        np.random.default_rng(1).normal(size=12),
        np.random.default_rng(2).normal(size=(4, 2)),
        None,
    ),
    "masked-multitarget-2d": SimplexProjectionCase(
        np.random.default_rng(3).normal(size=(12, 2)),
        np.random.default_rng(4).normal(size=(12, 3)),
        np.random.default_rng(5).normal(size=(4, 2)),
        np.array([True] * 10 + [False] * 2),
    ),
    "scalar-batched-3d": SimplexProjectionCase(
        np.random.default_rng(6).normal(size=(2, 12, 2)),
        np.random.default_rng(7).normal(size=(2, 12, 1)),
        np.random.default_rng(8).normal(size=(2, 4, 2)),
        None,
    ),
    "masked-multitarget-batched-3d": SimplexProjectionCase(
        np.random.default_rng(9).normal(size=(2, 12, 2)),
        np.random.default_rng(10).normal(size=(2, 12, 2)),
        np.random.default_rng(11).normal(size=(2, 4, 2)),
        np.array([[True] * 11 + [False], [True] * 10 + [False] * 2]),
    ),
    "self-query-single-output": SimplexProjectionCase(IDENTITY_X, IDENTITY_Y, IDENTITY_X[:1], None),
}

SIMPLEX_PROJECTION_MODES = {
    "numpy": check_simplex_projection,
    "tinygrad": pytest.param(check_simplex_projection_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_simplex_projection_tensor_gradient, marks=pytest.mark.gpu),
}

SIMPLEX_PROJECTION_INVALID = {
    "library-target-length": SimplexProjectionCase(np.zeros((5, 2)), np.zeros(4), np.zeros((2, 2)), None),
    "mixed-ranks": SimplexProjectionCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), None),
    "query-dimension": SimplexProjectionCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2, 3)), None),
    "too-few-unmasked-neighbors": SimplexProjectionCase(
        np.arange(10.0).reshape(5, 2),
        np.arange(5.0),
        np.array([[0.0, 0.0]]),
        np.array([True, True, False, False, False]),
    ),
}


@given(problem=simplex_projection_problems())
def test_simplex_projection_compatibility(problem: SimplexProjectionProblem) -> None:
    check_simplex_projection(problem.x, problem.y, problem.q, k=None, mask=problem.mask)


@pytest.mark.parametrize("mode", SIMPLEX_PROJECTION_MODES.values(), ids=SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("case", SIMPLEX_PROJECTION_VALID.values(), ids=SIMPLEX_PROJECTION_VALID.keys())
def test_simplex_projection_valid(case: SimplexProjectionCase, mode) -> None:
    mode(case.x, case.y, case.q, k=None, mask=case.mask)


@pytest.mark.parametrize("mode", SIMPLEX_PROJECTION_MODES.values(), ids=SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("k", [1, 5])
def test_simplex_projection_custom_k(k: int, mode) -> None:
    case = SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]
    mode(case.x, case.y, case.q, k=k, mask=case.mask)


def test_simplex_projection_none_k_preserves_default() -> None:
    case = SIMPLEX_PROJECTION_VALID["scalar-2d"]
    default = simplex_projection(case.x, case.y, case.q, mask=case.mask)
    explicit_none = simplex_projection(case.x, case.y, case.q, k=None, mask=case.mask)
    np.testing.assert_array_equal(explicit_none, default)


@pytest.mark.parametrize("k", [0, -1])
def test_simplex_projection_invalid_k(k: int) -> None:
    case = SIMPLEX_PROJECTION_VALID["scalar-2d"]
    with pytest.raises(ValueError, match="k must be positive"):
        simplex_projection(case.x, case.y, case.q, k=k, mask=case.mask)


@pytest.mark.parametrize("case", SIMPLEX_PROJECTION_INVALID.values(), ids=SIMPLEX_PROJECTION_INVALID.keys())
def test_simplex_projection_invalid(case: SimplexProjectionCase) -> None:
    with pytest.raises(ValueError):
        simplex_projection(case.x, case.y, case.q, mask=case.mask)


SOFT_SIMPLEX_PROJECTION_VALID = {
    "scalar-2d": SoftSimplexProjectionCase(
        np.random.default_rng(12).normal(size=(12, 2)),
        np.random.default_rng(13).normal(size=12),
        np.random.default_rng(14).normal(size=(4, 2)),
        None,
        0.02,
    ),
    "masked-multitarget-batched-3d": SoftSimplexProjectionCase(
        np.random.default_rng(15).normal(size=(2, 12, 2)),
        np.random.default_rng(16).normal(size=(2, 12, 2)),
        np.random.default_rng(17).normal(size=(2, 4, 2)),
        np.array([[True] * 11 + [False], [True] * 10 + [False] * 2]),
        0.02,
    ),
    "constant-target": SoftSimplexProjectionCase(
        np.random.default_rng(30).normal(size=(8, 2)),
        np.tile([3.5, -2.0], (8, 1)),
        np.random.default_rng(31).normal(size=(3, 2)),
        None,
        0.5,
    ),
    "hard-limit": SoftSimplexProjectionCase(
        np.random.default_rng(32).normal(size=(12, 2)),
        np.random.default_rng(33).normal(size=(12, 2)),
        np.random.default_rng(34).normal(size=(4, 2)),
        None,
        1e-8,
    ),
    "masked-equivalence": SoftSimplexProjectionCase(
        np.random.default_rng(35).normal(size=(12, 2)),
        np.random.default_rng(36).normal(size=(12, 2)),
        np.random.default_rng(37).normal(size=(4, 2)),
        np.array([True] * 10 + [False] * 2),
        0.1,
    ),
}

SOFT_SIMPLEX_PROJECTION_MODES = {
    "numpy": check_soft_simplex_projection,
    "tinygrad": pytest.param(check_soft_simplex_projection_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_soft_simplex_projection_tensor_gradient, marks=pytest.mark.gpu),
}

SOFT_SIMPLEX_PROJECTION_INVALID = {
    "insufficient-library": SoftSimplexProjectionCase(np.zeros((3, 2)), np.zeros(3), np.zeros((1, 2)), None, 0.02),
    "zero-softness": SoftSimplexProjectionCase(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None, 0.0),
    "negative-softness": SoftSimplexProjectionCase(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None, -0.1),
}


@given(problem=soft_simplex_projection_problems())
def test_soft_simplex_projection_compatibility(problem: SoftSimplexProjectionProblem) -> None:
    check_soft_simplex_projection(problem.x, problem.y, problem.q, k=None, mask=problem.mask, softness=problem.softness)


@pytest.mark.parametrize("mode", SOFT_SIMPLEX_PROJECTION_MODES.values(), ids=SOFT_SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("case", SOFT_SIMPLEX_PROJECTION_VALID.values(), ids=SOFT_SIMPLEX_PROJECTION_VALID.keys())
def test_soft_simplex_projection_valid(case: SoftSimplexProjectionCase, mode) -> None:
    mode(case.x, case.y, case.q, k=None, mask=case.mask, softness=case.softness)


@pytest.mark.parametrize("mode", SOFT_SIMPLEX_PROJECTION_MODES.values(), ids=SOFT_SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("k", [1, 5])
def test_soft_simplex_projection_custom_k(k: int, mode) -> None:
    case = SOFT_SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]
    mode(case.x, case.y, case.q, k=k, mask=case.mask, softness=case.softness)


def test_soft_simplex_projection_none_k_preserves_default() -> None:
    case = SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"]
    default = soft_simplex_projection(case.x, case.y, case.q, mask=case.mask, softness=case.softness)
    explicit_none = soft_simplex_projection(case.x, case.y, case.q, k=None, mask=case.mask, softness=case.softness)
    np.testing.assert_array_equal(explicit_none, default)


@pytest.mark.parametrize("k", [0, -1])
def test_soft_simplex_projection_invalid_k(k: int) -> None:
    case = SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"]
    with pytest.raises(ValueError, match="k must be positive"):
        soft_simplex_projection(case.x, case.y, case.q, k=k, mask=case.mask, softness=case.softness)


@pytest.mark.parametrize("case", SOFT_SIMPLEX_PROJECTION_INVALID.values(), ids=SOFT_SIMPLEX_PROJECTION_INVALID.keys())
def test_soft_simplex_projection_invalid(case: SoftSimplexProjectionCase) -> None:
    with pytest.raises(ValueError):
        soft_simplex_projection(case.x, case.y, case.q, mask=case.mask, softness=case.softness)


KNN_VALID = {
    "nearest-two": KNNCase(
        np.array([[0.0], [2.0], [5.0], [9.0]]),
        np.array([[1.5], [8.0]]),
        2,
        np.array([[0.5, 1.5], [1.0, 3.0]]),
        np.array([[1, 0], [3, 2]]),
    ),
}


@pytest.mark.parametrize("case", KNN_VALID.values(), ids=KNN_VALID.keys())
def test_knn_valid(case: KNNCase) -> None:
    check_knn(*case)


LOO_VALID = {
    "scalar-2d": LOOCase(
        np.random.default_rng(37).normal(size=(20, 2)),
        np.random.default_rng(38).normal(size=20),
        2,
    ),
    "multitarget-batched-3d": LOOCase(
        np.random.default_rng(39).normal(size=(2, 20, 2)),
        np.random.default_rng(40).normal(size=(2, 20, 2)),
        2,
    ),
}

LOO_INVALID = {
    "library-target-length": LOOCase(np.zeros((10, 2)), np.zeros(9), 1),
    "insufficient-library": LOOCase(np.zeros((10, 2)), np.zeros(10), 100),
}


@settings(deadline=5000)
@given(problem=loo_problems())
def test_loo_compatibility(problem: LOOProblem) -> None:
    check_loo(*problem)


@pytest.mark.parametrize("case", LOO_VALID.values(), ids=LOO_VALID.keys())
def test_loo_valid(case: LOOCase) -> None:
    check_loo(*case)


@pytest.mark.parametrize("case", LOO_INVALID.values(), ids=LOO_INVALID.keys())
def test_loo_invalid(case: LOOCase) -> None:
    with pytest.raises(ValueError):
        loo(case.x, case.y, theiler_window=case.theiler_window)
