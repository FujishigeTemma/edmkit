from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import expit

from edmkit.simplex_projection import knn, simplex_projection, soft_simplex_projection, theiler_window


class SimplexProjectionProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None


class SimplexProjectionCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None


class SoftSimplexProjectionProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None
    softness: float


class SoftSimplexProjectionCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None
    softness: float


class KNNProblem(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    k: int


class KNNCase(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    k: int


def simplex_projection_reference(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Brute-force simplex projection returning the canonical ``(M, targets)`` shape; `mask` is per-query with shape (M, N)."""
    y = y[:, None] if y.ndim == 1 else y

    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    if mask is not None:
        distances[~mask] = np.inf
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


def knn_reference(x: np.ndarray, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force k-nearest-neighbor search via full pairwise distances and argsort."""
    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    nearest = np.take_along_axis(distances, indices, axis=1)
    return nearest, indices


def check_simplex_projection(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None, mask: np.ndarray | None) -> None:
    actual = simplex_projection(x, y, q, k=k, mask=mask)
    if k is None:
        # an explicit k=None must be indistinguishable from omitting k entirely
        default = simplex_projection(x, y, q, mask=mask)
        np.testing.assert_array_equal(actual, default)
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
    k: int | None,
    mask: np.ndarray | None,
    softness: float,
) -> None:
    actual = soft_simplex_projection(x, y, q, k=k, mask=mask, softness=softness)
    if k is None:
        # an explicit k=None must be indistinguishable from omitting k entirely
        default = soft_simplex_projection(x, y, q, mask=mask, softness=softness)
        np.testing.assert_array_equal(actual, default)
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


def check_knn(x: np.ndarray, q: np.ndarray, k: int) -> None:
    distances, indices = knn(x, q, k)
    expected_distances, expected_indices = knn_reference(x, q, k)
    np.testing.assert_allclose(distances, expected_distances)
    np.testing.assert_array_equal(indices, expected_indices)


def check_simplex_projection_tensor(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    expected = simplex_projection(x, y, q, k=k, mask=mask)
    tensor_mask = None if mask is None else Tensor(mask)
    actual = simplex_projection(Tensor(x), Tensor(y), Tensor(q), k=k, mask=tensor_mask).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_simplex_projection_tensor_gradient(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None, mask: np.ndarray | None) -> None:
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
    k: int | None,
    mask: np.ndarray | None,
    softness: float,
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
    k: int | None,
    mask: np.ndarray | None,
    softness: float,
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
    # every query always retains at least e + 1 unmasked library points (see masking below),
    # so any k up to e + 1 is guaranteed valid regardless of masking or batching.
    k = draw(st.one_of(st.none(), st.integers(1, e + 1)))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    if batched:
        x = rng.normal(size=(batches, n, e))
        y = rng.normal(size=(batches, n, targets))
        q = rng.normal(size=(batches, m, e))
        mask = np.ones((batches, m, n), dtype=bool) if masked else None
        if mask is not None:
            for batch in range(batches):
                n_remove = min(batch + 1, n - (e + 1))
                for query in range(m):
                    mask[batch, query, rng.permutation(n)[:n_remove]] = False
    else:
        x = rng.normal(size=(n, e))
        y = rng.normal(size=(n, targets))
        q = rng.normal(size=(m, e))
        mask = np.ones((m, n), dtype=bool) if masked else None
        if mask is not None:
            for query in range(m):
                mask[query, rng.permutation(n)[:2]] = False
        if targets == 1:
            y = y[:, 0]
    return SimplexProjectionProblem(x, y, q, k, mask)


@st.composite
def soft_simplex_projection_problems(draw):
    e = draw(st.integers(1, 3))
    batches = draw(st.integers(1, 3))
    n = draw(st.integers(e + max(batches, 2) + 2, 18))
    m = draw(st.integers(2, 6))
    targets = draw(st.integers(1, 3))
    batched = draw(st.booleans())
    masked = draw(st.booleans())
    # a masked or batched library always retains at least e + 2 unmasked points (see masking below),
    # so any k up to e + 1 leaves the required k + 1 neighbors available.
    k = draw(st.one_of(st.none(), st.integers(1, e + 1)))
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
    return SoftSimplexProjectionProblem(x, y, q, k, mask, 0.02)


@st.composite
def knn_problems(draw):
    e = draw(st.integers(1, 3))
    n = draw(st.integers(1, 20))
    m = draw(st.integers(1, 5))
    k = draw(st.integers(1, n))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    x = rng.normal(size=(n, e))
    q = rng.normal(size=(m, e))
    return KNNProblem(x, q, k)


SIMPLEX_PROJECTION_VALID = {
    "scalar-2d": SimplexProjectionCase(
        np.random.default_rng(0).normal(size=(12, 2)),
        np.random.default_rng(1).normal(size=12),
        np.random.default_rng(2).normal(size=(4, 2)),
        None,
        None,
    ),
    "masked-multitarget-2d": SimplexProjectionCase(
        np.random.default_rng(3).normal(size=(12, 2)),
        np.random.default_rng(4).normal(size=(12, 3)),
        np.random.default_rng(5).normal(size=(4, 2)),
        None,
        np.array([[True] * 10 + [False] * 2] * 3 + [[False] * 2 + [True] * 10]),
    ),
    "scalar-batched-3d": SimplexProjectionCase(
        np.random.default_rng(6).normal(size=(2, 12, 2)),
        np.random.default_rng(7).normal(size=(2, 12, 1)),
        np.random.default_rng(8).normal(size=(2, 4, 2)),
        None,
        None,
    ),
    "masked-multitarget-batched-3d": SimplexProjectionCase(
        np.random.default_rng(9).normal(size=(2, 12, 2)),
        np.random.default_rng(10).normal(size=(2, 12, 2)),
        np.random.default_rng(11).normal(size=(2, 4, 2)),
        None,
        np.stack(
            [
                np.array([[True] * 11 + [False]] * 3 + [[False] + [True] * 11]),
                np.array([[True] * 10 + [False] * 2] * 3 + [[False] * 2 + [True] * 10]),
            ]
        ),
    ),
    "self-query-single-output": SimplexProjectionCase(
        np.array([[0.0], [2.0], [5.0], [9.0]]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([[0.0]]),
        None,
        None,
    ),
    "theiler-self-query-2d": SimplexProjectionCase(
        np.random.default_rng(37).normal(size=(20, 2)),
        np.random.default_rng(38).normal(size=20),
        np.random.default_rng(37).normal(size=(20, 2)),
        None,
        theiler_window(np.arange(20), np.arange(20), 2),
    ),
    "theiler-self-query-batched-3d": SimplexProjectionCase(
        np.random.default_rng(39).normal(size=(2, 20, 2)),
        np.random.default_rng(40).normal(size=(2, 20, 2)),
        np.random.default_rng(39).normal(size=(2, 20, 2)),
        None,
        np.tile(theiler_window(np.arange(20), np.arange(20), 2), (2, 1, 1)),
    ),
}
SIMPLEX_PROJECTION_VALID["custom-k-one"] = SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]._replace(k=1)
SIMPLEX_PROJECTION_VALID["custom-k-five"] = SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]._replace(k=5)

SIMPLEX_PROJECTION_MODES = {
    "numpy": check_simplex_projection,
    "tinygrad": pytest.param(check_simplex_projection_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_simplex_projection_tensor_gradient, marks=pytest.mark.gpu),
}

SIMPLEX_PROJECTION_INVALID = {
    "library-target-length": SimplexProjectionCase(np.zeros((5, 2)), np.zeros(4), np.zeros((2, 2)), None, None),
    "mixed-ranks": SimplexProjectionCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), None, None),
    "query-dimension": SimplexProjectionCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2, 3)), None, None),
    "too-few-unmasked-neighbors": SimplexProjectionCase(
        np.arange(10.0).reshape(5, 2),
        np.arange(5.0),
        np.array([[0.0, 0.0]]),
        None,
        np.array([[True, True, False, False, False]]),
    ),
    "mask-not-per-query": SimplexProjectionCase(
        np.zeros((12, 2)),
        np.zeros(12),
        np.zeros((4, 2)),
        None,
        np.ones(12, dtype=bool),
    ),
    "theiler-window-too-wide": SimplexProjectionCase(
        np.zeros((10, 2)),
        np.zeros(10),
        np.zeros((10, 2)),
        None,
        theiler_window(np.arange(10), np.arange(10), 100),
    ),
    "zero-k": SimplexProjectionCase(
        SIMPLEX_PROJECTION_VALID["scalar-2d"].x,
        SIMPLEX_PROJECTION_VALID["scalar-2d"].y,
        SIMPLEX_PROJECTION_VALID["scalar-2d"].q,
        0,
        None,
    ),
    "negative-k": SimplexProjectionCase(
        SIMPLEX_PROJECTION_VALID["scalar-2d"].x,
        SIMPLEX_PROJECTION_VALID["scalar-2d"].y,
        SIMPLEX_PROJECTION_VALID["scalar-2d"].q,
        -1,
        None,
    ),
}


@given(problem=simplex_projection_problems())
def test_simplex_projection_compatibility(problem: SimplexProjectionProblem) -> None:
    check_simplex_projection(*problem)


@pytest.mark.parametrize("mode", SIMPLEX_PROJECTION_MODES.values(), ids=SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("case", SIMPLEX_PROJECTION_VALID.values(), ids=SIMPLEX_PROJECTION_VALID.keys())
def test_simplex_projection_valid(case: SimplexProjectionCase, mode) -> None:
    mode(*case)


@pytest.mark.parametrize("case", SIMPLEX_PROJECTION_INVALID.values(), ids=SIMPLEX_PROJECTION_INVALID.keys())
def test_simplex_projection_invalid(case: SimplexProjectionCase) -> None:
    with pytest.raises(ValueError):
        simplex_projection(case.x, case.y, case.q, k=case.k, mask=case.mask)


SOFT_SIMPLEX_PROJECTION_VALID = {
    "scalar-2d": SoftSimplexProjectionCase(
        np.random.default_rng(12).normal(size=(12, 2)),
        np.random.default_rng(13).normal(size=12),
        np.random.default_rng(14).normal(size=(4, 2)),
        None,
        None,
        0.02,
    ),
    "masked-multitarget-batched-3d": SoftSimplexProjectionCase(
        np.random.default_rng(15).normal(size=(2, 12, 2)),
        np.random.default_rng(16).normal(size=(2, 12, 2)),
        np.random.default_rng(17).normal(size=(2, 4, 2)),
        None,
        np.array([[True] * 11 + [False], [True] * 10 + [False] * 2]),
        0.02,
    ),
    "constant-target": SoftSimplexProjectionCase(
        np.random.default_rng(30).normal(size=(8, 2)),
        np.tile([3.5, -2.0], (8, 1)),
        np.random.default_rng(31).normal(size=(3, 2)),
        None,
        None,
        0.5,
    ),
    "hard-limit": SoftSimplexProjectionCase(
        np.random.default_rng(32).normal(size=(12, 2)),
        np.random.default_rng(33).normal(size=(12, 2)),
        np.random.default_rng(34).normal(size=(4, 2)),
        None,
        None,
        1e-8,
    ),
    "masked-equivalence": SoftSimplexProjectionCase(
        np.random.default_rng(35).normal(size=(12, 2)),
        np.random.default_rng(36).normal(size=(12, 2)),
        np.random.default_rng(37).normal(size=(4, 2)),
        None,
        np.array([True] * 10 + [False] * 2),
        0.1,
    ),
}
SOFT_SIMPLEX_PROJECTION_VALID["custom-k-one"] = SOFT_SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]._replace(k=1)
SOFT_SIMPLEX_PROJECTION_VALID["custom-k-five"] = SOFT_SIMPLEX_PROJECTION_VALID["masked-multitarget-batched-3d"]._replace(k=5)

SOFT_SIMPLEX_PROJECTION_MODES = {
    "numpy": check_soft_simplex_projection,
    "tinygrad": pytest.param(check_soft_simplex_projection_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_soft_simplex_projection_tensor_gradient, marks=pytest.mark.gpu),
}

SOFT_SIMPLEX_PROJECTION_INVALID = {
    "insufficient-library": SoftSimplexProjectionCase(np.zeros((3, 2)), np.zeros(3), np.zeros((1, 2)), None, None, 0.02),
    "zero-softness": SoftSimplexProjectionCase(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None, None, 0.0),
    "negative-softness": SoftSimplexProjectionCase(np.zeros((4, 2)), np.zeros(4), np.zeros((1, 2)), None, None, -0.1),
    "zero-k": SoftSimplexProjectionCase(
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].x,
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].y,
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].q,
        0,
        None,
        0.02,
    ),
    "negative-k": SoftSimplexProjectionCase(
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].x,
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].y,
        SOFT_SIMPLEX_PROJECTION_VALID["scalar-2d"].q,
        -1,
        None,
        0.02,
    ),
}


@given(problem=soft_simplex_projection_problems())
def test_soft_simplex_projection_compatibility(problem: SoftSimplexProjectionProblem) -> None:
    check_soft_simplex_projection(*problem)


@pytest.mark.parametrize("mode", SOFT_SIMPLEX_PROJECTION_MODES.values(), ids=SOFT_SIMPLEX_PROJECTION_MODES.keys())
@pytest.mark.parametrize("case", SOFT_SIMPLEX_PROJECTION_VALID.values(), ids=SOFT_SIMPLEX_PROJECTION_VALID.keys())
def test_soft_simplex_projection_valid(case: SoftSimplexProjectionCase, mode) -> None:
    mode(*case)


@pytest.mark.parametrize("case", SOFT_SIMPLEX_PROJECTION_INVALID.values(), ids=SOFT_SIMPLEX_PROJECTION_INVALID.keys())
def test_soft_simplex_projection_invalid(case: SoftSimplexProjectionCase) -> None:
    with pytest.raises(ValueError):
        soft_simplex_projection(case.x, case.y, case.q, k=case.k, mask=case.mask, softness=case.softness)


KNN_VALID = {
    "nearest-two": KNNCase(
        np.array([[0.0], [2.0], [5.0], [9.0]]),
        np.array([[1.5], [8.0]]),
        2,
    ),
}

KNN_INVALID = {
    "k-exceeds-library-size": KNNCase(np.zeros((3, 2)), np.zeros((1, 2)), 4),
}


@given(problem=knn_problems())
def test_knn_compatibility(problem: KNNProblem) -> None:
    check_knn(*problem)


@pytest.mark.parametrize("case", KNN_VALID.values(), ids=KNN_VALID.keys())
def test_knn_valid(case: KNNCase) -> None:
    check_knn(*case)


@pytest.mark.parametrize("case", KNN_INVALID.values(), ids=KNN_INVALID.keys())
def test_knn_invalid(case: KNNCase) -> None:
    with pytest.raises(ValueError):
        knn(*case)
