from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.smap import smap, weights


class SmapProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


class SmapCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


class WeightsCase(NamedTuple):
    distances: np.ndarray
    theta: float
    mask: np.ndarray | None
    min_points: int


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


def check_smap(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    actual = smap(x, y, q, theta=theta, alpha=alpha, mask=mask)
    if x.ndim == 2:
        expected = weighted_lstsq_reference(
            x,
            y,
            q,
            theta=theta,
            alpha=alpha,
            mask=mask,
        ).squeeze()
    else:
        expected = np.stack(
            [
                weighted_lstsq_reference(
                    xi,
                    yi,
                    qi,
                    theta=theta,
                    alpha=alpha,
                    mask=None if mask is None else mask[batch],
                )
                for batch, (xi, yi, qi) in enumerate(zip(x, y, q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def weights_reference(distances: np.ndarray, theta: float, mask: np.ndarray | None, min_points: int) -> np.ndarray:
    valid = np.isfinite(distances) if mask is None else np.isfinite(distances) & mask[..., None, :]
    counts = valid.sum(axis=-1, keepdims=True)
    assert int(counts.min()) >= min_points
    if theta == 0.0:
        return np.where(valid, 1.0, 0.0)
    mean = np.maximum(np.where(valid, distances, 0.0).sum(axis=-1, keepdims=True) / counts, 1e-6)
    return np.where(valid, np.exp(-theta * distances / mean), 0.0)


def check_weights(distances: np.ndarray, theta: float, mask: np.ndarray | None, min_points: int) -> None:
    actual = weights(distances, theta, mask=mask, min_points=min_points)
    expected = weights_reference(distances, theta, mask, min_points)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def check_smap_tensor(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    expected = smap(x, y, q, theta=theta, alpha=alpha, mask=mask)
    actual = smap(
        Tensor(x),
        Tensor(y),
        Tensor(q),
        theta=theta,
        alpha=alpha,
        mask=None if mask is None else Tensor(mask),
    ).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_smap_tensor_gradient(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    coincident = min(q.shape[-2], 2)
    q[..., :coincident, :] = x[..., :coincident, :]  # coincident query and library points produce zero distances
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = smap(X, Y, Q, theta=theta, alpha=alpha).sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


@st.composite
def smap_problems(draw):
    e = draw(st.integers(1, 3))
    n = draw(st.integers(e + 5, 18))
    m = draw(st.integers(2, 5))
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
    return SmapProblem(
        x,
        y,
        q,
        draw(st.sampled_from((0.0, 0.5, 2.5))),
        draw(st.sampled_from((0.0, 1e-4, 0.25))),
        mask,
    )


@given(problem=smap_problems())
def test_smap_compatibility(problem: SmapProblem) -> None:
    check_smap(*problem)


SMAP_VALID = {
    "global-scalar-2d": SmapCase(
        np.random.default_rng(0).normal(size=(16, 2)),
        np.random.default_rng(1).normal(size=16),
        np.random.default_rng(2).normal(size=(4, 2)),
        0.0,
        0.0,
        None,
    ),
    "single-query-scalar-2d": SmapCase(
        np.random.default_rng(3).normal(size=(16, 2)),
        np.random.default_rng(4).normal(size=16),
        np.random.default_rng(5).normal(size=(1, 2)),
        1.0,
        1e-4,
        None,
    ),
    "local-multitarget-2d": SmapCase(
        np.random.default_rng(6).normal(size=(16, 2)),
        np.random.default_rng(7).normal(size=(16, 2)),
        np.random.default_rng(8).normal(size=(4, 2)),
        2.5,
        1e-4,
        None,
    ),
    "regularized-masked-scalar-2d": SmapCase(
        np.random.default_rng(9).normal(size=(16, 2)),
        np.random.default_rng(10).normal(size=16),
        np.random.default_rng(11).normal(size=(4, 2)),
        1.0,
        0.25,
        np.array([True] * 14 + [False] * 2),
    ),
    "local-scalar-batched-3d": SmapCase(
        np.random.default_rng(12).normal(size=(2, 16, 2)),
        np.random.default_rng(13).normal(size=(2, 16, 1)),
        np.random.default_rng(14).normal(size=(2, 4, 2)),
        1.5,
        1e-4,
        None,
    ),
    "regularized-masked-multitarget-batched-3d": SmapCase(
        np.random.default_rng(15).normal(size=(2, 16, 2)),
        np.random.default_rng(16).normal(size=(2, 16, 2)),
        np.random.default_rng(17).normal(size=(2, 4, 2)),
        2.0,
        0.1,
        np.array([[True] * 15 + [False], [True] * 14 + [False] * 2]),
    ),
}

SMAP_MODES = {
    "numpy": check_smap,
    "tinygrad": pytest.param(check_smap_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_smap_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", SMAP_MODES.values(), ids=SMAP_MODES.keys())
@pytest.mark.parametrize("case", SMAP_VALID.values(), ids=SMAP_VALID.keys())
def test_smap_valid(case: SmapCase, mode) -> None:
    mode(*case)


SMAP_INVALID = {
    "negative-theta": SmapCase(np.zeros((5, 2)), np.zeros(5), np.zeros((2, 2)), -1.0, 0.0, None),
    "library-target-length": SmapCase(np.zeros((5, 2)), np.zeros(4), np.zeros((2, 2)), 1.0, 0.0, None),
    "batched-library-target-length": SmapCase(np.zeros((2, 5, 2)), np.zeros((2, 4, 1)), np.zeros((2, 2, 2)), 1.0, 0.0, None),
    "mixed-ranks": SmapCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), 1.0, 0.0, None),
    "too-few-unmasked-points": SmapCase(
        np.arange(10.0).reshape(5, 2),
        np.arange(5.0),
        np.array([[0.0, 0.0]]),
        1.0,
        0.0,
        np.array([True, True, False, False, False]),
    ),
}


@pytest.mark.parametrize("case", SMAP_INVALID.values(), ids=SMAP_INVALID.keys())
def test_smap_invalid(case: SmapCase) -> None:
    with pytest.raises(ValueError):
        smap(case.x, case.y, case.q, theta=case.theta, alpha=case.alpha, mask=case.mask)


WEIGHTS_VALID = {
    "global-finite": WeightsCase(np.array([[0.0, 1.0, np.inf]]), 0.0, None, 2),
    "local-masked": WeightsCase(
        np.array([[0.0, 1.0, 2.0]]),
        2.0,
        np.array([True, True, False]),
        2,
    ),
}

WEIGHTS_INVALID = {
    "too-few-finite-points": WeightsCase(np.array([[0.0, np.inf, np.inf]]), 2.0, None, 2),
}


@pytest.mark.parametrize("case", WEIGHTS_VALID.values(), ids=WEIGHTS_VALID.keys())
def test_weights_valid(case: WeightsCase) -> None:
    check_weights(*case)


@pytest.mark.parametrize("case", WEIGHTS_INVALID.values(), ids=WEIGHTS_INVALID.keys())
def test_weights_invalid(case: WeightsCase) -> None:
    with pytest.raises(ValueError):
        weights(case.distances, case.theta, mask=case.mask, min_points=case.min_points)
