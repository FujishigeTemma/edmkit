from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.smap import coefficients, linear_operator, smap
from edmkit.theiler_window import theiler_window


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


def smap_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    theta: float,
    alpha: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Independent S-Map reference returning the canonical ``(M, targets)`` shape; `mask` is per-query with shape (M, N)."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        return np.concatenate([smap_reference(x[keep], y[keep], q[query : query + 1], theta=theta, alpha=alpha) for query, keep in enumerate(mask)])

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
        expected = smap_reference(
            x,
            y,
            q,
            theta=theta,
            alpha=alpha,
            mask=mask,
        )
        if y.ndim == 1:
            expected = expected[..., 0]
    else:
        expected = np.stack(
            [
                smap_reference(
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
        np.array([[True] * 14 + [False] * 2] * 3 + [[False] * 2 + [True] * 14]),
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
        np.stack(
            [
                np.array([[True] * 15 + [False]] * 3 + [[False] + [True] * 15]),
                np.array([[True] * 14 + [False] * 2] * 3 + [[False] * 2 + [True] * 14]),
            ]
        ),
    ),
    "single-query-multitarget-2d": SmapCase(
        np.random.default_rng(20).normal(size=(16, 2)),
        np.random.default_rng(21).normal(size=(16, 3)),
        np.random.default_rng(22).normal(size=(1, 2)),
        1.0,
        1e-4,
        None,
    ),
    "theiler-self-query-2d": SmapCase(
        np.random.default_rng(18).normal(size=(20, 2)),
        np.random.default_rng(19).normal(size=20),
        np.random.default_rng(18).normal(size=(20, 2)),
        1.5,
        1e-4,
        theiler_window(np.arange(20), np.arange(20), 2),
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
    "query-dimension": SmapCase(np.zeros((6, 2)), np.zeros(6), np.zeros((2, 3)), 1.0, 0.0, None),
    "batch-size-mismatch": SmapCase(np.zeros((1, 6, 2)), np.zeros((1, 6, 1)), np.zeros((4, 2, 2)), 1.0, 0.0, None),
    "mixed-ranks": SmapCase(np.zeros((2, 5, 2)), np.zeros((2, 5, 1)), np.zeros((2, 2)), 1.0, 0.0, None),
    "too-few-unmasked-points": SmapCase(
        np.arange(10.0).reshape(5, 2),
        np.arange(5.0),
        np.array([[0.0, 0.0]]),
        1.0,
        0.0,
        np.array([[True, True, False, False, False]]),
    ),
    "mask-not-per-query": SmapCase(
        np.zeros((5, 2)),
        np.zeros(5),
        np.zeros((2, 2)),
        1.0,
        0.0,
        np.ones(5, dtype=bool),
    ),
}


@pytest.mark.parametrize("case", SMAP_INVALID.values(), ids=SMAP_INVALID.keys())
def test_smap_invalid(case: SmapCase) -> None:
    with pytest.raises(ValueError):
        smap(case.x, case.y, case.q, theta=case.theta, alpha=case.alpha, mask=case.mask)


class LinearOperatorProblem(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


class LinearOperatorCase(NamedTuple):
    x: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


def check_linear_operator(x: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    actual = linear_operator(x, q, theta=theta, alpha=alpha, mask=mask)
    # projecting the identity target yields the operator itself
    identity = np.eye(x.shape[-2])
    if x.ndim == 2:
        expected = smap_reference(x, identity, q, theta=theta, alpha=alpha, mask=mask)
    else:
        expected = np.stack(
            [
                smap_reference(xi, identity, qi, theta=theta, alpha=alpha, mask=None if mask is None else mask[batch])
                for batch, (xi, qi) in enumerate(zip(x, q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def check_linear_operator_tensor(x: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, q = (array.astype(np.float32) for array in (x, q))
    expected = linear_operator(x, q, theta=theta, alpha=alpha, mask=mask)
    actual = linear_operator(Tensor(x), Tensor(q), theta=theta, alpha=alpha, mask=None if mask is None else Tensor(mask)).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_linear_operator_tensor_gradient(x: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, q = (array.astype(np.float32) for array in (x, q))
    coincident = min(q.shape[-2], 2)
    q[..., :coincident, :] = x[..., :coincident, :]  # coincident query and library points produce zero distances
    X, Q = Tensor(x), Tensor(q)
    gradients = linear_operator(X, Q, theta=theta, alpha=alpha).square().sum().gradient(X, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def linear_operator_problems():
    return smap_problems().map(lambda problem: LinearOperatorProblem(problem.x, problem.q, problem.theta, problem.alpha, problem.mask))


@given(problem=linear_operator_problems())
def test_linear_operator_compatibility(problem: LinearOperatorProblem) -> None:
    check_linear_operator(*problem)


LINEAR_OPERATOR_VALID = {name: LinearOperatorCase(case.x, case.q, case.theta, case.alpha, case.mask) for name, case in SMAP_VALID.items()}

LINEAR_OPERATOR_MODES = {
    "numpy": check_linear_operator,
    "tinygrad": pytest.param(check_linear_operator_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_linear_operator_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", LINEAR_OPERATOR_MODES.values(), ids=LINEAR_OPERATOR_MODES.keys())
@pytest.mark.parametrize("case", LINEAR_OPERATOR_VALID.values(), ids=LINEAR_OPERATOR_VALID.keys())
def test_linear_operator_valid(case: LinearOperatorCase, mode) -> None:
    mode(*case)


# the target plays no part in the operator, so drop the cases that are invalid only because of it
LINEAR_OPERATOR_INVALID = {
    name: LinearOperatorCase(case.x, case.q, case.theta, case.alpha, case.mask)
    for name, case in SMAP_INVALID.items()
    if name not in ("library-target-length", "batched-library-target-length")
}


@pytest.mark.parametrize("case", LINEAR_OPERATOR_INVALID.values(), ids=LINEAR_OPERATOR_INVALID.keys())
def test_linear_operator_invalid(case: LinearOperatorCase) -> None:
    with pytest.raises(ValueError):
        linear_operator(case.x, case.q, theta=case.theta, alpha=case.alpha, mask=case.mask)


class CoefficientsProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


class CoefficientsCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    theta: float
    alpha: float
    mask: np.ndarray | None


def coefficients_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    theta: float,
    alpha: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Independent S-Map coefficients reference returning the canonical ``(M, E+1, targets)`` shape; `mask` is per-query with shape (M, N)."""
    y = y[:, None] if y.ndim == 1 else y
    if mask is not None:
        return np.concatenate(
            [coefficients_reference(x[keep], y[keep], q[query : query + 1], theta=theta, alpha=alpha) for query, keep in enumerate(mask)]
        )

    x_aug = np.column_stack([np.ones(len(x)), x])
    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=-1)
    if theta == 0.0:
        query_weights = np.ones_like(distances)
    else:
        scale = np.maximum(distances.mean(axis=1, keepdims=True), 1e-6)
        query_weights = np.exp(-theta * distances / scale)

    penalty = np.eye(x_aug.shape[1])[1:]
    result = np.empty((len(q), x_aug.shape[1], y.shape[1]))
    for i, weight in enumerate(query_weights):
        sqrt_weight = np.sqrt(weight)[:, None]
        design = sqrt_weight * x_aug
        target = sqrt_weight * y
        trace = max(float(np.sum(design**2)), 1e-12)
        regularizer = np.sqrt(alpha * trace) * penalty
        design = np.vstack([design, regularizer])
        target = np.vstack([target, np.zeros((len(regularizer), y.shape[1]))])
        result[i], *_ = np.linalg.lstsq(design, target, rcond=None)
    return result


def check_coefficients(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    actual = coefficients(x, y, q, theta=theta, alpha=alpha, mask=mask)
    if x.ndim == 2:
        expected = coefficients_reference(x, y, q, theta=theta, alpha=alpha, mask=mask)
        if y.ndim == 1:
            expected = expected[..., 0]
    else:
        expected = np.stack(
            [
                coefficients_reference(xi, yi, qi, theta=theta, alpha=alpha, mask=None if mask is None else mask[batch])
                for batch, (xi, yi, qi) in enumerate(zip(x, y, q))
            ]
        )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def check_coefficients_tensor(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    expected = coefficients(x, y, q, theta=theta, alpha=alpha, mask=mask)
    actual = coefficients(Tensor(x), Tensor(y), Tensor(q), theta=theta, alpha=alpha, mask=None if mask is None else Tensor(mask)).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def check_coefficients_tensor_gradient(x: np.ndarray, y: np.ndarray, q: np.ndarray, theta: float, alpha: float, mask: np.ndarray | None) -> None:
    from tinygrad import Tensor

    x, y, q = (array.astype(np.float32) for array in (x, y, q))
    coincident = min(q.shape[-2], 2)
    q[..., :coincident, :] = x[..., :coincident, :]  # coincident query and library points produce zero distances
    X, Y, Q = Tensor(x), Tensor(y), Tensor(q)
    gradients = coefficients(X, Y, Q, theta=theta, alpha=alpha).square().sum().gradient(X, Y, Q)
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()


def coefficients_problems():
    return smap_problems().map(lambda problem: CoefficientsProblem(*problem))


@given(problem=coefficients_problems())
def test_coefficients_compatibility(problem: CoefficientsProblem) -> None:
    check_coefficients(*problem)


COEFFICIENTS_VALID = {name: CoefficientsCase(*case) for name, case in SMAP_VALID.items()}

COEFFICIENTS_MODES = {
    "numpy": check_coefficients,
    "tinygrad": pytest.param(check_coefficients_tensor, marks=pytest.mark.gpu),
    "gradient": pytest.param(check_coefficients_tensor_gradient, marks=pytest.mark.gpu),
}


@pytest.mark.parametrize("mode", COEFFICIENTS_MODES.values(), ids=COEFFICIENTS_MODES.keys())
@pytest.mark.parametrize("case", COEFFICIENTS_VALID.values(), ids=COEFFICIENTS_VALID.keys())
def test_coefficients_valid(case: CoefficientsCase, mode) -> None:
    mode(*case)


COEFFICIENTS_INVALID = {name: CoefficientsCase(*case) for name, case in SMAP_INVALID.items()}


@pytest.mark.parametrize("case", COEFFICIENTS_INVALID.values(), ids=COEFFICIENTS_INVALID.keys())
def test_coefficients_invalid(case: CoefficientsCase) -> None:
    with pytest.raises(ValueError):
        coefficients(case.x, case.y, case.q, theta=case.theta, alpha=case.alpha, mask=case.mask)
