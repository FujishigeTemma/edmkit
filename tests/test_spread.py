from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.spread import spread


class SpreadProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None


class SpreadCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    k: int | None
    mask: np.ndarray | None


def spread_reference(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Brute-force variance of the neighbouring outputs via `np.cov`, one query at a time; `mask` is per-query with shape (M, N)."""
    x = x[:, None] if x.ndim == 1 else x
    q = q[:, None] if q.ndim == 1 else q
    y = y[:, None] if y.ndim == 1 else y
    k = x.shape[1] + 1 if k is None else k

    V = np.empty((len(q), y.shape[1]))
    for m in range(len(q)):
        distances = np.linalg.norm(x - q[m], axis=1)
        if mask is not None:
            distances[~mask[m]] = np.inf
        nearest = np.argsort(distances, kind="stable")[:k]
        V[m] = np.cov(y[nearest], rowvar=False, bias=True).reshape(-1, y.shape[1]).diagonal() if k > 1 else 0.0
    return V


def check_spread(x: np.ndarray, y: np.ndarray, q: np.ndarray, k: int | None, mask: np.ndarray | None) -> None:
    actual = spread(x, y, q, k=k, mask=mask)
    expected = spread_reference(x, y, q, k=k, mask=mask)

    assert actual.shape == expected.shape
    assert (actual >= 0).all()
    scale = max(float(np.ptp(y)) ** 2, 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-9 * scale, rtol=1e-7)


@st.composite
def spread_problems(draw) -> SpreadProblem:
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    N = draw(st.integers(6, 40))
    M = draw(st.integers(1, 8))
    E = draw(st.integers(1, 4))
    outputs = draw(st.integers(1, 3))
    k = draw(st.one_of(st.none(), st.integers(1, 5)))
    offset = draw(st.sampled_from([0.0, 1e3]))

    x = rng.normal(size=(N, E))
    y = rng.normal(size=(N, outputs)) + offset
    q = rng.normal(size=(M, E))

    mask = None
    if draw(st.booleans()):
        mask = np.ones((M, N), dtype=bool)
        mask[np.arange(M), rng.integers(0, N, size=M)] = False
    return SpreadProblem(x, y[:, 0] if outputs == 1 and draw(st.booleans()) else y, q, k, mask)


_rng = np.random.default_rng(0)
_x = _rng.normal(size=(30, 2))
_y = _rng.normal(size=(30, 3))

SPREAD_VALID = {
    "several-outputs": SpreadCase(_x, _y, _x[:5] + 0.01, 4, None),
    "one-dimensional": SpreadCase(_x[:, 0], _y[:, 0], _x[:5, 0] + 0.01, 3, None),
    "single-query": SpreadCase(_x, _y, _x[:1] + 0.01, None, None),
    "constant-output": SpreadCase(_x, np.full((30, 2), 7.0), _x[:5] + 0.01, 4, None),
    "single-neighbour": SpreadCase(_x, _y, _x[:5] + 0.01, 1, None),
    "leave-one-out": SpreadCase(_x, _y, _x, 4, ~np.eye(30, dtype=bool)),
}

SPREAD_INVALID = {
    "three-dimensional-output": SpreadCase(_x, _y[:, :, None], _x[:5], 4, None),
    "length-mismatch": SpreadCase(_x, _y[:-1], _x[:5], 4, None),
    "non-positive-k": SpreadCase(_x, _y, _x[:5], 0, None),
    "too-few-unmasked": SpreadCase(_x[:4], _y[:4], _x[:4], 4, ~np.eye(4, dtype=bool)),
}


@given(spread_problems())
def test_spread_compatibility(problem: SpreadProblem) -> None:
    check_spread(*problem)


@pytest.mark.parametrize("case", SPREAD_VALID.values(), ids=SPREAD_VALID.keys())
def test_spread_valid(case: SpreadCase) -> None:
    check_spread(*case)


@pytest.mark.parametrize("case", SPREAD_INVALID.values(), ids=SPREAD_INVALID.keys())
def test_spread_invalid(case: SpreadCase) -> None:
    with pytest.raises(ValueError):
        spread(*case[:3], k=case.k, mask=case.mask)
