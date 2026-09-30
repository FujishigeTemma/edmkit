from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.embedding import embed


class EmbedProblem(NamedTuple):
    coordinates: np.ndarray
    X: np.ndarray


class EmbedCase(NamedTuple):
    coordinates: np.ndarray
    X: np.ndarray


def check_embed(coordinates: np.ndarray, X: np.ndarray) -> None:
    U, times = embed(coordinates, X)
    X = X[:, None] if X.ndim == 1 else X

    assert times.ndim == 1
    assert U.shape == (len(times), len(coordinates))
    np.testing.assert_array_equal(times, np.arange(times[0], times[0] + len(times)))

    # Every entry is the observation the coordinate names, read directly from X.
    for a, t in enumerate(times):
        for j, (variable, lag) in enumerate(coordinates):
            assert 0 <= t - lag < len(X)
            assert U[a, j] == X[t - lag, variable]

    # The returned times are exactly those for which that is possible.
    for t in (int(times[0]) - 1, int(times[-1]) + 1):
        assert any(not 0 <= t - lag < len(X) for lag in coordinates[:, 1])


@st.composite
def embed_problems(draw) -> EmbedProblem:
    d = draw(st.integers(1, 3))
    e = draw(st.integers(1, 4))
    lags = draw(st.lists(st.integers(-3, 5), min_size=e, max_size=e))
    variables = draw(st.lists(st.integers(0, d - 1), min_size=e, max_size=e))
    coordinates = np.array([variables, lags], dtype=np.int64).T
    span = max(lags) - min(min(lags), 0) + 1
    T = draw(st.integers(span, span + 20))
    X = draw(hnp.arrays(np.int64, (T, d), elements=st.integers(-10_000, 10_000)))
    return EmbedProblem(coordinates, X)


EMBED_VALID = {
    "single-coordinate": EmbedCase(np.array([[0, 0]]), np.arange(6).reshape(6, 1)),
    "classical-univariate": EmbedCase(np.array([[0, 0], [0, 2], [0, 4]]), np.arange(10).reshape(10, 1)),
    "mixed-variables-and-lags": EmbedCase(np.array([[0, 0], [1, 3], [1, 1]]), np.arange(20).reshape(10, 2)),
    "output-one-step-ahead": EmbedCase(np.array([[0, -1], [1, -1]]), np.arange(20).reshape(10, 2)),
    "lags-spanning-past-and-future": EmbedCase(np.array([[0, 2], [0, -2]]), np.arange(12).reshape(6, 2)),
    "one-dimensional-observations": EmbedCase(np.array([[0, 0], [0, 2]]), np.arange(6)),
    "exact-minimum-length": EmbedCase(np.array([[0, 0], [0, 3]]), np.arange(4).reshape(4, 1)),
}

EMBED_INVALID = {
    "one-dimensional-coordinates": EmbedCase(np.array([0, 1]), np.arange(6).reshape(6, 1)),
    "wrong-coordinate-width": EmbedCase(np.array([[0, 1, 2]]), np.arange(6).reshape(6, 1)),
    "empty-coordinates": EmbedCase(np.zeros((0, 2), dtype=np.int64), np.arange(6).reshape(6, 1)),
    "float-coordinates": EmbedCase(np.array([[0.0, 1.0]]), np.arange(6).reshape(6, 1)),
    "three-dimensional-observations": EmbedCase(np.array([[0, 0]]), np.zeros((6, 1, 1))),
    "variable-out-of-range": EmbedCase(np.array([[1, 0]]), np.arange(6).reshape(6, 1)),
    "negative-variable": EmbedCase(np.array([[-1, 0]]), np.arange(6).reshape(6, 1)),
    "insufficient-history": EmbedCase(np.array([[0, 0], [0, 6]]), np.arange(4).reshape(4, 1)),
    "insufficient-future": EmbedCase(np.array([[0, 3], [0, -3]]), np.arange(5).reshape(5, 1)),
}


@given(problem=embed_problems())
def test_embed_compatibility(problem: EmbedProblem) -> None:
    check_embed(*problem)


@pytest.mark.parametrize("case", EMBED_VALID.values(), ids=EMBED_VALID.keys())
def test_embed_valid(case: EmbedCase) -> None:
    check_embed(*case)


@pytest.mark.parametrize("case", EMBED_INVALID.values(), ids=EMBED_INVALID.keys())
def test_embed_invalid(case: EmbedCase) -> None:
    with pytest.raises(ValueError):
        embed(case.coordinates, case.X)
