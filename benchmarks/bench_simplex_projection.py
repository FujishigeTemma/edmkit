from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import edmkit.simplex_projection
from benchmarks.run import BACKENDS, DATA, SHAPES, Backend, Data, Func, Shape, named, run

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def simplex_projection(X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
@overload
def simplex_projection(X: Tensor, Y: Tensor, Q: Tensor) -> Tensor: ...
def simplex_projection(X, Y, Q):
    return edmkit.simplex_projection.simplex_projection(X, Y, Q)


@overload
def simplex_linear_operator(X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
@overload
def simplex_linear_operator(X: Tensor, Y: Tensor, Q: Tensor) -> Tensor: ...
def simplex_linear_operator(X, Y, Q):
    return edmkit.simplex_projection.linear_operator(X, Q)


FUNCS = [named(simplex_projection), named(simplex_linear_operator, pytest.mark.dense)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("data", DATA)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("func", FUNCS)
def test_simplex_projection(benchmark: BenchmarkFixture, func: Func, backend: Backend, data: Data, shape: Shape) -> None:
    run(benchmark, func, backend, data, shape)
