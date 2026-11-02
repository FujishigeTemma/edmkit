from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import edmkit.smap
from benchmarks.run import BACKENDS, DATA, SHAPES, Backend, Data, Func, Shape, named, run

if TYPE_CHECKING:
    from tinygrad import Tensor

THETA = 4.0


@overload
def smap(X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
@overload
def smap(X: Tensor, Y: Tensor, Q: Tensor) -> Tensor: ...
def smap(X, Y, Q):
    return edmkit.smap.smap(X, Y, Q, theta=THETA)


@overload
def smap_linear_operator(X: np.ndarray, Y: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
@overload
def smap_linear_operator(X: Tensor, Y: Tensor, Q: Tensor) -> Tensor: ...
def smap_linear_operator(X, Y, Q):
    return edmkit.smap.linear_operator(X, Q, theta=THETA)


FUNCS = [named(smap, pytest.mark.dense), named(smap_linear_operator, pytest.mark.dense)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("data", DATA)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("func", FUNCS)
def test_smap(benchmark: BenchmarkFixture, func: Func, backend: Backend, data: Data, shape: Shape) -> None:
    run(benchmark, func, backend, data, shape)
