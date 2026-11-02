"""
uv run pytest benchmarks -m "not (dense and huge)" --benchmark-autosave
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, NamedTuple, Protocol, overload

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from edmkit import generate
from edmkit.embedding import embed

if TYPE_CHECKING:
    from tinygrad import Tensor

type Call = Callable[[], object]


class Named(Protocol):
    @property
    def __name__(self) -> str: ...


class Func(Named, Protocol):
    """(X, Y, Q) -> predictions or operator, on either backend."""

    @overload
    def __call__(self, X: np.ndarray, Y: np.ndarray, Q: np.ndarray, /) -> np.ndarray: ...
    @overload
    def __call__(self, X: Tensor, Y: Tensor, Q: Tensor, /) -> Tensor: ...


class Data(Named, Protocol):
    """Shape -> (X, Y)."""

    def __call__(self, shape: Shape, /) -> tuple[np.ndarray, np.ndarray]: ...


class Backend(Named, Protocol):
    """Prepares a zero-argument call of `func` on (X, Y, X)."""

    def __call__(self, func: Func, X: np.ndarray, Y: np.ndarray, /) -> Call: ...


def named(f: Named, *marks: pytest.MarkDecorator) -> pytest.ParameterSet:
    return pytest.param(f, id=f.__name__, marks=marks)


class Shape(NamedTuple):
    """Input layout of `X`: (N,) if `E` is None, (N, E) if `B` is None, otherwise (B, N, E)."""

    N: int
    E: int | None
    B: int | None


def dims(shape: Shape) -> tuple[int, ...]:
    return tuple(d for d in (shape.B, shape.N, shape.E) if d is not None)


def layout(shape: Shape, a: np.ndarray) -> np.ndarray:
    """Drop the batch axis of `a` (B, N, E) when `shape.B` is None and the last axis when `shape.E` is None."""
    a = a if shape.B is not None else a[0]
    return a if shape.E is not None else a[..., 0]


@cache
def random(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Library `X` and target `Y` drawn from a standard normal."""
    rng = np.random.default_rng(0)
    B, E = shape.B or 1, shape.E or 1
    return layout(shape, rng.normal(size=(B, shape.N, E))), layout(shape, rng.normal(size=(B, shape.N, 1)))


TAU = 5
DT = 0.01
BURN_IN = 1000


@cache
def lorenz(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Delay embedding `X` of the Lorenz x-coordinate and its one-step-ahead value `Y`."""
    B, E = shape.B or 1, shape.E or 1
    shift = TAU * (E - 1)
    T = shape.N + shift + 1
    X0 = np.array([1.0, 1.0, 1.0]) + np.random.default_rng(0).normal(scale=0.1, size=(B, 3))
    _, states = generate.rk45(generate.lorenz(10.0, 28.0, 8 / 3), X0, dt=DT, t_max=(BURN_IN + T) * DT)
    x = states[:, BURN_IN:, 0]  # (B, T)
    coordinates = np.array([[0, TAU * j] for j in range(E)])
    X = np.stack([embed(coordinates, xb)[0][: shape.N] for xb in x])  # (B, N, E)
    Y = x[:, shift + 1 : shift + 1 + shape.N, None]  # (B, N, 1)
    return layout(shape, X), layout(shape, Y)


DATA = [named(lorenz), named(random)]


def shape_marks(shape: Shape) -> list[pytest.MarkDecorator]:
    slow = shape.N >= 10000 or (shape.B or 0) >= 64
    huge = shape.N >= 100000
    return [mark for mark, on in ((pytest.mark.slow, slow), (pytest.mark.huge, huge)) if on]


BASE = Shape(N=2000, E=3, B=None)
SHAPES = [
    pytest.param(shape, id="x".join(map(str, dims(shape))), marks=shape_marks(shape))
    for shape in dict.fromkeys(
        [
            *(BASE._replace(N=N) for N in (500, 2000, 5000, 10000, 100000)),
            *(BASE._replace(E=E) for E in (None, 1, 8, 27, 64, 125)),
            *(BASE._replace(B=B) for B in (None, 1, 16, 64, 128)),
        ]
    )
]


def numpy(func: Func, X: np.ndarray, Y: np.ndarray) -> Call:
    return lambda: func(X, Y, X)


def tensor(func: Func, X: np.ndarray, Y: np.ndarray) -> Call:
    """Jitted float32 call that waits for the device"""
    from tinygrad import Device, Tensor, TinyJit

    # Metal has no float64; Q gets its own buffer since TinyJit rejects duplicate inputs
    X_t, Y_t, Q_t = (Tensor(a.astype(np.float32)).realize() for a in (X, Y, X))

    def realized(X: Tensor, Y: Tensor, Q: Tensor) -> Tensor:
        return func(X, Y, Q).realize()

    jitted = TinyJit(realized)

    def call() -> Tensor:
        out = jitted(X_t, Y_t, Q_t)
        Device[Device.DEFAULT].synchronize()
        return out

    # TinyJit runs eagerly on the first call and captures on the second
    call()
    call()
    return call


BACKENDS = [named(numpy), named(tensor, pytest.mark.gpu, pytest.mark.dense)]


def run(benchmark: BenchmarkFixture, func: Func, backend: Backend, data: Data, shape: Shape) -> None:
    benchmark.extra_info.update(func=func.__name__, backend=backend.__name__, data=data.__name__, **shape._asdict())
    benchmark(backend(func, *data(shape)))
