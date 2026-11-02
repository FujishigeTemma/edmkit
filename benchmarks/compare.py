"""
uv run --group benchmark pytest benchmarks/compare.py --benchmark-autosave
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pyEDM
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from benchmarks.run import TAU, Shape, lorenz, shape_marks
from edmkit.embedding import embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap

E = 3
THETA = 2.0  # pyEDM's theta


def series(N: int) -> np.ndarray:
    """Lorenz x-coordinate of length 2N."""
    return lorenz(Shape(N=2 * N, E=None, B=None))[0]


def edmkit(f: Callable[..., np.ndarray], x: np.ndarray, **kwargs) -> np.ndarray:
    N = len(x) // 2
    X, t = embed(np.array([[0, TAU * j] for j in range(E)]), x)
    lib = t <= N - 2  # targets x[t + 1] stay in the library
    pred = (t >= N) & (t <= 2 * N - 2)  # targets x[t + 1] exist
    return f(X[lib], x[t[lib] + 1], X[pred], **kwargs)


def pyedm(f: Callable[..., pd.DataFrame], x: np.ndarray, **kwargs) -> np.ndarray:
    N = len(x) // 2
    df = pd.DataFrame({"time": np.arange(1, 2 * N + 1), "x": x})
    out = f(dataFrame=df, columns="x", target="x", lib=f"1 {N}", pred=f"{N + 1} {2 * N}", E=E, tau=-TAU, Tp=1, kdWorkers=1, **kwargs)
    out = out["predictions"] if isinstance(out, dict) else out
    # drop the NaN row before the first prediction and the one past the end of the series
    return out["Predictions"].to_numpy()[1:-1]


def simplex_edmkit(x: np.ndarray) -> np.ndarray:
    return edmkit(simplex_projection, x)


def simplex_pyedm(x: np.ndarray) -> np.ndarray:
    return pyedm(pyEDM.Simplex, x)


def smap_edmkit(x: np.ndarray) -> np.ndarray:
    return edmkit(smap, x, theta=2 * THETA)


def smap_pyedm(x: np.ndarray) -> np.ndarray:
    library_size = len(x) // 2 - 1 - TAU * (E - 1)
    return pyedm(pyEDM.SMap, x, theta=THETA, knn=library_size)


FUNCS = {"simplex": {"edmkit": simplex_edmkit, "pyEDM": simplex_pyedm}, "smap": {"edmkit": smap_edmkit, "pyEDM": smap_pyedm}}
SIZES = {"simplex": (500, 1000, 2000, 5000, 10000, 20000), "smap": (500, 1000, 2000, 5000)}
CASES = [pytest.param(method, N, id=f"{method}-{N}", marks=shape_marks(Shape(N=N, E=E, B=None))) for method, sizes in SIZES.items() for N in sizes]


@pytest.mark.parametrize(("method", "N"), CASES)
def test_agreement(method: str, N: int) -> None:
    x = series(N)
    ours, theirs = FUNCS[method]["edmkit"](x), FUNCS[method]["pyEDM"](x)
    # S-Map solves the normal equations in edmkit and calls lstsq in pyEDM
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=(1e-10 if method == "simplex" else 1e-4) * np.abs(theirs).max())


@pytest.mark.parametrize("library", ["edmkit", "pyEDM"])
@pytest.mark.parametrize(("method", "N"), CASES)
def test_compare(benchmark: BenchmarkFixture, method: str, N: int, library: str) -> None:
    benchmark.extra_info.update(func=method, backend=library, data="lorenz", N=N, E=E, B=None)
    x = series(N)
    benchmark(FUNCS[method][library], x)
