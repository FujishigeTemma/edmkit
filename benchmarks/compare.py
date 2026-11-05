"""
uv run --group benchmark pytest benchmarks/compare.py --benchmark-autosave
"""

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import pyEDM
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from benchmarks.run import TAU, Shape, shape_marks, trajectory
from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

E = 3
COORDINATES = np.array([[0, TAU * j] for j in range(E)])

SAMPLES = 20
LIB_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
WORKERS = os.cpu_count() or 1
# the Theiler window, |t_i - t_j| <= radius excluded, given as theiler_window's width and pyEDM's exclusionRadius
EXCLUSION_RADIUS = TAU * (E - 1)

# pyEDM's forkserver workers preload __main__, which already imports pyEDM in a user's script but not under pytest;
# without this, every parallel CCM call pays ~1 s of worker imports
multiprocessing.set_forkserver_preload(["pyEDM"])


def pair(N: int) -> np.ndarray:
    """Lorenz x- and y-coordinates of length N, of shape (2, N)."""
    return np.ascontiguousarray(trajectory(1, N)[0, :, :2].T)


def lib_sizes(N: int) -> np.ndarray:
    M = N - TAU * (E - 1)
    return np.array([round(fraction * M) for fraction in LIB_FRACTIONS])


def ccm_edmkit(xy: np.ndarray, workers: int) -> np.ndarray:
    """Mean correlations of x:y and y:x for each library size, of shape (len(LIB_FRACTIONS), 2)."""
    x, y = xy
    X, t = embed(COORDINATES, x)
    Y, _ = embed(COORDINATES, y)
    mask = theiler_window(t, t, EXCLUSION_RADIUS)
    sizes = lib_sizes(len(x))

    def score(job: tuple[tuple, np.ndarray, Callable[[], np.ndarray]]) -> tuple[tuple, np.ndarray]:
        index, target, samples = job
        predictions = samples()
        return index, pearson_correlation(predictions, np.broadcast_to(target[:, None], predictions.shape))

    jobs = [
        ((direction, *index), target, samples)
        for direction, (library, target) in enumerate(((X, y[t]), (Y, x[t])))
        for index, samples in bootstrap(
            library, target, library, sizes, predict_func=simplex_projection, n_samples=SAMPLES, replace=False, seed=direction, mask=mask
        )
    ]
    scores = np.empty((2, len(sizes), SAMPLES))
    with ThreadPoolExecutor(workers) as executor:
        for index, result in executor.map(score, jobs):
            scores[index] = result
    return scores.mean(axis=-1).T


def ccm_pyedm(xy: np.ndarray, workers: int) -> np.ndarray:
    x, y = xy
    N = len(x)
    df = pd.DataFrame({"time": np.arange(1, N + 1), "x": x, "y": y})
    out = pyEDM.CCM(
        dataFrame=df,
        columns="x",
        target="y",
        libSizes=lib_sizes(N).tolist(),
        sample=SAMPLES,
        E=E,
        tau=-TAU,
        Tp=0,
        exclusionRadius=EXCLUSION_RADIUS,
        seed=0,
        parallel=workers,
    )
    return out[["x:y", "y:x"]].to_numpy()


FUNCS = {
    "edmkit": partial(ccm_edmkit, workers=1),
    "pyEDM": partial(ccm_pyedm, workers=1),
    "edmkit-parallel": partial(ccm_edmkit, workers=WORKERS),
    "pyEDM-parallel": partial(ccm_pyedm, workers=WORKERS),
}
SIZES = (500, 1000, 2000, 5000, 10000)
CASES = [pytest.param(backend, N, id=f"ccm-{backend}-{N}", marks=shape_marks(Shape(N=N, E=E, B=None))) for N in SIZES for backend in FUNCS]


@pytest.mark.parametrize(("backend", "N"), CASES)
def test_compare(benchmark: BenchmarkFixture, backend: str, N: int) -> None:
    benchmark.extra_info.update(func="ccm", backend=backend, data="lorenz", N=N, E=E, B=None)
    benchmark(FUNCS[backend], pair(N))
