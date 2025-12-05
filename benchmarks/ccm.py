from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
from tabulate import tabulate

from edmkit.ccm import ccm
from edmkit.simplex_projection import simplex_projection

rng = np.random.default_rng(42)

N_POINTS = (100, 1000, 10000)
X_DIMS = (1, 10, 100)
Y_DIMS = (1, 10, 100)
N_SAMPLES = (10, 20, 30)
LIBRARY_RATIO = 0.7
LIB_SIZE_STEPS = 4
BATCH_LIMIT = 8


@dataclass(frozen=True)
class BenchmarkCase:
    n_points: int
    x_dim: int
    y_dim: int
    n_samples: int

    @property
    def name(self) -> str:
        return f"n{self.n_points}_x{self.x_dim}_y{self.y_dim}_s{self.n_samples}"

    @property
    def batch_size(self) -> int:
        return min(BATCH_LIMIT, self.n_samples)


def iter_cases() -> list[BenchmarkCase]:
    return [BenchmarkCase(n, xd, yd, ns) for n, xd, yd, ns in product(N_POINTS, X_DIMS, Y_DIMS, N_SAMPLES)]


def synthesize_series(n_points: int, x_dim: int, y_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate coupled synthetic data with controllable dimensionality."""

    drivers = rng.standard_normal((n_points, x_dim))
    weights = rng.standard_normal((x_dim, max(y_dim, 1)))
    responses = drivers @ weights + 0.1 * rng.standard_normal((n_points, max(y_dim, 1)))

    X = drivers if x_dim > 1 else drivers.reshape(n_points, 1)
    Y = responses[:, :y_dim] if y_dim > 1 else responses[:, 0]

    return X, Y


def compute_lib_sizes(min_size: int, pool_length: int, steps: int) -> np.ndarray:
    base = max(min_size, 2)
    max_candidate = max(pool_length, base)
    if max_candidate == base:
        max_candidate = base + max(2, base // 2)

    values = np.linspace(base, max_candidate, num=steps)
    lib_sizes = np.unique(np.maximum(values.astype(int), base))
    if lib_sizes.size == 0:
        lib_sizes = np.array([base])
    return lib_sizes


def main() -> None:
    predict_func = partial(simplex_projection, use_tensor=False)
    rows: list[list[str | int]] = []

    for case in iter_cases():
        X, Y = synthesize_series(case.n_points, case.x_dim, case.y_dim)
        split = max(2, min(case.n_points - 1, int(case.n_points * LIBRARY_RATIO)))
        library_pool = np.arange(split)
        prediction_pool = np.arange(split, case.n_points)
        lib_sizes = compute_lib_sizes(case.x_dim + 1, library_pool.size, LIB_SIZE_STEPS)

        start = time.perf_counter()
        ccm(
            X=X,
            Y=Y,
            lib_sizes=lib_sizes,
            predict_func=predict_func,
            n_samples=case.n_samples,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            batch_size=case.batch_size,
        )
        elapsed = time.perf_counter() - start

        rows.append(
            [
                case.name,
                case.n_points,
                case.x_dim,
                case.y_dim,
                case.n_samples,
                library_pool.size,
                f"{lib_sizes.tolist()}",
                case.batch_size,
                f"{elapsed:.3f}",
            ]
        )

    headers = [
        "scenario",
        "n_points",
        "x_dim",
        "y_dim",
        "n_samples",
        "pool",
        "lib sizes",
        "batch",
        "time (s)",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
