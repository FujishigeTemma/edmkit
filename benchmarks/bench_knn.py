"""Batched k-NN benchmark: kd-tree (current) vs NN-Descent vs RPF.

Run with: ``uv run python benchmarks/bench_knn.py``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from edmkit.simplex_projection.knn import knn as kdtree_knn
from edmkit.simplex_projection.nndescent import nndescent_knn
from edmkit.simplex_projection.rpf import rpf_knn


@dataclass
class Scenario:
    name: str
    B: int
    N: int
    M: int
    E: int
    k: int


SCENARIOS = [
    Scenario("small", B=8, N=500, M=200, E=3, k=4),
    Scenario("medium", B=16, N=2_000, M=500, E=4, k=5),
    Scenario("large", B=32, N=5_000, M=1_000, E=5, k=6),
    Scenario("wide-batch", B=128, N=1_000, M=300, E=3, k=4),
    Scenario("xwide-batch", B=512, N=1_000, M=300, E=3, k=4),
    Scenario("xxwide-batch", B=1024, N=500, M=200, E=3, k=4),
    Scenario("highdim", B=8, N=2_000, M=500, E=12, k=13),
]


def _make_data(s: Scenario, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal(size=(s.B, s.N, s.E)).astype(np.float64)
    Q = rng.standard_normal(size=(s.B, s.M, s.E)).astype(np.float64)
    return X, Q


def _kdtree_batched(X: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicates the current `simplex_projection` batched kd-tree path: a Python for-loop."""
    B, M = X.shape[0], Q.shape[1]
    distances = np.empty((B, M, k), dtype=np.float64)
    indices = np.empty((B, M, k), dtype=np.intp)
    for b in range(B):
        distances[b], indices[b] = kdtree_knn(X[b], Q[b], k)
    return distances, indices


_KD_EXECUTOR = ThreadPoolExecutor()


def _kdtree_batched_threaded(X: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``_kdtree_batched`` but distributes batches across a thread pool.

    Effective only if ``kdtree-rs`` releases the GIL during ``query``.
    """
    B, M = X.shape[0], Q.shape[1]
    distances = np.empty((B, M, k), dtype=np.float64)
    indices = np.empty((B, M, k), dtype=np.intp)

    def _work(b: int) -> None:
        distances[b], indices[b] = kdtree_knn(X[b], Q[b], k)

    list(_KD_EXECUTOR.map(_work, range(B)))
    return distances, indices


def _recall(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Per-row Jaccard-style recall: |approx ∩ exact| / k, averaged over rows."""
    B, M, k = exact_idx.shape
    total = B * M
    hits = 0.0
    for b in range(B):
        for m in range(M):
            hits += len(set(approx_idx[b, m].tolist()) & set(exact_idx[b, m].tolist()))
    return hits / (total * k)


def _dist_relerr(approx_d: np.ndarray, exact_d: np.ndarray) -> float:
    """Mean relative error of sorted distances."""
    ad = np.sort(approx_d, axis=-1)
    ed = np.sort(exact_d, axis=-1)
    denom = np.where(ed > 1e-12, ed, 1.0)
    return float(np.mean(np.abs(ad - ed) / denom))


def _time(fn, *args, repeats: int = 3, **kwargs) -> tuple[float, tuple]:
    out = fn(*args, **kwargs)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(scenarios: list[Scenario], *, seed: int = 0, repeats: int = 3) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for s in scenarios:
        X, Q = _make_data(s, rng)

        t_kd, (d_kd, i_kd) = _time(_kdtree_batched, X, Q, s.k, repeats=repeats)
        t_kt, _ = _time(_kdtree_batched_threaded, X, Q, s.k, repeats=repeats)
        t_nd, (d_nd, i_nd) = _time(
            nndescent_knn,
            X,
            Q,
            s.k,
            n_graph_iters=4,
            n_search_iters=4,
            seed=seed,
            repeats=repeats,
        )
        t_rpf, (d_rpf, i_rpf) = _time(
            rpf_knn,
            X,
            Q,
            s.k,
            n_trees=8,
            seed=seed,
            repeats=repeats,
        )

        rows.append(
            {
                "scenario": s.name,
                "B/N/M/E/k": f"{s.B}/{s.N}/{s.M}/{s.E}/{s.k}",
                "kd-tree loop (ms)": f"{t_kd * 1e3:8.2f}",
                "kd-tree threaded (ms)": f"{t_kt * 1e3:8.2f}",
                "NN-Descent (ms)": f"{t_nd * 1e3:8.2f}",
                "RPF (ms)": f"{t_rpf * 1e3:8.2f}",
                "ND recall": f"{_recall(i_nd, i_kd):.3f}",
                "RPF recall": f"{_recall(i_rpf, i_kd):.3f}",
                "ND speedup vs loop": f"{t_kd / t_nd:5.2f}x",
                "RPF speedup vs loop": f"{t_kd / t_rpf:5.2f}x",
                "threaded speedup": f"{t_kd / t_kt:5.2f}x",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rows = run(SCENARIOS, seed=args.seed, repeats=args.repeats)
    print(tabulate(rows, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
