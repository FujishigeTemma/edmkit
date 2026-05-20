import numpy as np
import pytest

from edmkit.simplex_projection.knn import knn
from edmkit.simplex_projection.nndescent import nndescent_knn
from edmkit.simplex_projection.rpf import rpf_knn


def _recall(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    flat_a = approx_idx.reshape(-1, approx_idx.shape[-1])
    flat_e = exact_idx.reshape(-1, exact_idx.shape[-1])
    rows, k = flat_e.shape
    hits = 0
    for r in range(rows):
        hits += len(set(flat_a[r].tolist()) & set(flat_e[r].tolist()))
    return hits / (rows * k)


class TestNNDescent:
    def test_self_knn_high_recall(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 4))
        k = 5
        _, i_kd = knn(X, X, k)
        _, i_nd = nndescent_knn(X, X, k, n_graph_iters=4, n_search_iters=4, seed=0)
        assert _recall(i_nd, i_kd) > 0.9

    def test_query_knn_high_recall(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 3))
        Q = rng.standard_normal((80, 3))
        k = 4
        _, i_kd = knn(X, Q, k)
        _, i_nd = nndescent_knn(X, Q, k, working_size=40, seed=0)
        assert _recall(i_nd, i_kd) > 0.85

    def test_batched_matches_loop(self):
        rng = np.random.default_rng(1)
        B, N, M, E, k = 3, 200, 50, 3, 4
        X = rng.standard_normal((B, N, E))
        Q = rng.standard_normal((B, M, E))
        d_batched, i_batched = nndescent_knn(X, Q, k, seed=42)
        # Distances and indices must be self-consistent.
        for b in range(B):
            recovered = np.sqrt(np.sum((X[b, i_batched[b]] - Q[b, :, None, :]) ** 2, axis=-1))
            np.testing.assert_allclose(d_batched[b], recovered, atol=1e-8, rtol=1e-8)

    def test_distances_are_sorted(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 2))
        Q = rng.standard_normal((20, 2))
        d, _ = nndescent_knn(X, Q, k=3, seed=0)
        assert np.all(d[:, :-1] <= d[:, 1:] + 1e-12)

    def test_rejects_batch_mismatch(self):
        X = np.zeros((2, 10, 3))
        Q = np.zeros((3, 4, 3))
        with pytest.raises(ValueError, match="batch size"):
            nndescent_knn(X, Q, k=2)


class TestRPF:
    def test_self_knn_high_recall(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 4))
        k = 5
        _, i_kd = knn(X, X, k)
        _, i_rpf = rpf_knn(X, X, k, n_trees=8, seed=0)
        assert _recall(i_rpf, i_kd) > 0.9

    def test_query_knn_high_recall(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 3))
        Q = rng.standard_normal((80, 3))
        k = 4
        _, i_kd = knn(X, Q, k)
        _, i_rpf = rpf_knn(X, Q, k, n_trees=8, seed=0)
        assert _recall(i_rpf, i_kd) > 0.9

    def test_batched_matches_loop(self):
        rng = np.random.default_rng(1)
        B, N, M, E, k = 3, 200, 50, 3, 4
        X = rng.standard_normal((B, N, E))
        Q = rng.standard_normal((B, M, E))
        d_batched, i_batched = rpf_knn(X, Q, k, seed=42)
        for b in range(B):
            recovered = np.sqrt(np.sum((X[b, i_batched[b]] - Q[b, :, None, :]) ** 2, axis=-1))
            np.testing.assert_allclose(d_batched[b], recovered, atol=1e-8, rtol=1e-8)

    def test_distances_are_sorted(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 2))
        Q = rng.standard_normal((20, 2))
        d, _ = rpf_knn(X, Q, k=3, seed=0)
        assert np.all(d[:, :-1] <= d[:, 1:] + 1e-12)

    def test_padded_library_returns_valid_indices(self):
        # N=50, leaf_size auto -> requires padding to power-of-two multiple.
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 3))
        Q = rng.standard_normal((10, 3))
        _, idx = rpf_knn(X, Q, k=3, n_trees=4, seed=0)
        assert np.all(idx >= 0)
        assert np.all(idx < 50)

    def test_rejects_batch_mismatch(self):
        X = np.zeros((2, 10, 3))
        Q = np.zeros((3, 4, 3))
        with pytest.raises(ValueError, match="batch size"):
            rpf_knn(X, Q, k=2)
