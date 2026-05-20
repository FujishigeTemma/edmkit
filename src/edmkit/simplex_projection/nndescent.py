"""Batched NN-Descent for approximate k-NN, NumPy only.

The algorithm is the basic NN-Descent of Dong et al. (2011):

1. Initialise the K-NN graph of ``X`` with random neighbours per point.
2. Iteratively refine: for each point, propose its current neighbours'
   neighbours as candidates and keep the closest K.
3. For queries ``Q`` that are not equal to ``X``, perform greedy graph
   search from random seeds, expanding the working set along graph edges.

All operations are vectorised over the batch dimension ``B`` and the
library/query dimensions, so no Python-level loop over ``B`` is required.
"""

from __future__ import annotations

import numpy as np


def _to_3d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    if X.ndim == 2:
        return X[None, ...], True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"expected 2D or 3D array, got ndim={X.ndim}")


def _batched_gather(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``X[b, idx[b, ...], :]`` for every batch ``b``.

    ``X`` has shape ``(B, N, E)`` and ``idx`` has shape ``(B, *trailing)``;
    the result has shape ``(B, *trailing, E)``.
    """
    B = X.shape[0]
    shape = (B,) + (1,) * (idx.ndim - 1)
    return X[np.arange(B).reshape(shape), idx]


def _squared_dists_to(
    X: np.ndarray,
    idx_set: np.ndarray,
    query: np.ndarray,
    X_norm_sq: np.ndarray | None = None,
    query_norm_sq: np.ndarray | None = None,
) -> np.ndarray:
    """``query`` has shape ``(B, M, E)``; ``idx_set`` has shape ``(B, M, K)``.

    Returns the squared Euclidean distances from each query to its referenced
    library points: shape ``(B, M, K)``.

    Optional precomputed squared norms avoid materialising the ``(B, M, K, E)``
    difference tensor; we instead use the polar identity
    ``|a - b|^2 = |a|^2 + |b|^2 - 2 a·b`` with the dot product done via einsum.
    """
    if X_norm_sq is None or query_norm_sq is None:
        cand_X = _batched_gather(X, idx_set)
        diffs = cand_X - query[:, :, None, :]
        return np.einsum("bmke,bmke->bmk", diffs, diffs)

    cand_X = _batched_gather(X, idx_set)  # (B, M, K, E)
    B = X.shape[0]
    bshape = (B,) + (1,) * (idx_set.ndim - 1)
    cand_norm = X_norm_sq[np.arange(B).reshape(bshape), idx_set]  # (B, M, K)
    dot = np.einsum("bmke,bme->bmk", cand_X, query)
    dist_sq = cand_norm + query_norm_sq[..., None] - 2.0 * dot
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


def _topk_sorted(values: np.ndarray, indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the smallest-k entries of ``values`` along the last axis, sorted ascending."""
    last = values.shape[-1]
    if last <= k:
        order = np.argsort(values, axis=-1)
        return np.take_along_axis(values, order, axis=-1), np.take_along_axis(indices, order, axis=-1)

    part = np.argpartition(values, k, axis=-1)[..., :k]
    sub_v = np.take_along_axis(values, part, axis=-1)
    sub_i = np.take_along_axis(indices, part, axis=-1)
    order = np.argsort(sub_v, axis=-1)
    return np.take_along_axis(sub_v, order, axis=-1), np.take_along_axis(sub_i, order, axis=-1)


def _dedupe_inplace(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set the values of duplicate indices (per row) to +inf so top-k yields uniques.

    Both arrays are modified out-of-place: a copy of ``values`` is returned with
    duplicate slots replaced by ``+inf``. ``indices`` itself is returned unchanged.
    """
    order = np.argsort(indices, kind="stable", axis=-1)
    sorted_idx = np.take_along_axis(indices, order, axis=-1)
    is_dup = np.zeros_like(sorted_idx, dtype=bool)
    is_dup[..., 1:] = sorted_idx[..., 1:] == sorted_idx[..., :-1]
    # Scatter the duplicate mask back to original positions.
    dup_mask = np.empty_like(is_dup)
    np.put_along_axis(dup_mask, order, is_dup, axis=-1)
    return np.where(dup_mask, np.inf, values), indices


def _reverse_neighbors(forward: np.ndarray, k_rev: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate reverse-neighbour candidates per point, fully batched.

    ``forward`` has shape ``(B, N, K)``; the returned array has shape
    ``(B, N, k_rev)``. For each destination ``m`` it collects up to
    ``k_rev`` sources that point at ``m`` in the forward graph. Slots that
    are not filled by real reverse edges fall back to random library
    indices so the union step always sees diverse candidates.
    """
    B, N, K = forward.shape
    src = np.broadcast_to(np.arange(N)[:, None], (N, K))
    src_bc = np.broadcast_to(src, (B, N, K))
    flat_src = src_bc.reshape(B, N * K)
    flat_dst = forward.reshape(B, N * K)

    order = np.argsort(flat_dst, kind="stable", axis=-1)
    sorted_dst = np.take_along_axis(flat_dst, order, axis=-1)
    sorted_src = np.take_along_axis(flat_src, order, axis=-1)

    boundary = np.empty((B, N * K), dtype=bool)
    boundary[:, 0] = True
    boundary[:, 1:] = sorted_dst[:, 1:] != sorted_dst[:, :-1]
    positions = np.broadcast_to(np.arange(N * K), (B, N * K))
    block_start = np.where(boundary, positions, 0)
    np.maximum.accumulate(block_start, axis=-1, out=block_start)
    pos_in_block = positions - block_start  # (B, N*K)

    keep = pos_in_block < k_rev
    target = sorted_dst * k_rev + pos_in_block

    out = rng.integers(0, N, size=(B, N, k_rev), dtype=forward.dtype)
    flat_out = out.reshape(B, N * k_rev)
    safe_target = np.where(keep, target, 0)
    batch_idx = np.broadcast_to(np.arange(B)[:, None], (B, N * K))
    flat_out[batch_idx, safe_target] = np.where(keep, sorted_src, flat_out[batch_idx, safe_target])
    return flat_out.reshape(B, N, k_rev)


def _build_graph(
    X: np.ndarray,
    k: int,
    n_iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an approximate k-NN graph on ``X`` of shape ``(B, N, E)``.

    Returns ``(indices, sq_distances)`` of shape ``(B, N, k)``, sorted ascending
    per row. The graph may contain self at distance 0 if it is discovered.
    """
    B, N, _ = X.shape
    X_norm_sq = np.einsum("bne,bne->bn", X, X)

    indices = rng.integers(0, N, size=(B, N, k), dtype=np.intp)
    distances = _squared_dists_to(X, indices, X, X_norm_sq, X_norm_sq)
    distances, indices = _topk_sorted(distances, indices, k)

    for _ in range(n_iters):
        batch_idx = np.arange(B)[:, None, None, None]
        nn_idx = indices[batch_idx, indices[:, :, :, None], :]
        nn_idx_flat = nn_idx.reshape(B, N, k * k)

        rev = _reverse_neighbors(indices, k, rng)

        all_idx = np.concatenate([indices, nn_idx_flat, rev], axis=-1)
        all_dist = _squared_dists_to(X, all_idx, X, X_norm_sq, X_norm_sq)
        all_dist, all_idx = _dedupe_inplace(all_dist, all_idx)
        new_dist, new_idx = _topk_sorted(all_dist, all_idx, k)

        if np.array_equal(new_idx, indices):
            indices = new_idx
            distances = new_dist
            break
        indices = new_idx
        distances = new_dist

    return indices, distances


def _greedy_search(
    X: np.ndarray,
    graph_indices: np.ndarray,
    Q: np.ndarray,
    k: int,
    working_size: int,
    n_iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Best-first graph search for k-NN of ``Q`` in ``X``."""
    B, N, _ = X.shape
    M = Q.shape[1]

    X_norm_sq = np.einsum("bne,bne->bn", X, X)
    Q_norm_sq = np.einsum("bme,bme->bm", Q, Q)

    ws = max(working_size, k)
    cur_idx = rng.integers(0, N, size=(B, M, ws), dtype=np.intp)
    cur_dist = _squared_dists_to(X, cur_idx, Q, X_norm_sq, Q_norm_sq)
    cur_dist, cur_idx = _topk_sorted(cur_dist, cur_idx, ws)

    for _ in range(n_iters):
        batch_idx = np.arange(B)[:, None, None, None]
        nbr_idx = graph_indices[batch_idx, cur_idx[:, :, :, None], :]  # (B, M, ws, k_graph)
        nbr_idx = nbr_idx.reshape(B, M, -1)

        nbr_dist = _squared_dists_to(X, nbr_idx, Q, X_norm_sq, Q_norm_sq)

        all_idx = np.concatenate([cur_idx, nbr_idx], axis=-1)
        all_dist = np.concatenate([cur_dist, nbr_dist], axis=-1)
        all_dist, all_idx = _dedupe_inplace(all_dist, all_idx)
        new_dist, new_idx = _topk_sorted(all_dist, all_idx, ws)

        if np.array_equal(new_idx, cur_idx):
            cur_idx = new_idx
            cur_dist = new_dist
            break
        cur_idx = new_idx
        cur_dist = new_dist

    return cur_dist[..., :k], cur_idx[..., :k]


def nndescent_knn(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    *,
    k_graph: int | None = None,
    n_graph_iters: int = 4,
    n_search_iters: int = 4,
    working_size: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate k-NN of ``Q`` in ``X`` via NN-Descent.

    Parameters
    ----------
    X : np.ndarray
        Library points of shape ``(N, E)`` or ``(B, N, E)``.
    Q : np.ndarray
        Query points of shape ``(M, E)`` or ``(B, M, E)``. Must share the
        batch dimension with ``X`` when 3D.
    k : int
        Number of neighbours to return per query.
    k_graph : int, optional
        Size of the NN-Descent graph. Larger values give higher recall at
        the cost of more memory and build time. Defaults to ``max(k + 5, 2k)``.
    n_graph_iters : int, optional
        Number of NN-Descent refinement iterations during graph construction.
    n_search_iters : int, optional
        Number of greedy-search expansion steps during query.
    working_size : int, optional
        Number of best candidates maintained during greedy search. Defaults
        to ``max(2k, 10)``.
    seed : int, optional
        Seed for the internal :class:`numpy.random.Generator`.

    Returns
    -------
    distances : np.ndarray
        Euclidean distances of shape ``(M, k)`` or ``(B, M, k)``.
    indices : np.ndarray
        Library indices of the same shape.
    """
    X3, was_2d_X = _to_3d(X)
    Q3, _ = _to_3d(Q)
    B, N, _ = X3.shape
    if Q3.shape[0] != B:
        raise ValueError(f"batch size mismatch: X has {B}, Q has {Q3.shape[0]}")

    if k_graph is None:
        k_graph = max(k + 5, 2 * k)
    if working_size is None:
        working_size = max(4 * k, 20)

    rng = np.random.default_rng(seed)
    graph_indices, _ = _build_graph(X3, k_graph, n_graph_iters, rng)
    dist_sq, idx = _greedy_search(X3, graph_indices, Q3, k, working_size, n_search_iters, rng)
    distances = np.sqrt(dist_sq)

    if was_2d_X:
        return distances[0], idx[0]
    return distances, idx
