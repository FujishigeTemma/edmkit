"""Batched Random Projection Forest for approximate k-NN, NumPy only.

Each tree recursively splits the library by projecting points onto a random
direction (per node) and routing them by the median projection. Splits are
balanced by position, so leaves always contain ``N / 2**depth`` points,
which lets the whole build (and the query traversal) be expressed as a
sequence of vectorised operations over the batch dimension.
"""

from __future__ import annotations

import numpy as np

from edmkit.simplex_projection.nndescent import _dedupe_inplace


def _to_3d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    if X.ndim == 2:
        return X[None, ...], True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"expected 2D or 3D array, got ndim={X.ndim}")


def _build_tree(
    X: np.ndarray,
    depth: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Build one balanced random-projection tree per batch.

    ``X`` has shape ``(B, N, E)`` where ``N`` must be ``2 ** depth * leaf_size``.
    Returns the per-level projection directions and split values, plus a
    ``(B, n_leaves, leaf_size)`` table of library indices per leaf.
    """
    B, N, E = X.shape
    n_leaves = 1 << depth
    if N % n_leaves != 0:
        raise ValueError(f"N={N} must be a multiple of 2**depth={n_leaves}")
    leaf_size = N // n_leaves

    # `perm[b]` is the permutation that groups points by their current node id.
    # After each level it is reshuffled so that points sharing a node remain
    # contiguous; this makes per-group median and split fully vectorisable.
    perm = np.broadcast_to(np.arange(N), (B, N)).copy()

    directions_all: list[np.ndarray] = []
    medians_all: list[np.ndarray] = []
    batch_idx = np.arange(B)[:, None]

    for d in range(depth):
        n_groups = 1 << d
        group_size = N // n_groups
        half = group_size // 2

        # Random projection direction per (batch, current group).
        directions = rng.standard_normal(size=(B, n_groups, E)).astype(X.dtype, copy=False)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-30

        # Gather X by the current permutation so points in group g sit at
        # positions [g * group_size, (g + 1) * group_size).
        X_grouped = X[batch_idx, perm, :].reshape(B, n_groups, group_size, E)
        # Project each point onto its group's direction.
        proj = np.einsum("bgse,bge->bgs", X_grouped, directions)  # (B, n_groups, group_size)

        # Sort within each group by projection value.
        order = np.argsort(proj, axis=-1)  # (B, n_groups, group_size)
        sorted_proj = np.take_along_axis(proj, order, axis=-1)
        # Median: the value at position `half` of the sorted group.
        medians = sorted_proj[..., half]  # (B, n_groups)

        # Reorder perm within each group by `order`.
        perm = perm.reshape(B, n_groups, group_size)
        perm = np.take_along_axis(perm, order, axis=-1)
        perm = perm.reshape(B, N)

        directions_all.append(directions)
        medians_all.append(medians)

    # After all levels, perm[b] lists library indices grouped by leaf id (0..n_leaves-1),
    # each leaf contiguous and of length leaf_size.
    leaf_points = perm.reshape(B, n_leaves, leaf_size)
    return directions_all, medians_all, leaf_points


def _query_tree(
    Q: np.ndarray,
    directions_all: list[np.ndarray],
    medians_all: list[np.ndarray],
    leaf_points: np.ndarray,
) -> np.ndarray:
    """Traverse a tree per query and return the leaf indices (per-query).

    Shape: ``(B, M, leaf_size)``.
    """
    B, M, _ = Q.shape
    depth = len(directions_all)
    batch_col = np.arange(B)[:, None]

    group_id = np.zeros((B, M), dtype=np.int64)
    for d in range(depth):
        directions = directions_all[d]
        medians = medians_all[d]
        point_dir = directions[batch_col, group_id, :]  # (B, M, E)
        proj = np.einsum("bme,bme->bm", Q, point_dir)
        threshold = medians[batch_col, group_id]
        group_id = group_id * 2 + (proj >= threshold).astype(np.int64)

    return leaf_points[batch_col, group_id, :]


def _pad_to_multiple(X: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    B, N, _ = X.shape
    rem = N % multiple
    if rem == 0:
        return X, 0
    pad = multiple - rem
    return np.concatenate([X, X[:, :pad, :]], axis=1), pad


def rpf_knn(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    *,
    n_trees: int = 4,
    leaf_size: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate k-NN of ``Q`` in ``X`` via Random Projection Forest.

    Parameters
    ----------
    X : np.ndarray
        Library points of shape ``(N, E)`` or ``(B, N, E)``.
    Q : np.ndarray
        Query points of shape ``(M, E)`` or ``(B, M, E)``.
    k : int
        Number of neighbours to return.
    n_trees : int, optional
        Number of independent trees in the forest. More trees give higher
        recall at the cost of build and query time.
    leaf_size : int, optional
        Target number of points per leaf. Larger leaves give higher recall
        per tree but cost more candidate-distance work. Defaults to
        ``max(4 * k, 16)``.
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

    if leaf_size is None:
        leaf_size = max(4 * k, 16)
    leaf_size = min(leaf_size, N)

    depth = max(1, int(np.floor(np.log2(max(N // leaf_size, 1)))))
    n_leaves = 1 << depth
    X_pad, pad = _pad_to_multiple(X3, n_leaves)

    rng = np.random.default_rng(seed)
    cand_list = []
    for _ in range(n_trees):
        directions, medians, leaf_points = _build_tree(X_pad, depth, rng)
        cand_list.append(_query_tree(Q3, directions, medians, leaf_points))
    candidates = np.concatenate(cand_list, axis=-1)  # (B, M, n_trees * actual_leaf_size)

    B_idx = np.arange(B)[:, None, None]
    cand_X = X_pad[B_idx, candidates, :]
    diffs = cand_X - Q3[:, :, None, :]
    dist_sq = np.sum(diffs * diffs, axis=-1)

    if pad > 0:
        dist_sq = np.where(candidates < N, dist_sq, np.inf)

    dist_sq, candidates = _dedupe_inplace(dist_sq, candidates)

    last = dist_sq.shape[-1]
    if last <= k:
        topk_v, topk_i = dist_sq, candidates
    else:
        part = np.argpartition(dist_sq, k, axis=-1)[..., :k]
        topk_v = np.take_along_axis(dist_sq, part, axis=-1)
        topk_i = np.take_along_axis(candidates, part, axis=-1)

    order = np.argsort(topk_v, axis=-1)
    topk_v = np.take_along_axis(topk_v, order, axis=-1)
    topk_i = np.take_along_axis(topk_i, order, axis=-1)
    distances = np.sqrt(topk_v)

    # If a wrapped-pad index slipped through (only possible when k exceeds the
    # number of valid candidates), remap it back to its source point.
    topk_i = np.where(topk_i < N, topk_i, topk_i - N)

    if was_2d_X:
        return distances[0], topk_i[0]
    return distances, topk_i
