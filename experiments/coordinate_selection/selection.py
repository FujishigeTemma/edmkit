"""Greedy selection of input delay coordinates from pointwise, multi-horizon spread profiles.

Every quantity is kept per query point and per horizon until the single place where a decision
needs a scalar: `profiles` returns the tensor, `select` weights and sums it once per cycle.
"""

import numpy as np

from edmkit.embedding import embed
from edmkit.spread import spread
from edmkit.theiler_window import theiler_window

__all__ = ["horizons", "profiles", "select"]


def horizons(variables: int, longest: int) -> np.ndarray:
    """Output coordinates: every variable at horizons 1, 2, 4, ... up to `longest`."""
    steps = 2 ** np.arange(int(np.log2(longest)) + 1)
    return np.array([[i, -h] for i in range(variables) for h in steps])


def profiles(
    coordinate_sets: list[np.ndarray], outputs: np.ndarray, x: np.ndarray, t: np.ndarray, *, k: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out log spread at each time in `t`, for each set of input coordinates.

    Returns
    -------
    S : np.ndarray
        Shape ``(C, M, H)``. ``S[c, m, h]`` is the log variance of output ``h`` among the `k` nearest
        neighbours of time ``t[m]`` under ``coordinate_sets[c]``. Outputs are standardized, so 0 is the
        spread of an input that carries no information. The neighbourhood holds a fixed number of
        points, which makes `S` comparable between coordinate sets of different dimension.
    D : np.ndarray
        Shape ``(C, M)``. Log variance of the same neighbours in the input coordinates, summed over them.
        ``S - D[..., None]`` is the log of the local noise amplification of Uzal et al. (2011).
    """
    Y = x[t[:, None] - outputs[:, 1], outputs[:, 0]]
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    mask = theiler_window(t, t, width)

    S = np.empty((len(coordinate_sets), len(t), len(outputs)))
    D = np.empty((len(coordinate_sets), len(t)))
    for c, coordinates in enumerate(coordinate_sets):
        X = x[t[:, None] - coordinates[:, 1], coordinates[:, 0]]
        S[c] = np.log(np.fmax(spread(X, Y, X, k=k, mask=mask), 1e-300))
        D[c] = np.log(np.fmax(spread(X, X, X, k=k, mask=mask).sum(axis=1), 1e-300))  # the input as its own output
    return S, D


def _jackknife(weights: np.ndarray, gains: np.ndarray, blocks: int) -> tuple[float, float]:
    """Weighted mean of `gains` and its delete-one-block jackknife standard error over contiguous blocks of time."""
    edges = np.linspace(0, len(gains), blocks + 1).astype(int)
    total_w, total_wg = weights.sum(), (weights * gains).sum()
    values = np.empty(blocks)
    for b in range(blocks):
        block = slice(edges[b], edges[b + 1])
        values[b] = (total_wg - (weights[block] * gains[block]).sum()) / (total_w - weights[block].sum())
    return float(total_wg / total_w), float(np.sqrt((blocks - 1) / blocks * ((values - values.mean()) ** 2).sum()))


def select(
    candidates: np.ndarray,
    outputs: np.ndarray,
    x: np.ndarray,
    *,
    k: int,
    width: int,
    weighting: str = "defect",
    z: float = 2.0,
    blocks: int = 50,
    max_cycles: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    """Add one candidate coordinate per cycle until none gives a gain of `z` standard errors.

    The gain of a candidate at a point is the reduction of the log spread there, averaged over horizons.
    With ``weighting="defect"`` each point is weighted by the noise amplification of the current
    coordinates at that point, so the cycle asks which candidate repairs the points that are currently
    defective. With ``weighting="uniform"`` every point counts equally.

    Returns the selected coordinates ``(E, 2)`` and one record per cycle.
    """
    _, t = embed(np.vstack([candidates, outputs]), x)

    selected = np.empty((0, 2), dtype=candidates.dtype)
    base = np.zeros(len(t))  # no input: the spread is the variance of the standardized output
    weights = np.ones(len(t))
    history = []
    for _ in range(max_cycles):
        remaining = [c for c in candidates if not (selected == c).all(axis=1).any()]
        if not selected.size:  # the first coordinate fixes the time origin
            remaining = [c for c in remaining if c[1] == 0]
        S, D = profiles([np.vstack([selected, c]) for c in remaining], outputs, x, t, k=k, width=width)
        points = S.mean(axis=2)  # (C, M): horizons averaged in log scale

        gains = ((base - points) * weights).sum(axis=1) / weights.sum()
        best = int(gains.argmax())
        gain, error = _jackknife(weights, base - points[best], blocks)
        record = {"coordinate": remaining[best].tolist(), "gain": gain, "error": error, "accepted": gain >= z * error}
        history.append(record)
        if not record["accepted"]:
            break
        selected = np.vstack([selected, remaining[best]])
        base = points[best]
        if weighting != "uniform":
            weights = np.exp(points[best] - D[best])
            weights /= weights.mean()
        if weighting == "mixed":
            weights = (1 + weights) / 2
    return selected, history
