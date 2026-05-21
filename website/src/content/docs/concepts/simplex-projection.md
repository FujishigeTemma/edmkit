---
title: Simplex projection
description: Nearest-neighbor prediction in reconstructed state space, the workhorse of EDM forecasting and the standard tool for picking E.
---

Simplex projection is the simplest EDM predictor. For each query in the embedded space, it finds the `E + 1` nearest library points (the vertices of a simplex around the query) and predicts the target as a distance-weighted average of their targets. It's the workhorse for forecasting nonlinear series and for choosing `E`.

## The algorithm

Given library embeddings `X` of shape `(N, E)`, targets `Y` of shape `(N,)` or `(N, D)`, and one query `q`:

1. Find the `k = E + 1` nearest neighbors `x_{i_1}, ..., x_{i_k}` of `q` in `X`, with Euclidean distances `d_1 <= ... <= d_k`.
2. Form weights `w_j = exp(-d_j / max(d_1, eps))`. The closest neighbor gets `1`; the rest decay exponentially.
3. Predict `y_hat = sum_j w_j * y_{i_j} / sum_j w_j`.

Two design choices:

- **Exactly `E + 1` neighbors** — the minimum to span an `E`-dimensional simplex around the query. Fewer leaves it degenerate; more dilutes the local geometry.
- **Self-normalizing weights.** Dividing by the minimum distance makes weights scale-free, so rescaling the library does not change the prediction.

## Why this picks `E`

Plot held-out `rho(E)` against `E`. The curve rises until `E` matches the attractor's effective dimensionality, then plateaus or declines as higher `E` only adds noise to the neighbor search. The peak is the recommended `E`. `edmkit.embedding.scan` and `select` package this as a grid search.

## Using `simplex_projection`

The function takes library embeddings `X`, targets `Y`, and queries `Q`.

```python
from edmkit.simplex_projection import simplex_projection

# X: (N, E) library embeddings
# Y: (N,) or (N, D) library targets
# Q: (M, E) query embeddings
predictions = simplex_projection(X, Y, Q)   # (M,) or (M, D)
```

Conventions:

- `Y` can be 1-D (scalar) or 2-D (vector). The prediction's rank matches `Y`.
- `Q` is independent of `X`. For CCM, `Q` is a different signal's embedding.
- Targets need not be future values of the embedding's source — they can be any quantity predicted from the state.

## Batched evaluation

For several independent simplex problems at once (e.g. `scan` over many `(E, tau)` cells), pass arrays with a leading batch dimension:

```python
# X: (B, N, E), Y: (B, N, D), Q: (B, M, E)
predictions = simplex_projection(X, Y, Q)   # (B, M, D)
```

Batched calls share allocation overhead and dispatch into the same KDTree, so they're much faster than a Python loop. This is what makes `scan` tractable.

## Excluding library points: `mask` and `loo`

- `mask` is a boolean array of shape `(N,)` or `(B, N)` that hides library points from the neighbor search. Use it for train/validation separation inside a batched call.
- `loo(X, Y, theiler_window=...)` performs leave-one-out simplex projection across `X`, excluding library points within `theiler_window` time steps of the query. For a lagged embedding, set the window to `(E - 1) * tau` so overlapping embeddings cannot trivially predict each other.

```python
from edmkit.simplex_projection import loo

predictions = loo(embedded, target, theiler_window=(E - 1) * tau)
```

Forgetting the Theiler window is a common reason reported correlations look "too good to be true" — overlapping embeddings act as near-duplicates.

## When simplex projection is enough

- A quick forecast on a nonlinear series.
- Choosing `E` from data.
- A baseline for comparison against other predictors (linear regression, S-Map, neural nets).

Reach for [S-Map](/edmkit/concepts/smap/) when you need to **quantify nonlinearity** via the `theta` knob or recover local Jacobians.

## Limitations

- **Library coverage.** Queries outside the library's convex hull extrapolate, and the weighted average loses meaning.
- **Equal-distance ties.** When multiple library points share the minimum distance, predictions become sensitive to floating-point order. Rare for continuous-valued data.
- **Noise amplification at small distances.** When the minimum distance is below the noise floor, the exponential decay collapses to a near-uniform average. Increasing `E` or switching to S-Map helps.
