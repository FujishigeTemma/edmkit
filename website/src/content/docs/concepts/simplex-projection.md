---
title: Simplex Projection
description: Nearest-neighbor prediction in reconstructed state space.
---

Simplex projection is the core prediction algorithm in EDM. It uses nearest neighbors in the embedded state space to make predictions.

## How It Works

For a query point **q** in _E_-dimensional space:

1. Find the _E+1_ nearest neighbors in the library set (forming a simplex)
2. Compute weights inversely proportional to distance: _w_i = exp(-d_i / d_min)_
3. Predict as the weighted average of the neighbors' target values

## Usage

```python
from edmkit.simplex_projection import simplex_projection

# X: library embeddings (N, E)
# Y: library targets (N,) or (N, D)
# Q: query embeddings (M, E)
predictions = simplex_projection(X, Y, Q)
```

### Batched Operations

All inputs support a leading batch dimension for parallel evaluation:

```python
# X: (B, N, E), Y: (B, N, D), Q: (B, M, E)
predictions = simplex_projection(X, Y, Q)  # -> (B, M, D)
```

### Masking

Use the `mask` parameter to exclude specific library points:

```python
mask = np.ones(N, dtype=bool)
mask[100:110] = False  # Exclude points 100-109

predictions = simplex_projection(X, Y, Q, mask=mask)
```

## Leave-One-Out

The `loo` function performs leave-one-out prediction with Theiler window exclusion, which prevents temporal neighbors from being used as predictors:

```python
from edmkit.simplex_projection import loo

predictions = loo(X, Y, theiler_window=5)
# Each point predicted using all others except those within ±5 time steps
```

## Nearest-Neighbor Search

The `knn` function handles neighbor finding, automatically choosing the best backend:

- **KDTree** (SciPy) for lower dimensions or smaller datasets
- **usearch** for high-dimensional or large-scale data (E >= 15 and N >= 10,000)

```python
from edmkit.simplex_projection import knn

distances, indices = knn(X, Q, k=10)
```
