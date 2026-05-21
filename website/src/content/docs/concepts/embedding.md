---
title: Time-delay embedding
description: How edmkit reconstructs a multidimensional state space from a single observed time series, and how to choose the embedding parameters.
---

Time-delay embedding turns a scalar series into points in a higher-dimensional space whose geometry mirrors the underlying system. Every EDM algorithm in edmkit operates on its output, so the embedding parameters are the first decision in any analysis.

## The construction

Given a scalar series `x(0), x(1), ..., x(N-1)`, the lagged embedding with dimension `E` and delay `tau` collects, for each time `t >= (E - 1) * tau`, the vector

```
v(t) = ( x(t),  x(t - tau),  x(t - 2*tau),  ...,  x(t - (E - 1)*tau) )
```

Earlier times lack enough history. The result is a matrix of shape `(N - (E - 1) * tau, E)`.

`edmkit.embedding.lagged_embed` returns this matrix:

```python
import numpy as np
from edmkit.embedding import lagged_embed

x = np.arange(10)             # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
embedded = lagged_embed(x, tau=2, e=3)
# array([[4, 2, 0],
#        [5, 3, 1],
#        [6, 4, 2],
#        [7, 5, 3],
#        [8, 6, 4],
#        [9, 7, 5]])
```

Row `i` corresponds to time `(E - 1) * tau + i` in the original series. The leftmost column is the present; each column to its right is one delay further back. Every alignment between embedded and non-embedded arrays depends on this layout.

## Why it works

Takens' theorem: for a generic smooth system on a compact manifold, the lagged-coordinate map is a diffeomorphism from the attractor onto its image in `R^E`, provided `E >= 2*d + 1` where `d` is the box-counting dimension. The reconstructed cloud has the same topology and local geometry as the true state space, from one observed variable.

Two consequences matter more than the proof:

- Points close in the reconstructed cloud were close in the true state space, so their futures behave similarly.
- The attractor's dimensionality is recovered by the smallest `E` at which prediction skill plateaus — this is how we pick `E`.

## Aligning embedded and non-embedded arrays

Misaligned indices between the embedded matrix and an auxiliary series (target, known cause, label) cause most subtle EDM bugs. The rule:

```
embedded[i]  corresponds to time  x[(E - 1) * tau + i]
```

To pair each embedded state with its 1-step-ahead value:

```python
shift = (E - 1) * tau
target = x[shift + 1 : shift + len(embedded) + 1]   # length len(embedded)
# embedded[i] <-> target[i] = x[shift + i + 1]
```

To align a second series `y` (e.g. a candidate cause for CCM) with the embedded `x`:

```python
y_aligned = y[shift:]   # same length as embedded
```

## Choosing E

`E` is the minimum number of lagged coordinates that "unfold" the attractor — the smallest dimension at which the reconstructed cloud no longer self-intersects. Two states that look identical at `E = 1` may sit at different positions in the true state space; lifting the dimension separates them.

The standard recipe: fit a simplex-projection forecast over a range of `E` and pick the peak in held-out skill. `edmkit.embedding.scan` automates this:

```python
from edmkit.embedding import scan, select

scores = scan(x, E=list(range(1, 11)), tau=[1])
E, _, rho = select(scores, E=list(range(1, 11)), tau=[1])
```

Plotting `rho` against `E` typically shows a rise, a peak, then a slow decline once the dimension exceeds the true dimensionality.

## Choosing tau

`tau` controls how much each coordinate adds to the previous one.

| `tau` regime | Symptom |
| --- | --- |
| Too small | Successive coordinates are nearly identical; the cloud collapses onto a diagonal. |
| Just right | Coordinates are independent enough to spread the cloud, still close enough to share a trajectory. |
| Too large | Coordinates behave as independent observations; geometric structure disappears. |

Two heuristics:

- **Autocorrelation drop.** Pick the smallest `tau` for which the autocorrelation of `x` falls below `1/e`. `edmkit.util.autocorrelation` returns the values.
- **Cross-validated scan.** Add `tau` as a second grid axis in `scan` and let `select` pick. See the [parameter selection guide](/edmkit/guides/choosing-parameters/).

For chaotic continuous-time systems sampled at high resolution, useful `tau` is usually tens of samples. For maps and other discrete-time systems, `tau = 1` is often correct.

## The `scan` and `select` helpers

`scan(x, E=..., tau=...)` returns a `(len(E), len(tau), K)` array of per-fold prediction scores — Pearson correlation by default. `select` picks the `(E, tau)` that maximizes `mean - SE` across folds, penalizing combinations whose performance varies wildly. The returned `best_score` is the raw mean, so it stays directly interpretable.

```python
from edmkit.embedding import scan, select

E_grid = list(range(1, 11))
tau_grid = [1, 2, 3, 5]
scores = scan(x, E=E_grid, tau=tau_grid)
E, tau, rho = select(scores, E=E_grid, tau=tau_grid)
```

See the [parameter selection guide](/edmkit/guides/choosing-parameters/) for a full walk-through, including custom splits and prediction functions.

## Things to watch for

- **Length budget.** Embedding consumes `(E - 1) * tau` samples upfront. Large `E` or `tau` on a short series leaves too few states to train on.
- **Multivariate inputs.** `lagged_embed` accepts only 1-D arrays. Build multivariate embeddings by concatenating per-variable embeddings before calling the predictors.
- **Stationarity.** The reconstruction is meaningful only if the system is roughly stationary across the window. Trends or regime shifts mean the library no longer represents the query.
