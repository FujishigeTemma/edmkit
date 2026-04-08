---
title: embedding
description: Time-delay embedding and parameter selection.
sidebar:
  order: 1
---

## `embedding`

**Functions:**

Name | Description
---- | -----------
[`lagged_embed`](#edmkit.embedding.lagged_embed) | Lagged embedding of a time series `x`.
[`scan`](#edmkit.embedding.scan) | Grid search over (E, tau) with cross-validation.
[`select`](#edmkit.embedding.select) | Select best (E, tau) from scan results.

### `lagged_embed`

```python
lagged_embed(x: np.ndarray, tau: int, e: int)
```

Lagged embedding of a time series `x`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#numpy.ndarray)</code> | 1D time series of shape ``(N,)``. | *required*
`tau` | <code>[int](#int)</code> | Time delay. | *required*
`e` | <code>[int](#int)</code> | Embedding dimension. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Embedded array of shape ``(N - (e - 1) * tau, e)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `x` is not a 1D array. - If `tau` or `e` is not positive. - If `e * tau >= len(x)`.

<details class="note" open markdown="1">
<summary>Notes</summary>

- While open to interpretation, it's generally more intuitive to consider the embedding as starting from the `(e - 1) * tau`th element of the original time series and ending at the `len(x) - 1`th element (the last value), rather than starting from the 0th element and ending at `len(x) - 1 - (e - 1) * tau`.
- This distinction reflects whether we think of "attaching past values to the present" or "attaching future values to the present". The information content of the result is the same either way.
- The use of `reversed` in the implementation emphasizes this perspective.

</details>

**Examples:**

```
import numpy as np
from edm.embedding import lagged_embed

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tau = 2
e = 3

E = lagged_embed(x, tau, e)
print(E)
print(E.shape)
# [[4 2 0]
#  [5 3 1]
#  [6 4 2]
#  [7 5 3]
#  [8 6 4]
#  [9 7 5]]
# (6, 3)
```

### `scan`

```python
scan(x: np.ndarray, Y: np.ndarray | None = None, *, E: list[int], tau: list[int], n_ahead: int = 1, split: SplitFunc | None = None, predict: PredictFunc | None = None, metric: MetricFunc | None = None) -> np.ndarray
```

Grid search over (E, tau) with cross-validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>([ndarray](#numpy.ndarray), [shape](#shape)([N](#N)))</code> | Time series to embed. | *required*
`Y` | <code>([ndarray](#numpy.ndarray) or None, [shape](#shape)([N](#N)) or ([N](#N), [M](#M)))</code> | Prediction target. If None, self-prediction (Y = x). | <code>None</code>
`E` | <code>[list](#list)[[int](#int)]</code> | Embedding dimension candidates. | *required*
`tau` | <code>[list](#list)[[int](#int)]</code> | Time delay candidates. | *required*
`n_ahead` | <code>[int](#int)</code> | Prediction horizon (steps ahead). | <code>1</code>
`split` | <code>[SplitFunc](#edmkit.splits.SplitFunc) or None</code> | Callable ``(n: int) -> list[Fold]``. Defaults to sliding_folds. | <code>None</code>
`predict` | <code>[PredictFunc](#edmkit.types.PredictFunc) or None</code> | Prediction function. Defaults to ``simplex_projection``. | <code>None</code>
`metric` | <code>[MetricFunc](#edmkit.metrics.MetricFunc) or None</code> | Evaluation metric. Defaults to ``mean_rho``. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`scores` | <code>([ndarray](#numpy.ndarray), [shape](#shape)([len](#len)([E](#E)), [len](#len)([tau](#tau)), [K_max](#K_max)))</code> | Per-fold CV metric for each (E, tau) combination. K_max is the maximum number of folds across all E values. Entries where the fold does not exist are NaN.

### `select`

```python
select(scores: np.ndarray, *, E: list[int], tau: list[int]) -> tuple[int, int, float]
```

Select best (E, tau) from scan results.

Ranks each (E, tau) by ``mean - SE`` where SE is the standard error
of the mean across folds.  This penalises combinations whose scores
vary widely across folds (unstable predictions) and those with fewer
valid folds (less certainty), favouring parameters we are *confident*
perform well.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`scores` | <code>([ndarray](#numpy.ndarray), [shape](#shape)([len](#len)([E](#E)), [len](#len)([tau](#tau)), [K_max](#K_max)))</code> | Output of ``scan``. | *required*
`E` | <code>[list](#list)[[int](#int)]</code> | Embedding dimension candidates (same as passed to ``scan``). | *required*
`tau` | <code>[list](#list)[[int](#int)]</code> | Time delay candidates (same as passed to ``scan``). | *required*

**Returns:**

Type | Description
---- | -----------
<code>([best_E](#best_E), [best_tau](#best_tau), [best_score](#best_score))</code> | ``best_score`` is the mean over folds (not the adjusted value) so that it remains directly interpretable.

