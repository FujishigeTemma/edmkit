---
title: simplex_projection
description: Simplex projection, leave-one-out, and k-nearest neighbors.
sidebar:
  order: 2
---

**Functions:**

Name | Description
---- | -----------
[`loo`](#loo) | Leave-one-out simplex projection: predict each point in `X` from its neighbors, excluding temporally close points.
[`simplex_projection`](#simplex_projection) | Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`.
[`soft_simplex_projection`](#soft_simplex_projection) | Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`, with a soft boundary between neighbors and non-neighbors.

## `loo`

```python
loo(X: np.ndarray, Y: np.ndarray, *, theiler_window: int) -> np.ndarray
```

Leave-one-out simplex projection: predict each point in `X` from its neighbors, excluding temporally close points.

Equivalent to ``simplex_projection(X, Y, X)`` with Theiler window exclusion,
but with the correct temporal index handling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | The input data of shape (N,) or (N, E) or (B, N, E). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | The target data of shape (N,) or (N, E') or (B, N, E'). | *required*
`theiler_window` | <code>[int](#int)</code> | Theiler window half-width. Library points ``j`` where ``|i - j| <= theiler_window`` are excluded when predicting point ``i``. For lagged embedding, use ``(E - 1) * tau``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | The predicted values of shape (N,) or (N, E') or (B, N, E').

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If the input arrays `X` and `Y` do not have the same number of points. - If there are not enough library points outside the Theiler window.



## `simplex_projection`

```python
simplex_projection(X, Y, Q, *, k = None, mask = None)
```

Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The input data of shape (N,) or (N, E) or (B, N, E) | *required*
`Y` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The target data of shape (N,) or (N, E') or (B, N, E') | *required*
`Q` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The query points of shape (M,) or (M, E) or (B, M, E) for which to find the nearest neighbors in `X`. | *required*
`k` | <code>[int](#int) or None</code> | The number of nearest neighbors to use. If None, uses E + 1, where E is the dimension of `X`. | <code>None</code>
`mask` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor) or None</code> | Boolean mask of shape (N,) or (B, N) indicating which library points to include when finding nearest neighbors for the queries in `Q`. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The predicted values based on the weighted mean of the nearest neighbors in `Y`.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `k` is not positive. - If the input arrays `X` and `Y` do not have the same number of points.

**Examples:**

```python
import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection

# Generate a simple time series (logistic map)
N = 300
x = np.zeros(N)
x[0] = 0.4
for i in range(1, N):
    x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

tau = 2
E = 3

embedding = lagged_embed(x, tau=tau, e=E)
shift = tau * (E - 1)

lib_size = 200
Tp = 1
X = embedding[:lib_size - shift]
Y = x[shift + Tp : lib_size + Tp]
Q = embedding[lib_size - shift : -Tp]
actual = x[lib_size + Tp :]

predictions = simplex_projection(X, Y, Q)

correlation = np.corrcoef(predictions, actual)[0, 1]
print(f"Correlation: {correlation:.3f}")
```



## `soft_simplex_projection`

```python
soft_simplex_projection(X, Y, Q, *, k = None, mask = None, softness = 0.02)
```

Perform simplex projection from `X` to `Y` using the nearest neighbors of the points specified by `Q`, with a soft boundary between neighbors and non-neighbors.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The input data of shape (N,) or (N, E) or (B, N, E) | *required*
`Y` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The target data of shape (N,) or (N, E') or (B, N, E') | *required*
`Q` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The query points of shape (M,) or (M, E) or (B, M, E) for which to find the nearest neighbors in `X`. | *required*
`k` | <code>[int](#int) or None</code> | The number of nearest neighbors to use. If None, uses E + 1, where E is the dimension of `X`. | <code>None</code>
`mask` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor) or None</code> | Boolean mask of shape (N,) or (B, N) indicating which library points to include when finding nearest neighbors for the queries in `Q`. | <code>None</code>
`softness` | <code>[float](#float)</code> | Width of the boundary between neighbors and non-neighbors, as a fraction of the neighborhood radius. For distinct boundary distances: ``soft_simplex_projection(X, Y, Q, softness) -> simplex_projection(X, Y, Q) as softness -> 0`` | <code>0.02</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The predicted values based on the weighted mean of the nearest neighbors in `Y`.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `k` is not positive. - If `softness` is not positive. - If the input arrays `X` and `Y` do not have the same number of points. - If `X` does not contain at least `k + 1` points.

**Examples:**

```python
import numpy as np
from tinygrad import Tensor

from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import soft_simplex_projection

# Generate a simple time series (logistic map)
N = 300
x = np.zeros(N, dtype=np.float32)
x[0] = 0.4
for i in range(1, N):
    x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

tau = 2
E = 3

embedding = lagged_embed(x, tau=tau, e=E)
shift = tau * (E - 1)

lib_size = 200
Tp = 1
X = Tensor(embedding[:lib_size - shift])
Y = Tensor(x[shift + Tp : lib_size + Tp])
Q = Tensor(embedding[lib_size - shift : -Tp])
actual = x[lib_size + Tp :]

predictions = soft_simplex_projection(X, Y, Q).numpy()

correlation = np.corrcoef(predictions, actual)[0, 1]
print(f"Correlation: {correlation:.3f}")
```

