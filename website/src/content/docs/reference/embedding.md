---
title: embedding
description: Time-delay embedding.
sidebar:
  order: 1
---

## `embedding`

**Functions:**

Name | Description
---- | -----------
[`embed`](#edmkit.embedding.embed) | Delay vectors for an arbitrary choice of variables and lags.

### `embed`

```python
embed(coordinates: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]
```

Delay vectors for an arbitrary choice of variables and lags.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coordinates` | <code>[ndarray](#numpy.ndarray)</code> | Integer array of shape ``(E, 2)``. Row ``j`` is ``(variable, lag)``, selecting observation ``x[t - lag, variable]`` for the delay vector at time ``t``. A negative lag selects an observation taken after time ``t``. | *required*
`x` | <code>[ndarray](#numpy.ndarray)</code> | Observations of shape ``(T, d)``, or ``(T,)``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Delay vectors of shape ``(N, E)``, one row per time in `t`.
`t` | <code>[ndarray](#numpy.ndarray)</code> | Times of shape ``(N,)`` at which every coordinate is available, that is ``lags.max() <= t <= T - 1 + lags.min()``. A time outside ``[0, T - 1]`` is included when every coordinate still reads an observation inside `x`.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `coordinates` is not an integer array of shape ``(E, 2)`` with ``E >= 1``. - If `x` is not a 1D or 2D array. - If a variable index lies outside `x`. - If no time has every coordinate available.

**Examples:**

```python
import numpy as np

from edmkit.embedding import embed

x = np.arange(20).reshape(10, 2)  # ten times, two variables

# Classical univariate delay coordinates.
tau, E = 2, 3
coordinates = np.array([[0, tau * j] for j in range(E)])
X, t = embed(coordinates, x)
print(X)
print(t)
# [[ 8  4  0]
#  [10  6  2]
#  [12  8  4]
#  [14 10  6]
#  [16 12  8]
#  [18 14 10]]
# [4 5 6 7 8 9]

# One step ahead of the same coordinates, as iterated forecasting needs.
Y, t_ahead = embed(coordinates - [0, 1], x)
```

