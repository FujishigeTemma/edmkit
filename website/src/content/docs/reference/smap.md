---
title: smap
description: S-Map local linear prediction.
sidebar:
  order: 3
---

## `smap`

**Functions:**

Name | Description
---- | -----------
[`smap`](#edmkit.smap.smap) | Perform S-Map (local linear regression) from `X` to `Y`.
[`weights`](#edmkit.smap.weights) | Compute S-Map exponential weights, zeroing out masked library points.

### `smap`

```python
smap(X: np.ndarray, Y: np.ndarray, Q: np.ndarray, *, theta: float, alpha: float = 1e-10, mask: np.ndarray | None = None, use_tensor: bool = False) -> np.ndarray
```

Perform S-Map (local linear regression) from `X` to `Y`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | The input data | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | The target data | *required*
`Q` | <code>[ndarray](#numpy.ndarray)</code> | The query points for which to make predictions. | *required*
`theta` | <code>[float](#float)</code> | Locality parameter. (0: global linear, >0: local linear) | *required*
`alpha` | <code>[float](#float)</code> | Regularization parameter to stabilize the inversion. | <code>1e-10</code>
`use_tensor` | <code>[bool](#bool)</code> | Whether to use `tinygrad.Tensor` for computation. **This may be slower than the NumPy implementation in most cases for now.** | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | The predicted values based on the weighted linear regression.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If the input arrays `X` and `Y` do not have the same number of points. - If `theta` is negative.

**Examples:**

```python
import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.smap import smap

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

# Local linear with theta=4.0
predictions = smap(X, Y, Q, theta=4.0)
correlation = np.corrcoef(predictions, actual)[0, 1]
print(f"Correlation (theta=4.0): {correlation:.3f}")

# Global linear with theta=0.0
predictions_global = smap(X, Y, Q, theta=0.0)
correlation_global = np.corrcoef(predictions_global, actual)[0, 1]
print(f"Correlation (theta=0.0): {correlation_global:.3f}")
```

### `weights`

```python
weights(D: np.ndarray, theta: float, *, mask: np.ndarray | None = None, min_points: int) -> np.ndarray
```

Compute S-Map exponential weights, zeroing out masked library points.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`D` | <code>[ndarray](#numpy.ndarray)</code> | Distance matrix — (M, N) or (B, M, N). | *required*
`theta` | <code>[float](#float)</code> | Locality parameter. | *required*
`mask` | <code>[ndarray](#numpy.ndarray) \| None</code> | Boolean mask over the library axis — (N,) or (B, N). | <code>None</code>
`min_points` | <code>[int](#int)</code> | Minimum number of valid library points required. | *required*

