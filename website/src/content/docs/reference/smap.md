---
title: smap
description: S-Map local linear prediction.
sidebar:
  order: 3
---

**Functions:**

Name | Description
---- | -----------
[`coefficients`](#coefficients) | Compute the S-Map coefficients `C` of the local linear regression from `X` to `Y` around each query in `Q`.
[`linear_operator`](#linear_operator) | Build the linear operator `S` of S-Map, such that ``S @ Y`` equals ``smap(X, Y, Q)`` for any target `Y`.
[`smap`](#smap) | Perform S-Map (local linear regression) from `X` to `Y`.

## `coefficients`

```python
coefficients(X, Y, Q, *, theta, alpha = 1e-10, mask = None)
```

Compute the S-Map coefficients `C` of the local linear regression from `X` to `Y` around each query in `Q`.

``C[m, 0]`` is the intercept and ``C[m, 1 + i, j]`` is the slope of target `j` with respect to coordinate `i` around ``Q[m]``,
so ``smap(X, Y, Q)[m]`` equals ``C[m, 0] + Q[m] @ C[m, 1:]``.
The slopes estimate the local Jacobian of the map from `X` to `Y`, shrunk by the regularization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The input data of shape (N,) or (N, E) or (B, N, E) | *required*
`Y` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The target data of shape (N,) or (N, E') or (B, N, E') | *required*
`Q` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The query points of shape (M,) or (M, E) or (B, M, E) around which to fit the regressions. | *required*
`theta` | <code>[float](#float)</code> | Locality parameter. (0: global linear, >0: local linear) | *required*
`alpha` | <code>[float](#float)</code> | Regularization parameter to stabilize the inversion. | <code>1e-10</code>
`mask` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor) or None</code> | Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include in the regression. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`C` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The coefficients of shape (M, E+1) if `Y` is 1D, (M, E+1, E') if `Y` is 2D, or (B, M, E+1, E') if `Y` is 3D.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `theta` is negative. - If the input arrays `X` and `Y` do not have the same number of points. - If the batch size or dimension of `X` and `Q` do not match. - If `mask` does not have shape (M, N) or (B, M, N). - If fewer than E + 1 library points are unmasked for some query.

**Examples:**

```python
import numpy as np

from edmkit.smap import coefficients, smap

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
Q = rng.normal(size=(50, 3))
Y = rng.normal(size=(200, 4))

C = coefficients(X, Y, Q, theta=4.0)  # (50, 4, 4)
jacobians = C[:, 1:].swapaxes(-1, -2)  # (50, 4, 3): d Y_j / d X_i around each query

assert np.allclose(C[:, 0] + np.einsum("mi,mij->mj", Q, C[:, 1:]), smap(X, Y, Q, theta=4.0))
```



## `linear_operator`

```python
linear_operator(X, Q, *, theta, alpha = 1e-10, mask = None)
```

Build the linear operator `S` of S-Map, such that ``S @ Y`` equals ``smap(X, Y, Q)`` for any target `Y`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The input data of shape (N,) or (N, E) or (B, N, E) | *required*
`Q` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The query points of shape (M,) or (M, E) or (B, M, E) for which to make predictions. | *required*
`theta` | <code>[float](#float)</code> | Locality parameter. (0: global linear, >0: local linear) | *required*
`alpha` | <code>[float](#float)</code> | Regularization parameter to stabilize the inversion. | <code>1e-10</code>
`mask` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor) or None</code> | Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include in the regression. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`S` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The operator of shape (M, N) or (B, M, N).

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `theta` is negative. - If the batch size or dimension of `X` and `Q` do not match. - If `mask` does not have shape (M, N) or (B, M, N). - If fewer than E + 1 library points are unmasked for some query.

**Examples:**

```python
import numpy as np

from edmkit.smap import linear_operator, smap

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
Q = rng.normal(size=(50, 3))
Y = rng.normal(size=(200, 4))

assert np.allclose(linear_operator(X, Q, theta=4.0) @ Y, smap(X, Y, Q, theta=4.0))
```



## `smap`

```python
smap(X, Y, Q, *, theta, alpha = 1e-10, mask = None)
```

Perform S-Map (local linear regression) from `X` to `Y`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The input data of shape (N,) or (N, E) or (B, N, E) | *required*
`Y` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The target data of shape (N,) or (N, E') or (B, N, E') | *required*
`Q` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The query points of shape (M,) or (M, E) or (B, M, E) for which to make predictions. | *required*
`theta` | <code>[float](#float)</code> | Locality parameter. (0: global linear, >0: local linear) | *required*
`alpha` | <code>[float](#float)</code> | Regularization parameter to stabilize the inversion. | <code>1e-10</code>
`mask` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor) or None</code> | Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include in the regression. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | The predicted values based on the weighted linear regression. Shape (M,) if `Y` is 1D, (M, E') if `Y` is 2D, or (B, M, E') if `Y` is 3D.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `theta` is negative. - If the input arrays `X` and `Y` do not have the same number of points. - If the batch size or dimension of `X` and `Q` do not match. - If `mask` does not have shape (M, N) or (B, M, N). - If fewer than E + 1 library points are unmasked for some query.

**Examples:**

```python
import numpy as np

from edmkit.embedding import embed
from edmkit.smap import smap

# Generate a simple time series (logistic map)
N = 300
x = np.zeros(N)
x[0] = 0.4
for i in range(1, N):
    x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

tau = 2
E = 3

coordinates = np.array([[0, tau * j] for j in range(E)])
embedding, _ = embed(coordinates, x)
shift = tau * (E - 1)

lib_size = 200
Tp = 1
X = embedding[: lib_size - shift]
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

