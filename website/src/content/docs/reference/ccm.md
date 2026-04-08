---
title: ccm
description: Convergent Cross Mapping for causal inference.
sidebar:
  order: 4
---

## `ccm`

**Functions:**

Name | Description
---- | -----------
[`make_sample_func`](#edmkit.ccm.make_sample_func) | Create a sample function with its own independent RNG.
[`bootstrap`](#edmkit.ccm.bootstrap) | Perform Convergent Cross Mapping and return per-sample correlations.
[`ccm`](#edmkit.ccm.ccm) | Perform Convergent Cross Mapping using a custom prediction function.
[`pearson_correlation`](#edmkit.ccm.pearson_correlation) | Compute vectorized Pearson correlation between X and Y.
[`with_simplex_projection`](#edmkit.ccm.with_simplex_projection) | Perform Convergent Cross Mapping using simplex projection.
[`with_smap`](#edmkit.ccm.with_smap) | Perform Convergent Cross Mapping using S-Map (local linear regression).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`SampleFunc`](#edmkit.ccm.SampleFunc) | <code>[TypeAlias](#typing.TypeAlias)</code> | SampleFunc is a function that takes (pool, size) and returns a sampled array.
[`AggregateFunc`](#edmkit.ccm.AggregateFunc) | <code>[TypeAlias](#typing.TypeAlias)</code> | AggregateFunc is a function that takes an array of values and returns a single value.

### `SampleFunc`

```python
SampleFunc: TypeAlias = Callable[[np.ndarray, int], np.ndarray]
```

SampleFunc is a function that takes (pool, size) and returns a sampled array.

### `AggregateFunc`

```python
AggregateFunc: TypeAlias = Callable[[np.ndarray], float]
```

AggregateFunc is a function that takes an array of values and returns a single value.

### `make_sample_func`

```python
make_sample_func(seed: int | None = 42) -> SampleFunc
```

Create a sample function with its own independent RNG.

### `bootstrap`

```python
bootstrap(X: np.ndarray, Y: np.ndarray, lib_sizes: np.ndarray, predict_func: PredictFunc, n_samples: int = 20, *, library_pool: np.ndarray, prediction_pool: np.ndarray, sample_func: SampleFunc | None = None, batch_size: int | None = 10) -> np.ndarray
```

Perform Convergent Cross Mapping and return per-sample correlations.

Same as :func:`ccm` but returns the raw per-sample scores instead of
aggregating them.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Library time series (potential response) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target time series (potential driver) | *required*
`lib_sizes` | <code>[ndarray](#numpy.ndarray)</code> | Array of library sizes to test convergence. | *required*
`predict_func` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature (X, Y, Q) -> predictions. | *required*
`n_samples` | <code>[int](#int)</code> | Number of random samples per library size for bootstrapping. | <code>20</code>
`library_pool` | <code>[ndarray](#numpy.ndarray)</code> | 1-D array of integer indices from which library members are sampled. | *required*
`prediction_pool` | <code>[ndarray](#numpy.ndarray)</code> | 1-D array of integer indices that are predicted. | *required*
`sample_func` | <code>[SampleFunc](#edmkit.ccm.SampleFunc) or None</code> | Function responsible for drawing a library sample of a given size. When None, a fresh RNG-backed sampler is created per call. | <code>None</code>
`batch_size` | <code>[int](#int) or None</code> | If specified, predictions are made in batches to limit memory usage. | <code>10</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`samples` | <code>[ndarray](#numpy.ndarray)</code> | Per-sample correlation coefficients of shape ``(n_samples, len(lib_sizes))``.

### `ccm`

```python
ccm(X: np.ndarray, Y: np.ndarray, lib_sizes: np.ndarray, predict_func: PredictFunc, n_samples: int = 20, *, library_pool: np.ndarray, prediction_pool: np.ndarray, sample_func: SampleFunc | None = None, aggregate_func: AggregateFunc = np.mean, batch_size: int | None = 10) -> np.ndarray
```

Perform Convergent Cross Mapping using a custom prediction function.

CCM tests for causality from X to Y by using the attractor reconstructed from Y
to predict values of X. If X causes Y, then Y's attractor contains information
about X, allowing cross-mapping from Y to X.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Library time series (potential response) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target time series (potential driver) | *required*
`lib_sizes` | <code>[ndarray](#numpy.ndarray)</code> | Array of library sizes to test convergence. | *required*
`predict_func` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | Prediction function with signature (X, Y, Q) -> predictions. Can be `simplex_projection`, `smap` with partial application, or a custom function. | *required*
`n_samples` | <code>[int](#int)</code> | Number of random samples per library size for bootstrapping. | <code>100</code>
`library_pool` | <code>[ndarray](#numpy.ndarray)</code> | 1-D array of integer indices from which library members are sampled. | *required*
`prediction_pool` | <code>[ndarray](#numpy.ndarray)</code> | 1-D array of integer indices that are predicted. | *required*
`sample_func` | <code>[SampleFunc](#edmkit.ccm.SampleFunc) or None</code> | Function responsible for drawing a library sample of a given size. It receives ``(pool, size)`` and returns an array of indices. When None, a fresh RNG-backed sampler is created per call. | <code>None</code>
`aggregate_func` | <code>[AggregateFunc](#edmkit.ccm.AggregateFunc)</code> | Reducer applied to the correlation samples for each library size. | <code>np.mean</code>
`batch_size` | <code>[int](#int) or None</code> | If not specified, batch_size == n_samples. If specified, predictions are made in batches to limit memory usage. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`correlations` | <code>[ndarray](#numpy.ndarray)</code> | Mean correlation coefficient for each library size.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `X` and `Y` have different lengths. - If `lib_sizes` contains non-positive values. - If `predict_func` is not callable. - If `n_samples` is not positive. - If `aggregate_func` is not callable. - If `library_pool` or `prediction_pool` is invalid.

<details class="note" open markdown="1">
<summary>Notes</summary>

- Higher correlation at larger library sizes indicates convergence and suggests X influences Y (X -> Y causality)
- Convergence is the key signature of causality in CCM
- The method uses Y's attractor to predict X (cross-mapping)

</details>

**Examples:**

```python
from functools import partial

import numpy as np

from edmkit.ccm import ccm, simplex_projection, smap
from edmkit.embedding import lagged_embed

# Generate coupled logistic maps (X drives Y)
N = 1000
rx, ry, Bxy = 3.8, 3.5, 0.02
X = np.zeros(N)
Y = np.zeros(N)
X[0], Y[0] = 0.4, 0.2
for i in range(1, N):
    X[i] = X[i - 1] * (rx - rx * X[i - 1])
    Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]

tau = 1
E = 2

# To test X -> Y causality, cross-map from Y's attractor to X
Y_embedding = lagged_embed(Y, tau=tau, e=E)
shift = tau * (E - 1)
X_aligned = X[shift:]

library_pool = np.arange(Y_embedding.shape[0] // 2)
prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

# logarithmic within range 10 to max library size
lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=5, dtype=int)

# Using simplex projection
correlations = ccm(
    Y_embedding,
    X_aligned,
    lib_sizes=lib_sizes,
    predict_func=simplex_projection,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)

# Using S-Map with partial application
correlations = ccm(
    Y_embedding,
    X_aligned,
    lib_sizes=lib_sizes,
    predict_func=partial(smap, theta=2.0, alpha=1e-10),
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
```

### `pearson_correlation`

```python
pearson_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray
```

Compute vectorized Pearson correlation between X and Y.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | 1D or 2D array of shape (L,) or (B, L) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | 1D or 2D array of shape (L,) or (B, L) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`correlation` | <code>[ndarray](#numpy.ndarray)</code> | Pearson correlation coefficient(s) between X and Y. Shape (B,) if inputs are 2D, else scalar.

### `with_simplex_projection`

```python
with_simplex_projection(X: np.ndarray, Y: np.ndarray, lib_sizes: np.ndarray, n_samples: int = 100, use_tensor: bool = False, *, library_pool: np.ndarray, prediction_pool: np.ndarray, sample_func: SampleFunc | None = None, aggregate_func: AggregateFunc = np.mean) -> np.ndarray
```

Perform Convergent Cross Mapping using simplex projection.

This is a convenience wrapper around the general ccm function using
simplex_projection as the prediction method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Library time series (potential response) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target time series (potential driver) | *required*
`lib_sizes` | <code>[ndarray](#numpy.ndarray)</code> | Array of library sizes to test convergence | *required*
`n_samples` | <code>[int](#int)</code> | Number of random samples per library size for bootstrapping | <code>100</code>
`use_tensor` | <code>[bool](#bool)</code> | Whether to use tinygrad tensors for computation | <code>False</code>
`library_pool` | <code>[ndarray](#numpy.ndarray)</code> | Indices that can be used to draw library samples. Defaults to the full range. | *required*
`prediction_pool` | <code>[ndarray](#numpy.ndarray)</code> | Indices that should be predicted (leave-one-out over this set). Defaults to the full range. | *required*
`sample_func` | <code>[callable](#callable)</code> | Function responsible for drawing a library sample of a given size. When omitted, a fresh RNG-backed sampler is created per call. | <code>None</code>
`aggregate_func` | <code>[callable](#callable)</code> | Reducer applied to the correlation samples for each library size. Falls back to `np.mean` when omitted. | <code>[mean](#numpy.mean)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`correlations` | <code>[ndarray](#numpy.ndarray)</code> | Mean correlation coefficient for each library size

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If the underlying ccm call detects invalid arguments

**Examples:**

```python
import numpy as np

from edmkit import ccm
from edmkit.embedding import lagged_embed

# Generate coupled logistic maps (X drives Y)
N = 1000
rx, ry, Bxy = 3.8, 3.5, 0.02
X = np.zeros(N)
Y = np.zeros(N)
X[0], Y[0] = 0.4, 0.2
for i in range(1, N):
    X[i] = X[i - 1] * (rx - rx * X[i - 1])
    Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]

tau = 1
E = 2

# To test X -> Y causality, cross-map from Y's attractor to X
Y_embedding = lagged_embed(Y, tau=tau, e=E)
shift = tau * (E - 1)
X_aligned = X[shift:]

library_pool = np.arange(Y_embedding.shape[0] // 2)
prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

# logarithmic within range 10 to max library size
lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=5, dtype=int)

correlations = ccm.with_simplex_projection(
    Y_embedding,
    X_aligned,
    lib_sizes=lib_sizes,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
```

### `with_smap`

```python
with_smap(X: np.ndarray, Y: np.ndarray, lib_sizes: np.ndarray, theta: float, alpha: float = 1e-10, n_samples: int = 100, use_tensor: bool = False, *, library_pool: np.ndarray, prediction_pool: np.ndarray, sample_func: SampleFunc | None = None, aggregate_func: AggregateFunc = np.mean) -> np.ndarray
```

Perform Convergent Cross Mapping using S-Map (local linear regression).

This is a convenience wrapper around the general ccm function using
smap as the prediction method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Library time series (potential response) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target time series (potential driver) | *required*
`lib_sizes` | <code>[ndarray](#numpy.ndarray)</code> | Array of library sizes to test convergence | *required*
`theta` | <code>[float](#float)</code> | Nonlinearity parameter for S-Map | *required*
`alpha` | <code>[float](#float)</code> | Regularization parameter for S-Map | <code>1e-10</code>
`n_samples` | <code>[int](#int)</code> | Number of random samples per library size for bootstrapping | <code>100</code>
`use_tensor` | <code>[bool](#bool)</code> | Whether to use tinygrad tensors for computation | <code>False</code>
`library_pool` | <code>[ndarray](#numpy.ndarray)</code> | Indices that can be used to draw library samples. Defaults to the full range. | *required*
`prediction_pool` | <code>[ndarray](#numpy.ndarray)</code> | Indices that should be predicted (leave-one-out over this set). Defaults to the full range. | *required*
`sample_func` | <code>[callable](#callable)</code> | Function responsible for drawing a library sample of a given size. When omitted, a fresh RNG-backed sampler is created per call. | <code>None</code>
`aggregate_func` | <code>[callable](#callable)</code> | Reducer applied to the correlation samples for each library size. Falls back to `np.mean` when omitted. | <code>[mean](#numpy.mean)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`correlations` | <code>[ndarray](#numpy.ndarray)</code> | Mean correlation coefficient for each library size

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If the underlying ccm call detects invalid arguments

**Examples:**

```python
import numpy as np

from edmkit import ccm
from edmkit.embedding import lagged_embed

# Generate coupled logistic maps (X drives Y)
N = 1000
rx, ry, Bxy = 3.8, 3.5, 0.02
X = np.zeros(N)
Y = np.zeros(N)
X[0], Y[0] = 0.4, 0.2
for i in range(1, N):
    X[i] = X[i - 1] * (rx - rx * X[i - 1])
    Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]

tau = 1
E = 2

# To test X -> Y causality, cross-map from Y's attractor to X
Y_embedding = lagged_embed(Y, tau=tau, e=E)
shift = tau * (E - 1)
X_aligned = X[shift:]

library_pool = np.arange(Y_embedding.shape[0] // 2)
prediction_pool = np.arange(Y_embedding.shape[0] // 2, Y_embedding.shape[0])

# logarithmic within range 10 to max library size
lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=5, dtype=int)

correlations = ccm.with_smap(
    Y_embedding,
    X_aligned,
    lib_sizes=lib_sizes,
    theta=2.0,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
```

