---
title: ccm
description: Convergent Cross Mapping for causal inference.
sidebar:
  order: 5
---

## `ccm`

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#edmkit.ccm.bootstrap) | Iterate `predict_func(X[library], Y[library], Q)` for `n_samples` times for each library size in `lib_sizes`.

### `bootstrap`

```python
bootstrap(X: np.ndarray, Y: np.ndarray, Q: np.ndarray, lib_sizes: np.ndarray, *, predict_func: PredictFunc, n_samples: int = 20, replace: bool = True, seed: int | None = 42, mask: Mask | None = None, batch_size: int | None = 20) -> Iterator[tuple[tuple[int, slice], Callable[[], np.ndarray]]]
```

Iterate `predict_func(X[library], Y[library], Q)` for `n_samples` times for each library size in `lib_sizes`.

By gathering the results of the returned tasks, you can get `(len(lib_sizes), n_samples, M, E')` array of predictions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | The input data of shape (N,) or (N, E) | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | The target data of shape (N,) or (N, E'), sampled together with `X`. | *required*
`Q` | <code>[ndarray](#numpy.ndarray)</code> | The query points of shape (M,) or (M, E), shared by all sampled libraries as a read-only view. | *required*
`lib_sizes` | <code>[ndarray](#numpy.ndarray)</code> | The 1D array of library sizes, yielded in the given order. Sizes may exceed N only if `replace` is True. | *required*
`predict_func` | <code>[PredictFunc](#edmkit.types.PredictFunc)</code> | The batched prediction function, such as `simplex_projection` or ``functools.partial(smap, theta=...)``. | *required*
`n_samples` | <code>[int](#int)</code> | The number of libraries to sample for each size in `lib_sizes`. | <code>20</code>
`replace` | <code>[bool](#bool)</code> | Whether to sample the library points with replacement. | <code>True</code>
`seed` | <code>[int](#int) or None</code> | The seed of the random generator. If None, uses fresh entropy. | <code>42</code>
`mask` | <code>[Mask](#edmkit.types.Mask) or None</code> | ``mask(i, j)`` indicating, for each query in `Q`, which points of `X` to include. Use `theiler_window` for temporal exclusion. | <code>None</code>
`batch_size` | <code>[int](#int) or None</code> | The maximum number of libraries per task. If None, uses `n_samples`. | <code>20</code>

**Yields:**

Name | Type | Description
---- | ---- | -----------
`index` | <code>[tuple](#tuple)[[int](#int), [slice](#slice)]</code> | The position of the task's libraries in an array of shape (len(`lib_sizes`), `n_samples`), such as the scores.
`samples` | <code>[Callable](#collections.abc.Callable)[[], [ndarray](#numpy.ndarray)]</code> | The task that returns the predictions on B sampled libraries, of shape (B, M, E'), where B <= `batch_size` and E' = 1 if `Y` is 1D.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `X`, `Y`, or `Q` is empty or not 1D or 2D. - If the input arrays `X` and `Y` do not have the same number of points. - If the dimension of `X` and `Q` do not match. - If `lib_sizes` is not a nonempty 1D array of positive integers. - If some size in `lib_sizes` exceeds N when `replace` is False. - If `n_samples` or `batch_size` is not positive. - If `predict_func` raises when a task is executed.

**Examples:**

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection

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

pool_size = 200
Tp = 1
X = embedding[: pool_size - shift]
Y = x[shift + Tp : pool_size + Tp]
Q = embedding[pool_size - shift : -Tp]
actual = x[pool_size + Tp :, None]

lib_sizes = np.array([10, 50, 190])
n_samples = 100


def score(samples):
    predictions = samples()  # (B, M, 1)
    return pearson_correlation(predictions, np.broadcast_to(actual, predictions.shape))


scores = np.empty((len(lib_sizes), n_samples))
tasks = bootstrap(X, Y, Q, lib_sizes, predict_func=simplex_projection, n_samples=n_samples)
indices, samples = zip(*tasks)
with ThreadPoolExecutor() as executor:
    for index, result in zip(indices, executor.map(score, samples)):
        scores[index] = result

for lib_size, correlation in zip(lib_sizes, scores.mean(axis=1)):
    print(f"Correlation (L={lib_size}): {correlation:.3f}")
```

