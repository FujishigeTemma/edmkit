---
title: Getting Started
description: Install edmkit and make your first prediction.
---

## Install

edmkit requires Python 3.13 or later.

```bash
uv add edmkit
# or
pip install edmkit
```

## The model

The prediction methods (`simplex_projection`, `soft_simplex_projection`, and `smap`) take three arrays.

Name | Shape | Meaning
---- | ----- | -------
`X` | `(N, E)` | Library inputs, usually delay vectors
`Y` | `(N, E')` | Library targets, paired row by row with `X`
`Q` | `(M, E)` | Query points to predict at

The result has one row per query. Each prediction is a weighted sum of the rows of `Y`, so each method is linear in `Y` and has a matching `linear_operator`:

```python
import numpy as np

from edmkit.simplex_projection import linear_operator, simplex_projection

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))  # (N, E)
Y = rng.normal(size=(200, 2))  # (N, E')
Q = rng.normal(size=(50, 3))  # (M, E)

assert np.allclose(simplex_projection(X, Y, Q), linear_operator(X, Q) @ Y)
```

An optional `mask(i, j)` decides whether query `i` may use library point `j`.

Arrays may also carry a leading batch axis, `(B, N, E)`, and may be tinygrad Tensors instead of NumPy arrays.

## Forecast a time series

Observe one coordinate of the Lorenz system, embed it, and predict one step ahead.

```python
import numpy as np

from edmkit import generate
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 40.0)
x = trajectory[500:, 0]

# Delay vectors (x[t], x[t - 2], x[t - 4]) and the value one step ahead.
coordinates = np.array([[0, 0], [0, 2], [0, 4]])
X, t = embed(coordinates, x)
X, t = X[:-1], t[:-1]
Y = x[t + 1]

# The first half is the library, the second half is queried.
n = len(X) // 2
predictions = simplex_projection(X[:n], Y[:n], X[n:])
print(pearson_correlation(predictions, Y[n:]))  # 0.9991...

predictions = smap(X[:n], Y[:n], X[n:], theta=2.0)
print(pearson_correlation(predictions, Y[n:]))  # 0.9999...
```

## Leave-one-out with a mask

To query the library itself, pass `Q = X` and exclude points that are close in time. `theiler_window` builds that mask.

```python
import numpy as np

from edmkit import generate
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 40.0)
x = trajectory[500:, 0]

coordinates = np.array([[0, 0], [0, 2], [0, 4]])
X, t = embed(coordinates, x)
X, t = X[:-1], t[:-1]
Y = x[t + 1]

mask = theiler_window(t, t, width=16)  # see Pitfalls
predictions = simplex_projection(X, Y, X, mask=mask)
print(pearson_correlation(predictions, Y))  # 0.9996...
```

## Convergent cross mapping

`ccm.bootstrap` yields tasks that predict on random libraries of each size. The tasks are independent, so run them in a thread pool. If `y` drives `x`, the embedding of `x` can predict `y`, and the skill rises as the library grows.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit import generate
from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 40.0)
x, y = trajectory[500:, 0], trajectory[500:, 1]

# Embed x, and target y at the same time (Tp = 0).
coordinates = np.array([[0, 0], [0, 2], [0, 4]])
X, t = embed(coordinates, x)
Y = y[t]
mask = theiler_window(t, t, width=16)  # see Pitfalls


def score(task):
    predictions = task()  # (B, M, 1)
    return pearson_correlation(predictions, np.broadcast_to(Y[:, None], predictions.shape))


lib_sizes = np.array([10, 100, 1000])
rho = np.empty((len(lib_sizes), 20))
indices, tasks = zip(*bootstrap(X, Y, X, lib_sizes, predict_func=simplex_projection, mask=mask))
with ThreadPoolExecutor() as executor:
    for index, result in zip(indices, executor.map(score, tasks)):
        rho[index] = result

print(rho.mean(axis=1).round(3))  # [0.834 0.977 0.997]
```

## Next

- [Embedding search](../examples/embedding-search/) chooses `E`, `tau`, and `Tp`.
- [Pitfalls](../pitfalls/) covers leakage and false positives in CCM.
- The [API reference](../reference/embedding/) lists every function and its parameters.
