---
title: Pitfalls
description: Keep evaluation honest and avoid false positives in CCM.
---

## Leakage in evaluation

A row at time `t` uses the observations `x[t - (E - 1) tau], ..., x[t]` and the target `x[t + Tp]`. Library rows close in time to a query inflate the score in two ways.

1. **Shared observations**: a library row within `(E - 1) * tau + Tp` of the query shares an observation with it, so part of the query or its answer is in the library. When the target comes from another variable, as in CCM, only the delay vector is shared, and `(E - 1) * tau` is enough.
2. **Autocorrelation**: rows beyond that range share no observation but still lie close to the query, because the trajectory moves little between samples, and their targets are close to the answer for that reason alone. This happens whether library and queries overlap in time or meet at the boundary of separate periods. The autocorrelation time covers it, for example the lag where `autocorrelation(x)` falls below 1/e.

Exclude the library rows within the larger of the two ranges: with `theiler_window` when library and query times overlap, and with the `gap` argument of the functions in `splits` when they are separate periods.

```python
import numpy as np

from edmkit import generate
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 60.0)
x = trajectory[500:, 0]

E, tau, Tp = 3, 4, 20
coordinates = np.array([[0, tau * j] for j in range(E)])
X, t = embed(coordinates, x)
X, t = X[:-Tp], t[:-Tp]
Y = x[t + Tp]

print(pearson_correlation(simplex_projection(X, Y, X), Y))  # 1.0

mask = theiler_window(t, t, width=(E - 1) * tau + Tp)
print(pearson_correlation(simplex_projection(X, Y, X, mask=mask), Y))  # 0.9956...
```

## Data

- **Length and dimension**: each prediction needs `E + 1` neighbors that are close in `E` dimensions. With short series, skill falls as `E` grows. Observation noise has the same effect.
- **Stationarity**: library and queries must follow the same dynamics. A trend or a regime shift makes neighbors meaningless. Remove the trend, or build the library from the same regime.

## CCM

If `y` drives `x`, the embedding of `x` can predict `y`. The converse does not hold: high cross-map skill can also appear without any causal link.

- **Shared driver**: if `z` drives both `x` and `y`, each embedding carries `z`, and skill can be high in both directions.
- **Shared seasonality**: a common periodic forcing is a shared driver. Two independent logistic maps plus the same sine wave cross map each other at 0.96 or more, even with a library of 20. Without the sine wave, skill is about 0.
- **Strong forcing**: if `y` drives `x` so strongly that `x` follows `y` (synchrony), skill appears in both directions.
- **Leakage**: temporal neighbors inflate skill, as described above. Pass the mask to `bootstrap`.

Check these before reading skill as causation.

1. **Convergence**: skill must rise with library size. High skill at the smallest library points to a shared component or synchrony.
2. **Lag**: embed `x`, set the targets to `y[t + Tp]`, and sweep `Tp` (`x xmap y`). If `y` drives `x`, `x` carries the past of `y`, so skill peaks at `Tp <= 0`. Under synchrony both directions show skill, and the direction without causation peaks at `Tp > 0`.
3. **Surrogates**: compare skill with surrogates that keep the suspected shared component, such as seasonal surrogates that keep the seasonal cycle and shuffle the rest. Fully shuffled surrogates destroy autocorrelation and make almost any result look significant.
4. **Unobserved drivers**: CCM on two series cannot rule out a third one. If you suspect one, measure it and cross map it against both series.

Convergence on coupled logistic maps, where `y` drives `x`:

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

# y drives x strongly, x drives y weakly.
x, y = np.empty(1000), np.empty(1000)
x[0], y[0] = 0.4, 0.2
for i in range(999):
    x[i + 1] = x[i] * (3.5 - 3.5 * x[i] - 0.1 * y[i])
    y[i + 1] = y[i] * (3.8 - 3.8 * y[i] - 0.02 * x[i])
x, y = x[100:], y[100:]

# Embed x, and cross map y[t].
E, tau = 2, 1
coordinates = np.array([[0, tau * j] for j in range(E)])
X, t = embed(coordinates, x)
Y = y[t]
mask = theiler_window(t, t, width=(E - 1) * tau)


def score(task):
    predictions = task()  # (B, M, 1)
    return pearson_correlation(predictions, np.broadcast_to(Y[:, None], predictions.shape))


lib_sizes = np.array([10, 50, 200, 800])
rho = np.empty((len(lib_sizes), 20))
indices, tasks = zip(*bootstrap(X, Y, X, lib_sizes, predict_func=simplex_projection, mask=mask))
with ThreadPoolExecutor() as executor:
    for index, result in zip(indices, executor.map(score, tasks)):
        rho[index] = result

print(rho.mean(axis=1).round(3))  # [0.299 0.688 0.867 0.948]
```

Lag under synchrony, with the model of Ye et al. (2015) where `y` drives `x` and `x` does not drive `y`. Both directions cross map almost perfectly, but only the true direction peaks at `Tp <= 0`:

```python
import numpy as np

from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

# y drives x so strongly that x synchronizes to y. x does not drive y.
x, y = np.empty(1000), np.empty(1000)
x[0], y[0] = 0.4, 0.2
for i in range(999):
    x[i + 1] = x[i] * (3.1 - 3.1 * x[i] - 0.8 * y[i])
    y[i + 1] = y[i] * (3.8 - 3.8 * y[i])
x, y = x[100:], y[100:]

E, tau = 2, 1
Tps = np.arange(-8, 13)


def scores(source, target):
    """Skill of the embedding of source at predicting target[t + Tp], for each Tp."""
    coordinates = np.array([[0, tau * j] for j in range(E)])
    X, t = embed(coordinates, source)
    valid = (t + Tps.min() >= 0) & (t + Tps.max() < len(target))
    X, t = X[valid], t[valid]
    Y = target[t[:, None] + Tps]  # one column per Tp
    mask = theiler_window(t, t, width=(E - 1) * tau)
    predictions = simplex_projection(X, Y, X, mask=mask)
    return pearson_correlation(predictions.T[..., None], Y.T[..., None])  # one score per Tp


for name, rho in [("x xmap y", scores(x, y)), ("y xmap x", scores(y, x))]:
    print(f"{name}: rho={rho.max():.2f} at Tp={Tps[rho.argmax()]}")
# x xmap y: rho=1.00 at Tp=-1
# y xmap x: rho=1.00 at Tp=7
```

Skill alone would suggest causation both ways. The positive peak of `y xmap x` shows that `y` only predicts the future of `x`, so `x` does not drive `y`.

## References

- Sugihara et al., Detecting causality in complex ecosystems, Science, 2012.
- Ye et al., Distinguishing time-delayed causal interactions using convergent cross mapping, Scientific Reports, 2015.
- Deyle et al., Global environmental drivers of influenza, PNAS, 2016.
- Cobey and Baskerville, Limits to causal inference with state-space reconstruction for infectious disease, PLOS ONE, 2016.
