---
title: Forecasting a time series
description: A full forecast pipeline with simplex projection — parameter selection, train/test split, prediction, evaluation, and a nonlinearity check.
---

This guide builds a one-step-ahead forecast for a scalar nonlinear series. The pipeline is generic: any series long enough to embed will work.

1. Pick `(E, tau)` from data.
2. Embed and align targets.
3. Split into library and query.
4. Predict with simplex projection.
5. Evaluate against held-out truth.
6. Verify nonlinearity by sweeping S-Map's `theta`.

Read [Time-delay embedding](/edmkit/concepts/embedding/) and [Simplex projection](/edmkit/concepts/simplex-projection/) first if needed.

## Set up the data

The Lorenz attractor stands in for any chaotic series. Substitute your own `x` to use the rest as-is.

```python
import numpy as np
from edmkit.generate import lorenz

_, trajectory = lorenz(
    sigma=10, rho=28, beta=8 / 3,
    X0=np.array([1.0, 1.0, 1.0]), dt=0.01, t_max=80,
)
x = trajectory[:, 0]
```

## Pick embedding parameters

```python
from edmkit.embedding import scan, select

E_grid = list(range(1, 11))
tau_grid = [1, 2, 3, 5, 8]
scores = scan(x, E=E_grid, tau=tau_grid)
E, tau, cv_rho = select(scores, E=E_grid, tau=tau_grid)
print(f"E={E}, tau={tau}, cross-validated rho={cv_rho:.3f}")
```

See the [parameter selection guide](/edmkit/guides/choosing-parameters/) for reading `scores` and customization.

## Embed and align

`embedded[i]` is time `(E - 1) * tau + i` in the original series, so the one-step-ahead target is `x[(E - 1) * tau + i + 1]`.

```python
from edmkit.embedding import lagged_embed

embedded = lagged_embed(x, tau=tau, e=E)
shift = (E - 1) * tau
n_pairs = len(embedded) - 1     # one less, because we need a 1-step-ahead target
```

For a horizon `Tp > 1`, replace `+1` below with `+Tp` and reduce `n_pairs` accordingly.

## Split the data

Temporal split: the first `n_train` pairs are the library, the rest is the test set.

```python
n_train = n_pairs * 2 // 3

library = embedded[:n_train]
target = x[shift + 1 : shift + n_train + 1]

query = embedded[n_train:n_pairs]
truth = x[shift + n_train + 1 : shift + n_pairs + 1]
```

Random splits leak across train and test through overlapping embeddings. Stick with temporal splits, or use leave-one-out with a Theiler window (below) when the data is too short for a held-out tail.

## Predict and evaluate

```python
from edmkit.metrics import mae, pearson_correlation, rmse
from edmkit.simplex_projection import simplex_projection

prediction = simplex_projection(library, target, query)

print(f"rho:  {pearson_correlation(prediction, truth):.3f}")
print(f"rmse: {rmse(prediction, truth):.3f}")
print(f"mae:  {mae(prediction, truth):.3f}")
```

Report at least one correlation-like metric (`pearson_correlation`) and one error-scale metric (`rmse` or `mae`). Correlation tells you about shape; error tells you about magnitude.

## Leave-one-out with a Theiler window

For short series, a held-out tail wastes data. Leave-one-out reuses every point as both library and query: query the library with itself, and pass a mask built by `theiler_window` so each query cannot see library points within `(E - 1) * tau` time steps of itself.

```python
from edmkit.theiler_window import theiler_window

library_all = embedded[:-1]
target_all = x[shift + 1 :]    # same length as library_all: every point has a 1-step-ahead target

times = np.arange(len(library_all))
mask = theiler_window(times, times, width=(E - 1) * tau)
prediction_all = simplex_projection(library_all, target_all, library_all, mask=mask)

print(f"LOO rho:  {pearson_correlation(prediction_all, target_all):.3f}")
```

Every query still needs `E + 1` library points outside its window, so a very short series may force a smaller `width` or `E`.

Leave-one-out is the right tool for diagnostics — picking `E` from a short series, comparing `theta`, or a quick baseline. Keep a held-out evaluation when you need a number no parameter choice has seen.

## Check that the system is nonlinear

If S-Map wins only at small `theta`, the system is essentially linear and simpler methods suffice. If skill rises with `theta`, EDM is buying something real.

```python
from edmkit.smap import smap

for theta in [0, 0.1, 0.3, 1, 2, 4, 8]:
    prediction_theta = smap(library, target, query, theta=theta)
    print(f"theta={theta:>4}: rho={pearson_correlation(prediction_theta, truth):.3f}")
```

Expected pattern on a chaotic Lorenz trace: modest `rho(0)`, climbing through `theta=1` to `3`, then plateauing.

## Common pitfalls

- **Misaligned target.** `embedded[i]` corresponds to time `shift + i` in `x`. Off by `(E - 1) * tau` ruins the forecast — always derive the target from `shift`.
- **Train/test overlap.** Splitting after embedding can leave overlapping coordinates at the boundary. Drop the last `(E - 1) * tau` rows of the training library to remove the overlap.
- **Reporting leave-one-out as a generalization score.** Leave-one-out is a diagnostic. The Theiler window prevents trivial leakage but does not match a held-out tail.
- **Forgetting to scale.** Distances are raw Euclidean. Normalize first if coordinates live on different scales (common for multivariate inputs).

## Where to go next

- [S-Map](/edmkit/concepts/smap/) — when the `theta` sweep is interesting on its own.
- [Causality with CCM](/edmkit/guides/causality-with-ccm/) — when you also have a candidate cause.
- [API reference](/edmkit/reference/simplex-projection/) — full signatures and edge cases.
