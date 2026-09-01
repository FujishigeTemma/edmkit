---
title: Choosing E and tau
description: A practical recipe for picking embedding dimension and delay with cross-validated simplex projection.
---

This guide picks `E` and `tau` for an unknown series using cross-validated simplex projection — the standard EDM approach. Read [Time-delay embedding](/edmkit/concepts/embedding/) and [Simplex projection](/edmkit/concepts/simplex-projection/) first if needed.

## The minimal recipe

`scan` evaluates simplex projection over a grid of `(E, tau)` with rolling cross-validation; `select` picks the most reliable cell.

```python
import numpy as np
from edmkit.embedding import scan, select

E_grid = list(range(1, 11))
tau_grid = [1, 2, 3, 5, 8]

scores = scan(x, E=E_grid, tau=tau_grid)   # shape (len(E), len(tau), K)
E, tau, rho = select(scores, E=E_grid, tau=tau_grid)
print(f"Selected E={E}, tau={tau}, mean rho={rho:.3f}")
```

`scores[i, j, k]` is the held-out Pearson correlation on fold `k` for `E = E_grid[i]`, `tau = tau_grid[j]`. NaN means the fold could not run — usually because `E` was too large for the available history. `select` ranks cells by `mean(rho) - SE(rho)`, penalizing unstable cells.

## Interpreting the scan output

Inspect beyond the "best" cell to catch surprises.

```python
import numpy as np

mean_rho = np.nanmean(scores, axis=2)   # (len(E), len(tau))
print(np.round(mean_rho, 3))
```

Patterns to look for:

| Pattern | Meaning | Action |
| --- | --- | --- |
| Sharp peak at small `E`, falling for large `E` | Low-dimensional, clean signal | Take the peak. |
| Plateau at moderate `E`, flat across `tau` | Signal near noise, or `tau` outside useful range | Widen `tau_grid` or check data quality. |
| Highest `rho` at the largest `E` | Grid did not reach true dimension | Extend `E_grid` upward. |
| NaN cells at large `E` and `tau` | Not enough samples after embedding | Shrink the grid or get more data. |

Sweep `tau` from `1` up to roughly the autocorrelation time of `x`. `edmkit.util.autocorrelation` finds that range.

```python
from edmkit.util import autocorrelation

ac = autocorrelation(x, max_lag=50)
candidate_tau = int(np.argmax(ac < 1 / np.e))   # first lag with autocorr < 1/e
```

Use `candidate_tau` to size `tau_grid`.

## Customizing the cross-validation

`scan` has three pluggable components:

| Parameter | Default | Replace when |
| --- | --- | --- |
| `split` | `sliding_folds`, `N/5` train, `N/10` validation | Trend or regime shifts — use `expanding_folds` to grow the training set. |
| `predict` | `simplex_projection` | Picking parameters for an S-Map workflow — pass `partial(smap, theta=...)`. |
| `metric` | `pearson_correlation` | Multidimensional target or you prefer absolute error — pass a sign-flipped `rmse` or `mae`. |

Switching to expanding-window CV with RMSE for a noisy non-stationary series:

```python
from functools import partial

from edmkit.embedding import scan, select
from edmkit.metrics import rmse
from edmkit.splits import expanding_folds

split = partial(
    expanding_folds,
    initial_train_size=len(x) // 5,
    validation_size=len(x) // 20,
    stride=len(x) // 20,
)

scores = scan(
    x,
    E=list(range(1, 11)),
    tau=[1, 2, 4, 8],
    split=split,
    metric=lambda prediction, truth: -rmse(prediction, truth),
)
```

`select` always maximizes, which is why the error metric is sign-flipped: smaller RMSE must score higher.

## When the chosen `(E, tau)` looks suspicious

- **Embed and look.** Plot the first two coordinates of `lagged_embed(x, tau=tau, e=2)`. A low-dimensional system shows structure (loop, butterfly, sheet), not a featureless cloud or diagonal.
- **Run leave-one-out with a Theiler window.** If held-out correlation drops sharply once temporal neighbors are excluded, the original CV was leaking through overlapping embeddings. Query the library with itself on a one-step-ahead target, masking library points within `(E - 1) * tau` time steps of each query.

```python
import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

embedded = lagged_embed(x, tau=tau, e=E)
shift = (E - 1) * tau

library = embedded[:-1]
target = x[shift + 1 :]    # same length as library: every point has a 1-step-ahead target

times = np.arange(len(library))
mask = theiler_window(times, times, width=(E - 1) * tau)
prediction = simplex_projection(library, target, library, mask=mask)
print(pearson_correlation(prediction, target))
```

- **Permuted baseline.** Shuffle `x` and re-run `scan`. The best `rho` on the shuffle is the noise floor; your real choice should beat it by a clear margin.

## Next steps

With `(E, tau)` in hand:

- Forecasting — see the [forecasting guide](/edmkit/guides/forecasting/).
- Nonlinearity — sweep S-Map's `theta` as in [S-Map](/edmkit/concepts/smap/).
- Causal direction — see the [causality guide](/edmkit/guides/causality-with-ccm/).
