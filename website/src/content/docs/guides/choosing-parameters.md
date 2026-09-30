---
title: Choosing E and tau
description: A practical recipe for picking embedding dimension and delay with cross-validated simplex projection.
---

This guide picks `E` and `tau` for an unknown series using cross-validated simplex projection — the standard EDM approach. Read [Time-delay embedding](/edmkit/concepts/embedding/) and [Simplex projection](/edmkit/concepts/simplex-projection/) first if needed.

## The minimal recipe

Score each `(E, tau)` by held-out one-step-ahead skill and take the best cell. `embed` builds the delay vectors, `sliding_folds` supplies the splits, and `simplex_projection` makes the forecast.

```python
from functools import partial

import numpy as np
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import sliding_folds


def cv_score(x, E, tau, folds):
    coordinates = np.array([[0, tau * j] for j in range(E)])
    embedded, times = embed(coordinates, x)
    embedded, times = embedded[:-1], times[:-1]  # keep rows whose one-step-ahead target exists
    target = x[times + 1]
    return np.mean(
        [
            pearson_correlation(
                simplex_projection(embedded[fold.train], target[fold.train], embedded[fold.validation]),
                target[fold.validation],
            )
            for fold in folds(len(times))
        ]
    )


folds = partial(sliding_folds, train_size=len(x) // 5, validation_size=len(x) // 10)
scores = {(E, tau): cv_score(x, E, tau, folds) for E in range(1, 11) for tau in (1, 2, 3, 5, 8)}

E, tau = max(scores, key=scores.get)
print(f"Selected E={E}, tau={tau} (CV rho={scores[(E, tau)]:.3f})")
```

Every part of the recipe is a parameter of the loop: swap `sliding_folds` for `expanding_folds` when the series trends, `simplex_projection` for `partial(smap, theta=...)` when tuning an S-Map workflow, and `pearson_correlation` for `rmse` or `mae` when magnitudes matter — then minimize instead of maximize.

## Reading the grid

Inspect beyond the "best" cell to catch surprises.

```python
table = np.array([[scores[(E, tau)] for tau in (1, 2, 3, 5, 8)] for E in range(1, 11)])
print(np.round(table, 3))
```

| Pattern | Meaning | Action |
| --- | --- | --- |
| Sharp peak at small `E`, falling for large `E` | Low-dimensional, clean signal | Take the peak. |
| Plateau at moderate `E`, flat across `tau` | Signal near noise, or `tau` outside useful range | Widen the `tau` range or check data quality. |
| Highest `rho` at the largest `E` | Grid did not reach true dimension | Extend the `E` range upward. |
| Skill near 1 at every cell, including `E = 1` | Consecutive samples are nearly identical | Sample more coarsely, or add a Theiler window (below). |

Sweep `tau` from `1` up to roughly the autocorrelation time of `x`. `edmkit.util.autocorrelation` finds that range.

```python
from edmkit.util import autocorrelation

ac = autocorrelation(x, max_lag=50)
candidate_tau = int(np.argmax(ac < 1 / np.e))  # first lag with autocorr < 1/e
```

## When the chosen `(E, tau)` looks suspicious

- **Embed and look.** Plot the first two coordinates of `embed(np.array([[0, 0], [0, tau]]), x)[0]`. A low-dimensional system shows structure (loop, butterfly, sheet), not a featureless cloud or diagonal.
- **Run leave-one-out with a Theiler window.** If held-out correlation drops sharply once temporal neighbors are excluded, the original CV was leaking through overlapping embeddings. Query the library with itself on a one-step-ahead target, masking library points within `(E - 1) * tau` time steps of each query.

```python
import numpy as np

from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.theiler_window import theiler_window

coordinates = np.array([[0, tau * j] for j in range(E)])
embedded, _ = embed(coordinates, x)
shift = (E - 1) * tau

library = embedded[:-1]
target = x[shift + 1 :]  # same length as library: every point has a 1-step-ahead target

times = np.arange(len(library))
mask = theiler_window(times, times, width=(E - 1) * tau)
prediction = simplex_projection(library, target, library, mask=mask)
print(pearson_correlation(prediction, target))
```

- **Permuted baseline.** Shuffle `x` and re-run the grid. The best `rho` on the shuffle is the noise floor; your real choice should beat it by a clear margin.

## Beyond a uniform grid

`(E, tau)` is the classical special case of a coordinate set: one variable at evenly spaced lags. `embed` accepts any list of `(variable, lag)` pairs, so a delay vector may mix variables and irregular lags. Searching that larger space — forward selection, beam search, cross-validated scoring of candidate coordinate sets — is what [edmkit-search](https://github.com/FujishigeTemma/edmkit-search) is for.

## Next steps

With `(E, tau)` in hand:

- Forecasting — see the [forecasting guide](/edmkit/guides/forecasting/).
- Nonlinearity — sweep S-Map's `theta` as in [S-Map](/edmkit/concepts/smap/).
- Causal direction — see the [causality guide](/edmkit/guides/causality-with-ccm/).
