---
title: Convergent Cross Mapping
description: Testing causal direction between two variables in a coupled dynamical system, based on convergence of cross-mapped predictions.
---

Convergent Cross Mapping (CCM) tests whether one variable causally influences another in a coupled deterministic system. Unlike Granger-style tests, it does not require linearity or independent noise. The idea is geometric: if `X` drives `Y`, then `Y`'s state encodes past values of `X`, and predictions of `X` from `Y`'s reconstructed attractor improve as the library grows. That **convergence** — not absolute skill — is the signature of causality.

## The intuition

When `X` drives `Y`, `Y`'s dynamics are shaped by `X`'s history, so `Y`'s reconstructed attractor contains the information needed to recover `X`. Variables merely correlated through a common driver do not embed each other's history.

The asymmetry is the test:

- `Y` -> `X` cross-mapping skill rises with library size  ⇒  `X` causes `Y`.
- `X` -> `Y` cross-mapping skill rises with library size  ⇒  `Y` causes `X`.
- Both rise  ⇒  bidirectional coupling.
- Neither rises  ⇒  no detectable causal coupling (or insufficient data).

"Cross-mapping from `Y` to `X`" sounds backwards at first. Read it as: use neighbors in the embedded `Y`-attractor to predict the corresponding values of `X`.

## The procedure

1. Embed the candidate effect `Y` with `embed` to get `Y_E`.
2. Time-align the candidate cause `X` so each row of `Y_E` pairs with the matching `X`.
3. For each library size `L`:
   - Sample `L` indices from a library pool; the corresponding `Y_E` rows form the library.
   - Predict `X` at every index in the prediction pool by simplex projection (or S-Map).
   - Score the prediction against the actual `X` values.
4. Repeat each `L` with several random subsamples; take the mean (or a robust aggregate).

A library/prediction pool split with a handful of bootstrap subsamples is usually enough; full leave-one-out is not required.

## Convergence is the signature

Plot mean correlation against library size on a log axis. The shape carries the diagnostic.

| Curve shape | Reading |
| --- | --- |
| Monotone increase that flattens | Convergence — the candidate cause is supported |
| Flat near zero at all library sizes | No detectable cross-mapping |
| High but flat across library sizes | Skill from confounders or shared seasonality — not causation |
| Curve rising then falling | Probably non-stationarity or a poor embedding for the effect |

Raw skill alone is not informative: confounded or correlated variables can both produce high but flat skill. Convergence is what separates them.

## Using `edmkit.ccm`

`bootstrap` samples libraries and yields `(index, samples)` tasks. Calling `samples()` returns the predictions of a batch of sampled libraries, and `index` is where that batch belongs in a `(len(lib_sizes), n_samples)` array of scores.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection

# Test "x causes y": embed y, align x, then cross-map.
tau, E = 1, 3
coordinates = np.array([[0, tau * j] for j in range(E)])
y_embedded, _ = embed(coordinates, y)
x_aligned = x[tau * (E - 1) :]

mid = len(y_embedded) // 2
lib_sizes = np.geomspace(10, mid, num=10).astype(int)
n_samples = 100
actual = x_aligned[mid:, None]


def score(samples):
    predictions = samples()  # (B, M, 1)
    return pearson_correlation(predictions, np.broadcast_to(actual, predictions.shape))


scores = np.empty((len(lib_sizes), n_samples))
tasks = bootstrap(
    y_embedded[:mid],
    x_aligned[:mid],
    y_embedded[mid:],
    lib_sizes,
    predict_func=simplex_projection,
    n_samples=n_samples,
    replace=False,
)
indices, samples = zip(*tasks)
with ThreadPoolExecutor() as executor:
    for index, result in zip(indices, executor.map(score, samples)):
        scores[index] = result

rho = scores.mean(axis=1)
```

Plot `rho` against `lib_sizes` to read convergence.

## Sampling and temporal exclusion

The example reserves disjoint halves for the library and the queries. When they overlap, exclude self-points and temporally close neighbors with a mask:

```python
from edmkit.theiler_window import theiler_window

times = np.arange(len(y_embedded))
mask = theiler_window(times, times, width=(E - 1) * tau)
tasks = bootstrap(
    y_embedded,
    x_aligned,
    y_embedded,
    lib_sizes,
    predict_func=simplex_projection,
    n_samples=n_samples,
    replace=False,
    mask=mask,
)
```

The mask refers to the original library, and each task selects the columns of its sampled points. If a sampled library has too few valid neighbors, the task raises when it is executed.

`replace=True` (the default) draws with replacement; `replace=False` draws random subsets. A fixed `seed` gives the same libraries regardless of `batch_size` or execution order.

## Bootstrap distributions

The scores support any aggregation:

```python
median = np.median(scores, axis=1)
low, high = np.quantile(scores, [0.05, 0.95], axis=1)
```

These bands describe variability across sampled libraries. They are not, by themselves, confidence intervals for a causal effect.

## Things to watch for

- **Direction.** "Test `X` causes `Y`" means embed `Y` and predict `X` from it. In `bootstrap`, `X` is the embedded effect and `Y` is the candidate cause.
- **Embedding quality.** CCM stands or falls on the effect's embedding. Choose `(E, tau)` for the effect first — see [Choosing E and tau](/edmkit/guides/choosing-parameters/).
- **Synchronous coupling.** With strong delay-zero coupling, both directions converge and CCM cannot disambiguate. Time-shifted CCM helps; build it by shifting the target before calling `bootstrap`.
- **Shared external forcing.** A common driver `Z` can make both directions converge without direct coupling. Surrogate tests (shuffled or seasonality-matched) are the usual safeguard.
- **Sample size.** Small library sizes produce noisy correlations that swing across subsamples. Use a log grid from tens up to the pool size, and 50–200 bootstrap samples for stable means.

See the [causality guide](/edmkit/guides/causality-with-ccm/) for a full walk-through.
