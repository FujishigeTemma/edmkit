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

1. Embed the candidate effect `Y` with `lagged_embed` to get `Y_E`.
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

The module offers two wrappers (`with_simplex_projection`, `with_smap`) and a general entry point `ccm` that accepts any `(X, Y, Q) -> predictions` function.

```python
import numpy as np
from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed

# Test "x causes y": embed y, align x, then cross-map.
tau, E = 1, 3
y_embedded = lagged_embed(y, tau=tau, e=E)
x_aligned = x[tau * (E - 1):]

n_states = y_embedded.shape[0]
library_pool = np.arange(n_states // 2)
prediction_pool = np.arange(n_states // 2, n_states)
lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]),
                        num=10, dtype=int)

rho = with_simplex_projection(
    y_embedded, x_aligned,
    lib_sizes=lib_sizes,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
```

`rho` has shape `(len(lib_sizes),)` — one mean per library size. Plot on a log-x axis to read convergence.

## `library_pool` and `prediction_pool`

These decide which indices can be sampled into the library and which the function predicts.

- **Disjoint halves.** First half library, second half prediction. Prevents the library from memorizing its own predictions.
- **Full pool.** Both pools cover the full range. Combined with random subsampling and the Theiler effect, this is closer to the original Sugihara et al. formulation. Use it when the data is short.

`sample_func` controls the random sampling. Pass `make_sample_func(seed=42)` for reproducibility:

```python
from edmkit.ccm import make_sample_func, with_simplex_projection

rho = with_simplex_projection(
    y_embedded, x_aligned,
    lib_sizes=lib_sizes,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
    sample_func=make_sample_func(seed=42),
)
```

## Bootstrap distributions

For a confidence band instead of a single mean, switch to `bootstrap`. It returns raw per-sample correlations.

```python
from edmkit.ccm import bootstrap
from edmkit.simplex_projection import simplex_projection

samples = bootstrap(
    y_embedded, x_aligned,
    lib_sizes=lib_sizes,
    predict_func=simplex_projection,
    n_samples=100,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
# samples.shape == (100, len(lib_sizes))
```

Plot the median with a percentile band against `lib_sizes` to show uncertainty alongside convergence.

## Things to watch for

- **Direction.** "Test `X` causes `Y`" means embed `Y` and predict `X` from it. In `with_simplex_projection`, `X` is the embedded effect and `Y` is the candidate cause.
- **Embedding quality.** CCM stands or falls on the effect's embedding. Run `scan`/`select` on the effect first.
- **Synchronous coupling.** With strong delay-zero coupling, both directions converge and CCM cannot disambiguate. Time-shifted CCM helps; build it by shifting the target before calling `ccm`.
- **Shared external forcing.** A common driver `Z` can make both directions converge without direct coupling. Surrogate tests (shuffled or seasonality-matched) are the usual safeguard.
- **Sample size.** Small library sizes produce noisy correlations that swing across subsamples. Use a log grid from tens up to the pool size, and 50–200 bootstrap samples for stable means.

See the [causality guide](/edmkit/guides/causality-with-ccm/) for a full walk-through.
