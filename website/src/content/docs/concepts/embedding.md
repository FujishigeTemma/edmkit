---
title: Embedding
description: Time-delay embedding and parameter selection in edmkit.
---

Time-delay embedding is the foundation of EDM. It transforms a scalar time series into a multidimensional representation that reconstructs the dynamics of the underlying system.

## Lagged Embedding

Given a time series _x(t)_, the embedding vector at time _t_ is:

```
[x(t), x(t - tau), x(t - 2*tau), ..., x(t - (E-1)*tau)]
```

where:

- **E** (embedding dimension) — the number of lagged coordinates
- **tau** (time delay) — the spacing between lags

```python
from edmkit.embedding import lagged_embed

embedded = lagged_embed(x, tau=2, e=3)
# Each row is [x(t), x(t-2), x(t-4)]
```

## Choosing E and tau

The `scan` function performs a grid search over candidate values of _E_ and _tau_, evaluating prediction skill using cross-validation:

```python
from edmkit.embedding import scan, select

scores = scan(
    x, None,
    E=list(range(1, 11)),
    tau=[1, 2, 3, 5],
)

best_E, best_tau, best_score = select(scores, E=list(range(1, 11)), tau=[1, 2, 3, 5])
```

`scan` returns a 3D array of shape `(len(E), len(tau), K)` where _K_ is the maximum number of cross-validation folds. `select` picks the combination that maximizes _mean - SE_ (standard error), which favors stable performance across folds.

### Customization

`scan` accepts pluggable components:

- **split**: Cross-validation strategy (default: `sliding_folds`)
- **predict**: Prediction function (default: `simplex_projection`)
- **metric**: Evaluation metric (default: `mean_rho`, i.e., Pearson correlation)
- **n_ahead**: Prediction horizon (default: 1 step)
