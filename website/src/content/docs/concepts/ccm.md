---
title: Convergent Cross Mapping
description: Testing causal relationships between variables with CCM.
---

Convergent Cross Mapping (CCM) tests whether variable _X_ causally influences variable _Y_ in a coupled dynamical system. It is based on the idea that if _X_ drives _Y_, then the attractor reconstructed from _Y_ will contain information about _X_.

## How It Works

1. Reconstruct the attractor from the **effect** variable _Y_
2. Use nearest neighbors in this attractor to predict the **cause** variable _X_
3. Repeat at increasing library sizes
4. If prediction skill **converges** (improves with more data), _X_ causes _Y_

Convergence is the key signature — it distinguishes true causation from mere correlation.

## Usage

### Using Convenience Wrappers

The simplest way to run CCM:

```python
import numpy as np
from edmkit.ccm import with_simplex_projection

# X, Y: embedded time series
n = len(X)
lib_sizes = np.linspace(10, n, 20, dtype=int)

rho = with_simplex_projection(
    X, Y,
    lib_sizes=lib_sizes,
    library_pool=np.arange(n),
    prediction_pool=np.arange(n),
)
# rho[i] = mean correlation at lib_sizes[i]
```

For S-Map-based CCM:

```python
from edmkit.ccm import with_smap

rho = with_smap(
    X, Y,
    lib_sizes=lib_sizes,
    theta=3.0,
    library_pool=np.arange(n),
    prediction_pool=np.arange(n),
)
```

### Custom Prediction Function

Use the `ccm` function directly with any prediction function:

```python
from edmkit.ccm import ccm

def my_predictor(X, Y, Q, *, mask=None):
    # Custom prediction logic
    ...

rho = ccm(
    X, Y,
    lib_sizes=lib_sizes,
    predict_func=my_predictor,
    library_pool=np.arange(n),
    prediction_pool=np.arange(n),
)
```

### Bootstrap Analysis

Get per-sample correlation values for statistical testing:

```python
from edmkit.ccm import bootstrap

samples = bootstrap(
    X, Y,
    lib_sizes=lib_sizes,
    predict_func=my_predictor,
    library_pool=np.arange(n),
    prediction_pool=np.arange(n),
    n_samples=100,
)
# samples shape: (100, len(lib_sizes))
```

### Reproducibility

Use `make_sample_func` for reproducible random sampling:

```python
from edmkit.ccm import make_sample_func

rho = with_simplex_projection(
    X, Y,
    lib_sizes=lib_sizes,
    library_pool=np.arange(n),
    prediction_pool=np.arange(n),
    sample_func=make_sample_func(seed=42),
)
```
