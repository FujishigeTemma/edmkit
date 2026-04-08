---
title: Getting Started
description: Install edmkit and run your first EDM analysis.
---

## Installation

Install from PyPI:

```bash
pip install edmkit
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add edmkit
```

:::note
edmkit requires Python 3.13 or later.
:::

## Requirements

edmkit depends on:

- **NumPy** >= 2.4 — core array operations
- **SciPy** >= 1.17 — KDTree for nearest-neighbor search
- **usearch** >= 2.23 — fast approximate nearest-neighbor search for high-dimensional data
- **tinygrad** >= 0.11 — optional GPU acceleration

## Your First Analysis

Here is a minimal example that generates a chaotic time series, finds optimal embedding parameters, and makes predictions:

```python
import numpy as np
from edmkit.generate import lorenz
from edmkit.embedding import lagged_embed, scan, select
from edmkit.simplex_projection import simplex_projection

# 1. Generate a Lorenz attractor time series
t, X = lorenz(sigma=10, rho=28, beta=8/3,
              X0=np.array([1.0, 1.0, 1.0]), dt=0.01, t_max=50)
x = X[:, 0]

# 2. Search for best embedding dimension (E) and time delay (tau)
scores = scan(x, None, E=list(range(1, 11)), tau=[1, 2, 3, 5])
best_E, best_tau, best_score = select(
    scores, E=list(range(1, 11)), tau=[1, 2, 3, 5]
)
print(f"Best E={best_E}, tau={best_tau}, score={best_score:.4f}")

# 3. Embed the time series
embedded = lagged_embed(x, tau=best_tau, e=best_E)

# 4. Split into library and query sets
n = len(embedded)
half = n // 2
offset = best_tau * (best_E - 1)

lib = embedded[:half]
target = x[offset + 1 : half + offset + 1]
query = embedded[half:]

# 5. Predict using simplex projection
predictions = simplex_projection(lib, target, query)
```

## What's Next?

- Learn the theory behind EDM in the [Concepts](/concepts/edm/) section
- Browse the full [API Reference](/reference/embedding/)
