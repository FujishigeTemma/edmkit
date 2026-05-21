---
title: Getting Started
description: Install edmkit, run a forecast on a chaotic test system, and test for causality between two coupled signals.
---

Install edmkit, forecast a chaotic series, and test a causal hypothesis between two coupled signals. The [Concepts](/edmkit/concepts/edm/) section explains each step in depth.

## Install

```bash
pip install edmkit
```

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv add edmkit
```

:::note
Requires Python 3.13+. Core dependencies: NumPy 2.4+, SciPy 1.17+, and Rust-backed `kdtree-rs`. `tinygrad` is optional for the GPU backend.
:::

## Forecast a chaotic time series

The Lorenz system is deterministic but chaotic — hard for parametric models, ideal for EDM. The example observes only its `x` component and predicts one step ahead from the reconstructed dynamics.

```python
import numpy as np
from edmkit.embedding import lagged_embed, scan, select
from edmkit.generate import lorenz
from edmkit.metrics import mean_rho
from edmkit.simplex_projection import simplex_projection

# 1. Generate a Lorenz trajectory and keep only the x component.
_, trajectory = lorenz(
    sigma=10, rho=28, beta=8 / 3,
    X0=np.array([1.0, 1.0, 1.0]), dt=0.01, t_max=50,
)
x = trajectory[:, 0]

# 2. Pick (E, tau) by cross-validated grid search.
E_grid = list(range(1, 11))
tau_grid = [1, 2, 3, 5]
scores = scan(x, E=E_grid, tau=tau_grid)
E, tau, cv_rho = select(scores, E=E_grid, tau=tau_grid)
print(f"Selected E={E}, tau={tau} (CV rho={cv_rho:.3f})")

# 3. Embed once with the chosen parameters.
embedded = lagged_embed(x, tau=tau, e=E)
shift = tau * (E - 1)               # embedded[i] corresponds to time x[shift + i]
n_pairs = len(embedded) - 1         # one less because we need a 1-step-ahead target

# 4. Split into library (training) and query (test) sets.
half = n_pairs // 2
library = embedded[:half]
target = x[shift + 1 : shift + half + 1]
query = embedded[half:n_pairs]
truth = x[shift + half + 1 : shift + n_pairs + 1]

# 5. Predict and evaluate.
prediction = simplex_projection(library, target, query)
print(f"Test rho: {mean_rho(prediction, truth):.3f}")
```

Expect a held-out correlation above 0.9 at this series length. The same pipeline works for any scalar series — see the [forecasting guide](/edmkit/guides/forecasting/) for cross-validated variants.

## Test a causal hypothesis

CCM predicts a candidate *cause* from the *effect's* reconstructed attractor. If skill improves as the library grows ("convergence"), the cause-to-effect direction is supported.

The example couples two logistic maps so `x` drives `y` but not the reverse, then asks CCM to recover that asymmetry.

```python
import numpy as np
from edmkit.ccm import with_simplex_projection
from edmkit.embedding import lagged_embed

# 1. Build coupled logistic maps. x drives y with coupling 0.02; the reverse is zero.
N, rx, ry, beta = 1000, 3.8, 3.5, 0.02
x = np.zeros(N)
y = np.zeros(N)
x[0], y[0] = 0.4, 0.2
for i in range(1, N):
    x[i] = x[i - 1] * (rx - rx * x[i - 1])
    y[i] = y[i - 1] * (ry - ry * y[i - 1]) + beta * x[i - 1]

# 2. To test "x causes y", cross-map from y's attractor to x.
tau, E = 1, 2
y_embedded = lagged_embed(y, tau=tau, e=E)
x_aligned = x[tau * (E - 1):]

# 3. Reserve disjoint pools so the library cannot trivially memorize the queries.
mid = y_embedded.shape[0] // 2
library_pool = np.arange(mid)
prediction_pool = np.arange(mid, y_embedded.shape[0])
lib_sizes = np.logspace(np.log10(10), np.log10(library_pool[-1]), num=8, dtype=int)

# 4. Sweep library sizes and read the convergence pattern.
rho_x_drives_y = with_simplex_projection(
    y_embedded, x_aligned,
    lib_sizes=lib_sizes,
    library_pool=library_pool,
    prediction_pool=prediction_pool,
)
print("x -> y skill:", rho_x_drives_y.round(3))
```

`rho` should rise with library size and then saturate. That convergence is the signature of `x -> y` coupling. See [Convergent Cross Mapping](/edmkit/concepts/ccm/) for the theory and the [causality guide](/edmkit/guides/causality-with-ccm/) for interpretation.

## Next steps

- [What is EDM?](/edmkit/concepts/edm/) — the intuition and where the framework fits.
- [Time-delay embedding](/edmkit/concepts/embedding/) — the reconstruction step every algorithm relies on.
- [Forecasting a time series](/edmkit/guides/forecasting/) — a full pipeline with cross-validation.
- [Testing causality with CCM](/edmkit/guides/causality-with-ccm/) — reading convergence and avoiding pitfalls.
