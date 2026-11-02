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
from edmkit.embedding import embed
from edmkit.generate import lorenz, rk45
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection

# 1. Generate a Lorenz trajectory and keep only the x component.
_, trajectory = rk45(
    lorenz(sigma=10, rho=28, beta=8 / 3),
    X0=np.array([1.0, 1.0, 1.0]),
    dt=0.01,
    t_max=50,
)
x = trajectory[:, 0]

# 2. Fix the delay coordinates. Pick them by held-out skill for your own data —
#    see the parameter selection guide.
E, tau = 3, 2

# 3. Embed once with the chosen parameters.
coordinates = np.array([[0, tau * j] for j in range(E)])
embedded, _ = embed(coordinates, x)
shift = tau * (E - 1)  # embedded[i] corresponds to time x[shift + i]
n_pairs = len(embedded) - 1  # one less because we need a 1-step-ahead target

# 4. Split into library (training) and query (test) sets.
half = n_pairs // 2
library = embedded[:half]
target = x[shift + 1 : shift + half + 1]
query = embedded[half:n_pairs]
truth = x[shift + half + 1 : shift + n_pairs + 1]

# 5. Predict and evaluate.
prediction = simplex_projection(library, target, query)
print(f"Test rho: {pearson_correlation(prediction, truth):.3f}")
```

Expect a held-out correlation above 0.9 at this series length. The same pipeline works for any scalar series — see the [forecasting guide](/edmkit/guides/forecasting/) for cross-validated variants.

## Test a causal hypothesis

CCM predicts a candidate *cause* from the *effect's* reconstructed attractor. If skill improves as the library grows ("convergence"), the cause-to-effect direction is supported.

The example couples two logistic maps so `x` drives `y` but not the reverse, then asks CCM to recover that asymmetry.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection

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
coordinates = np.array([[0, tau * j] for j in range(E)])
y_embedded, _ = embed(coordinates, y)
x_aligned = x[tau * (E - 1) :]

# 3. Reserve disjoint library and query sets.
mid = len(y_embedded) // 2
lib_sizes = np.geomspace(10, mid, num=8).astype(int)
n_samples = 100
actual = x_aligned[mid:, None]


# 4. Score each batch of predictions in a thread pool.
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

print("x -> y skill:", scores.mean(axis=1).round(3))
```

`rho` should rise with library size and then saturate. That convergence is the signature of `x -> y` coupling. See [Convergent Cross Mapping](/edmkit/concepts/ccm/) for the theory and the [causality guide](/edmkit/guides/causality-with-ccm/) for interpretation.

## Next steps

- [What is EDM?](/edmkit/concepts/edm/) — the intuition and where the framework fits.
- [Time-delay embedding](/edmkit/concepts/embedding/) — the reconstruction step every algorithm relies on.
- [Forecasting a time series](/edmkit/guides/forecasting/) — a full pipeline with cross-validation.
- [Testing causality with CCM](/edmkit/guides/causality-with-ccm/) — reading convergence and avoiding pitfalls.
