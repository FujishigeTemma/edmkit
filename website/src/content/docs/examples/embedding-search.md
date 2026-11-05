---
title: Embedding search
description: Choose the embedding dimension and delay of a single variable by prediction skill.
---

Choose `E` and `tau` by how well the delay vectors `(x[t], x[t - tau], ..., x[t - (E - 1) tau])` predict `x[t + Tp]`. Here every pair is scored on `x` of the Lorenz system. The search splits its data into a library and queries, and a separate test period, never seen during the search, evaluates the chosen pair.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit import generate
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.splits import temporal_fold
from edmkit.util import autocorrelation

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 60.0)
x = trajectory[500:, 0]
Tp = 20  # see Choosing Tp
acf_time = int(np.argmax(autocorrelation(x, max_lag=100) < 1 / np.e))  # 15 steps

# Search on the first part, and keep the rest for the final evaluation.
# Each part is embedded on its own, so the gap only needs to cover autocorrelation.
search, test = temporal_fold(len(x), 0.75, gap=acf_time)


def delay_vectors(x, E, tau):
    """Delay vectors of x and the values Tp steps ahead."""
    coordinates = np.array([[0, tau * j] for j in range(E)])
    X, t = embed(coordinates, x)
    X, t = X[:-Tp], t[:-Tp]
    return X, x[t + Tp]


def score(E, tau):
    X, Y = delay_vectors(x[search], E, tau)
    # Keep library and query rows from sharing an observation or being autocorrelated.
    library, query = temporal_fold(len(X), 0.5, gap=max((E - 1) * tau + Tp, acf_time))
    return pearson_correlation(simplex_projection(X[library], Y[library], X[query]), Y[query])


Es, taus = np.arange(1, 8), np.arange(1, 11)
E_grid, tau_grid = np.meshgrid(Es, taus, indexing="ij")
with ThreadPoolExecutor() as executor:
    rho = np.reshape(list(executor.map(score, E_grid.flat, tau_grid.flat)), E_grid.shape)

for E, row in zip(Es, rho):
    print(f"E={E}  best tau={taus[row.argmax()]}  rho={row.max():.3f}")

# Evaluate the best pair once on the held-out test period.
a, b = np.unravel_index(rho.argmax(), rho.shape)
X, Y = delay_vectors(x[search], Es[a], taus[b])
X_test, Y_test = delay_vectors(x[test], Es[a], taus[b])
print(f"test: E={Es[a]}  tau={taus[b]}  rho={pearson_correlation(simplex_projection(X, Y, X_test), Y_test):.3f}")
```

```text
E=1  best tau=1  rho=0.570
E=2  best tau=5  rho=0.982
E=3  best tau=3  rho=0.983
E=4  best tau=3  rho=0.981
E=5  best tau=2  rho=0.981
E=6  best tau=2  rho=0.979
E=7  best tau=1  rho=0.977
test: E=3  tau=3  rho=0.996
```

Skill rises with `E` and then levels off, so a larger `E` adds nothing. The best `tau` shrinks as `E` grows, so search the two together. The search scores picked the pair, so only the test score measures how well it predicts new data.

## Choosing `Tp`

`Tp` sets the target `Y = x[t + Tp]`. Its value depends on the goal.

Goal | `Tp`
---- | ----
Forecast | The horizon you need.
Choose `E` and `tau` | The shortest `Tp` at which skill clearly beats the baseline below. Check that nearby `Tp` give the same choice.
Test nonlinearity with S-Map `theta` | The same `Tp` used to choose `E`.
Measure how far ahead the system is predictable | Sweep `Tp` and watch skill decay.
CCM | `0` by default. Sweep negative and positive values to check direction, as described in [Pitfalls](../../pitfalls/#ccm).

`Tp` counts samples, not time. With dense sampling, predicting no change already scores near 1 at small `Tp`, so every embedding looks good. The autocorrelation `acf[Tp]` is the score of that baseline:

```python
import numpy as np

from edmkit import generate
from edmkit.util import autocorrelation

_, trajectory = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.02, 60.0)
x = trajectory[500:, 0]

acf = autocorrelation(x, max_lag=50)  # acf[Tp] is the baseline for Tp
print(acf[[5, 20]].round(3))  # [0.864 0.222]
```

At `Tp = 5`, every `E >= 2` reaches 0.998 and the search cannot tell them apart. At `Tp = 20`, the baseline drops to 0.22 while the best embedding scores 0.983.
