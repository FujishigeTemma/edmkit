---
title: S-Map
description: Locally weighted linear regression in reconstructed state space. The standard EDM tool for quantifying nonlinearity.
---

S-Map (Sequential Locally Weighted Global Linear Map) replaces simplex projection's weighted average with a distance-weighted linear regression. A scalar `theta` controls how sharply weights drop with distance, interpolating between a global linear model (`theta = 0`) and a strongly local one (large `theta`). Sweeping `theta` is the standard EDM diagnostic for nonlinearity.

## The model

For each query `q`, S-Map fits a linear regression on the full library, weighting each point by

```
w_i = exp(-theta * d_i / d_mean)
```

where `d_i` is the Euclidean distance from `q` to `x_i` and `d_mean` is the average library distance to `q`. The regression solves

```
beta_q = argmin_beta  sum_i  w_i * ( y_i - beta^T * [1, x_i] )^2  +  alpha * ||beta||^2
```

with a small Tikhonov `alpha` for stability, and predicts `y_hat = beta_q^T * [1, q]`. The intercept is not regularized.

S-Map fits a different linear model around every query; `theta` decides how local "around" is.

## Reading the `theta` knob

| `theta` | What it means | When it predicts well |
| --- | --- | --- |
| `0` | All library points share one global linear fit | The system is genuinely linear |
| Small (0–1) | Smoothly varying local linear models | The system is mildly nonlinear |
| Large (5+) | Sharply local; only nearby points carry weight | The system is strongly nonlinear with state-dependent dynamics |

If the system is linear, raising `theta` cannot improve skill. If it is nonlinear, skill rises and plateaus around the `theta` that captures the dynamics' locality.

## Detecting nonlinearity

Evaluate the same forecast at several `theta` values and compare `rho(theta)`.

```python
from edmkit.metrics import mean_rho
from edmkit.smap import smap

for theta in [0, 0.1, 0.3, 1, 2, 4, 8]:
    prediction = smap(library, target, query, theta=theta)
    print(f"theta={theta:>4}: rho={mean_rho(prediction, truth):.3f}")
```

If `rho(0)` is already best, the data look linear. If `rho` keeps climbing with `theta`, the system has nonlinear dynamics and local methods carry real predictive content.

## Using `smap`

```python
from edmkit.smap import smap

# X: (N, E) library embeddings
# Y: (N,) or (N, D) library targets
# Q: (M, E) query embeddings
predictions = smap(X, Y, Q, theta=3.0)   # (M,) or (M, D)
```

Shapes match `simplex_projection` — 2-D and batched 3-D layouts both work, and `mask` hides library points from the regression. Two more parameters:

- **`alpha`** controls the Tikhonov ridge. The default `1e-10` stabilizes numerics; raise it for ill-conditioned cases (small libraries or redundant coordinates).
- **`use_tensor`** routes through tinygrad. Rarely faster than NumPy today; mainly for GPU experiments.

## S-Map vs simplex projection

| Aspect | Simplex projection | S-Map |
| --- | --- | --- |
| Local model | Distance-weighted mean of `E + 1` neighbors | Distance-weighted linear regression on all `N` library points |
| Free parameter | None | `theta` (locality) |
| Cost per query | `O(log N)` for KDTree + `O(E)` for the weighted mean | `O(N * E^2)` for the weighted regression |
| Primary use | Forecasting and choosing `E` | Quantifying nonlinearity, recovering local Jacobians |

Simplex is faster and usually the right baseline. Pick S-Map when you need the `theta` sweep, or when small-library prediction benefits from the regression's interpolation.

## Things to watch for

- **Ill-conditioned regression.** When embedded coordinates are nearly redundant (e.g. `tau` too small), `X^T W X` becomes singular and the coefficients blow up. The Tikhonov term mitigates; revisiting `tau` is the real fix.
- **Outliers.** Distant library points still contribute with small but nonzero weight. For heavy-tailed targets, trim or pre-process before calling `smap`.
- **Cross-validation cost.** S-Map is `O(N)` per query, so sweeping `theta` across folds costs more than the simplex equivalent. Start with a coarse grid, then refine around the peak.
