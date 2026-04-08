---
title: S-Map
description: Locally weighted linear regression for nonlinear time series.
---

S-Map (Sequential Locally Weighted Global Linear Map) is a prediction method that fits a local linear model at each query point, with weights that decay exponentially with distance.

## How It Works

For each query point **q**:

1. Compute distances from **q** to all library points
2. Assign weights: _w_i = exp(-theta _ d_i / d_mean)\*
3. Fit a weighted linear regression (with Tikhonov regularization) to predict the target

The **theta** parameter controls locality:

| theta | Behavior                                          |
| ----- | ------------------------------------------------- |
| 0     | Global linear model (all points weighted equally) |
| Small | Weakly local — smooth transition                  |
| Large | Strongly local — only nearby points matter        |

## Usage

```python
from edmkit.smap import smap

predictions = smap(X, Y, Q, theta=3.0)
```

### Parameters

- **theta**: Locality parameter (>= 0). Start with values in [0, 1, 2, 3, 5, 8].
- **alpha**: Tikhonov regularization strength (default: 1e-10). Increase if the system is ill-conditioned.
- **mask**: Boolean mask to exclude library points.
- **use_tensor**: Use tinygrad backend for GPU acceleration.

### Detecting Nonlinearity

Compare prediction skill across theta values. If performance improves substantially for theta > 0, the system exhibits nonlinear dynamics:

```python
from edmkit.smap import smap
from edmkit.metrics import mean_rho

for theta in [0, 1, 2, 4, 8]:
    preds = smap(lib, target, query, theta=theta)
    rho = mean_rho(preds, observations)
    print(f"theta={theta}: rho={rho:.4f}")
```

### Batched Operations

Like simplex projection, S-Map supports batched inputs with a leading batch dimension:

```python
# X: (B, N, E), Y: (B, N, D), Q: (B, M, E)
predictions = smap(X, Y, Q, theta=3.0)  # -> (B, M, D)
```
