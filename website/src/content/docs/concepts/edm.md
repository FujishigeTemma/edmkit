---
title: What is EDM?
description: A non-parametric framework for analyzing nonlinear time series. The intuition, the workflow, and where it fits.
---

Empirical Dynamic Modeling (EDM) is a non-parametric approach for nonlinear time series. Instead of assuming a parametric form like ARIMA or VAR, EDM reconstructs the system's state space from observed data and performs forecasting and causal inference geometrically inside that space.

Background in dynamical systems is not required; comfort with NumPy is.

## The core insight

If you observe a single variable `x(t)` from a deterministic system long enough, its unobserved state is encoded in the recent history of `x`. **Takens' embedding theorem** makes this precise: under mild conditions, the map

```
t  -->  ( x(t),  x(t - tau),  x(t - 2*tau),  ...,  x(t - (E - 1)*tau) )
```

preserves the geometry of the true attractor inside `R^E`. Lagged copies of one observed variable recover the underlying dynamics up to a smooth deformation.

Two practical consequences follow.

- **States close in lagged coordinates evolve similarly into the future.** This justifies nearest-neighbor forecasting in the reconstructed space — the building block of every EDM algorithm.
- **The reconstructed attractor of an effect carries information about its drivers.** This is the basis of Convergent Cross Mapping.

EDM never asks you to guess the equations. It reuses the geometry the system already draws.

## The EDM workflow

Almost every EDM analysis follows three steps.

1. **Embed.** Pick a dimension `E` and a delay `tau`, then turn the scalar series into points in `R^E` using lagged coordinates.
2. **Predict.** For each query, find its nearest neighbors in the library and combine their futures (simplex projection), or fit a local linear model (S-Map).
3. **Interpret.** Compare prediction skill across parameters or library sizes to answer the scientific question — system dimensionality, nonlinearity, or whether `X` drives `Y`.

edmkit ships one module per step.

| Module | Purpose | Typical question |
| --- | --- | --- |
| [`embedding`](/edmkit/reference/embedding/) | Lagged embedding and parameter selection | "What `E` and `tau` reconstruct this system?" |
| [`simplex_projection`](/edmkit/reference/simplex-projection/) | Nearest-neighbor prediction | "How well can I forecast the next step?" |
| [`smap`](/edmkit/reference/smap/) | Local linear prediction | "Is the system nonlinear?" |
| [`ccm`](/edmkit/reference/ccm/) | Convergent Cross Mapping | "Does `X` drive `Y`?" |

## When EDM helps

- **Nonlinear dynamics** that a linear model misses.
- **Low-dimensional but unknown systems** — no governing equations, and a high-capacity model would overfit.
- **Short, noisy series.** Local methods regularize implicitly; parametric models burn degrees of freedom.
- **Causal direction** between coupled variables, where Granger-style tests fail due to nonlinearity or shared noise.

## When EDM struggles

- **Stochasticity dominates.** EDM exploits geometric regularity; noise has none.
- **The series is too short.** As a heuristic, plan for `N` larger than `5 * 10^E` samples.
- **The sampling rate is wrong.** Too small `tau` makes coordinates redundant; too large makes them independent.
- **The system visits regimes that never repeat.** EDM extrapolates poorly outside the library's coverage.

## Recommended reading order

1. [Time-delay embedding](/edmkit/concepts/embedding/) — the foundational construction.
2. [Simplex projection](/edmkit/concepts/simplex-projection/) — the core prediction algorithm and how it picks `E`.
3. [S-Map](/edmkit/concepts/smap/) — a refinement that quantifies nonlinearity.
4. [Convergent Cross Mapping](/edmkit/concepts/ccm/) — the causal inference method.

Already know EDM? Jump to the [guides](/edmkit/guides/choosing-parameters/).

## Primary references

- Takens, F. (1981). *Detecting strange attractors in turbulence.* In *Dynamical Systems and Turbulence* (pp. 366–381). Springer.
- Sugihara, G., & May, R. M. (1990). *Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series.* Nature, 344(6268), 734–741.
- Sugihara, G., May, R., Ye, H., Hsieh, C., Deyle, E., Fogarty, M., & Munch, S. (2012). *Detecting causality in complex ecosystems.* Science, 338(6106), 496–500.
