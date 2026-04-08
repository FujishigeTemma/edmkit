---
title: What is EDM?
description: Introduction to Empirical Dynamic Modeling and its core ideas.
---

Empirical Dynamic Modeling (EDM) is a nonparametric framework for analyzing time series data. Unlike traditional statistical approaches that assume a fixed model structure (e.g., ARIMA, VAR), EDM reconstructs the underlying dynamics directly from observed data using **Takens' embedding theorem**.

## Core Idea

A single observed time series can be used to reconstruct the multidimensional state space of the system that generated it. By embedding the time series into a higher-dimensional space using lagged copies of itself, EDM recovers the attractor — the geometric structure that governs the system's dynamics.

## The EDM Workflow

1. **Embedding** — Transform a scalar time series into a multidimensional state space using lagged coordinates
2. **Prediction** — Use nearest neighbors in the reconstructed state space to forecast future states
3. **Inference** — Distinguish deterministic chaos from stochasticity, estimate nonlinearity, and test causal relationships

## Key Algorithms

### Simplex Projection

Simplex projection is the simplest EDM prediction method. It finds the _E+1_ nearest neighbors (forming a simplex in _E_-dimensional space) around a query point and produces a weighted-average prediction. The weights are inversely proportional to distance.

Simplex projection is primarily used to **determine the optimal embedding dimension** _E_ — the dimension at which prediction skill peaks indicates the true dimensionality of the attractor.

### S-Map

Sequential Locally Weighted Global Linear Map (S-Map) extends simplex projection by fitting a local linear model at each prediction point. A locality parameter _theta_ controls how strongly distance-weighted the regression is:

- **theta = 0**: Global linear regression (equivalent to a standard linear model)
- **theta > 0**: Increasingly local, nonlinear behavior

The degree to which prediction improves with increasing _theta_ quantifies the **nonlinearity** of the system.

### Convergent Cross Mapping (CCM)

CCM tests for **causal relationships** between variables in coupled dynamical systems. If variable _X_ causally influences variable _Y_, then the attractor reconstructed from _Y_ should contain information about _X_.

CCM measures this by checking whether predictions of _X_ from the reconstructed state space of _Y_ **converge** (improve) as more data is used. Convergence with increasing library size is the signature of causality.

## When to Use EDM

EDM is particularly suited for:

- **Nonlinear dynamical systems** where linear models fail
- **Short, noisy time series** where parametric models overfit
- **Causal inference** in coupled systems (e.g., ecology, neuroscience, climate)
- **State-dependent dynamics** where relationships change over time

## Further Reading

- Sugihara & May (1990). _Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series._ Nature, 344, 734–741.
- Sugihara et al. (2012). _Detecting causality in complex ecosystems._ Science, 338, 496–500.
