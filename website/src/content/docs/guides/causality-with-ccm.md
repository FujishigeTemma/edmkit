---
title: Testing causality with CCM
description: A full Convergent Cross Mapping workflow, including both directions, surrogate testing, and a checklist of common pitfalls.
---

This guide tests a causal hypothesis between two scalar series with CCM. It tests both directions, plots convergence, and compares against a shuffled surrogate. The pipeline applies to any two series long enough to embed.

Read [Convergent Cross Mapping](/edmkit/concepts/ccm/) first for the intuition.

## Set up coupled test data

Two coupled logistic maps as ground truth: `x` drives `y` with coupling `beta`; the reverse is zero. The right answer is "x causes y" and "y does not cause x".

```python
import numpy as np

N, rx, ry, beta = 2000, 3.8, 3.5, 0.04
x = np.zeros(N)
y = np.zeros(N)
x[0], y[0] = 0.4, 0.2
for i in range(1, N):
    x[i] = x[i - 1] * (rx - rx * x[i - 1])
    y[i] = y[i - 1] * (ry - ry * y[i - 1]) + beta * x[i - 1]
```

Substitute your own pair for `x` and `y` to use the rest as-is.

## Pick embedding parameters per direction

CCM relies on the embedding of the *effect*. Each direction needs its own `(E, tau)`, chosen against the variable being reconstructed.

```python
from edmkit.embedding import scan, select

def best_params(series, E_grid=range(1, 11), tau_grid=(1, 2, 3, 5)):
    scores = scan(series, E=list(E_grid), tau=list(tau_grid))
    E, tau, _ = select(scores, E=list(E_grid), tau=list(tau_grid))
    return E, tau

Ey, tauy = best_params(y)   # used when testing "x -> y"
Ex, taux = best_params(x)   # used when testing "y -> x"
print(f"Embedding for y: E={Ey}, tau={tauy}")
print(f"Embedding for x: E={Ex}, tau={taux}")
```

:::caution
`scan`/`select` maximize **self-prediction** skill. CCM often needs a slightly larger `E` to capture the cause's influence. If cross-mapping looks suspiciously flat, raise `E` by one or two and re-run before concluding "no causality".
:::

For the coupled logistic map below, the cause enters `y` at lag 1, so `E = 2` is the smallest dimension that lets `y`'s attractor recover `x`:

```python
Ey, tauy = 2, 1   # override; self-prediction would have picked E=1
Ex, taux = 2, 1
```

## Cross-map both directions

For each hypothesis: embed the effect, align the cause, sweep library size with `with_simplex_projection`.

```python
from edmkit.ccm import make_sample_func, with_simplex_projection
from edmkit.embedding import lagged_embed


def cross_map(effect, cause, E, tau, seed):
    embedded = lagged_embed(effect, tau=tau, e=E)
    aligned = cause[(E - 1) * tau :]

    mid = embedded.shape[0] // 2
    library_pool = np.arange(mid)
    prediction_pool = np.arange(mid, embedded.shape[0])

    lib_sizes = np.logspace(
        np.log10(10),
        np.log10(library_pool[-1]),
        num=12,
        dtype=int,
    )

    rho = with_simplex_projection(
        embedded,
        aligned,
        lib_sizes=lib_sizes,
        n_samples=100,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(seed=seed),
    )
    return lib_sizes, rho


sizes_xy, rho_xy = cross_map(effect=y, cause=x, E=Ey, tau=tauy, seed=42)
sizes_yx, rho_yx = cross_map(effect=x, cause=y, E=Ex, tau=taux, seed=43)

print("x -> y:", rho_xy.round(3))
print("y -> x:", rho_yx.round(3))
```

Direction is encoded in the first two arguments: the first is the embedded effect, the second is the candidate cause to predict. Read `cross_map(effect=y, cause=x, ...)` as "test x -> y by cross-mapping y to x".

## Read the convergence

A convergent curve rises with library size and plateaus. A non-causal direction stays low and flat.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogx(sizes_xy, rho_xy, marker="o", label="x -> y")
ax.semilogx(sizes_yx, rho_yx, marker="s", label="y -> x")
ax.set_xlabel("library size")
ax.set_ylabel("cross-map skill (rho)")
ax.set_ylim(-0.1, 1)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
```

For these coupling parameters, expect `x -> y` to converge near `0.8` while `y -> x` stays near `0`.

## Bootstrap confidence intervals

`with_simplex_projection` returns one mean per library size. Use `bootstrap` for raw per-sample correlations and percentile bands.

```python
from edmkit.ccm import bootstrap
from edmkit.simplex_projection import simplex_projection

embedded_y = lagged_embed(y, tau=tauy, e=Ey)
x_aligned = x[(Ey - 1) * tauy :]
mid = embedded_y.shape[0] // 2

samples = bootstrap(
    embedded_y,
    x_aligned,
    lib_sizes=sizes_xy,
    predict_func=simplex_projection,
    n_samples=200,
    library_pool=np.arange(mid),
    prediction_pool=np.arange(mid, embedded_y.shape[0]),
)
# samples.shape == (200, len(sizes_xy))

median = np.median(samples, axis=0)
lo = np.quantile(samples, 0.05, axis=0)
hi = np.quantile(samples, 0.95, axis=0)
```

Plot `median` with `lo`/`hi` as a shaded band. Convergence above a surrogate's band is the publication-grade claim.

## Surrogate test

Shuffling the candidate cause breaks any genuine coupling while preserving its marginal distribution. The cross-map skill on the shuffle is the noise floor the real curve must clear.

```python
rng = np.random.default_rng(0)
x_shuffled = rng.permutation(x)

_, rho_surrogate = cross_map(effect=y, cause=x_shuffled, E=Ey, tau=tauy, seed=44)
print("surrogate x -> y:", rho_surrogate.round(3))
```

The surrogate curve should be flat near zero. If it tracks the real curve, the apparent convergence is an artifact — usually shared trends, periodicity, or non-stationarity.

For stricter tests, use phase-randomized surrogates (preserves the spectrum) or season-matched surrogates. edmkit does not generate these, but `numpy` and `scipy.signal` suffice.

## Checklist before reporting causal direction

- [ ] Both directions tested, with separate embeddings per effect.
- [ ] Convergence read visually on a log-x axis, not just from endpoints.
- [ ] Surrogate cleared by a clear margin.
- [ ] Library and prediction pools disjoint, unless matching the original Sugihara protocol.
- [ ] Same hyperparameters across both directions and the surrogate.
- [ ] At least several hundred embedded states at the largest library size.

## Common pitfalls

- **Bidirectional convergence with instant coupling.** When delay-zero coupling is strong, both directions converge and CCM cannot disambiguate. Cross-map at non-zero target lags and compare positive vs negative lag.
- **Common driver.** A confounding third variable produces convergence in both directions. Surrogates do not fix this; use conditional CCM (not yet shipped) or domain controls.
- **Non-stationarity.** Trends and regime shifts produce convergence-looking curves for non-causal pairs. Detrend and de-seasonalize before embedding.
- **Embedding mismatch.** `tau = 1` on a slowly varying continuous-time series makes coordinates near-identical and the cross-map looks suspicious. Cross-validate the embedding per effect.
- **Too few bootstrap samples.** `n_samples=20` lets per-library variance dominate. Use 100–200 for reported numbers; 20 is fine for exploratory plots.

## Where to go next

- [Convergent Cross Mapping](/edmkit/concepts/ccm/) — theory and notation.
- [Choosing E and tau](/edmkit/guides/choosing-parameters/) — the embedding step CCM depends on.
- [`ccm` API reference](/edmkit/reference/ccm/) — `bootstrap`, `with_smap`, and the generic `ccm`.
