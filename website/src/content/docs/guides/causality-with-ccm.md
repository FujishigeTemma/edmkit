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

Score each variable's own `(E, tau)` by held-out skill as in the [parameter selection guide](/edmkit/guides/choosing-parameters/), once for `y` and once for `x`.

:::caution
That recipe maximizes **self-prediction** skill. CCM often needs a slightly larger `E` to capture the cause's influence. If cross-mapping looks suspiciously flat, raise `E` by one or two and re-run before concluding "no causality".
:::

For the coupled logistic map below, the cause enters `y` at lag 1, so `E = 2` is the smallest dimension that lets `y`'s attractor recover `x`:

```python
Ey, tauy = 2, 1  # override; self-prediction would have picked E=1
Ex, taux = 2, 1
```

## Cross-map both directions

For each hypothesis: embed the effect, align the cause, sweep library size with `bootstrap`, and score each batch of predictions in a thread pool.

```python
from concurrent.futures import ThreadPoolExecutor

from edmkit.ccm import bootstrap
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection


def cross_map(effect, cause, E, tau, seed, n_samples=100):
    coordinates = np.array([[0, tau * j] for j in range(E)])
    embedded, _ = embed(coordinates, effect)
    aligned = cause[(E - 1) * tau :]

    mid = len(embedded) // 2
    lib_sizes = np.geomspace(10, mid, num=12).astype(int)
    actual = aligned[mid:, None]

    def score(samples):
        predictions = samples()  # (B, M, 1)
        return pearson_correlation(predictions, np.broadcast_to(actual, predictions.shape))

    scores = np.empty((len(lib_sizes), n_samples))
    tasks = bootstrap(
        embedded[:mid],
        aligned[:mid],
        embedded[mid:],
        lib_sizes,
        predict_func=simplex_projection,
        n_samples=n_samples,
        replace=False,
        seed=seed,
    )
    indices, samples = zip(*tasks)
    with ThreadPoolExecutor() as executor:
        for index, result in zip(indices, executor.map(score, samples)):
            scores[index] = result
    return lib_sizes, scores


sizes_xy, scores_xy = cross_map(effect=y, cause=x, E=Ey, tau=tauy, seed=42)
sizes_yx, scores_yx = cross_map(effect=x, cause=y, E=Ex, tau=taux, seed=43)
rho_xy, rho_yx = scores_xy.mean(axis=1), scores_yx.mean(axis=1)

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

## Bootstrap distributions

`cross_map` keeps the per-sample scores, so their distribution across sampled libraries needs no extra run.

```python
median = np.median(scores_xy, axis=1)
lo, hi = np.quantile(scores_xy, [0.05, 0.95], axis=1)
```

Plot `median` with `lo`/`hi` as a shaded band. These bands describe variation across library samples; they are not confidence intervals for a causal effect.

To use S-Map, pass `predict_func=partial(smap, theta=2.0)` after importing `partial` from `functools` and `smap` from `edmkit.smap`. Scoring stays the same.

## Surrogate test

Shuffling the candidate cause breaks any genuine coupling while preserving its marginal distribution. The cross-map skill on the shuffle is the noise floor the real curve must clear.

```python
rng = np.random.default_rng(0)
x_shuffled = rng.permutation(x)

_, scores_surrogate = cross_map(effect=y, cause=x_shuffled, E=Ey, tau=tauy, seed=44)
print("surrogate x -> y:", scores_surrogate.mean(axis=1).round(3))
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
- [`ccm` API reference](/edmkit/reference/ccm/) — task generation, shapes, and sampling options.
