---
name: edmkit-testing
description: Add, revise, and review tests for the edmkit repository with repo-specific guidance for pytest, Hypothesis, numerical tolerances, synthetic dynamical systems, slow or gpu markers, and smoke-test constraints. Use when Codex is changing code under src/edmkit/, touching tests/, or needs to decide the right test strategy for embedding, simplex projection, S-Map, CCM, metrics, splits, or generators in this repository.
---

# Edmkit Testing

## Overview

Apply the general Python testing guidance from the user's global testing skill, then specialize it to edmkit's numerical and dynamical-systems behavior. Prefer small deterministic assertions first, then add property or trend tests only where the algorithm semantics justify them.

Read [repo-test-map.md](./references/repo-test-map.md) for the current repository-specific map, commands, and module-level invariants.

## Start Here

Before writing tests, read:

- `AGENTS.md`
- `pyproject.toml`
- `tests/conftest.py`
- the target module under `src/edmkit/`
- the nearest matching file in `tests/`

Assume these local constraints:

- Use `uv` with Python 3.13.
- `pytest` and `hypothesis` are already part of the dev group.
- Valid markers are `slow` and `gpu`.
- Hypothesis profiles are loaded from `tests/conftest.py`, defaulting to `dev` and switching with `HYPOTHESIS_PROFILE`.
- `tests/smoke_test.py` must work under `uv run --isolated --no-project` and cannot depend on `tests` helpers or dev-only packages.

## Test Selection For Edmkit

Choose the narrowest test that proves the behavior:

- Example-based exact test:
  Use for hand-computable embeddings, constant-target predictions, shape checks, and argument validation.
- Property-based test:
  Use for algebraic and structural invariants such as shape, permutation invariance, convex-range bounds, round-trips, or correlation symmetries.
- Trend test:
  Use when theory predicts direction rather than exact value, such as forecast degradation with larger horizons, noise sensitivity, or CCM convergence with larger library sizes.
- Smoke test:
  Use only for importability and one minimal end-to-end call path per public surface.

Do not overuse trend tests when a deterministic exact or property test exists. Trend tests are more brittle and should check sign, rank trend, or relative ordering, not overfit exact numbers.

## Numerical Assertion Rules

For numerical code in this repo:

- Use `np.testing.assert_array_equal` only when the operation is mathematically exact and implemented with integer-style indexing or reshaping.
- Use `np.testing.assert_allclose` for floating-point algorithms. Pick tolerances from the problem structure, not habit.
- Prefer absolute tolerances for values expected near zero.
- Prefer relative tolerances when the scale grows with the signal.
- When comparing correlations, RMSE, or ranks, assert a threshold with a short justification rather than snapshotting exact values.

When a test depends on random data, seed the RNG locally inside the test with `np.random.default_rng(...)`.

## Module Guidance

Use these defaults unless the concrete change suggests a better target:

- `embedding.py`:
  Test output shape, indexing semantics, minimal valid length, and invalid parameter rejection. Good property tests: each output cell maps to the correct lagged source index.
- `simplex_projection.py`:
  Test exact recovery for identity queries, constant-target behavior, prediction staying inside target range, permutation invariance of the library, and consistency between masked and explicitly filtered libraries.
- `smap.py`:
  Test `theta=0` against ordinary least squares, constant-target recovery, monotone locality effects as `theta` increases, and regularization behavior through relative shrinkage rather than exact coefficients.
- `ccm.py`:
  Test helper correctness first (`pearson_correlation`, custom sampler, aggregator, reproducibility), then convergence trends on synthetic coupled systems. For CCM, assert directional or monotone evidence, not exact curves.
- `generate/`:
  Test output shapes, finiteness, and qualitative constraints from the generator definition. Avoid brittle golden trajectories unless the solver is intentionally frozen.
- `splits.py`, `metrics.py`, `util.py`:
  Prefer deterministic unit tests and Hypothesis properties over long dynamical runs.

## Hypothesis In This Repo

Follow the existing style:

- Keep custom strategies near the test module unless many files reuse them.
- Use `hypothesis.extra.numpy` for shaped arrays.
- Exclude `NaN` and `inf` unless the target behavior explicitly handles them.
- Use `assume(...)` sparingly; prefer generating only valid inputs up front.
- Respect the existing `dev` and `ci` profiles instead of forcing large per-test settings.

For edmkit specifically, Hypothesis is most valuable for:

- shape preservation
- permutation invariance
- boundedness or convex-combination properties
- symmetry or self-consistency of metrics
- agreement between batched and unbatched paths

## Fixtures And Synthetic Data

`tests/conftest.py` already provides reusable synthetic systems such as bounded linear series, logistic maps, Lorenz trajectories, and coupled pairs.

Prefer:

- reusing those fixtures when the same dynamical system semantics matter
- adding a fixture only when multiple files need it
- generating local data inline when the test only needs one small synthetic example

When adding new shared fixtures, keep them deterministic, documented, and cheap enough for broad reuse.

## Markers And Runtime Budget

Mark a test `slow` when it needs long trajectories, many bootstrap samples, or expensive numerical scans. Mark a test `gpu` only for tinygrad-backed behavior.

During implementation, run:

```bash
uv run pytest path/to/test_file.py
uv run pytest -m "not slow"
```

Before finishing a substantial testing change, run:

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

Use CI-style Hypothesis coverage when a property test is central to the change:

```bash
HYPOTHESIS_PROFILE=ci uv run pytest
```

## Smoke Test Rules

If you touch `tests/smoke_test.py`, keep it isolated:

- import only runtime dependencies
- do not import from `tests`
- do not rely on `hypothesis`, `pytest`, or helper fixtures
- cover import plus one minimal successful API path

Treat smoke tests as packaging and installation checks, not correctness proofs.

## Review Checklist

Before finalizing, verify:

- the new test would fail on a plausible implementation bug
- tolerance choices are justified by algorithm structure
- random data is locally seeded
- trend tests assert direction or ranking, not unstable exact numbers
- shared fixtures are reused instead of copied
- `slow` and `gpu` markers are applied consistently
- smoke-test constraints remain intact
