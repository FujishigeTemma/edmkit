# Edmkit Test Map

Verified against the repository on 2026-03-23.

## Table Of Contents

1. Repository Facts
2. Current Commands
3. Existing Test Patterns
4. Module-Specific Invariants
5. External Sources

## Repository Facts

- Runtime package: `src/edmkit/`
- Main algorithm modules:
  `embedding.py`, `simplex_projection.py`, `smap.py`, `ccm.py`, `metrics.py`, `splits.py`, `util.py`
- Synthetic generators:
  `src/edmkit/generate/`
- Tests:
  `tests/test_*.py`
- Special smoke test:
  `tests/smoke_test.py`

## Current Commands

From `AGENTS.md` and `README.md`:

- `uv sync --group dev`
- `uv run pytest`
- `uv run pytest -m "not slow"`
- `HYPOTHESIS_PROFILE=ci uv run pytest`
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run ty check`

## Existing Test Patterns

Observed in the current suite:

- example-based exact tests for hand-computable cases
- Hypothesis strategies built close to each module's tests
- deterministic RNG seeding with `np.random.default_rng`
- trend assertions with Spearman rank correlation for qualitative degradation
- domain fixtures in `tests/conftest.py`
- direct use of `np.testing.assert_allclose` and `assert_array_equal`

Current repo-wide pytest facts from `pyproject.toml` and `tests/conftest.py`:

- `pytest>=9.0.2`
- `hypothesis>=6.151.9`
- `addopts = "-v --tb=short"`
- markers: `slow`, `gpu`
- Hypothesis profile `dev`:
  `max_examples=50`, `deadline=500`
- Hypothesis profile `ci`:
  `max_examples=200`, `deadline=None`, suppresses `HealthCheck.too_slow`

## Module-Specific Invariants

Use these as starting points when adding coverage.

### `embedding.lagged_embed`

- output shape is `(len(x) - (e - 1) * tau, e)`
- `e=1, tau=1` is a reshape to column vector
- each output column corresponds to the expected lagged slice
- invalid dimensions, non-positive `tau` or `e`, and insufficient input length must raise `ValueError`

### `simplex_projection.simplex_projection`

- if a query equals a library point, the prediction should recover that target for exact-nearest cases
- constant targets stay constant
- predictions stay inside the target value range because the result is a weighted average
- library permutation should not change predictions
- masked batched calls should agree with explicitly filtered library calls

### `smap.smap`

- `theta=0` should match or closely approximate global linear regression depending on `alpha`
- constant targets remain constant even with large regularization
- larger `theta` increases locality and should downweight far points
- larger `alpha` should shrink variation toward simpler behavior without changing intercept semantics
- library permutation should not change predictions

### `ccm`

- helper functions should be tested independently from convergence behavior
- fixed seeds should make sampling reproducible
- custom samplers and aggregators should be observable in tests
- convergence tests should focus on directionality or monotone evidence, not exact sampled values

### Generators

- outputs should have expected dimensionality and finite values
- tests should prefer structural and physical sanity checks over exact trajectory snapshots

## External Sources

The parent skill at `~/.codex/skills/python-testing-best-practices` contains the broader pytest, Hypothesis, and AI-era testing guidance.

Primary external references used while preparing this repo-local skill:

- Pytest good practices
  https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Pytest parametrization
  https://docs.pytest.org/en/stable/how-to/parametrize.html
- Pytest temporary paths
  https://docs.pytest.org/en/stable/how-to/tmp_path.html
- Pytest monkeypatch
  https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- Pytest skip and xfail
  https://docs.pytest.org/en/stable/how-to/skipping.html
- Hypothesis quickstart
  https://hypothesis.readthedocs.io/en/latest/quickstart.html
- Hypothesis settings
  https://hypothesis.readthedocs.io/en/latest/tutorial/settings.html
- Hypothesis replaying failures
  https://hypothesis.readthedocs.io/en/latest/tutorial/replaying-failures.html
