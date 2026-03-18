# Repository Guidelines

## Project Structure & Module Organization
Core library code lives under `src/edmkit/`. Top-level modules such as `embedding.py`, `simplex_projection.py`, `smap.py`, and `ccm.py` expose the main EDM algorithms, while `src/edmkit/generate/` contains synthetic data generators. Tests live in `tests/` and generally mirror module names, for example `src/edmkit/embedding.py` and `tests/test_embedding.py`. Use `benchmarks/` for ad hoc performance scripts, not correctness checks. CI workflows are in `.github/workflows/`.

## Build, Test, and Development Commands
Use `uv` with Python 3.13 for local work.

- `uv sync --group dev`: install runtime and development dependencies into the project environment.
- `uv run pytest`: run the full test suite.
- `uv run pytest -m "not slow"`: skip slow tests during quick iteration.
- `HYPOTHESIS_PROFILE=ci uv run pytest`: match CI’s heavier property-based test profile.
- `uv run ruff check .`: run lint checks.
- `uv run ruff format .`: apply formatting.
- `uv run ty check`: run static type checks.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, explicit imports, and module-level functions for numerical routines. Ruff enforces formatting and a maximum line length of 150 via `ruff.toml`; run it before opening a PR. Use `snake_case` for modules, functions, variables, and test names. Keep new code under `src/edmkit/` and match filenames to the public API they implement.

## Testing Guidelines
Tests use `pytest` plus `hypothesis`. Name files `test_*.py` and keep focused assertions near the behavior under test. Reuse `tests/conftest.py` when adding shared fixtures or utilities. Mark expensive cases with `@pytest.mark.slow` and tinygrad-backed cases with `@pytest.mark.gpu`. `tests/smoke_test.py` is special: it must run with `uv run --isolated --no-project`, so do not import `tests` helpers or rely on dev-only packages there.

## Commit & Pull Request Guidelines
Recent commits are short, imperative, and lower-case, for example `bump`, `use usearch`, and `fix lint and type check CI failures`. Keep commit subjects concise and action-oriented. PRs should explain the user-visible or numerical impact, mention any API changes, and list the validation you ran (`pytest`, `ruff`, `ty`). Include benchmark notes when performance-sensitive code changes.
