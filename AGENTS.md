# Repository Guidelines

## Project Structure & Module Organization
Core library code lives under `src/edmkit/`. Top-level modules such as `simplex_projection/*.py`, `smap.py`, and `ccm.py` expose the main EDM algorithms, while `src/edmkit/generate/` contains synthetic data generators. Tests live in `tests/` and generally mirror module names, for example `src/edmkit/smap.py` and `tests/test_smap.py`. Use `benchmarks/` for ad hoc performance scripts, not correctness checks. CI workflows are in `.github/workflows/`.

## Build, Test, and Development Commands
Use `uv`, never use `python`, `pip`, `uv pip`.

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
Tests use `pytest` plus `hypothesis`. Name files `test_*.py`, mirroring the module under test. Less is more: test only functions whose behavior is worth pinning down — trivial glue, thin wrappers, and code already covered through its callers need no dedicated tests. Every function under test `xxx` gets exactly this set of definitions, written inline in its test file; anything outside this pattern is forbidden:

- `check_xxx(...)`: calls `xxx` and asserts what the output must satisfy. Focus on properties the result provably holds; when comparing against an expected value, the reference must be an oracle (an existing library or a genuinely independent implementation), not a restatement of the implementation. If no meaningful property or oracle exists, do not force a test (prefer a clean, weaker assertion over a complex one).
- `XxxProblem` + `xxx_problems`: a NamedTuple of inputs and a `hypothesis` strategy generating them; `test_xxx_compatibility` runs `check_xxx` over it (property-based testing).
- `XxxCase` + `XXX_VALID` / `XXX_INVALID`: the same inputs as concrete edge cases in name-keyed dicts; `test_xxx_valid` runs `check_xxx` over `XXX_VALID`, and `test_xxx_invalid` asserts each `XXX_INVALID` case raises `ValueError`.

Backend variants (`check_xxx_tensor`, `check_xxx_tensor_gradient`) are extra check functions run over the same cases. Mark expensive cases with `@pytest.mark.slow` and tinygrad-backed cases with `@pytest.mark.gpu`. `tests/smoke_test.py` is special: it must run with `uv run --isolated --no-project`, so do not import `tests` helpers or rely on dev-only packages there.

## Commit & Pull Request Guidelines
Recent commits are short, imperative, and lower-case, for example `bump`, `use usearch`, and `fix lint and type check CI failures`. Keep commit subjects concise and action-oriented. PRs should explain the user-visible or numerical impact, mention any API changes, and list the validation you ran (`pytest`, `ruff`, `ty`). Include benchmark notes when performance-sensitive code changes.
