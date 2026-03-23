---
name: python-testing-best-practices
description: Design, implement, review, and modernize Python tests with an emphasis on pytest, Hypothesis, and AI-era evaluation workflows. Use when Codex needs to add or fix tests, improve flaky or slow suites, review pytest structure and fixture usage, introduce property-based testing, or separate deterministic code tests from eval-style checks for LLM or agent behavior.
---

# Python Testing Best Practices

## Overview

Design tests that stay trustworthy under AI-assisted implementation. Prefer fast deterministic checks at the lowest stable layer, then add higher-level integration coverage only where it catches unique risk.

Read [official-sources.md](./references/official-sources.md) when you need the supporting rationale, version-sensitive pytest guidance, or source links.

## Workflow

1. Inspect the local test stack before writing anything.
2. Map the change to the cheapest reliable test layer.
3. Write or revise tests using pytest defaults first, Hypothesis where invariants matter, and evals only for nondeterministic model behavior.
4. Run the smallest validating subset first, then the project-standard full checks.

## Inspect First

Read the local test configuration before proposing changes:

- `pyproject.toml`, `pytest.ini`, or `tox.ini`
- `tests/conftest.py`
- one or two representative test files near the target module
- CI commands or local task runner config if present

Determine:

- test discovery conventions and import mode
- fixture scope and shared helpers
- markers already in use
- Hypothesis profiles or test-environment toggles
- whether the repo treats warnings as failures, disables plugin autoload, or enables pytest strictness

## Choose The Right Test Type

Use this decision rule:

- Unit test: prefer for pure functions, numerical routines, parsing, shape checks, validation, and error handling.
- Property-based test: prefer when behavior is defined by invariants, round-trips, monotonicity, idempotence, bounds, or algebraic relationships.
- Integration test: prefer when correctness depends on module interaction, filesystem, subprocesses, network boundaries, or package wiring.
- E2E or smoke test: prefer for install, import, CLI, packaging guarantees, and one critical happy path.
- Evals: prefer only for nondeterministic LLM or agent behavior that cannot be reduced to deterministic assertions.

Do not spend an expensive integration or eval test on logic that can be proven with a unit or property test.

## Implement Pytest Tests

Prefer:

- plain `assert` statements with precise failures emerging from values
- `@pytest.mark.parametrize` instead of duplicated tests
- `tmp_path` and `tmp_path_factory` instead of legacy temp helpers
- `monkeypatch` for environment variables, process state, and imported functions
- `caplog` or `capsys` when asserting logs or console behavior
- explicit marker registration and strict marker usage when the repo supports it

Keep fixtures narrow:

- return simple test data or lightweight setup
- use `yield` fixtures for cleanup
- avoid broad autouse fixtures unless they protect the whole suite from shared global state
- move shared fixtures into `tests/conftest.py` only when multiple files need them

Avoid:

- asserting on incidental formatting, ordering, timestamps, random seeds, or exact wording unless required behavior depends on it
- heavy mocks when a temporary directory, fake input object, or thin seam gives a more realistic test
- new custom markers without config registration
- `xfail` as a parking lot for unknown failures; require a reason and remove it when the bug is fixed

## Add Property-Based Tests Deliberately

Use Hypothesis when you can express behavior as a property, not just as a list of examples.

Good fits:

- encode/decode or serialize/parse round-trips
- projection or normalization preserving invariants
- agreement between optimized and reference implementations
- masks, missing-data handling, monotonic constraints, or shape-preserving transforms
- numerical edge cases where hand-picked examples are incomplete

Start with existing domain constraints. Only widen strategy space after the property is stable.

Use `@example` for must-run regressions. Treat the Hypothesis database as a replay cache, not as correctness.

When the suite has different local and CI budgets, configure Hypothesis profiles and load them through environment or pytest options instead of hard-coding large `max_examples` values everywhere.

## Handle AI-Era Testing

Separate these concerns:

- Deterministic code: test with normal pytest assertions and Hypothesis.
- LLM or agent output quality: test with eval datasets, graders, and explicit acceptance criteria.
- Tooling or orchestration around LLM calls: test deterministically by stubbing model I/O and asserting request construction, parsing, retries, and fallbacks.

For AI-assisted code generation, assume generated tests may be overfit to the implementation. Counter this by checking:

- whether the test asserts observable behavior instead of copied internals
- whether a small bug in the implementation would still fail the test
- whether the test covers at least one edge case or invariant the implementation did not spell out
- whether fixtures and parametrization reduce copy-paste without hiding intent

## Review Checklist

Before finishing, check:

- Does each test protect a real behavior or regression?
- Is the chosen layer the cheapest one that can reliably catch the bug?
- Are temporary files, env vars, logs, and global state isolated per test?
- Are skips and xfails justified, not masking unknown problems?
- Are Hypothesis settings and profiles appropriate for local and CI runtimes?
- If models are involved, are deterministic tests and evals clearly separated?

## Validation

Run repo-native commands first. If none exist, use the minimum set:

```bash
pytest path/to/test_file.py
pytest -m "not slow"
pytest
```

When relevant, also run lint and type checks that validate test code or config.
