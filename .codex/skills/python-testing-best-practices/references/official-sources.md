# Official Sources

Verified on 2026-03-23.

## Table Of Contents

1. Pytest
2. Hypothesis
3. AI And LLM Evaluation
4. Distilled Guidance

## Pytest

- Good Integration Practices
  https://docs.pytest.org/en/stable/explanation/goodpractices.html
  Notes:
  New projects are recommended to use `importlib` import mode.
  `python setup.py test` and `pytest-runner` are not recommended.
  Recent docs also describe pytest strict mode and the individual strict options.

- Parametrization
  https://docs.pytest.org/en/stable/how-to/parametrize.html
  Notes:
  Prefer parametrization instead of duplicated tests.
  Parameter values are passed as-is, so mutable values can leak mutation across cases.

- Temporary paths
  https://docs.pytest.org/en/stable/how-to/tmp_path.html
  Notes:
  Prefer `tmp_path` and `tmp_path_factory`.
  `tmpdir` is legacy.

- Monkeypatch
  https://docs.pytest.org/en/stable/how-to/monkeypatch.html
  Notes:
  Use `monkeypatch` for env vars, attributes, dict items, `sys.path`, and cwd.
  Use `MonkeyPatch.context()` when patching risky stdlib or library behavior for a narrow scope.

- Logging
  https://docs.pytest.org/en/stable/how-to/logging.html
  Notes:
  `caplog` is the right tool for asserting logs.
  Root logger rewrites can break log capture.

- Skip and xfail
  https://docs.pytest.org/en/stable/how-to/skipping.html
  Notes:
  Skip when preconditions are absent.
  Xfail when a known bug or missing feature is intentionally tolerated.

- Plugins
  https://docs.pytest.org/en/stable/how-to/plugins.html
  Notes:
  `--disable-plugin-autoload` was added in pytest 8.4.
  Use it when test environments need stricter isolation from ambient plugins.

## Hypothesis

- Quickstart
  https://hypothesis.readthedocs.io/en/latest/quickstart.html
  Notes:
  Hypothesis tests are still normal pytest tests.
  Default behavior generates 100 examples.
  Falsifying examples are minimized for easier debugging.

- Configuring settings
  https://hypothesis.readthedocs.io/en/latest/tutorial/settings.html
  Notes:
  Use `@settings` for per-test tuning.
  Use settings profiles for suite-wide local versus CI tradeoffs.
  Profiles can be loaded via `HYPOTHESIS_PROFILE` or pytest integration.

- Replaying failed tests
  https://hypothesis.readthedocs.io/en/latest/tutorial/replaying-failures.html
  Notes:
  The example database replays failures quickly.
  The database is not a correctness guarantee.
  Use `@example` for permanent must-run regressions.

- Flaky failures
  https://hypothesis.readthedocs.io/en/latest/tutorial/flaky.html
  Notes:
  Hypothesis treats flaky behavior as a real failure.
  Deterministic pass or fail behavior matters, even if internal implementation uses randomness.

- Ghostwriter
  https://hypothesis.readthedocs.io/en/latest/_modules/hypothesis/extra/ghostwriter.html
  Notes:
  Ghostwriter is a bootstrap tool for property tests.
  Treat generated tests as drafts; refine strategy quality and assertions manually.

## AI And LLM Evaluation

- OpenAI evals guide
  https://developers.openai.com/api/docs/guides/evals
  Notes:
  Evals are for testing model outputs against explicit criteria.
  The documented loop is define task, run on test inputs, analyze results, then iterate.
  Use this for nondeterministic model behavior, not as a replacement for deterministic code tests.

## Distilled Guidance

- Put most effort into fast deterministic tests that can run on every change.
- Add property-based tests when the behavior is an invariant, relation, or round-trip.
- Use integration tests to verify seams, not to duplicate every unit-level behavior.
- Keep fixtures narrow and obvious; shared fixtures should remove duplication, not hide setup logic.
- Treat AI-generated tests as untrusted until they prove they would fail on a plausible bug.
- For LLM systems, split deterministic orchestration tests from eval-driven model quality checks.
