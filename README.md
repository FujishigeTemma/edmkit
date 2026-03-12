# edmkit

This library is a collection of tools and utilities that are useful for Empirical Data Modeling (EDM) and related tasks. The library is designed to be fast and lightweight, and easy to use.

::: warning
This library is still under intensive development so API may change in the future.
:::

## Installation

To install the library, you can use pip:

```bash
pip install edmkit
```

Or you can also use [uv](https://docs.astral.sh/uv/):

```bash
uv add edmkit
```

## Testing

```bash
# Run all tests
uv run pytest

# Skip slow tests
uv run pytest -m "not slow"

# Run with CI profile (more hypothesis examples)
HYPOTHESIS_PROFILE=ci uv run pytest
```
