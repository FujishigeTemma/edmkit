# edmkit

This library is a collection of tools and utilities that are useful for Empirical Data Modeling (EDM) and related tasks. The library is designed to be fast and lightweight, and easy to use.

[benchmarks](./benchmarks)

![](./benchmarks/benchmarks.png)

[compare.py](./benchmarks/compare.py)

![](./benchmarks/compare.png)

| simplex N | edmkit [ms] | pyEDM [ms] | speedup |
|---:|---:|---:|---:|
| 500 | 0.2 | 1.5 | 7x |
| 1,000 | 0.5 | 2.4 | 5x |
| 2,000 | 1.0 | 4.4 | 5x |
| 5,000 | 2.4 | 11.4 | 5x |
| 10,000 | 5.2 | 22.8 | 4x |
| 20,000 | 10.9 | 49.2 | 5x |

| smap N | edmkit [ms] | pyEDM [ms] | speedup |
|---:|---:|---:|---:|
| 500 | 2.0 | 38.8 | 20x |
| 1,000 | 7.4 | 132.8 | 18x |
| 2,000 | 21.5 | 522.0 | 24x |
| 5,000 | 183.2 | 3,430 | 19x |

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
