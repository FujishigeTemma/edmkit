# edmkit

This library is a collection of tools and utilities that are useful for Empirical Data Modeling (EDM) and related tasks. The library is designed to be fast and lightweight, and easy to use.

[benchmarks](./benchmarks)

![](./benchmarks/benchmarks.png)

[compare.py](./benchmarks/compare.py)

![](./benchmarks/compare.png)

Measured on an Apple M4 Pro on November 4, 2026. Times are medians from `pytest-benchmark`. Parallel CCM uses 14 workers. Peak memory is process-tree RSS in MiB, sampled separately from timing for each case. It includes the interpreter, input, and worker processes. Summed RSS can count shared pages in more than one process.

| simplex N | edmkit [ms] | pyEDM [ms] | speedup | edmkit peak [MiB] | pyEDM peak [MiB] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.17 | 1.5 | 9.0× | 168 | 169 |
| 1,000 | 0.39 | 2.4 | 6.1× | 169 | 170 |
| 2,000 | 0.81 | 4.4 | 5.5× | 168 | 170 |
| 5,000 | 2.2 | 11.5 | 5.3× | 170 | 175 |
| 10,000 | 4.7 | 23.8 | 5.0× | 172 | 180 |
| 20,000 | 10.2 | 50.3 | 4.9× | 176 | 193 |

| smap N | edmkit [ms] | pyEDM [ms] | speedup | edmkit peak [MiB] | pyEDM peak [MiB] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 2.1 | 36.6 | 17.9× | 176 | 180 |
| 1,000 | 8.7 | 128.3 | 14.7× | 192 | 216 |
| 2,000 | 22.0 | 513.6 | 23.4× | 261 | 357 |
| 5,000 | 193.9 | 3,376.7 | 17.4× | 744 | 1,344 |

| ccm N | edmkit [ms] | pyEDM [ms] | speedup | edmkit parallel [ms] | pyEDM parallel [ms] | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 106.4 | 151.7 | 1.4× | 19.5 | 68.0 | 3.5× |
| 1,000 | 168.4 | 327.0 | 1.9× | 26.3 | 91.9 | 3.5× |
| 2,000 | 329.7 | 745.1 | 2.3× | 48.8 | 148.3 | 3.0× |
| 5,000 | 668.9 | 1,945.7 | 2.9× | 97.4 | 397.5 | 4.1× |
| 10,000 | 1,253.6 | 4,176.7 | 3.3× | 164.6 | 585.4 | 3.6× |

| ccm N | edmkit peak [MiB] | pyEDM peak [MiB] | edmkit parallel peak [MiB] | pyEDM parallel peak [MiB] |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 171 | 170 | 191 | 1,417 |
| 1,000 | 173 | 176 | 196 | 1,429 |
| 2,000 | 179 | 186 | 207 | 1,460 |
| 5,000 | 204 | 224 | 239 | 1,624 |
| 10,000 | 230 | 362 | 297 | 1,804 |


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
