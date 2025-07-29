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

## Usage

Most of the functions accept and return `numpy` arrays or `edmkit.Tensor`(alias to `tinygrad.Tensor`).
