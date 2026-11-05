---
title: util
description: Utility functions for distance computation, padding, and more.
sidebar:
  order: 10
---

## `util`

**Functions:**

Name | Description
---- | -----------
[`materialize`](#edmkit.util.materialize) | Evaluate `mask` on every index pair, giving the dense boolean array of `shape`, such as (M, N) or (B, M, N).
[`pad`](#edmkit.util.pad) | Pad the `np.ndarray` in `Xs` to merge them into a single `np.ndarray`.
[`pairwise_distance`](#edmkit.util.pairwise_distance) | Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).
[`topk`](#edmkit.util.topk) | Computes the top-k elements of the tensor along the specified `dim`.
[`pairwise_distance_np`](#edmkit.util.pairwise_distance_np) | Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).
[`dtw`](#edmkit.util.dtw) | Computes the Dynamic Time Warping (DTW) distance between two sequences `x` and `y`.
[`autocorrelation`](#edmkit.util.autocorrelation) | Computes the autocorrelation of a given 1D numpy array at lags below a specified maximum lag.

### `materialize`

```python
materialize(mask: Mask, shape: Sequence[SupportsInt]) -> np.ndarray
```

Evaluate `mask` on every index pair, giving the dense boolean array of `shape`, such as (M, N) or (B, M, N).

### `pad`

```python
pad(As: list[np.ndarray])
```

Pad the `np.ndarray` in `Xs` to merge them into a single `np.ndarray`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`As` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of arrays of shape ``(N, D_i)``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Single array of shape ``(B, N, max(D))`` where B is ``len(As)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If any array in `As` is not 2D. - If the first dimension of all arrays in `As` are not equal.

### `pairwise_distance`

```python
pairwise_distance(A: Tensor, B: Tensor | None = None) -> Tensor
```

Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[Tensor](#tinygrad.Tensor)</code> | Shape ``(N, D)`` or ``(B, N, D)``. ``B`` is batch size, ``N`` is number of points, ``D`` is dimension of each point. | *required*
`B` | <code>[Tensor](#tinygrad.Tensor)</code> | Shape ``(M, D)`` or ``(B, M, D)``. ``B`` is batch size, ``M`` is number of points, ``D`` is dimension of each point. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[Tensor](#tinygrad.Tensor)</code> | When `A` is of shape ``(N, D)``: shape ``(N, N)`` [or ``(N, M)``] where the element at position ``(i, j)`` is the squared Euclidean distance between ``A[i]`` and ``A[j]`` [or between ``A[i]`` and ``B[j]``]. When `A` is of shape ``(B, N, D)``: shape ``(B, N, N)`` [or ``(B, N, M)``] where the element at position ``(b, i, j)`` is the squared Euclidean distance between ``A[b, i]`` and ``A[b, j]``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `A` is not a 2D or 3D tensor. - If `B` is not `None` and `A` and `B` have different number of dimensions.

### `topk`

```python
topk(A: Tensor, k: int, dim: int = -1, largest: bool = True, sorted_: bool = True) -> tuple[Tensor, Tensor]
```

Computes the top-k elements of the tensor along the specified `dim`.

Order of indices for equivalent elements is always preserved.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[Tensor](#tinygrad.Tensor)</code> | Input tensor. | *required*
`k` | <code>[int](#int)</code> | The number of elements to take, at most ``A.shape[dim]``. | *required*
`dim` | <code>[int](#int)</code> | The dimension along which to take the elements. | <code>-1</code>
`largest` | <code>[bool](#bool)</code> | Take the largest elements if True, otherwise the smallest. | <code>True</code>
`sorted_` | <code>[bool](#bool)</code> | Must be True; the elements are returned in order, as `Tensor.topk` requires. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`values` | <code>[Tensor](#tinygrad.Tensor)</code> | The top-k elements, ordered from the most extreme, with ``A.shape[dim]`` replaced by `k`.
`indices` | <code>[Tensor](#tinygrad.Tensor)</code> | Their indices along `dim`, of the same shape as `values`. Equal elements keep their order.

**Raises:**

Type | Description
---- | -----------
<code>[NotImplementedError](#NotImplementedError)</code> | - If `sorted_` is False.
<code>[ValueError](#ValueError)</code> | - If `k` exceeds ``A.shape[dim]``.

### `pairwise_distance_np`

```python
pairwise_distance_np(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray
```

Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[ndarray](#numpy.ndarray)</code> | Shape ``(N, D)`` or ``(B, N, D)``. ``B`` is batch size, ``N`` is number of points, ``D`` is dimension of each point. | *required*
`B` | <code>[ndarray](#numpy.ndarray)</code> | Shape ``(M, D)`` or ``(B, M, D)``. ``B`` is batch size, ``M`` is number of points, ``D`` is dimension of each point. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | When `A` is of shape ``(N, D)``: shape ``(N, N)`` [or ``(N, M)``] where the element at position ``(i, j)`` is the squared Euclidean distance between ``A[i]`` and ``A[j]`` [or between ``A[i]`` and ``B[j]``]. When `A` is of shape ``(B, N, D)``: shape ``(B, N, N)`` [or ``(B, N, M)``] where the element at position ``(b, i, j)`` is the squared Euclidean distance between ``A[b, i]`` and ``A[b, j]``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `A` is not a 2D or 3D array. - If `B` is not `None` and `A` and `B` have different number of dimensions.

### `dtw`

```python
dtw(A: np.ndarray, B: np.ndarray)
```

Computes the Dynamic Time Warping (DTW) distance between two sequences `x` and `y`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[ndarray](#numpy.ndarray)</code> | Sequence of shape ``(N, D)``. | *required*
`B` | <code>[ndarray](#numpy.ndarray)</code> | Sequence of shape ``(M, D)``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`distance` | <code>[float](#float)</code> | The DTW distance between the two sequences.

### `autocorrelation`

```python
autocorrelation(x: np.ndarray, max_lag: int, step: int = 1)
```

Computes the autocorrelation of a given 1D numpy array at lags below a specified maximum lag.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#numpy.ndarray)</code> | The input array for which to compute the autocorrelation. | *required*
`max_lag` | <code>[int](#int)</code> | Exclusive upper bound of the lags, also capped at ``len(x)``. | *required*
`step` | <code>[int](#int)</code> | The step size for the lag. Default is 1. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Autocorrelation at lags ``range(0, min(max_lag, len(x)), step)``.

