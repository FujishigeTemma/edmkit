---
title: theiler_window
description: Per-query masks that exclude temporally close library points.
sidebar:
  order: 4
---

## `theiler_window`

**Functions:**

Name | Description
---- | -----------
[`theiler_window`](#edmkit.theiler_window.theiler_window) | Build a per-query mask that excludes temporally close library points.

### `theiler_window`

```python
theiler_window(t1: np.ndarray, t2: np.ndarray, width: int) -> Mask
```

Build a per-query mask that excludes temporally close library points.

Passing the result as `mask` to `simplex_projection`, `soft_simplex_projection`, or `smap`
gives leave-one-out prediction with Theiler window exclusion when `Q` is `X` and `t1` is `t2`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t1` | <code>[ndarray](#numpy.ndarray)</code> | Time indices of the query points, shape (M,) or (B, M). | *required*
`t2` | <code>[ndarray](#numpy.ndarray)</code> | Time indices of the library points, shape (N,) or (B, N). | *required*
`width` | <code>[int](#int)</code> | Theiler window half-width. Library points ``j`` where ``|t1[i] - t2[j]| <= width`` are excluded when predicting query ``i``. For lagged embedding, use ``(E - 1) * tau``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mask` | <code>[Mask](#edmkit.types.Mask)</code> | ``mask(i, j)``, or ``mask(b, i, j)``, True where the library point lies outside the window.

