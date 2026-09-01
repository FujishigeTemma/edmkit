---
title: metrics
description: Prediction evaluation metrics.
sidebar:
  order: 6
---

**Functions:**

Name | Description
---- | -----------
[`MetricFunc`](#MetricFunc) | Metric function protocol.
[`pearson_correlation`](#pearson_correlation) | Mean Pearson correlation over the target dimensions.
[`rmse`](#rmse) | Root Mean Squared Error.
[`mae`](#mae) | Mean Absolute Error.

## `MetricFunc`

Bases: <code>[Protocol](#typing.Protocol)</code>

Metric function protocol.

Accepts predictions and observations of the same shape and returns a metric value.



## `pearson_correlation`

```python
pearson_correlation(predictions, observations)
```

Mean Pearson correlation over the target dimensions.

The correlation is computed per target dimension along the sample axis,
then averaged over the dimensions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | Same shape as `predictions`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `predictions` and `observations` have different shapes. - If the inputs are not 1D, 2D, or 3D.



## `rmse`

```python
rmse(predictions, observations)
```

Root Mean Squared Error.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | Same shape as `predictions`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `predictions` and `observations` have different shapes. - If the inputs are not 1D, 2D, or 3D.



## `mae`

```python
mae(predictions, observations)
```

Mean Absolute Error.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | Same shape as `predictions`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) or [Tensor](#tinygrad.Tensor)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `predictions` and `observations` have different shapes. - If the inputs are not 1D, 2D, or 3D.

