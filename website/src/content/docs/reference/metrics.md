---
title: metrics
description: Prediction evaluation metrics.
sidebar:
  order: 5
---

## `metrics`

**Functions:**

Name | Description
---- | -----------
[`validate_and_promote`](#edmkit.metrics.validate_and_promote) | Validate shape match and promote 1D to 2D.
[`rhos`](#edmkit.metrics.rhos) | Pearson correlation per dimension.
[`mean_rho`](#edmkit.metrics.mean_rho) | Mean Pearson correlation.
[`rmse`](#edmkit.metrics.rmse) | Root Mean Squared Error.
[`mae`](#edmkit.metrics.mae) | Mean Absolute Error.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MetricFunc`](#edmkit.metrics.MetricFunc) | <code>[TypeAlias](#typing.TypeAlias)</code> | MetricFunc is a function that takes (predictions, observations) and returns a metric value.

### `MetricFunc`

```python
MetricFunc: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
```

MetricFunc is a function that takes (predictions, observations) and returns a metric value.

### `validate_and_promote`

```python
validate_and_promote(predictions: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]
```

Validate shape match and promote 1D to 2D.

### `rhos`

```python
rhos(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray
```

Pearson correlation per dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray)</code> | Same shape as predictions. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | ``(1,)`` for 1D input, ``(D,)`` for 2D, ``(B, D)`` for 3D.

### `mean_rho`

```python
mean_rho(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray
```

Mean Pearson correlation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray)</code> | Same shape as predictions. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

### `rmse`

```python
rmse(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray
```

Root Mean Squared Error.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray)</code> | Same shape as *predictions*. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

### `mae`

```python
mae(predictions: np.ndarray, observations: np.ndarray) -> np.ndarray
```

Mean Absolute Error.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predictions` | <code>[ndarray](#numpy.ndarray)</code> | ``(N,)``, ``(N, D)``, or ``(B, N, D)``. | *required*
`observations` | <code>[ndarray](#numpy.ndarray)</code> | Same shape as predictions. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | ``()`` for 1D/2D input, ``(B,)`` for 3D input.

