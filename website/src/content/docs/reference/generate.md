---
title: generate
description: Synthetic chaotic time series generators.
sidebar:
  order: 7
---

**Functions:**

Name | Description
---- | -----------
[`double_pendulum`](#double_pendulum) | Generate double pendulum dynamics via forward Euler integration.
[`to_xy`](#to_xy) | Convert double pendulum angles to Cartesian coordinates.
[`lorenz`](#lorenz) | Generate a Lorenz system trajectory via forward Euler integration.
[`mackey_glass`](#mackey_glass) | Generate a Mackey-Glass chaotic time series via forward Euler integration.

## `double_pendulum`

```python
double_pendulum(m1: float, m2: float, L1: float, L2: float, g: float, X0: np.ndarray, dt: float, t_max: int)
```

Generate double pendulum dynamics via forward Euler integration.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m1` | <code>[float](#float)</code> | Mass of first pendulum. | *required*
`m2` | <code>[float](#float)</code> | Mass of second pendulum. | *required*
`L1` | <code>[float](#float)</code> | Length of first pendulum. | *required*
`L2` | <code>[float](#float)</code> | Length of second pendulum. | *required*
`g` | <code>[float](#float)</code> | Gravitational acceleration. | *required*
`X0` | <code>[ndarray](#numpy.ndarray)</code> | Initial state ``(theta1, theta2, omega1, omega2)`` of shape ``(4,)``. | *required*
`dt` | <code>[float](#float)</code> | Integration time step. | *required*
`t_max` | <code>[int](#int)</code> | Maximum time. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`t` | <code>[ndarray](#numpy.ndarray)</code> | Time array.
`X` | <code>[ndarray](#numpy.ndarray)</code> | State trajectory of shape ``(N, 4)``.



## `to_xy`

```python
to_xy(L1: float, L2: float, theta1: np.ndarray, theta2: np.ndarray)
```

Convert double pendulum angles to Cartesian coordinates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`L1` | <code>[float](#float)</code> | Length of first pendulum. | *required*
`L2` | <code>[float](#float)</code> | Length of second pendulum. | *required*
`theta1` | <code>[ndarray](#numpy.ndarray)</code> | Angle of first pendulum. | *required*
`theta2` | <code>[ndarray](#numpy.ndarray)</code> | Angle of second pendulum. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`x1` | <code>[ndarray](#numpy.ndarray)</code> | x-coordinate of first pendulum.
`y1` | <code>[ndarray](#numpy.ndarray)</code> | y-coordinate of first pendulum.
`x2` | <code>[ndarray](#numpy.ndarray)</code> | x-coordinate of second pendulum.
`y2` | <code>[ndarray](#numpy.ndarray)</code> | y-coordinate of second pendulum.



## `lorenz`

```python
lorenz(sigma: float, rho: float, beta: float, X0: np.ndarray, dt: float, t_max: int)
```

Generate a Lorenz system trajectory via forward Euler integration.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sigma` | <code>[float](#float)</code> | Prandtl number (typical: 10). | *required*
`rho` | <code>[float](#float)</code> | Rayleigh number (typical: 28). | *required*
`beta` | <code>[float](#float)</code> | Geometric factor (typical: 8/3). | *required*
`X0` | <code>[ndarray](#numpy.ndarray)</code> | Initial condition of shape ``(3,)``. | *required*
`dt` | <code>[float](#float)</code> | Integration time step. | *required*
`t_max` | <code>[int](#int)</code> | Maximum time. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`t` | <code>[ndarray](#numpy.ndarray)</code> | Time array.
`X` | <code>[ndarray](#numpy.ndarray)</code> | Trajectory of shape ``(N, 3)`` for ``(x, y, z)``.



## `mackey_glass`

```python
mackey_glass(tau: float, n: int, beta: float, gamma: float, x0: float, dt: float, t_max: int)
```

Generate a Mackey-Glass chaotic time series via forward Euler integration.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tau` | <code>[float](#float)</code> | Delay parameter (typical: 17 for chaos). | *required*
`n` | <code>[int](#int)</code> | Nonlinearity exponent (typical: 10). | *required*
`beta` | <code>[float](#float)</code> | Feedback strength (typical: 0.2). | *required*
`gamma` | <code>[float](#float)</code> | Decay rate (typical: 0.1). | *required*
`x0` | <code>[float](#float)</code> | Initial condition. | *required*
`dt` | <code>[float](#float)</code> | Integration time step. | *required*
`t_max` | <code>[int](#int)</code> | Maximum time. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`t` | <code>[ndarray](#numpy.ndarray)</code> | Time array.
`x` | <code>[ndarray](#numpy.ndarray)</code> | 1D time series.

