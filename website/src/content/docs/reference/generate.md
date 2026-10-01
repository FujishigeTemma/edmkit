---
title: generate
description: Synthetic chaotic time series generators.
sidebar:
  order: 8
---

**Functions:**

Name | Description
---- | -----------
[`ctrnn`](#ctrnn) | Vector field of a continuous-time recurrent neural network ``dx/dt = -x + g J tanh(x)``.
[`double_pendulum`](#double_pendulum) | Vector field of the double pendulum.
[`duffing`](#duffing) | Vector field of the forced Duffing oscillator.
[`lorenz`](#lorenz) | Vector field of the Lorenz system.
[`lorenz96`](#lorenz96) | Vector field of the Lorenz 96 system.
[`mackey_glass`](#mackey_glass) | Generate a Mackey-Glass chaotic time series via forward Euler integration.
[`rk45`](#rk45) | Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.
[`rossler`](#rossler) | Vector field of the Rössler system.
[`to_xy`](#to_xy) | Convert double pendulum angles to Cartesian coordinates.

## `ctrnn`

```python
ctrnn(g: float, J: np.ndarray)
```

Vector field of a continuous-time recurrent neural network ``dx/dt = -x + g J tanh(x)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`g` | <code>[float](#float)</code> | Gain (chaos for ``g > 1`` when ``J`` is large and random with variance ``1 / N``). | *required*
`J` | <code>[ndarray](#numpy.ndarray)</code> | Connectivity of shape ``(N, N)``; ``J[i, j]`` is the weight with which unit ``j`` drives unit ``i``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(N,)``.



## `double_pendulum`

```python
double_pendulum(m1: float, m2: float, L1: float, L2: float, g: float)
```

Vector field of the double pendulum.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m1` | <code>[float](#float)</code> | Mass of first pendulum. | *required*
`m2` | <code>[float](#float)</code> | Mass of second pendulum. | *required*
`L1` | <code>[float](#float)</code> | Length of first pendulum. | *required*
`L2` | <code>[float](#float)</code> | Length of second pendulum. | *required*
`g` | <code>[float](#float)</code> | Gravitational acceleration. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(theta1, theta2, omega1, omega2)`` of shape ``(4,)``.



## `duffing`

```python
duffing(alpha: float, beta: float, delta: float, gamma: float, omega: float)
```

Vector field of the forced Duffing oscillator.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>[float](#float)</code> | Linear stiffness (typical: -1). | *required*
`beta` | <code>[float](#float)</code> | Cubic stiffness (typical: 1). | *required*
`delta` | <code>[float](#float)</code> | Damping coefficient (typical: 0.3). | *required*
`gamma` | <code>[float](#float)</code> | Forcing amplitude (typical: 0.5 for chaos). | *required*
`omega` | <code>[float](#float)</code> | Forcing angular frequency (typical: 1.2). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, v)`` of shape ``(2,)``.



## `lorenz`

```python
lorenz(sigma: float, rho: float, beta: float)
```

Vector field of the Lorenz system.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sigma` | <code>[float](#float)</code> | Prandtl number (typical: 10). | *required*
`rho` | <code>[float](#float)</code> | Rayleigh number (typical: 28). | *required*
`beta` | <code>[float](#float)</code> | Geometric factor (typical: 8/3). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``.



## `lorenz96`

```python
lorenz96(F: float)
```

Vector field of the Lorenz 96 system.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`F` | <code>[float](#float)</code> | Forcing constant (typical: 8 for chaos). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(D,)`` with ``D >= 4``.



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



## `rk45`

```python
rk45(f: Callable[[float, np.ndarray], np.ndarray], X0: np.ndarray, dt: float, t_max: int, rtol: float = 1e-06, atol: float = 1e-09)
```

Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.

The step size adapts between samples, while the output stays on the uniform grid of spacing ``dt``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`f` | <code>[Callable](#collections.abc.Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side of the system. | *required*
`X0` | <code>[ndarray](#numpy.ndarray)</code> | Initial condition of shape ``(D,)``. | *required*
`dt` | <code>[float](#float)</code> | Sampling time step. | *required*
`t_max` | <code>[int](#int)</code> | Maximum time. | *required*
`rtol` | <code>[float](#float)</code> | Relative tolerance of the local error. | <code>1e-06</code>
`atol` | <code>[float](#float)</code> | Absolute tolerance of the local error. | <code>1e-09</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`t` | <code>[ndarray](#numpy.ndarray)</code> | Time array.
`X` | <code>[ndarray](#numpy.ndarray)</code> | Trajectory of shape ``(N, D)``.



## `rossler`

```python
rossler(a: float, b: float, c: float)
```

Vector field of the Rössler system.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`a` | <code>[float](#float)</code> | Spiral growth rate (typical: 0.2). | *required*
`b` | <code>[float](#float)</code> | Constant injection into z (typical: 0.2). | *required*
`c` | <code>[float](#float)</code> | Folding threshold (typical: 5.7). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``.



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

