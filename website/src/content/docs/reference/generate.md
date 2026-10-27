---
title: generate
description: Synthetic chaotic time series generators.
sidebar:
  order: 8
---

**Functions:**

Name | Description
---- | -----------
[`chua`](#chua) | Vector field of Chua's circuit, with the piecewise-linear diode ``h(x) = m1 x + (m0 - m1) (|x + 1| - |x - 1|) / 2``.
[`ctrnn`](#ctrnn) | Vector field of a continuous-time recurrent neural network ``dx/dt = -x + g J tanh(x)``.
[`double_pendulum`](#double_pendulum) | Vector field of the double pendulum.
[`duffing`](#duffing) | Vector field of the forced Duffing oscillator.
[`euler_spiral`](#euler_spiral) | Vector field of the Euler spiral (clothoid).
[`fitzhugh_nagumo`](#fitzhugh_nagumo) | Vector field of the FitzHugh-Nagumo neuron model.
[`hindmarsh_rose`](#hindmarsh_rose) | Vector field of the Hindmarsh-Rose neuron model.
[`kuramoto`](#kuramoto) | Vector field of the Kuramoto model ``dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)``.
[`lorenz`](#lorenz) | Vector field of the Lorenz system.
[`lorenz96`](#lorenz96) | Vector field of the Lorenz 96 system.
[`lotka_volterra`](#lotka_volterra) | Vector field of the generalized Lotka-Volterra system ``dx/dt = x (sigma - rho x) + S``.
[`mackey_glass`](#mackey_glass) | Generate a Mackey-Glass chaotic time series via forward Euler integration.
[`morris_lecar`](#morris_lecar) | Vector field of the Morris-Lecar neuron model.
[`pyloric`](#pyloric) | Vector field of the three-neuron pyloric circuit of the crustacean stomatogastric ganglion (Prinz, Bucher & Marder 2004), in ms and mV.
[`rk45`](#rk45) | Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.
[`rossler`](#rossler) | Vector field of the Rössler system.
[`rulkov_map`](#rulkov_map) | Generate a time series of globally coupled Rulkov map neurons.
[`three_scroll`](#three_scroll) | Vector field of Li's three-scroll system.
[`to_xy`](#to_xy) | Convert double pendulum angles to Cartesian coordinates.

## `chua`

```python
chua(alpha: float, beta: float, m0: float, m1: float)
```

Vector field of Chua's circuit, with the piecewise-linear diode ``h(x) = m1 x + (m0 - m1) (|x + 1| - |x - 1|) / 2``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>[float](#float)</code> | Ratio of capacitances (typical: 15.6). | *required*
`beta` | <code>[float](#float)</code> | Ratio of inductance to resistance (typical: 28). | *required*
`m0` | <code>[float](#float)</code> | Diode slope for ``|x| < 1`` (typical: -8/7). | *required*
`m1` | <code>[float](#float)</code> | Diode slope for ``|x| > 1`` (typical: -5/7). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(N,)``, or ``(B, N)`` for a batch.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(theta1, theta2, omega1, omega2)`` of shape ``(4,)``, or ``(B, 4)`` for a batch.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, v)`` of shape ``(2,)``, or ``(B, 2)`` for a batch.



## `euler_spiral`

```python
euler_spiral(a: float)
```

Vector field of the Euler spiral (clothoid).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`a` | <code>[float](#float)</code> | Rate at which the curvature grows (typical: 1). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, theta)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.



## `fitzhugh_nagumo`

```python
fitzhugh_nagumo(a: float, b: float, epsilon: float, I: float)
```

Vector field of the FitzHugh-Nagumo neuron model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`a` | <code>[float](#float)</code> | Offset of the recovery variable (typical: 0.7). | *required*
`b` | <code>[float](#float)</code> | Damping of the recovery variable (typical: 0.8). | *required*
`epsilon` | <code>[float](#float)</code> | Time scale of the recovery variable (typical: 0.08). | *required*
`I` | <code>[float](#float)</code> | Applied current (typical: 0.5 for spiking). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(v, w)`` of shape ``(2,)``, or ``(B, 2)`` for a batch.



## `hindmarsh_rose`

```python
hindmarsh_rose(a: float, b: float, c: float, d: float, r: float, s: float, x_R: float, I: float)
```

Vector field of the Hindmarsh-Rose neuron model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`a` | <code>[float](#float)</code> | Cubic coefficient of the membrane potential (typical: 1). | *required*
`b` | <code>[float](#float)</code> | Quadratic coefficient of the membrane potential (typical: 3). | *required*
`c` | <code>[float](#float)</code> | Constant drive of the recovery variable (typical: 1). | *required*
`d` | <code>[float](#float)</code> | Quadratic coupling of the potential into the recovery variable (typical: 5). | *required*
`r` | <code>[float](#float)</code> | Time scale of the slow adaptation current (typical: 0.006). | *required*
`s` | <code>[float](#float)</code> | Adaptation strength (typical: 4). | *required*
`x_R` | <code>[float](#float)</code> | Resting potential (typical: -1.6). | *required*
`I` | <code>[float](#float)</code> | Applied current (typical: 3.25 for chaotic bursting). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch: membrane potential, recovery variable and adaptation current.



## `kuramoto`

```python
kuramoto(omega: np.ndarray, K: np.ndarray)
```

Vector field of the Kuramoto model ``dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`omega` | <code>[ndarray](#numpy.ndarray)</code> | Natural frequencies of shape ``(N,)``. | *required*
`K` | <code>[ndarray](#numpy.ndarray)</code> | Coupling of shape ``(N, N)``; ``K[i, j]`` is the strength with which oscillator ``j`` drives oscillator ``i`` (typical: ``k / N`` everywhere). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for unwrapped phases of shape ``(N,)``, or ``(B, N)`` for a batch.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(D,)``, or ``(B, D)`` for a batch, with ``D >= 4``.



## `lotka_volterra`

```python
lotka_volterra(rho: np.ndarray, sigma: float | np.ndarray, S: float | np.ndarray)
```

Vector field of the generalized Lotka-Volterra system ``dx/dt = x (sigma - rho x) + S``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`rho` | <code>[ndarray](#numpy.ndarray)</code> | Interaction matrix of shape ``(N, N)``; ``rho[i, j]`` is the effect of species ``j`` on species ``i``. | *required*
`sigma` | <code>[float](#float) \| [ndarray](#numpy.ndarray)</code> | Growth rate, scalar or of shape ``(N,)`` (typical: 1). | *required*
`S` | <code>[float](#float) \| [ndarray](#numpy.ndarray)</code> | Constant input, scalar or of shape ``(N,)`` (typical: 0; small positive values keep winnerless competition switching). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(N,)``, or ``(B, N)`` for a batch.



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



## `morris_lecar`

```python
morris_lecar(C: float, g_L: float, g_Ca: float, g_K: float, V_L: float, V_Ca: float, V_K: float, V1: float, V2: float, V3: float, V4: float, phi: float, I: float)
```

Vector field of the Morris-Lecar neuron model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`C` | <code>[float](#float)</code> | Membrane capacitance (typical: 20). | *required*
`g_L` | <code>[float](#float)</code> | Leak conductance (typical: 2). | *required*
`g_Ca` | <code>[float](#float)</code> | Calcium conductance (typical: 4.4). | *required*
`g_K` | <code>[float](#float)</code> | Potassium conductance (typical: 8). | *required*
`V_L` | <code>[float](#float)</code> | Leak reversal potential (typical: -60). | *required*
`V_Ca` | <code>[float](#float)</code> | Calcium reversal potential (typical: 120). | *required*
`V_K` | <code>[float](#float)</code> | Potassium reversal potential (typical: -84). | *required*
`V1` | <code>[float](#float)</code> | Half-activation voltage of the calcium channel (typical: -1.2). | *required*
`V2` | <code>[float](#float)</code> | Slope of the calcium activation (typical: 18). | *required*
`V3` | <code>[float](#float)</code> | Half-activation voltage of the potassium channel (typical: 2). | *required*
`V4` | <code>[float](#float)</code> | Slope of the potassium activation (typical: 30). | *required*
`phi` | <code>[float](#float)</code> | Rate of the potassium activation (typical: 0.04). | *required*
`I` | <code>[float](#float)</code> | Applied current (typical: 100 for spiking). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(V, n)`` of shape ``(2,)``, or ``(B, 2)`` for a batch: membrane potential and potassium activation.



## `pyloric`

```python
pyloric(g_mem: np.ndarray, g_glu: np.ndarray, g_ach: np.ndarray)
```

Vector field of the three-neuron pyloric circuit of the crustacean stomatogastric ganglion (Prinz, Bucher & Marder 2004), in ms and mV.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`g_mem` | <code>[ndarray](#numpy.ndarray)</code> | Maximal conductances in mS/cm^2 of shape ``(3, 8)``: rows are AB/PD, LP, PY and columns are Na, CaT, CaS, A, KCa, Kd, H, leak (typical: ``[[400, 2.5, 6, 50, 10, 100, 0.01, 0], [100, 0, 8, 40, 5, 75, 0.05, 0.02], [100, 2.5, 2, 50, 0, 125, 0.05, 0.01]]``). | *required*
`g_glu` | <code>[ndarray](#numpy.ndarray)</code> | Fast glutamatergic synapse strengths in nS of shape ``(3, 3)``; ``g_glu[i, j]`` is the synapse from neuron ``j`` onto neuron ``i`` (typical: ``g_glu[1, 0] = 30`` (AB to LP), ``g_glu[2, 0] = 3`` (AB to PY), ``g_glu[0, 1] = 10`` (LP to AB/PD), ``g_glu[2, 1] = 3`` (LP to PY), ``g_glu[1, 2] = 3`` (PY to LP), the rest 0, for a PD, LP, PY rhythm of period about 1.7 s). | *required*
`g_ach` | <code>[ndarray](#numpy.ndarray)</code> | Slow cholinergic synapse strengths in nS of shape ``(3, 3)``, indexed like ``g_glu`` (typical: ``g_ach[1, 0] = 3`` (PD to LP), ``g_ach[2, 0] = 30`` (PD to PY), the rest 0). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state of shape ``(45,)``, or ``(B, 45)`` for a batch; ``x.reshape(15, 3)`` has one column per neuron and rows: membrane potential, intracellular [Ca] in uM, the gates m and h of Na, m and h of CaT, m and h of CaS, m and h of A, m of KCa, n of Kd, m of H, and the glutamatergic and cholinergic synaptic activations of the neuron as presynaptic cell (typical initial state: V = -50, [Ca] = 0.05, all gates and synaptic activations 0, discard the first 3 s as transient).



## `rk45`

```python
rk45(f: Callable[[float, np.ndarray], np.ndarray], X0: np.ndarray, dt: float, t_max: float, rtol: float = 1e-06, atol: float = 1e-09)
```

Integrate ``dx/dt = f(t, x)`` via the adaptive Dormand-Prince RK45 method.

Steps are chosen by the error alone, and the uniform grid of spacing ``dt`` is filled in by the method's dense output.
A batch of initial conditions is integrated together, with steps that keep every member within the tolerance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`f` | <code>[Callable](#collections.abc.Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side of the system, called with states of the shape of ``X0``. | *required*
`X0` | <code>[ndarray](#numpy.ndarray)</code> | Initial condition of shape ``(D,)``, or ``(B, D)`` for a batch. | *required*
`dt` | <code>[float](#float)</code> | Sampling time step. | *required*
`t_max` | <code>[float](#float)</code> | End time, excluded from the grid. | *required*
`rtol` | <code>[float](#float)</code> | Relative tolerance of the local error. | <code>1e-06</code>
`atol` | <code>[float](#float)</code> | Absolute tolerance of the local error. | <code>1e-09</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`t` | <code>[ndarray](#numpy.ndarray)</code> | Time array of shape ``(N,)``.
`X` | <code>[ndarray](#numpy.ndarray)</code> | Trajectory of shape ``(N, D)``, or ``(B, N, D)`` for a batch.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ``X0`` is not of shape ``(D,)`` or ``(B, D)``.



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
`f` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``f(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.



## `rulkov_map`

```python
rulkov_map(alpha: float, mu: float, sigma: float, epsilon: float, X0: np.ndarray, n_max: int)
```

Generate a time series of globally coupled Rulkov map neurons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>[float](#float)</code> | Nonlinearity of the fast variable (typical: 4.1 for chaotic bursting). | *required*
`mu` | <code>[float](#float)</code> | Time scale of the slow variable (typical: 0.001). | *required*
`sigma` | <code>[float](#float)</code> | Drive of the slow variable (typical: -1.5). | *required*
`epsilon` | <code>[float](#float)</code> | Coupling through the mean of the fast variables (typical: 0). | *required*
`X0` | <code>[ndarray](#numpy.ndarray)</code> | Initial condition of shape ``(N, 2)``, one ``(x, y)`` per unit. | *required*
`n_max` | <code>[int](#int)</code> | Number of steps. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`n` | <code>[ndarray](#numpy.ndarray)</code> | Step array.
`X` | <code>[ndarray](#numpy.ndarray)</code> | Time series of shape ``(n_max, N, 2)``.



## `three_scroll`

```python
three_scroll(a: float, k: float, c: float, d: float, e: float, f: float)
```

Vector field of Li's three-scroll system.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`a` | <code>[float](#float)</code> | Coupling of ``x`` towards ``y`` (typical: 40). | *required*
`k` | <code>[float](#float)</code> | Gain of ``x`` in ``dy/dt`` (typical: 55). | *required*
`c` | <code>[float](#float)</code> | Gain of ``z`` in ``dz/dt`` (typical: 1.9 for chaos; Li's 11/6 gives a limit cycle). | *required*
`d` | <code>[float](#float)</code> | Coefficient of ``x z`` in ``dx/dt`` (typical: 0.16). | *required*
`e` | <code>[float](#float)</code> | Coefficient of ``x^2`` in ``dz/dt`` (typical: 0.65). | *required*
`f` | <code>[float](#float)</code> | Gain of ``y`` in ``dy/dt`` (typical: 20). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`rhs` | <code>[Callable](#Callable)[[[float](#float), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)]</code> | Right-hand side ``rhs(t, x)`` for a state ``(x, y, z)`` of shape ``(3,)``, or ``(B, 3)`` for a batch.



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

