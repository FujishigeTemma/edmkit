---
title: wavelet
description: Continuous and discrete wavelet transforms.
sidebar:
  order: 9
---

**Functions:**

Name | Description
---- | -----------
[`ContinuousWavelet`](#ContinuousWavelet) | Mother wavelet of `cwt`, described in the frequency domain.
[`DiscreteWavelet`](#DiscreteWavelet) | Orthonormal wavelet of `modwt` and `mra`, described by its scaling filter.
[`cwt`](#cwt) | Continuous wavelet transform of `x` at the given frequencies.
[`daubechies`](#daubechies) | Daubechies extremal phase wavelet with `p` vanishing moments.
[`dog`](#dog) | Derivative of Gaussian wavelet of order `m`, a real wavelet whose coefficients peak at the extrema of the signal.
[`icwt`](#icwt) | Inverse continuous wavelet transform, summing the coefficients of `cwt` over frequency (Torrence & Compo 1998).
[`imodwt`](#imodwt) | Inverse maximal overlap discrete wavelet transform (Percival & Walden 2000).
[`modwt`](#modwt) | Maximal overlap discrete wavelet transform of `x` (Percival & Walden 2000).
[`morlet`](#morlet) | Complex Morlet wavelet, an analytic wavelet made of a complex exponential under a Gaussian envelope.
[`mra`](#mra) | Multiresolution analysis of `x` by the MODWT (Percival & Walden 2000).
[`paul`](#paul) | Paul wavelet of order `m`, an analytic wavelet with a short envelope.
[`symlet`](#symlet) | Least asymmetric wavelet with `p` vanishing moments.

## `ContinuousWavelet`

Bases: <code>[NamedTuple](#typing.NamedTuple)</code>

Mother wavelet of `cwt`, described in the frequency domain.

The wavelet dilated to the frequency ``f`` has the Fourier transform ``spectrum(nu / f)`` at ``nu``. The spectrum peaks
at 2 for an analytic wavelet and at 1 on either side of zero for a real one.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`spectrum`](#edmkit.wavelet.continuous.ContinuousWavelet.spectrum) | <code>[Callable](#collections.abc.Callable)[[[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]], [NDArray](#numpy.typing.NDArray)[[complexfloating](#numpy.complexfloating)]]</code> | Fourier transform of the wavelet at frequencies in units of its peak frequency.
[`width`](#edmkit.wavelet.continuous.ContinuousWavelet.width) | <code>[float](#float)</code> | e-folding time of the wavelet power for a spike, in units of the period ``1 / f``.

### `spectrum`

```python
spectrum: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.complexfloating]]
```

Fourier transform of the wavelet at frequencies in units of its peak frequency.

### `width`

```python
width: float
```

e-folding time of the wavelet power for a spike, in units of the period ``1 / f``.



## `DiscreteWavelet`

Bases: <code>[NamedTuple](#typing.NamedTuple)</code>

Orthonormal wavelet of `modwt` and `mra`, described by its scaling filter.

The wavelet filter is the quadrature mirror ``h[l] = (-1) ** l * g[L - 1 - l]`` of the scaling filter ``g``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`scaling_filter`](#edmkit.wavelet.discrete.DiscreteWavelet.scaling_filter) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Scaling filter of shape ``(L,)``, of even length, summing to ``sqrt(2)`` and orthonormal to its even shifts.

### `scaling_filter`

```python
scaling_filter: npt.NDArray[np.floating]
```

Scaling filter of shape ``(L,)``, of even length, summing to ``sqrt(2)`` and orthonormal to its even shifts.



## `cwt`

```python
cwt(x: npt.NDArray[np.floating], *, dt: float, frequencies: npt.NDArray[np.floating], wavelet: ContinuousWavelet) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.bool_]]
```

Continuous wavelet transform of `x` at the given frequencies.

A sinusoid of amplitude ``A`` and frequency ``f`` gives coefficients whose envelope is ``A`` at ``f``, whatever the
wavelet. The transform is computed by FFT with `x` zero-padded, so coefficients near either end are biased; those
within the cone of influence (Torrence & Compo 1998) are marked unreliable.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#numpy.ndarray)</code> | Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series transformed independently. | *required*
`dt` | <code>[float](#float)</code> | Sampling interval. | *required*
`frequencies` | <code>[ndarray](#numpy.ndarray)</code> | Frequencies of shape ``(F,)``, in cycles per unit time, within ``(0, 1 / (2 dt)]``. | *required*
`wavelet` | <code>[ContinuousWavelet](#edmkit.wavelet.continuous.ContinuousWavelet)</code> | Mother wavelet, such as ``morlet()``, ``paul()`` or ``dog()``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`W` | <code>[ndarray](#numpy.ndarray)</code> | Complex coefficients of shape ``(T, F)`` or ``(T, d, F)``. For a real wavelet, ``W.real`` holds the coefficients.
`reliable` | <code>[ndarray](#numpy.ndarray)</code> | Boolean mask of shape ``(T, F)``, True where the coefficient lies outside the cone of influence, that is where both ends are more than the e-folding time ``wavelet.width / f`` away.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `x` is not a nonempty, finite 1D or 2D array. - If `dt` is not positive. - If `frequencies` is not a nonempty 1D array within ``(0, 1 / (2 dt)]``.

**Examples:**

```python
import numpy as np

from edmkit.wavelet import cwt, morlet

dt = 0.01
t = np.arange(0, 20, dt)
x = np.where(t < 10, np.sin(2 * np.pi * 1.0 * t), 0.5 * np.sin(2 * np.pi * 4.0 * t))

W, reliable = cwt(x, dt=dt, frequencies=np.geomspace(0.25, 16, 64), wavelet=morlet())
amplitude = np.where(reliable, np.abs(W), np.nan)  # about 1 near (t < 10, f = 1) and 0.5 near (t > 10, f = 4)
```



## `daubechies`

```python
daubechies(p: int) -> DiscreteWavelet
```

Daubechies extremal phase wavelet with `p` vanishing moments.

Its scaling filter keeps every zero inside the unit circle (minimum phase). ``daubechies(1)`` is the Haar wavelet.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>[int](#int)</code> | Number of vanishing moments of the wavelet. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`wavelet` | <code>[DiscreteWavelet](#edmkit.wavelet.discrete.DiscreteWavelet)</code> | Wavelet whose scaling filter has shape ``(2 p,)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If `p` is not positive.

**Examples:**

```python
from edmkit.wavelet import daubechies

print(daubechies(2).scaling_filter)
# [ 0.48296291  0.8365163   0.22414387 -0.12940952]
```



## `dog`

```python
dog(m: int = 2) -> ContinuousWavelet
```

Derivative of Gaussian wavelet of order `m`, a real wavelet whose coefficients peak at the extrema of the signal.

Order 2 is the Mexican hat (Ricker) wavelet.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` | <code>[int](#int)</code> | Order of the derivative. Larger values resolve frequency more finely and time more coarsely. | <code>2</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`wavelet` | <code>[ContinuousWavelet](#edmkit.wavelet.continuous.ContinuousWavelet)</code> | 

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If `m` is not positive.



## `icwt`

```python
icwt(W: npt.NDArray[np.complexfloating], *, frequencies: npt.NDArray[np.floating], wavelet: ContinuousWavelet) -> npt.NDArray[np.floating]
```

Inverse continuous wavelet transform, summing the coefficients of `cwt` over frequency (Torrence & Compo 1998).

The sum is linear in `W`, so setting the coefficients outside a band to zero gives the part of `x` in that band. To
recover `x`, `frequencies` must extend well beyond its content (further for broad wavelets such as ``dog(1)``) and be
dense in log frequency. Near the ends, as in `cwt`, the result is biased.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`W` | <code>[ndarray](#numpy.ndarray)</code> | Coefficients of shape ``(T, F)`` or ``(T, d, F)``, as `cwt` returns them. | *required*
`frequencies` | <code>[ndarray](#numpy.ndarray)</code> | Frequencies of shape ``(F,)`` with ``F >= 2``, those passed to `cwt`. | *required*
`wavelet` | <code>[ContinuousWavelet](#edmkit.wavelet.continuous.ContinuousWavelet)</code> | Mother wavelet passed to `cwt`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`x` | <code>[ndarray](#numpy.ndarray)</code> | Series of shape ``(T,)`` or ``(T, d)``, without its mean, which no wavelet sees.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `W` is not a finite 2D or 3D array with ``T >= 1``. - If `frequencies` is not a 1D array of at least two finite positive frequencies, one per column of `W`.

**Examples:**

```python
import numpy as np

from edmkit.wavelet import cwt, icwt, morlet

dt = 0.01
t = np.arange(0, 60, dt)
x = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 4.0 * t)

frequencies = np.geomspace(0.01, 50, 128)
W, reliable = cwt(x, dt=dt, frequencies=frequencies, wavelet=morlet())
slow = icwt(np.where(frequencies < 2, W, 0), frequencies=frequencies, wavelet=morlet())  # about the 0.5 Hz sinusoid
```



## `imodwt`

```python
imodwt(C: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet) -> npt.NDArray[np.floating]
```

Inverse maximal overlap discrete wavelet transform (Percival & Walden 2000).

Each level is filtered back through the adjoint of its `modwt` filter and the levels are summed, so
``imodwt(modwt(x, wavelet=wavelet, levels=J), wavelet=wavelet)`` is `x`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`C` | <code>[ndarray](#numpy.ndarray)</code> | Finite coefficients of shape ``(T, J + 1)`` or ``(T, d, J + 1)``, laid out as `modwt` returns them. | *required*
`wavelet` | <code>[DiscreteWavelet](#edmkit.wavelet.discrete.DiscreteWavelet)</code> | Wavelet, as for `modwt`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`x` | <code>[ndarray](#numpy.ndarray)</code> | Series of shape ``(T,)`` or ``(T, d)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `C` is not a finite 2D or 3D array with ``T >= 1`` and at least two levels. - If the scaling filter is invalid, as for `modwt`. - If the level-``J`` filter is longer than ``T``, as for `modwt`.

**Examples:**

```python
import numpy as np

from edmkit.wavelet import imodwt, modwt, symlet

x = np.random.default_rng(0).normal(size=1000)
C = modwt(x, wavelet=symlet(4), levels=5)

assert np.allclose(imodwt(C, wavelet=symlet(4)), x)
```



## `modwt`

```python
modwt(x: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet, levels: int) -> npt.NDArray[np.floating]
```

Maximal overlap discrete wavelet transform of `x` (Percival & Walden 2000).

Level ``j`` passes roughly the band ``[1 / (2 ** (j + 1) dt), 1 / (2 ** j dt)]``. `x` is treated as periodic, so the
first ``(2 ** j - 1) (L - 1)`` coefficients of level ``j`` mix the start of `x` with its end.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#numpy.ndarray)</code> | Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series transformed independently. | *required*
`wavelet` | <code>[DiscreteWavelet](#edmkit.wavelet.discrete.DiscreteWavelet)</code> | Wavelet with a scaling filter of length ``L``, such as ``daubechies(p)`` or ``symlet(p)``. | *required*
`levels` | <code>[int](#int)</code> | Number of levels ``J``. The level-``J`` filter, of ``(2 ** J - 1) (L - 1) + 1`` taps, must fit within ``T``. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`C` | <code>[ndarray](#numpy.ndarray)</code> | Coefficients of shape ``(T, J + 1)`` or ``(T, d, J + 1)``: the wavelet coefficients ``W_1, ..., W_J`` from fine to coarse, then the scaling coefficients ``V_J``. Their squares sum to those of `x`.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | - If `x` is not a nonempty, finite 1D or 2D array. - If the scaling filter is not of even length, does not sum to ``sqrt(2)`` or is not orthonormal to its even shifts. - If `levels` is not positive, or its filter is longer than ``T``.

**Examples:**

```python
import numpy as np

from edmkit.wavelet import modwt, symlet

x = np.random.default_rng(0).normal(size=1000)
C = modwt(x, wavelet=symlet(4), levels=5)

variance = (C**2).mean(axis=0)  # wavelet variance per level, summing to (x ** 2).mean()
```



## `morlet`

```python
morlet(omega0: float = 6.0) -> ContinuousWavelet
```

Complex Morlet wavelet, an analytic wavelet made of a complex exponential under a Gaussian envelope.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`omega0` | <code>[float](#float)</code> | Radians of oscillation per e-folding of the envelope. Larger values resolve frequency more finely and time more coarsely. Below about 4 the wavelet is far from admissible and `icwt` is biased. | <code>6.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`wavelet` | <code>[ContinuousWavelet](#edmkit.wavelet.continuous.ContinuousWavelet)</code> | 

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If `omega0` is not positive.



## `mra`

```python
mra(x: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet, levels: int) -> npt.NDArray[np.floating]
```

Multiresolution analysis of `x` by the MODWT (Percival & Walden 2000).

Component ``j`` is `imodwt` of the `modwt` coefficients with every level but ``j`` set to zero, so the components sum
to `x` and stay aligned with it in time. As in `modwt`, `x` is treated as periodic, so a trend leaks into every
component near both ends.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#numpy.ndarray)</code> | Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series analysed independently. | *required*
`wavelet` | <code>[DiscreteWavelet](#edmkit.wavelet.discrete.DiscreteWavelet)</code> | Wavelet, as for `modwt`. | *required*
`levels` | <code>[int](#int)</code> | Number of levels ``J``, as for `modwt`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`D` | <code>[ndarray](#numpy.ndarray)</code> | Components of shape ``(T, J + 1)`` or ``(T, d, J + 1)``: the details ``D_1, ..., D_J`` from fine to coarse, then the smooth ``S_J``. They sum to `x` along the last axis.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | As for `modwt`.

**Examples:**

```python
import numpy as np

from edmkit.wavelet import mra, symlet

t = np.arange(2048)
x = np.sin(2 * np.pi * t / 512) + 0.3 * np.sin(2 * np.pi * t / 8)
D = mra(x, wavelet=symlet(4), levels=6)

assert np.allclose(D.sum(axis=-1), x)
fast, slow = D[:, :3].sum(axis=-1), D[:, 3:].sum(axis=-1)  # the period-8 and period-512 sinusoids
```



## `paul`

```python
paul(m: int = 4) -> ContinuousWavelet
```

Paul wavelet of order `m`, an analytic wavelet with a short envelope.

Compared with the Morlet wavelet, it resolves time more finely and frequency more coarsely.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` | <code>[int](#int)</code> | Order. Larger values resolve frequency more finely and time more coarsely. | <code>4</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`wavelet` | <code>[ContinuousWavelet](#edmkit.wavelet.continuous.ContinuousWavelet)</code> | 

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If `m` is not positive.



## `symlet`

```python
symlet(p: int) -> DiscreteWavelet
```

Least asymmetric wavelet with `p` vanishing moments.

Of the scaling filters that differ from that of `daubechies` only in which of each pair of zeros ``z, 1 / z`` they
keep, this one has the phase closest, in mean square, to linear. Of it and its time reversal, the one whose energy
comes first is taken.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>[int](#int)</code> | Number of vanishing moments of the wavelet. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`wavelet` | <code>[DiscreteWavelet](#edmkit.wavelet.discrete.DiscreteWavelet)</code> | Wavelet whose scaling filter has shape ``(2 p,)``.

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If `p` is not positive.

<details class="note" open markdown="1">
<summary>Notes</summary>

The search runs over about ``2 ** (p // 2)`` candidates.

</details>

