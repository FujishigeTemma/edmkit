import itertools
from math import comb
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ["DiscreteWavelet", "daubechies", "imodwt", "modwt", "mra", "symlet"]


class DiscreteWavelet(NamedTuple):
    """Orthonormal wavelet of `modwt` and `mra`, described by its scaling filter.

    The wavelet filter is the quadrature mirror ``h[l] = (-1) ** l * g[L - 1 - l]`` of the scaling filter ``g``.
    """

    scaling_filter: npt.NDArray[np.floating]
    """Scaling filter of shape ``(L,)``, of even length, summing to ``sqrt(2)`` and orthonormal to its even shifts."""


def daubechies(p: int) -> DiscreteWavelet:
    """Daubechies extremal phase wavelet with `p` vanishing moments.

    Its scaling filter keeps every zero inside the unit circle (minimum phase). ``daubechies(1)`` is the Haar wavelet.

    Parameters
    ----------
    p : int
        Number of vanishing moments of the wavelet.

    Returns
    -------
    wavelet : DiscreteWavelet
        Wavelet whose scaling filter has shape ``(2 p,)``.

    Raises
    ------
    ValueError
        If `p` is not positive.

    Examples
    --------
    ```python
    from edmkit.wavelet import daubechies

    print(daubechies(2).scaling_filter)
    # [ 0.48296291  0.8365163   0.22414387 -0.12940952]
    ```
    """
    if p < 1:
        raise ValueError(f"p must be positive, got p={p}")

    # roots of P(y) = sum_k C(p - 1 + k, k) y ** k, the squared gain over cos(omega / 2) ** (2 p) in y = sin(omega / 2) ** 2
    roots = np.roots([comb(p - 1 + k, k) for k in range(p)][::-1])
    # each root gives the zeros z, 1 / z solving z + 1 / z = 2 - 4 y; keep the one inside the unit circle
    b = 2.0 - 4.0 * roots
    z = (b + np.sqrt(b * b - 4.0)) / 2.0
    zeros = np.where(np.abs(z) < 1, z, 1.0 / z)

    g = np.real(np.poly(np.concatenate([-np.ones(p), zeros])))  # p zeros at z = -1 for the vanishing moments

    return DiscreteWavelet(g * np.sqrt(2.0) / g.sum())


def symlet(p: int) -> DiscreteWavelet:
    """Least asymmetric wavelet with `p` vanishing moments.

    Of the scaling filters that differ from that of `daubechies` only in which of each pair of zeros ``z, 1 / z`` they
    keep, this one has the phase closest, in mean square, to linear. Of it and its time reversal, the one whose energy
    comes first is taken.

    Parameters
    ----------
    p : int
        Number of vanishing moments of the wavelet.

    Returns
    -------
    wavelet : DiscreteWavelet
        Wavelet whose scaling filter has shape ``(2 p,)``.

    Raises
    ------
    ValueError
        If `p` is not positive.

    Notes
    -----
    The search runs over about ``2 ** (p // 2)`` candidates.
    """
    if p < 1:
        raise ValueError(f"p must be positive, got p={p}")

    # zeros inside the unit circle, as in daubechies
    roots = np.roots([comb(p - 1 + k, k) for k in range(p)][::-1])
    b = 2.0 - 4.0 * roots
    z = (b + np.sqrt(b * b - 4.0)) / 2.0
    zeros = np.where(np.abs(z) < 1, z, 1.0 / z)

    # a conjugate pair of zeros is flipped together, keeping the filter real
    choices = [np.array([zero.real]) for zero in zeros[roots.imag == 0]] + [np.array([zero, zero.conjugate()]) for zero in zeros[roots.imag > 0]]

    lags = np.arange(2 * p)
    omega = np.linspace(0.0, np.pi, 1024, endpoint=False)[1:]  # the filter vanishes at pi, where the phase is undefined
    response = np.exp(-1j * np.outer(omega, lags))  # (1023, 2 p)
    design = np.stack([omega, np.ones_like(omega)], axis=1)  # linear fit of the phase

    filters, deviations = [], []
    for flips in itertools.product((False, True), repeat=len(choices)):
        kept = [1.0 / choice if flip else choice for choice, flip in zip(choices, flips)]
        g = np.real(np.poly(np.concatenate([-np.ones(p), *kept])))
        g = g * np.sqrt(2.0) / g.sum()
        phase = np.unwrap(np.angle(response @ g))
        filters.append(g)
        deviations.append(np.linalg.lstsq(design, phase, rcond=None)[1][0])  # residual sum of squares
    best = filters[int(np.argmin(deviations))]

    return DiscreteWavelet(best if lags @ best**2 <= lags @ best[::-1] ** 2 else best[::-1])


def modwt(x: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet, levels: int) -> npt.NDArray[np.floating]:
    """Maximal overlap discrete wavelet transform of `x` (Percival & Walden 2000).

    Level ``j`` passes roughly the band ``[1 / (2 ** (j + 1) dt), 1 / (2 ** j dt)]``. `x` is treated as periodic, so the
    first ``(2 ** j - 1) (L - 1)`` coefficients of level ``j`` mix the start of `x` with its end.

    Parameters
    ----------
    x : np.ndarray
        Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series transformed independently.
    wavelet : DiscreteWavelet
        Wavelet with a scaling filter of length ``L``, such as ``daubechies(p)`` or ``symlet(p)``.
    levels : int
        Number of levels ``J``. The level-``J`` filter, of ``(2 ** J - 1) (L - 1) + 1`` taps, must fit within ``T``.

    Returns
    -------
    C : np.ndarray
        Coefficients of shape ``(T, J + 1)`` or ``(T, d, J + 1)``: the wavelet coefficients ``W_1, ..., W_J`` from fine to
        coarse, then the scaling coefficients ``V_J``. Their squares sum to those of `x`.

    Raises
    ------
    ValueError
        - If `x` is not a nonempty, finite 1D or 2D array.
        - If the scaling filter is not of even length, does not sum to ``sqrt(2)`` or is not orthonormal to its even shifts.
        - If `levels` is not positive, or its filter is longer than ``T``.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.wavelet import modwt, symlet

    x = np.random.default_rng(0).normal(size=1000)
    C = modwt(x, wavelet=symlet(4), levels=5)

    variance = (C**2).mean(axis=0)  # wavelet variance per level, summing to (x ** 2).mean()
    ```
    """
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError(f"x must be a nonempty array of shape (T,) or (T, d), got x.shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"x must be finite, got {np.count_nonzero(~np.isfinite(x))} NaN or infinite values")
    g = np.asarray(wavelet.scaling_filter, dtype=float)
    if g.ndim != 1 or g.size == 0 or g.size % 2 != 0:
        raise ValueError(f"scaling_filter must be a 1D array of even length, got scaling_filter.shape={g.shape}")
    if not np.isclose(g.sum(), np.sqrt(2.0), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must sum to sqrt(2), got {g.sum()}")
    shifts = [g[2 * m :] @ g[: g.size - 2 * m] for m in range(g.size // 2)]
    if not np.allclose(shifts, [1.0] + [0.0] * (g.size // 2 - 1), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must be orthonormal to its even shifts, got inner products {shifts}")
    if levels < 1:
        raise ValueError(f"levels must be positive, got levels={levels}")
    # a longer filter wraps around the whole series, leaving no coefficient free of the ends
    width = (2**levels - 1) * (g.size - 1) + 1
    if width > x.shape[0]:
        raise ValueError(
            f"levels={levels} gives a filter of {width} taps, longer than the series of x.shape={x.shape}; use fewer levels or a shorter filter"
        )

    T = x.shape[0]
    h = (-1.0) ** np.arange(g.size) * g[::-1]  # wavelet filter
    G = np.fft.fft(g / np.sqrt(2.0), n=T)
    H = np.fft.fft(h / np.sqrt(2.0), n=T)

    # frequency responses of W_1, ..., W_J, then V_J, at the nonnegative frequencies (the filters are real)
    responses = np.empty((T // 2 + 1, levels + 1), dtype=complex)
    smooth = np.ones(T // 2 + 1, dtype=complex)  # response of V_j
    for j in range(levels):
        upsampled = (2**j * np.arange(T // 2 + 1)) % T  # level j + 1 upsamples the filters by 2 ** j
        responses[:, j] = smooth * H[upsampled]
        smooth = smooth * G[upsampled]
    responses[:, levels] = smooth

    if x.ndim == 2:
        responses = responses[:, None, :]  # (T, 1, J + 1)

    return np.fft.irfft(np.fft.rfft(x, axis=0)[..., None] * responses, n=T, axis=0)


def imodwt(C: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet) -> npt.NDArray[np.floating]:
    """Inverse maximal overlap discrete wavelet transform (Percival & Walden 2000).

    Each level is filtered back through the adjoint of its `modwt` filter and the levels are summed, so
    ``imodwt(modwt(x, wavelet=wavelet, levels=J), wavelet=wavelet)`` is `x`.

    Parameters
    ----------
    C : np.ndarray
        Finite coefficients of shape ``(T, J + 1)`` or ``(T, d, J + 1)``, laid out as `modwt` returns them.
    wavelet : DiscreteWavelet
        Wavelet, as for `modwt`.

    Returns
    -------
    x : np.ndarray
        Series of shape ``(T,)`` or ``(T, d)``.

    Raises
    ------
    ValueError
        - If `C` is not a finite 2D or 3D array with ``T >= 1`` and at least two levels.
        - If the scaling filter is invalid, as for `modwt`.
        - If the level-``J`` filter is longer than ``T``, as for `modwt`.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.wavelet import imodwt, modwt, symlet

    x = np.random.default_rng(0).normal(size=1000)
    C = modwt(x, wavelet=symlet(4), levels=5)

    assert np.allclose(imodwt(C, wavelet=symlet(4)), x)
    ```
    """
    C = np.asarray(C, dtype=float)
    if C.ndim not in (2, 3) or C.shape[0] == 0 or C.shape[-1] < 2:
        raise ValueError(f"C must have shape (T, J + 1) or (T, d, J + 1) with T >= 1 and J >= 1, got C.shape={C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError(f"C must be finite, got {np.count_nonzero(~np.isfinite(C))} NaN or infinite values")
    g = np.asarray(wavelet.scaling_filter, dtype=float)
    if g.ndim != 1 or g.size == 0 or g.size % 2 != 0:
        raise ValueError(f"scaling_filter must be a 1D array of even length, got scaling_filter.shape={g.shape}")
    if not np.isclose(g.sum(), np.sqrt(2.0), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must sum to sqrt(2), got {g.sum()}")
    shifts = [g[2 * m :] @ g[: g.size - 2 * m] for m in range(g.size // 2)]
    if not np.allclose(shifts, [1.0] + [0.0] * (g.size // 2 - 1), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must be orthonormal to its even shifts, got inner products {shifts}")
    levels = C.shape[-1] - 1
    # a longer filter wraps around the whole series, as in modwt
    width = (2**levels - 1) * (g.size - 1) + 1
    if width > C.shape[0]:
        raise ValueError(
            f"{levels} levels give a filter of {width} taps, longer than the series of C.shape={C.shape}; use fewer levels or a shorter filter"
        )

    T = C.shape[0]
    h = (-1.0) ** np.arange(g.size) * g[::-1]  # wavelet filter
    G = np.fft.fft(g / np.sqrt(2.0), n=T)
    H = np.fft.fft(h / np.sqrt(2.0), n=T)

    # frequency responses of W_1, ..., W_J, then V_J, as in modwt
    responses = np.empty((T // 2 + 1, levels + 1), dtype=complex)
    smooth = np.ones(T // 2 + 1, dtype=complex)
    for j in range(levels):
        upsampled = (2**j * np.arange(T // 2 + 1)) % T
        responses[:, j] = smooth * H[upsampled]
        smooth = smooth * G[upsampled]
    responses[:, levels] = smooth

    # the adjoint of each level's filter, summed over the levels
    if C.ndim == 3:
        responses = responses[:, None, :]  # (T, 1, J + 1)

    return np.fft.irfft((np.conj(responses) * np.fft.rfft(C, axis=0)).sum(axis=-1), n=T, axis=0)


def mra(x: npt.NDArray[np.floating], *, wavelet: DiscreteWavelet, levels: int) -> npt.NDArray[np.floating]:
    """Multiresolution analysis of `x` by the MODWT (Percival & Walden 2000).

    Component ``j`` is `imodwt` of the `modwt` coefficients with every level but ``j`` set to zero, so the components sum
    to `x` and stay aligned with it in time. As in `modwt`, `x` is treated as periodic, so a trend leaks into every
    component near both ends.

    Parameters
    ----------
    x : np.ndarray
        Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series analysed independently.
    wavelet : DiscreteWavelet
        Wavelet, as for `modwt`.
    levels : int
        Number of levels ``J``, as for `modwt`.

    Returns
    -------
    D : np.ndarray
        Components of shape ``(T, J + 1)`` or ``(T, d, J + 1)``: the details ``D_1, ..., D_J`` from fine to coarse, then
        the smooth ``S_J``. They sum to `x` along the last axis.

    Raises
    ------
    ValueError
        As for `modwt`.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.wavelet import mra, symlet

    t = np.arange(2048)
    x = np.sin(2 * np.pi * t / 512) + 0.3 * np.sin(2 * np.pi * t / 8)
    D = mra(x, wavelet=symlet(4), levels=6)

    assert np.allclose(D.sum(axis=-1), x)
    fast, slow = D[:, :3].sum(axis=-1), D[:, 3:].sum(axis=-1)  # the period-8 and period-512 sinusoids
    ```
    """
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError(f"x must be a nonempty array of shape (T,) or (T, d), got x.shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"x must be finite, got {np.count_nonzero(~np.isfinite(x))} NaN or infinite values")
    g = wavelet.scaling_filter
    if g.ndim != 1 or g.size == 0 or g.size % 2 != 0:
        raise ValueError(f"scaling_filter must be a 1D array of even length, got scaling_filter.shape={g.shape}")
    if not np.isclose(g.sum(), np.sqrt(2.0), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must sum to sqrt(2), got {g.sum()}")
    shifts = [g[2 * m :] @ g[: g.size - 2 * m] for m in range(g.size // 2)]
    if not np.allclose(shifts, [1.0] + [0.0] * (g.size // 2 - 1), rtol=0, atol=1e-8):
        raise ValueError(f"scaling_filter must be orthonormal to its even shifts, got inner products {shifts}")
    if levels < 1:
        raise ValueError(f"levels must be positive, got levels={levels}")
    # a longer filter wraps around the whole series, leaving no coefficient free of the ends
    width = (2**levels - 1) * (g.size - 1) + 1
    if width > x.shape[0]:
        raise ValueError(
            f"levels={levels} gives a filter of {width} taps, longer than the series of x.shape={x.shape}; use fewer levels or a shorter filter"
        )

    T = x.shape[0]
    h = (-1.0) ** np.arange(g.size) * g[::-1]  # wavelet filter
    G = np.fft.fft(g / np.sqrt(2.0), n=T)
    H = np.fft.fft(h / np.sqrt(2.0), n=T)

    # frequency responses of W_1, ..., W_J, then V_J, as in modwt
    responses = np.empty((T // 2 + 1, levels + 1), dtype=complex)
    smooth = np.ones(T // 2 + 1, dtype=complex)
    for j in range(levels):
        upsampled = (2**j * np.arange(T // 2 + 1)) % T
        responses[:, j] = smooth * H[upsampled]
        smooth = smooth * G[upsampled]
    responses[:, levels] = smooth

    if x.ndim == 2:
        responses = responses[:, None, :]  # (T // 2 + 1, 1, J + 1)

    # imodwt of each level of modwt alone: the level's filter followed by its adjoint
    return np.fft.irfft(np.fft.rfft(x, axis=0)[..., None] * np.abs(responses) ** 2, n=T, axis=0)
