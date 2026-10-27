from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ["ContinuousWavelet", "cwt", "dog", "icwt", "morlet", "paul"]


class ContinuousWavelet(NamedTuple):
    """Mother wavelet of `cwt`, described in the frequency domain.

    The wavelet dilated to the frequency ``f`` has the Fourier transform ``spectrum(nu / f)`` at ``nu``. The spectrum peaks
    at 2 for an analytic wavelet and at 1 on either side of zero for a real one.
    """

    spectrum: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.complexfloating]]
    """Fourier transform of the wavelet at frequencies in units of its peak frequency."""
    width: float
    """e-folding time of the wavelet power for a spike, in units of the period ``1 / f``."""


def morlet(omega0: float = 6.0) -> ContinuousWavelet:
    """Complex Morlet wavelet, an analytic wavelet made of a complex exponential under a Gaussian envelope.

    Parameters
    ----------
    omega0 : float, default 6.0
        Radians of oscillation per e-folding of the envelope. Larger values resolve frequency more finely and time more coarsely.
        Below about 4 the wavelet is far from admissible and `icwt` is biased.

    Returns
    -------
    wavelet : ContinuousWavelet

    Raises
    ------
    ValueError
        If `omega0` is not positive.
    """
    if not omega0 > 0:
        raise ValueError(f"omega0 must be positive, got omega0={omega0}")

    def spectrum(u: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        return 2.0 * np.exp(-0.5 * omega0**2 * (u - 1.0) ** 2) * (u > 0)

    # e-folding time sqrt(2) s at the scale s (Torrence & Compo 1998), in periods 2 pi s / omega0
    return ContinuousWavelet(spectrum, np.sqrt(2.0) * omega0 / (2.0 * np.pi))


def paul(m: int = 4) -> ContinuousWavelet:
    """Paul wavelet of order `m`, an analytic wavelet with a short envelope.

    Compared with the Morlet wavelet, it resolves time more finely and frequency more coarsely.

    Parameters
    ----------
    m : int, default 4
        Order. Larger values resolve frequency more finely and time more coarsely.

    Returns
    -------
    wavelet : ContinuousWavelet

    Raises
    ------
    ValueError
        If `m` is not positive.
    """
    if m < 1:
        raise ValueError(f"m must be positive, got m={m}")

    def spectrum(u: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        positive = u > 0
        v = np.where(positive, u, 1.0)
        # u ** m * exp(m (1 - u)), in logarithms to avoid overflow
        return 2.0 * np.exp(m * (np.log(v) + 1.0 - v)) * positive

    # e-folding time s / sqrt(2) at the scale s (Torrence & Compo 1998), in periods 2 pi s / m
    return ContinuousWavelet(spectrum, m / (np.sqrt(2.0) * 2.0 * np.pi))


def dog(m: int = 2) -> ContinuousWavelet:
    """Derivative of Gaussian wavelet of order `m`, a real wavelet whose coefficients peak at the extrema of the signal.

    Order 2 is the Mexican hat (Ricker) wavelet.

    Parameters
    ----------
    m : int, default 2
        Order of the derivative. Larger values resolve frequency more finely and time more coarsely.

    Returns
    -------
    wavelet : ContinuousWavelet

    Raises
    ------
    ValueError
        If `m` is not positive.
    """
    if m < 1:
        raise ValueError(f"m must be positive, got m={m}")

    def spectrum(u: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        a = np.where(u != 0, np.abs(u), 1.0)  # sign(0) = 0 zeroes u = 0 below
        # |u| ** m * exp(m (1 - u ** 2) / 2), in logarithms to avoid overflow
        magnitude = np.exp(m * (np.log(a) + 0.5 * (1.0 - a**2)))
        return (-1.0) ** (m + 1) * 1j**m * np.sign(u) ** m * magnitude

    # e-folding time sqrt(2) s at the scale s (Torrence & Compo 1998), in periods 2 pi s / sqrt(m)
    return ContinuousWavelet(spectrum, np.sqrt(2.0 * m) / (2.0 * np.pi))


def fft_length(m: int) -> int:
    """Smallest length >= `m` of the form ``2 ** a * 3 ** b * 5 ** c``, which the FFT handles fast."""
    best = 2 ** (m - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p2 = 2 ** (-(-m // p35) - 1).bit_length()  # smallest power of two with p35 * p2 >= m
            best = min(best, p35 * p2)
            p35 *= 3
        p5 *= 5
    return best


def cwt(
    x: npt.NDArray[np.floating], *, dt: float, frequencies: npt.NDArray[np.floating], wavelet: ContinuousWavelet
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.bool_]]:
    """Continuous wavelet transform of `x` at the given frequencies.

    A sinusoid of amplitude ``A`` and frequency ``f`` gives coefficients whose envelope is ``A`` at ``f``, whatever the
    wavelet. The transform is computed by FFT with `x` zero-padded, so coefficients near either end are biased; those
    within the cone of influence (Torrence & Compo 1998) are marked unreliable.

    Parameters
    ----------
    x : np.ndarray
        Finite observations of shape ``(T,)``, or ``(T, d)`` for ``d`` series transformed independently.
    dt : float
        Sampling interval.
    frequencies : np.ndarray
        Frequencies of shape ``(F,)``, in cycles per unit time, within ``(0, 1 / (2 dt)]``.
    wavelet : ContinuousWavelet
        Mother wavelet, such as ``morlet()``, ``paul()`` or ``dog()``.

    Returns
    -------
    W : np.ndarray
        Complex coefficients of shape ``(T, F)`` or ``(T, d, F)``. For a real wavelet, ``W.real`` holds the coefficients.
    reliable : np.ndarray
        Boolean mask of shape ``(T, F)``, True where the coefficient lies outside the cone of influence, that is where both
        ends are more than the e-folding time ``wavelet.width / f`` away.

    Raises
    ------
    ValueError
        - If `x` is not a nonempty, finite 1D or 2D array.
        - If `dt` is not positive.
        - If `frequencies` is not a nonempty 1D array within ``(0, 1 / (2 dt)]``.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.wavelet import cwt, morlet

    dt = 0.01
    t = np.arange(0, 20, dt)
    x = np.where(t < 10, np.sin(2 * np.pi * 1.0 * t), 0.5 * np.sin(2 * np.pi * 4.0 * t))

    W, reliable = cwt(x, dt=dt, frequencies=np.geomspace(0.25, 16, 64), wavelet=morlet())
    amplitude = np.where(reliable, np.abs(W), np.nan)  # about 1 near (t < 10, f = 1) and 0.5 near (t > 10, f = 4)
    ```
    """
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError(f"x must be a nonempty array of shape (T,) or (T, d), got x.shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"x must be finite, got {np.count_nonzero(~np.isfinite(x))} NaN or infinite values")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive, got dt={dt}")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError(f"frequencies must be a nonempty array of shape (F,), got frequencies.shape={frequencies.shape}")
    if not (np.all(frequencies > 0) and np.all(frequencies <= 0.5 / dt)):
        raise ValueError(f"frequencies must lie in (0, {0.5 / dt}] for dt={dt}, got {frequencies.min()} to {frequencies.max()}")

    T = x.shape[0]
    n = fft_length(2 * T)  # zero padding to >= 2 T, so the series does not wrap onto itself
    X = np.fft.fft(x, n=n, axis=0)  # (n,) or (n, d)
    nu = np.fft.fftfreq(n, dt)  # (n,)

    # correlation with the dilated wavelet
    psi = np.conj(wavelet.spectrum(nu[:, None] / frequencies[None, :]))  # (n, F)
    if x.ndim == 2:
        psi = psi[:, None, :]  # (n, 1, F)
    W = np.fft.ifft(X[..., None] * psi, axis=0)[:T]

    distance = np.minimum(np.arange(T), np.arange(T)[::-1]) * dt  # time to the nearer end
    reliable = distance[:, None] >= wavelet.width / frequencies[None, :]  # (T, F)
    return W, reliable


def icwt(W: npt.NDArray[np.complexfloating], *, frequencies: npt.NDArray[np.floating], wavelet: ContinuousWavelet) -> npt.NDArray[np.floating]:
    """Inverse continuous wavelet transform, summing the coefficients of `cwt` over frequency (Torrence & Compo 1998).

    The sum is linear in `W`, so setting the coefficients outside a band to zero gives the part of `x` in that band. To
    recover `x`, `frequencies` must extend well beyond its content (further for broad wavelets such as ``dog(1)``) and be
    dense in log frequency. Near the ends, as in `cwt`, the result is biased.

    Parameters
    ----------
    W : np.ndarray
        Coefficients of shape ``(T, F)`` or ``(T, d, F)``, as `cwt` returns them.
    frequencies : np.ndarray
        Frequencies of shape ``(F,)`` with ``F >= 2``, those passed to `cwt`.
    wavelet : ContinuousWavelet
        Mother wavelet passed to `cwt`.

    Returns
    -------
    x : np.ndarray
        Series of shape ``(T,)`` or ``(T, d)``, without its mean, which no wavelet sees.

    Raises
    ------
    ValueError
        - If `W` is not a finite 2D or 3D array with ``T >= 1``.
        - If `frequencies` is not a 1D array of at least two finite positive frequencies, one per column of `W`.

    Examples
    --------
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
    """
    W = np.asarray(W, dtype=complex)
    if W.ndim not in (2, 3) or W.shape[0] == 0:
        raise ValueError(f"W must have shape (T, F) or (T, d, F) with T >= 1, got W.shape={W.shape}")
    if not np.all(np.isfinite(W)):
        raise ValueError(f"W must be finite, got {np.count_nonzero(~np.isfinite(W))} NaN or infinite values")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2 or not np.all(np.isfinite(frequencies) & (frequencies > 0)):
        raise ValueError(f"frequencies must be an array of shape (F,) of at least two finite positive frequencies, got {frequencies}")
    if frequencies.size != W.shape[-1]:
        raise ValueError(f"frequencies must have one frequency per column of W, got {frequencies.size} for W.shape={W.shape}")

    T = W.shape[0]
    # weight of each frequency: the width of its cell in log frequency
    order = np.argsort(frequencies)
    weights = np.empty(frequencies.size)
    weights[order] = np.gradient(np.log(frequencies[order]))
    S = W @ weights  # (T,) or (T, d)

    # the sum over log frequency multiplies each positive frequency of x by the integral of the conjugate spectrum over ln u
    s = np.linspace(-40.0, 40.0, 4001)
    gain = np.trapezoid(np.conj(wavelet.spectrum(np.exp(s))), s)

    n = fft_length(2 * T)  # zero padding as in cwt
    X = np.fft.fft(S, n=n, axis=0)[: n // 2 + 1] / gain  # positive frequencies of x
    X[0] = 0.0  # the mean
    return np.fft.irfft(X, n=n, axis=0)[:T]
