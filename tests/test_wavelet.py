from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import pywt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from pywt import Wavelet as PyWavelet  # ty: ignore[unresolved-import]

from edmkit.wavelet import ContinuousWavelet, DiscreteWavelet, cwt, daubechies, dog, icwt, imodwt, modwt, morlet, mra, paul, symlet


class DaubechiesProblem(NamedTuple):
    p: int


class DaubechiesCase(NamedTuple):
    p: int


def check_daubechies(p: int) -> None:
    g = daubechies(p).scaling_filter

    assert g.shape == (2 * p,)
    np.testing.assert_allclose(g, PyWavelet(f"db{p}").rec_lo, atol=1e-10)


@st.composite
def daubechies_problems(draw) -> DaubechiesProblem:
    return DaubechiesProblem(draw(st.integers(1, 20)))


DAUBECHIES_VALID = {
    "haar": DaubechiesCase(1),
    "d4": DaubechiesCase(2),
    "ten-moments": DaubechiesCase(10),
}

DAUBECHIES_INVALID = {
    "zero": DaubechiesCase(0),
}


@given(problem=daubechies_problems())
def test_daubechies_compatibility(problem: DaubechiesProblem) -> None:
    check_daubechies(*problem)


@pytest.mark.parametrize("case", DAUBECHIES_VALID.values(), ids=DAUBECHIES_VALID.keys())
def test_daubechies_valid(case: DaubechiesCase) -> None:
    check_daubechies(*case)


@pytest.mark.parametrize("case", DAUBECHIES_INVALID.values(), ids=DAUBECHIES_INVALID.keys())
def test_daubechies_invalid(case: DaubechiesCase) -> None:
    with pytest.raises(ValueError):
        daubechies(case.p)


class SymletProblem(NamedTuple):
    p: int


class SymletCase(NamedTuple):
    p: int


def check_symlet(p: int) -> None:
    g = symlet(p).scaling_filter
    h = (-1.0) ** np.arange(2 * p) * g[::-1]

    assert g.shape == (2 * p,)
    # orthonormal, with p vanishing moments and the gain of daubechies(p)
    np.testing.assert_allclose(g.sum(), np.sqrt(2.0))
    np.testing.assert_allclose([g[2 * m :] @ g[: 2 * p - 2 * m] for m in range(p)], np.eye(1, p)[0], atol=1e-9)
    np.testing.assert_allclose([(np.arange(2 * p) / (2 * p)) ** k @ h for k in range(p)], 0, atol=1e-9)
    np.testing.assert_allclose(np.abs(np.fft.rfft(g, 256)), np.abs(np.fft.rfft(daubechies(p).scaling_filter, 256)), atol=1e-9)

    # PyWavelets' symN up to time reversal, for the p where every measure of asymmetry agrees
    if 2 <= p <= 5:
        expected = np.asarray(PyWavelet(f"sym{p}").rec_lo)
        assert np.allclose(g, expected, atol=1e-9) or np.allclose(g[::-1], expected, atol=1e-9)


@st.composite
def symlet_problems(draw) -> SymletProblem:
    return SymletProblem(draw(st.integers(1, 10)))


SYMLET_VALID = {
    "haar": SymletCase(1),
    "la8": SymletCase(4),
    "beyond-pywavelets-agreement": SymletCase(10),
}

SYMLET_INVALID = {
    "zero": SymletCase(0),
}


@given(problem=symlet_problems())
def test_symlet_compatibility(problem: SymletProblem) -> None:
    check_symlet(*problem)


@pytest.mark.parametrize("case", SYMLET_VALID.values(), ids=SYMLET_VALID.keys())
def test_symlet_valid(case: SymletCase) -> None:
    check_symlet(*case)


@pytest.mark.parametrize("case", SYMLET_INVALID.values(), ids=SYMLET_INVALID.keys())
def test_symlet_invalid(case: SymletCase) -> None:
    with pytest.raises(ValueError):
        symlet(case.p)


class CwtProblem(NamedTuple):
    x: np.ndarray
    dt: float
    frequencies: np.ndarray
    wavelet: ContinuousWavelet
    amplitude: float
    reference: tuple[str, float] | None


class CwtCase(NamedTuple):
    x: np.ndarray
    dt: float
    frequencies: np.ndarray
    wavelet: ContinuousWavelet
    amplitude: float
    reference: tuple[str, float] | None


def check_cwt(
    x: np.ndarray, dt: float, frequencies: np.ndarray, wavelet: ContinuousWavelet, amplitude: float, reference: tuple[str, float] | None
) -> None:
    """`x` holds sinusoids of `amplitude` at ``frequencies[0]``; `reference` is the PyWavelets name and peak frequency, if any."""
    W, reliable = cwt(x, dt=dt, frequencies=frequencies, wavelet=wavelet)

    assert W.shape == (*x.shape, len(frequencies))
    assert reliable.shape == (len(x), len(frequencies))
    if x.ndim == 2:
        np.testing.assert_allclose(W, np.stack([cwt(column, dt=dt, frequencies=frequencies, wavelet=wavelet)[0] for column in x.T], axis=1))
        return

    # the envelope over a period about the centre is the amplitude
    centre = slice(len(x) // 2, len(x) // 2 + int(1 / (frequencies[0] * dt)) + 1)
    assert reliable[centre].all()
    np.testing.assert_allclose(np.abs(W[centre, 0]).max(), amplitude, rtol=1e-2)

    if reference is None:
        return
    name, peak = reference
    scales = peak / (frequencies * dt)
    expected, _ = pywt.cwt(x, scales, name, sampling_period=dt, method="fft", precision=16)
    # PyWavelets normalizes in L2, we in L1
    a, b = W[centre].ravel(), (expected.T / np.sqrt(scales))[centre].ravel()
    assert abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.9999


@st.composite
def cwt_problems(draw) -> CwtProblem:
    family = draw(st.sampled_from(["morlet", "paul", "dog"]))
    if family == "morlet":
        omega0 = draw(st.floats(5.0, 8.0))
        wavelet, reference = morlet(omega0), (f"cmor2.0-{omega0 / (2 * np.pi)}", omega0 / (2 * np.pi))
    elif family == "paul":
        wavelet, reference = paul(draw(st.integers(1, 6))), None
    else:  # PyWavelets' gausM differentiates exp(-t ** 2), peaking at sqrt(2 m) radians per unit time
        m = draw(st.integers(1, 6))
        wavelet, reference = dog(m), (f"gaus{m}", np.sqrt(2 * m) / (2 * np.pi))
    amplitude = draw(st.floats(0.1, 10.0))
    frequency = draw(st.floats(0.5, 2.0))
    # fine sampling keeps PyWavelets' discretization error small
    dt = 1 / (frequency * draw(st.floats(500.0, 800.0)))
    frequencies = frequency * np.array([1.0, *draw(st.lists(st.floats(0.6, 1.6), min_size=1, max_size=3))])
    # a few periods beyond six e-folding times, as the Paul wavelet of low order decays only as a power of time
    T = int(2 * (6 * wavelet.width + 3) / frequencies.min() / dt)
    phases = np.array(draw(st.lists(st.floats(0.0, 2 * np.pi), min_size=1, max_size=2)))
    x = amplitude * np.cos(2 * np.pi * frequency * np.arange(T)[:, None] * dt + phases)
    return CwtProblem(x[:, 0] if len(phases) == 1 else x, dt, frequencies, wavelet, amplitude, reference)


CWT_VALID = {
    "morlet": CwtCase(
        1.5 * np.sin(2 * np.pi * np.arange(30000) * 0.002),
        0.002,
        np.array([1.0, 0.7, 1.4]),
        morlet(6.0),
        1.5,
        (f"cmor2.0-{6.0 / (2 * np.pi)}", 6.0 / (2 * np.pi)),
    ),
    "paul": CwtCase(0.3 * np.sin(2 * np.pi * np.arange(20000) * 0.002), 0.002, np.array([1.0, 0.5, 2.0]), paul(4), 0.3, None),
    "mexican-hat": CwtCase(
        2.0 * np.sin(2 * np.pi * np.arange(20000) * 0.002), 0.002, np.array([1.0, 0.8]), dog(2), 2.0, ("mexh", np.sqrt(2) / (2 * np.pi))
    ),
    "odd-dog": CwtCase(np.sin(2 * np.pi * np.arange(20000) * 0.002), 0.002, np.array([1.0, 1.3]), dog(3), 1.0, ("gaus3", np.sqrt(6) / (2 * np.pi))),
    "two-series": CwtCase(np.cos(2 * np.pi * np.arange(500)[:, None] * 0.05 + np.array([0.0, 1.0])), 0.05, np.array([1.0]), morlet(6.0), 1.0, None),
}

CWT_INVALID = {
    "empty": CwtCase(np.zeros(0), 0.1, np.array([1.0]), morlet(), 1.0, None),
    "three-dimensional": CwtCase(np.zeros((10, 2, 2)), 0.1, np.array([1.0]), morlet(), 1.0, None),
    "zero-dt": CwtCase(np.zeros(10), 0.0, np.array([1.0]), morlet(), 1.0, None),
    "no-frequencies": CwtCase(np.zeros(10), 0.1, np.zeros(0), morlet(), 1.0, None),
    "zero-frequency": CwtCase(np.zeros(10), 0.1, np.array([0.0, 1.0]), morlet(), 1.0, None),
    "above-nyquist": CwtCase(np.zeros(10), 0.1, np.array([1.0, 6.0]), morlet(), 1.0, None),
    "two-dimensional-frequencies": CwtCase(np.zeros(10), 0.1, np.ones((2, 2)), morlet(), 1.0, None),
    "not-finite": CwtCase(np.array([0.0] * 9 + [np.nan]), 0.1, np.array([1.0]), morlet(), 1.0, None),
    "nan-frequency": CwtCase(np.zeros(10), 0.1, np.array([np.nan]), morlet(), 1.0, None),
}


@given(problem=cwt_problems())
def test_cwt_compatibility(problem: CwtProblem) -> None:
    check_cwt(*problem)


@pytest.mark.parametrize("case", CWT_VALID.values(), ids=CWT_VALID.keys())
def test_cwt_valid(case: CwtCase) -> None:
    check_cwt(*case)


@pytest.mark.parametrize("case", CWT_INVALID.values(), ids=CWT_INVALID.keys())
def test_cwt_invalid(case: CwtCase) -> None:
    with pytest.raises(ValueError):
        cwt(case.x, dt=case.dt, frequencies=case.frequencies, wavelet=case.wavelet)


class IcwtProblem(NamedTuple):
    W: np.ndarray
    frequencies: np.ndarray
    wavelet: ContinuousWavelet


class IcwtCase(NamedTuple):
    W: np.ndarray
    frequencies: np.ndarray
    wavelet: ContinuousWavelet


def check_icwt(W: np.ndarray, frequencies: np.ndarray, wavelet: ContinuousWavelet) -> None:
    """`frequencies` spans a factor of 100 on either side of its geometric centre, and `W` is long enough for a sinusoid there."""
    x = icwt(W, frequencies=frequencies, wavelet=wavelet)

    assert x.shape == W.shape[:-1]
    # linear in W, so the parts of complementary bands sum to the whole
    low = frequencies < np.median(frequencies)
    parts = icwt(np.where(low, W, 0), frequencies=frequencies, wavelet=wavelet) + icwt(np.where(low, 0, W), frequencies=frequencies, wavelet=wavelet)
    np.testing.assert_allclose(parts, x, atol=1e-9)

    # a sinusoid at the centre of the band is recovered away from the ends
    T = W.shape[0]
    dt = 0.4 / frequencies.max()
    centre_frequency = np.sqrt(frequencies.min() * frequencies.max())
    y = np.cos(2 * np.pi * centre_frequency * dt * np.arange(T))
    Y, _ = cwt(y, dt=dt, frequencies=frequencies, wavelet=wavelet)
    centre = slice(T // 2, T // 2 + int(1 / (centre_frequency * dt)) + 1)
    np.testing.assert_allclose(icwt(Y, frequencies=frequencies, wavelet=wavelet)[centre], y[centre], atol=5e-2)


@st.composite
def icwt_problems(draw) -> IcwtProblem:
    family = draw(st.sampled_from([morlet, paul, dog]))
    wavelet = family(draw(st.floats(5.0, 8.0))) if family is morlet else family(draw(st.integers(1, 6)))
    frequencies = draw(st.floats(0.1, 10.0)) * 2.0 ** np.arange(-np.log2(100), np.log2(100) + 1e-9, 1 / draw(st.integers(8, 12)))
    # six e-folding times plus a few periods of the centre frequency, of 250 samples each in check_icwt, on either side
    T = int(2 * (6 * wavelet.width + 3) * 250)
    shape = (T, len(frequencies)) if draw(st.booleans()) else (T, draw(st.integers(1, 2)), len(frequencies))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return IcwtProblem(rng.normal(size=shape) + 1j * rng.normal(size=shape), frequencies, wavelet)


ICWT_VALID = {
    "morlet": IcwtCase(np.random.default_rng(0).normal(size=(5600, 107)), 2.0 ** np.arange(-np.log2(100), np.log2(100) + 1e-9, 1 / 8), morlet()),
    "paul": IcwtCase(np.ones((3000, 107)), 3.0 * 2.0 ** np.arange(-np.log2(100), np.log2(100) + 1e-9, 1 / 8), paul()),
    "mexican-hat": IcwtCase(np.zeros((3000, 2, 160)), 0.5 * 2.0 ** np.arange(-np.log2(100), np.log2(100) + 1e-9, 1 / 12), dog()),
    "descending-frequencies": IcwtCase(np.ones((5600, 107)), 2.0 ** np.arange(np.log2(100), -np.log2(100) - 1e-9, -1 / 8), morlet()),
}

ICWT_INVALID = {
    "one-dimensional": IcwtCase(np.zeros(10), np.array([1.0, 2.0]), morlet()),
    "empty": IcwtCase(np.zeros((0, 2)), np.array([1.0, 2.0]), morlet()),
    "not-finite": IcwtCase(np.array([[np.nan, 0.0]] * 4), np.array([1.0, 2.0]), morlet()),
    "single-frequency": IcwtCase(np.zeros((10, 1)), np.array([1.0]), morlet()),
    "zero-frequency": IcwtCase(np.zeros((10, 2)), np.array([0.0, 1.0]), morlet()),
    "infinite-frequency": IcwtCase(np.zeros((10, 2)), np.array([1.0, np.inf]), morlet()),
    "frequencies-not-matching-columns": IcwtCase(np.zeros((10, 3)), np.array([1.0, 2.0]), morlet()),
}


@given(problem=icwt_problems())
def test_icwt_compatibility(problem: IcwtProblem) -> None:
    check_icwt(*problem)


@pytest.mark.parametrize("case", ICWT_VALID.values(), ids=ICWT_VALID.keys())
def test_icwt_valid(case: IcwtCase) -> None:
    check_icwt(*case)


@pytest.mark.parametrize("case", ICWT_INVALID.values(), ids=ICWT_INVALID.keys())
def test_icwt_invalid(case: IcwtCase) -> None:
    with pytest.raises(ValueError):
        icwt(case.W, frequencies=case.frequencies, wavelet=case.wavelet)


class ModwtProblem(NamedTuple):
    x: np.ndarray
    wavelet: DiscreteWavelet
    levels: int


class ModwtCase(NamedTuple):
    x: np.ndarray
    wavelet: DiscreteWavelet
    levels: int


def check_modwt(x: np.ndarray, wavelet: DiscreteWavelet, levels: int) -> None:
    C = modwt(x, wavelet=wavelet, levels=levels)

    assert C.shape == (*x.shape, levels + 1)
    # energy is preserved, and shifting x in time shifts every level alike
    np.testing.assert_allclose((C**2).sum(axis=(0, -1)), (x**2).sum(axis=0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(modwt(np.roll(x, 1, axis=0), wavelet=wavelet, levels=levels), np.roll(C, 1, axis=0), atol=1e-9)

    # PyWavelets' SWT differs only by time-reversed filters and circular shifts, neither of which changes |DFT|
    if x.ndim == 1 and len(x) % 2**levels == 0:
        g = np.asarray(wavelet.scaling_filter)
        h = (-1.0) ** np.arange(len(g)) * g[::-1]
        reference = PyWavelet("given", filter_bank=[g[::-1], h[::-1], g, h])
        reference.orthogonal = True
        approximation, *details = pywt.swt(x, reference, level=levels, norm=True, trim_approx=True)  # V_J, W_J, ..., W_1
        expected = np.stack([*details[::-1], approximation], axis=-1)
        np.testing.assert_allclose(np.abs(np.fft.rfft(C, axis=0)), np.abs(np.fft.rfft(expected, axis=0)), atol=1e-8)


@st.composite
def modwt_problems(draw) -> ModwtProblem:
    wavelet = draw(st.sampled_from([daubechies, symlet]))(draw(st.integers(1, 5)))
    levels = draw(st.integers(1, 4))
    # the level-J filter must fit within the series
    width = (2**levels - 1) * (len(wavelet.scaling_filter) - 1) + 1
    T = draw(st.integers(width, width + 64))
    shape = (T,) if draw(st.booleans()) else (T, draw(st.integers(1, 3)))
    x = draw(hnp.arrays(np.float64, shape, elements=st.floats(-100, 100)))
    return ModwtProblem(x, wavelet, levels)


MODWT_VALID = {
    "haar": ModwtCase(np.arange(16.0), daubechies(1), 4),
    "la8": ModwtCase(np.sin(np.arange(64.0)), symlet(4), 3),
    "length-not-a-power-of-two": ModwtCase(np.cos(np.arange(37.0) ** 1.5), daubechies(3), 2),
    "constant": ModwtCase(np.full(32, 3.0), daubechies(2), 2),
    "several-series": ModwtCase(np.arange(48.0).reshape(24, 2) ** 2 % 7, symlet(3), 2),
    "filter-from-elsewhere": ModwtCase(np.arange(64.0) ** 2 % 5, DiscreteWavelet(np.asarray(PyWavelet("coif2").rec_lo)), 2),
}

MODWT_INVALID = {
    "three-dimensional": ModwtCase(np.zeros((16, 2, 2)), daubechies(2), 1),
    "odd-filter": ModwtCase(np.zeros(16), DiscreteWavelet(np.ones(3) * np.sqrt(2) / 3), 1),
    "filter-not-summing-to-sqrt2": ModwtCase(np.zeros(16), DiscreteWavelet(np.array([1.0, 1.0])), 1),
    "filter-not-orthonormal": ModwtCase(np.zeros(16), DiscreteWavelet(np.array([0.5, 0.5, 0.5, np.sqrt(2) - 1.5])), 1),
    "zero-levels": ModwtCase(np.zeros(16), daubechies(2), 0),
    "too-many-levels": ModwtCase(np.zeros(15), daubechies(2), 4),
    "filter-longer-than-series": ModwtCase(np.zeros(64), daubechies(10), 3),
    "not-finite": ModwtCase(np.array([0.0] * 15 + [np.nan]), daubechies(2), 1),
}


@given(problem=modwt_problems())
def test_modwt_compatibility(problem: ModwtProblem) -> None:
    check_modwt(*problem)


@pytest.mark.parametrize("case", MODWT_VALID.values(), ids=MODWT_VALID.keys())
def test_modwt_valid(case: ModwtCase) -> None:
    check_modwt(*case)


@pytest.mark.parametrize("case", MODWT_INVALID.values(), ids=MODWT_INVALID.keys())
def test_modwt_invalid(case: ModwtCase) -> None:
    with pytest.raises(ValueError):
        modwt(case.x, wavelet=case.wavelet, levels=case.levels)


class ImodwtProblem(NamedTuple):
    C: np.ndarray
    wavelet: DiscreteWavelet


class ImodwtCase(NamedTuple):
    C: np.ndarray
    wavelet: DiscreteWavelet


def check_imodwt(C: np.ndarray, wavelet: DiscreteWavelet) -> None:
    x = imodwt(C, wavelet=wavelet)
    y = np.cos(np.arange(x.size) ** 1.3).reshape(x.shape)
    C_y = modwt(y, wavelet=wavelet, levels=C.shape[-1] - 1)

    assert x.shape == C.shape[:-1]
    # the adjoint of modwt, which it inverts
    np.testing.assert_allclose(np.sum(C_y * C), np.sum(y * x), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(imodwt(C_y, wavelet=wavelet), y, atol=1e-9)


@st.composite
def imodwt_problems(draw) -> ImodwtProblem:
    wavelet = draw(st.sampled_from([daubechies, symlet]))(draw(st.integers(1, 5)))
    levels = draw(st.integers(1, 4))
    width = (2**levels - 1) * (len(wavelet.scaling_filter) - 1) + 1
    T = draw(st.integers(width, width + 64))
    shape = (T, levels + 1) if draw(st.booleans()) else (T, draw(st.integers(1, 3)), levels + 1)
    C = draw(hnp.arrays(np.float64, shape, elements=st.floats(-100, 100)))
    return ImodwtProblem(C, wavelet)


IMODWT_VALID = {
    "haar": ImodwtCase(np.arange(48.0).reshape(16, 3), daubechies(1)),
    "la8": ImodwtCase(np.sin(np.arange(256.0)).reshape(64, 4), symlet(4)),
    "single-level": ImodwtCase(np.cos(np.arange(74.0)).reshape(37, 2), daubechies(3)),
    "several-series": ImodwtCase(np.arange(144.0).reshape(24, 2, 3) % 7, symlet(3)),
}

IMODWT_INVALID = {
    "one-dimensional": ImodwtCase(np.zeros(16), daubechies(2)),
    "no-level": ImodwtCase(np.zeros((16, 1)), daubechies(2)),
    "empty": ImodwtCase(np.zeros((0, 3)), daubechies(1)),
    "not-finite": ImodwtCase(np.array([[0.0, 0.0]] * 15 + [[np.nan, 0.0]]), daubechies(2)),
    "filter-not-orthonormal": ImodwtCase(np.zeros((16, 2)), DiscreteWavelet(np.array([0.5, 0.5, 0.5, np.sqrt(2) - 1.5]))),
    "filter-longer-than-series": ImodwtCase(np.zeros((64, 4)), daubechies(10)),
}


@given(problem=imodwt_problems())
def test_imodwt_compatibility(problem: ImodwtProblem) -> None:
    check_imodwt(*problem)


@pytest.mark.parametrize("case", IMODWT_VALID.values(), ids=IMODWT_VALID.keys())
def test_imodwt_valid(case: ImodwtCase) -> None:
    check_imodwt(*case)


@pytest.mark.parametrize("case", IMODWT_INVALID.values(), ids=IMODWT_INVALID.keys())
def test_imodwt_invalid(case: ImodwtCase) -> None:
    with pytest.raises(ValueError):
        imodwt(case.C, wavelet=case.wavelet)


class MraProblem(NamedTuple):
    x: np.ndarray
    wavelet: DiscreteWavelet
    levels: int


class MraCase(NamedTuple):
    x: np.ndarray
    wavelet: DiscreteWavelet
    levels: int


def check_mra(x: np.ndarray, wavelet: DiscreteWavelet, levels: int) -> None:
    D = mra(x, wavelet=wavelet, levels=levels)

    assert D.shape == (*x.shape, levels + 1)
    # the components sum to x, and shifting x in time shifts every component alike
    np.testing.assert_allclose(D.sum(axis=-1), x, atol=1e-9)
    np.testing.assert_allclose(mra(np.roll(x, 1, axis=0), wavelet=wavelet, levels=levels), np.roll(D, 1, axis=0), atol=1e-9)

    if x.ndim == 1 and len(x) % 2**levels == 0:
        g = np.asarray(wavelet.scaling_filter)
        h = (-1.0) ** np.arange(len(g)) * g[::-1]
        reference = PyWavelet("given", filter_bank=[g[::-1], h[::-1], g, h])
        reference.orthogonal = True
        expected = pywt.mra(x, reference, level=levels, transform="swt")  # S_J, D_J, ..., D_1
        np.testing.assert_allclose(D, np.stack(expected[::-1], axis=-1), atol=1e-9)


@st.composite
def mra_problems(draw) -> MraProblem:
    return MraProblem(*draw(modwt_problems()))


MRA_VALID = {
    "haar": MraCase(np.arange(16.0), daubechies(1), 4),
    "la8": MraCase(np.sin(np.arange(64.0)), symlet(4), 3),
    "length-not-a-power-of-two": MraCase(np.cos(np.arange(37.0) ** 1.5), daubechies(3), 2),
    "constant": MraCase(np.full(32, 3.0), daubechies(2), 2),
    "several-series": MraCase(np.arange(48.0).reshape(24, 2) ** 2 % 7, symlet(3), 2),
}

MRA_INVALID = {
    "three-dimensional": MraCase(np.zeros((16, 2, 2)), daubechies(2), 1),
    "odd-filter": MraCase(np.zeros(16), DiscreteWavelet(np.ones(3) * np.sqrt(2) / 3), 1),
    "filter-not-orthonormal": MraCase(np.zeros(16), DiscreteWavelet(np.array([0.5, 0.5, 0.5, np.sqrt(2) - 1.5])), 1),
    "zero-levels": MraCase(np.zeros(16), daubechies(2), 0),
    "too-many-levels": MraCase(np.zeros(15), daubechies(2), 4),
    "filter-longer-than-series": MraCase(np.zeros(64), daubechies(10), 3),
    "not-finite": MraCase(np.array([0.0] * 15 + [np.inf]), daubechies(2), 1),
}


@given(problem=mra_problems())
def test_mra_compatibility(problem: MraProblem) -> None:
    check_mra(*problem)


@pytest.mark.parametrize("case", MRA_VALID.values(), ids=MRA_VALID.keys())
def test_mra_valid(case: MraCase) -> None:
    check_mra(*case)


@pytest.mark.parametrize("case", MRA_INVALID.values(), ids=MRA_INVALID.keys())
def test_mra_invalid(case: MraCase) -> None:
    with pytest.raises(ValueError):
        mra(case.x, wavelet=case.wavelet, levels=case.levels)
