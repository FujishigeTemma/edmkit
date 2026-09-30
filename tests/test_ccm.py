from __future__ import annotations

from typing import NamedTuple, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.ccm import AggregateFunc, SampleFunc, bootstrap, ccm, make_sample_func
from edmkit.embedding import embed
from edmkit.metrics import MetricFunc, pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.types import PredictFunc


class BootstrapProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc
    metric_func: MetricFunc
    n_samples: int
    library_pool: np.ndarray
    prediction_pool: np.ndarray
    seed: int
    batch_size: int | None


class BootstrapCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc | None
    metric_func: MetricFunc
    n_samples: int
    library_pool: np.ndarray
    prediction_pool: np.ndarray
    seed: int
    batch_size: int | None


class CCMProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc
    metric_func: MetricFunc
    n_samples: int
    library_pool: np.ndarray
    prediction_pool: np.ndarray
    seed: int
    aggregate_func: AggregateFunc
    batch_size: int | None
    minimum_gap: float
    maximum_final: float


class CCMCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc
    metric_func: MetricFunc
    n_samples: int
    library_pool: np.ndarray
    prediction_pool: np.ndarray
    seed: int
    aggregate_func: AggregateFunc | None
    batch_size: int | None
    minimum_gap: float
    maximum_final: float


def signed_linear_predictor(X, Y, Q, *, mask=None):
    """Deterministic PredictFunc for a `Y = 2X + 1` library: sign of the library mean times the true map."""
    assert mask is None
    np.testing.assert_array_equal(Y, 2.0 * X + 1.0)
    sign = np.where(X[..., 0].mean(axis=1) >= 0.0, 1.0, -1.0)
    return sign[:, None, None] * (2.0 * Q + 1.0)


def coupled_logistic_maps(length: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.zeros(length), np.zeros(length)
    x[0], y[0] = 0.4, 0.2
    for i in range(1, length):
        x[i] = x[i - 1] * (3.8 - 3.8 * x[i - 1])
        y[i] = y[i - 1] * (3.5 - 3.5 * y[i - 1]) + 0.02 * x[i - 1]
    return x[50:], y[50:]


def logistic_map(length: int, growth: float, initial: float) -> np.ndarray:
    x = np.zeros(length)
    x[0] = initial
    for i in range(1, length):
        x[i] = growth * x[i - 1] * (1.0 - x[i - 1])
    return x[50:]


def cross_mapped_series(source: np.ndarray, target: np.ndarray, tau: int, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed `source` and align `target` for a CCM cross-map, splitting indices into equal library/prediction pools."""
    embedding, _ = embed(np.array([[0, tau * j] for j in range(e)]), source)
    aligned_target = target[tau * (e - 1) :]
    half = len(embedding) // 2
    return embedding, aligned_target, np.arange(half), np.arange(half, len(embedding))


def check_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc | None,
    metric_func: MetricFunc,
    n_samples: int,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    seed: int,
    batch_size: int | None,
) -> None:
    """`bootstrap`'s scores for a fixed `sample_func` seed do not depend on how the draws for
    each library size are chunked into batches, and are reproducible for a deterministic
    predict_func/metric_func."""
    predict_func = cast(PredictFunc, predict_func)
    actual = bootstrap(
        x,
        y,
        lib_sizes,
        predict_func,
        metric_func,
        n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(seed),
        batch_size=batch_size,
    )
    assert actual.shape == (n_samples, len(lib_sizes))

    for alternate_batch_size in (None, 1, n_samples):
        alternate = bootstrap(
            x,
            y,
            lib_sizes,
            predict_func,
            metric_func,
            n_samples,
            library_pool=library_pool,
            prediction_pool=prediction_pool,
            sample_func=make_sample_func(seed),
            batch_size=alternate_batch_size,
        )
        np.testing.assert_array_equal(actual, alternate)


def ccm_reference(
    x: np.ndarray,
    y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    metric_func: MetricFunc,
    n_samples: int,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    sample_func: SampleFunc,
    aggregate_func: AggregateFunc,
    batch_size: int | None,
) -> np.ndarray:
    samples = bootstrap(
        x,
        y,
        lib_sizes,
        predict_func,
        metric_func,
        n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=sample_func,
        batch_size=batch_size,
    )
    return np.array([aggregate_func(samples[:, i]) for i in range(samples.shape[1])])


def check_ccm(
    x: np.ndarray,
    y: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    metric_func: MetricFunc,
    n_samples: int,
    library_pool: np.ndarray,
    prediction_pool: np.ndarray,
    seed: int,
    aggregate_func: AggregateFunc | None,
    batch_size: int | None,
    minimum_gap: float,
    maximum_final: float,
) -> None:
    aggregate_func = cast(AggregateFunc, aggregate_func)
    actual = ccm(
        x,
        y,
        lib_sizes,
        predict_func,
        metric_func,
        n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(seed),
        aggregate_func=aggregate_func,
        batch_size=batch_size,
    )
    expected = ccm_reference(
        x, y, lib_sizes, predict_func, metric_func, n_samples, library_pool, prediction_pool, make_sample_func(seed), aggregate_func, batch_size
    )
    assert actual.shape == expected.shape == (len(lib_sizes),)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    # a degenerate series (constant, so its correlation is undefined) legitimately scores NaN,
    # and a convergence gap says nothing there
    if not np.isfinite(actual).all():
        return

    # convergence signature: skill at the largest library size should exceed skill at the
    # smallest by at least `minimum_gap`, and should not exceed `maximum_final` (a control
    # for series with no real coupling, where skill should stay low)
    assert float(actual[-1]) - float(actual[0]) >= minimum_gap
    assert float(actual[-1]) <= maximum_final


FINITE = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)

LINEAR_X = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
LINEAR_Y = 2.0 * LINEAR_X + 1.0
LINEAR_LIBRARY_POOL = np.arange(6)
LINEAR_PREDICTION_POOL = np.arange(6, 12)
LINEAR_LIB_SIZES = np.array([1, 3, 5])

COUPLED_X, COUPLED_Y = coupled_logistic_maps(500)
FORWARD_X, FORWARD_Y, FORWARD_LIBRARY_POOL, FORWARD_PREDICTION_POOL = cross_mapped_series(COUPLED_Y, COUPLED_X, tau=1, e=2)
FORWARD_LIB_SIZES = np.array([20, len(FORWARD_LIBRARY_POOL) - 1])

INDEPENDENT_X = logistic_map(1050, growth=3.8, initial=0.4)
INDEPENDENT_Y = logistic_map(1050, growth=3.7, initial=0.2)
NULL_X, NULL_Y, NULL_LIBRARY_POOL, NULL_PREDICTION_POOL = cross_mapped_series(INDEPENDENT_Y, INDEPENDENT_X, tau=1, e=2)
NULL_LIB_SIZES = np.array([20, 40, 80, 160])


@st.composite
def bootstrap_problems(draw) -> BootstrapProblem:
    total = draw(st.integers(8, 20))
    library_size = draw(st.integers(4, total - 2))
    x = draw(hnp.arrays(np.float64, total, elements=FINITE))
    y = 2.0 * x + 1.0
    library_pool = np.arange(library_size)
    prediction_pool = np.arange(library_size, total)
    lib_sizes = np.array(sorted(draw(st.lists(st.integers(1, library_size), min_size=1, max_size=3, unique=True))))
    n_samples = draw(st.integers(1, 6))
    batch_size = draw(st.one_of(st.none(), st.integers(1, n_samples)))
    seed = draw(st.integers(0, 2**32 - 1))
    return BootstrapProblem(x, y, lib_sizes, signed_linear_predictor, pearson_correlation, n_samples, library_pool, prediction_pool, seed, batch_size)


@given(problem=bootstrap_problems())
def test_bootstrap_compatibility(problem: BootstrapProblem) -> None:
    check_bootstrap(*problem)


@st.composite
def ccm_problems(draw) -> CCMProblem:
    total = draw(st.integers(8, 20))
    library_size = draw(st.integers(4, total - 2))
    x = draw(hnp.arrays(np.float64, total, elements=FINITE))
    y = 2.0 * x + 1.0
    library_pool = np.arange(library_size)
    prediction_pool = np.arange(library_size, total)
    lib_sizes = np.array(sorted(draw(st.lists(st.integers(1, library_size), min_size=1, max_size=3, unique=True))))
    n_samples = draw(st.integers(1, 6))
    batch_size = draw(st.one_of(st.none(), st.integers(1, n_samples)))
    seed = draw(st.integers(0, 2**32 - 1))
    aggregate_func = draw(st.sampled_from((np.mean, np.median)))
    return CCMProblem(
        x,
        y,
        lib_sizes,
        signed_linear_predictor,
        pearson_correlation,
        n_samples,
        library_pool,
        prediction_pool,
        seed,
        aggregate_func,
        batch_size,
        -np.inf,
        np.inf,
    )


@given(problem=ccm_problems())
def test_ccm_compatibility(problem: CCMProblem) -> None:
    check_ccm(*problem)


BOOTSTRAP_VALID = {
    "unbatched": BootstrapCase(
        LINEAR_X, LINEAR_Y, LINEAR_LIB_SIZES, signed_linear_predictor, pearson_correlation, 7, LINEAR_LIBRARY_POOL, LINEAR_PREDICTION_POOL, 19, None
    ),
    "batched": BootstrapCase(
        LINEAR_X, LINEAR_Y, LINEAR_LIB_SIZES, signed_linear_predictor, pearson_correlation, 7, LINEAR_LIBRARY_POOL, LINEAR_PREDICTION_POOL, 19, 2
    ),
}

BOOTSTRAP_INVALID = {
    "mismatched-lengths": BootstrapCase(
        LINEAR_X[:-1],
        LINEAR_Y,
        LINEAR_LIB_SIZES,
        signed_linear_predictor,
        pearson_correlation,
        100,
        LINEAR_LIBRARY_POOL,
        LINEAR_PREDICTION_POOL,
        19,
        None,
    ),
    "noncallable-predictor": BootstrapCase(
        LINEAR_X, LINEAR_Y, LINEAR_LIB_SIZES, None, pearson_correlation, 100, LINEAR_LIBRARY_POOL, LINEAR_PREDICTION_POOL, 19, None
    ),
    "nonpositive-samples": BootstrapCase(
        LINEAR_X, LINEAR_Y, LINEAR_LIB_SIZES, signed_linear_predictor, pearson_correlation, 0, LINEAR_LIBRARY_POOL, LINEAR_PREDICTION_POOL, 19, None
    ),
}

CCM_VALID = {
    "median-aggregate": CCMCase(
        LINEAR_X,
        LINEAR_Y,
        LINEAR_LIB_SIZES,
        signed_linear_predictor,
        pearson_correlation,
        7,
        LINEAR_LIBRARY_POOL,
        LINEAR_PREDICTION_POOL,
        23,
        np.median,
        3,
        -np.inf,
        np.inf,
    ),
    "coupled-direction-converges": CCMCase(
        FORWARD_X,
        FORWARD_Y,
        FORWARD_LIB_SIZES,
        simplex_projection,
        pearson_correlation,
        8,
        FORWARD_LIBRARY_POOL,
        FORWARD_PREDICTION_POOL,
        42,
        np.mean,
        None,
        0.25,
        np.inf,
    ),
    "independent-series-stay-uncorrelated": CCMCase(
        NULL_X,
        NULL_Y,
        NULL_LIB_SIZES,
        simplex_projection,
        pearson_correlation,
        20,
        NULL_LIBRARY_POOL,
        NULL_PREDICTION_POOL,
        7,
        np.mean,
        None,
        -np.inf,
        0.2,
    ),
}

CCM_INVALID = {
    "noncallable-aggregator": CCMCase(
        LINEAR_X,
        LINEAR_Y,
        LINEAR_LIB_SIZES,
        signed_linear_predictor,
        pearson_correlation,
        7,
        LINEAR_LIBRARY_POOL,
        LINEAR_PREDICTION_POOL,
        23,
        None,
        3,
        -np.inf,
        np.inf,
    ),
}


@pytest.mark.parametrize("case", BOOTSTRAP_VALID.values(), ids=BOOTSTRAP_VALID.keys())
def test_bootstrap_valid(case: BootstrapCase) -> None:
    check_bootstrap(*case)


@pytest.mark.parametrize("case", BOOTSTRAP_INVALID.values(), ids=BOOTSTRAP_INVALID.keys())
def test_bootstrap_invalid(case: BootstrapCase) -> None:
    with pytest.raises(ValueError):
        bootstrap(
            case.x,
            case.y,
            case.lib_sizes,
            case.predict_func,  # ty: ignore[invalid-argument-type]
            case.metric_func,
            case.n_samples,
            library_pool=case.library_pool,
            prediction_pool=case.prediction_pool,
            sample_func=make_sample_func(case.seed),
            batch_size=case.batch_size,
        )


@pytest.mark.parametrize("case", CCM_VALID.values(), ids=CCM_VALID.keys())
def test_ccm_valid(case: CCMCase) -> None:
    check_ccm(*case)


@pytest.mark.parametrize("case", CCM_INVALID.values(), ids=CCM_INVALID.keys())
def test_ccm_invalid(case: CCMCase) -> None:
    with pytest.raises(ValueError):
        ccm(
            case.x,
            case.y,
            case.lib_sizes,
            case.predict_func,
            case.metric_func,
            case.n_samples,
            library_pool=case.library_pool,
            prediction_pool=case.prediction_pool,
            sample_func=make_sample_func(case.seed),
            aggregate_func=case.aggregate_func,  # ty: ignore[invalid-argument-type]
            batch_size=case.batch_size,
        )
