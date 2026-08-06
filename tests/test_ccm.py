from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edmkit.ccm import AggregateFunc, bootstrap, ccm, make_sample_func, pearson_correlation, with_simplex_projection, with_smap
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap
from edmkit.types import PredictFunc


type Check = Callable[[], None]
type WrapperFunc = Callable[..., np.ndarray]


class Wiring(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    library_pool: np.ndarray
    prediction_pool: np.ndarray
    lib_sizes: np.ndarray


class PearsonCorrelationProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    scale: float
    shift: float


class PearsonCorrelationCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class MakeSampleFuncCase(NamedTuple):
    seed: int
    pool: np.ndarray
    sizes: tuple[int, ...]


class BootstrapCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    predict: PredictFunc | None
    n_samples: int
    batch_size: int | None


class CCMCase(NamedTuple):
    seed: int
    n_samples: int
    batch_size: int | None
    aggregate_func: Callable[..., np.ndarray] | None


def corrcoef_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vector = x.ndim == 1
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    correlations = np.asarray([correlation(xi, yi) for xi, yi in zip(x, y)])
    return correlations.squeeze() if vector or len(correlations) == 1 else correlations


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(x @ y / denominator)


def check_pearson_correlation(x: np.ndarray, y: np.ndarray) -> None:
    actual = np.asarray(pearson_correlation(x, y))
    expected = np.asarray(corrcoef_rows(x, y))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)


def check_pearson_correlation_invariants(x: np.ndarray, y: np.ndarray, scale: float, shift: float) -> None:
    actual = np.asarray(pearson_correlation(x, y))
    expected = np.asarray(corrcoef_rows(x, y))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)
    np.testing.assert_allclose(actual, pearson_correlation(y, x), atol=1e-12, rtol=1e-12)
    assert np.all(np.abs(actual) <= 1.0 + 1e-12)
    np.testing.assert_allclose(pearson_correlation(x, x), 1.0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(pearson_correlation(x, scale * x + shift), 1.0, atol=1e-12, rtol=1e-12)


def signed_linear_predictor(X, Y, Q, *, mask=None):
    assert mask is None
    np.testing.assert_array_equal(Y, 2.0 * X + 1.0)
    sign = np.where(X[..., 0].mean(axis=1) >= 0.0, 1.0, -1.0)
    return sign[:, None, None] * (2.0 * Q + 1.0)


WIRING = Wiring(
    x=np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
    y=np.array([-5.0, -3.0, -1.0, 3.0, 5.0, 9.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]),
    library_pool=np.arange(6),
    prediction_pool=np.arange(6, 12),
    lib_sizes=np.array([1, 3, 5]),
)


def expected_wiring_samples(seed: int, n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    columns = [
        [1.0 if WIRING.x[rng.choice(WIRING.library_pool, size=int(size), replace=True)].mean() >= 0.0 else -1.0 for _ in range(n_samples)]
        for size in WIRING.lib_sizes
    ]
    return np.asarray(columns).T


def check_make_sample_func(seed: int, pool: np.ndarray, sizes: tuple[int, ...]) -> None:
    actual_sample, expected_sample = make_sample_func(seed), np.random.default_rng(seed)
    for size in sizes:
        np.testing.assert_array_equal(actual_sample(pool, size), expected_sample.choice(pool, size=size, replace=True))


def check_bootstrap(x: np.ndarray, y: np.ndarray, predict: PredictFunc | None, n_samples: int, batch_size: int | None) -> None:
    predict = cast(PredictFunc, predict)
    actual = bootstrap(
        x,
        y,
        WIRING.lib_sizes,
        predict,
        n_samples=n_samples,
        library_pool=WIRING.library_pool,
        prediction_pool=WIRING.prediction_pool,
        sample_func=make_sample_func(19),
        batch_size=batch_size,
    )
    expected = expected_wiring_samples(19, n_samples)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def check_ccm(seed: int, n_samples: int, batch_size: int | None, aggregate_func: Callable[..., np.ndarray] | None) -> None:
    expected = np.median(expected_wiring_samples(seed, n_samples), axis=0)
    actual = ccm(
        WIRING.x,
        WIRING.y,
        WIRING.lib_sizes,
        signed_linear_predictor,
        n_samples=n_samples,
        library_pool=WIRING.library_pool,
        prediction_pool=WIRING.prediction_pool,
        sample_func=make_sample_func(seed),
        aggregate_func=cast(AggregateFunc, aggregate_func),
        batch_size=batch_size,
    )
    np.testing.assert_array_equal(actual, expected)


def check_wrapper(wrapper: WrapperFunc, predictor: PredictFunc) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 2))
    y = 0.5 * x[:, 0] - x[:, 1] ** 2
    sizes = np.array([8, 16])
    library_pool, prediction_pool = np.arange(22), np.arange(22, 32)
    actual = wrapper(
        x,
        y,
        sizes,
        n_samples=3,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(5),
    )
    expected = ccm(
        x,
        y,
        sizes,
        predictor,
        n_samples=3,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(5),
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, strict=True)


def coupled_logistic_maps(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = 0.4, 0.2
    for i in range(1, n):
        x[i] = x[i - 1] * (3.8 - 3.8 * x[i - 1])
        y[i] = y[i - 1] * (3.5 - 3.5 * y[i - 1]) + 0.02 * x[i - 1]
    return x[50:], y[50:]


def logistic_map(n: int, growth: float, initial: float) -> np.ndarray:
    x = np.zeros(n)
    x[0] = initial
    for i in range(1, n):
        x[i] = growth * x[i - 1] * (1.0 - x[i - 1])
    return x[50:]


def cross_map(source: np.ndarray, target: np.ndarray, lib_sizes: np.ndarray, *, n_samples: int, seed: int) -> np.ndarray:
    embedding = lagged_embed(source, tau=1, e=2)
    target = target[1:]
    library_pool = np.arange(len(embedding) // 2)
    prediction_pool = np.arange(len(embedding) // 2, len(embedding))
    return with_simplex_projection(
        embedding,
        target,
        lib_sizes,
        n_samples=n_samples,
        library_pool=library_pool,
        prediction_pool=prediction_pool,
        sample_func=make_sample_func(seed),
    )


def check_coupled(length: int, n_samples: int, seed: int, minimum_gap: float) -> None:
    x, y = coupled_logistic_maps(length)
    lib_sizes = np.array([20, (len(x) - 1) // 2])
    forward = cross_map(y, x, lib_sizes, n_samples=n_samples, seed=seed)
    reverse = cross_map(x, y, lib_sizes, n_samples=n_samples, seed=seed)
    assert forward[-1] > forward[0] + minimum_gap
    assert forward[-1] > reverse[-1] + minimum_gap


def check_independent(length: int, n_samples: int, seed: int, upper_bound: float) -> None:
    x = logistic_map(length, growth=3.8, initial=0.4)
    y = logistic_map(length, growth=3.7, initial=0.2)
    correlations = cross_map(y, x, np.array([20, 40, 80, 160]), n_samples=n_samples, seed=seed)
    assert float(correlations.max()) < upper_bound


@st.composite
def pearson_correlation_problems(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    batch, length = draw(st.integers(1, 4)), draw(st.integers(3, 30))
    rng = np.random.default_rng(seed)
    return PearsonCorrelationProblem(
        rng.standard_normal((batch, length)),
        rng.standard_normal((batch, length)),
        draw(st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)),
        draw(st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)),
    )


@given(problem=pearson_correlation_problems())
def test_pearson_correlation_compatibility(problem: PearsonCorrelationProblem) -> None:
    check_pearson_correlation_invariants(*problem)


PEARSON_CORRELATION_VALID = {
    "positive": PearsonCorrelationCase(np.array([1.0, 2.0, 4.0, 8.0]), np.array([3.0, 5.0, 9.0, 17.0])),
    "negative": PearsonCorrelationCase(np.array([-2.0, -1.0, 1.0, 4.0]), np.array([7.0, 5.0, 1.0, -5.0])),
    "batched": PearsonCorrelationCase(
        np.array([[1.0, 3.0, 2.0, 5.0], [-2.0, 0.0, 4.0, 3.0]]),
        np.array([[4.0, -1.0, 2.0, 0.0], [3.0, 2.0, -1.0, 5.0]]),
    ),
    "constant": PearsonCorrelationCase(np.ones(5), np.arange(5.0)),
}

MAKE_SAMPLE_FUNC_VALID = {
    "seeded": MakeSampleFuncCase(7, WIRING.library_pool, (1, 5, 12)),
}

BOOTSTRAP_VALID = {
    "unbatched": BootstrapCase(WIRING.x, WIRING.y, signed_linear_predictor, 7, None),
    "batched": BootstrapCase(WIRING.x, WIRING.y, signed_linear_predictor, 7, 2),
}

BOOTSTRAP_INVALID = {
    "mismatched-lengths": BootstrapCase(WIRING.x[:-1], WIRING.y, signed_linear_predictor, 100, None),
    "noncallable-predictor": BootstrapCase(WIRING.x, WIRING.y, None, 100, None),
    "nonpositive-samples": BootstrapCase(WIRING.x, WIRING.y, signed_linear_predictor, 0, None),
}

CCM_VALID = {
    "median": CCMCase(23, 7, 3, np.median),
}

CCM_INVALID = {
    "noncallable-aggregator": CCMCase(23, 7, 3, None),
}

WITH_SIMPLEX_PROJECTION_VALID: dict[str, Check] = {
    "wrapper": partial(check_wrapper, with_simplex_projection, simplex_projection),
    "coupled-direction": partial(check_coupled, length=500, n_samples=8, seed=42, minimum_gap=0.25),
    "independent-null-control": partial(check_independent, length=1050, n_samples=20, seed=7, upper_bound=0.2),
}

WITH_SMAP_VALID: dict[str, Check] = {
    "wrapper": partial(check_wrapper, partial(with_smap, theta=2.0, alpha=1e-4), partial(smap, theta=2.0, alpha=1e-4)),
}


@pytest.mark.parametrize("case", PEARSON_CORRELATION_VALID.values(), ids=PEARSON_CORRELATION_VALID.keys())
def test_pearson_correlation_valid(case: PearsonCorrelationCase) -> None:
    check_pearson_correlation(*case)


@pytest.mark.parametrize("case", MAKE_SAMPLE_FUNC_VALID.values(), ids=MAKE_SAMPLE_FUNC_VALID.keys())
def test_make_sample_func_valid(case: MakeSampleFuncCase) -> None:
    check_make_sample_func(*case)


@pytest.mark.parametrize("case", BOOTSTRAP_VALID.values(), ids=BOOTSTRAP_VALID.keys())
def test_bootstrap_valid(case: BootstrapCase) -> None:
    check_bootstrap(*case)


@pytest.mark.parametrize("case", BOOTSTRAP_INVALID.values(), ids=BOOTSTRAP_INVALID.keys())
def test_bootstrap_invalid(case: BootstrapCase) -> None:
    with pytest.raises(ValueError):
        bootstrap(
            case.x,
            case.y,
            WIRING.lib_sizes,
            case.predict,  # ty: ignore[invalid-argument-type]
            n_samples=case.n_samples,
            batch_size=case.batch_size,
            library_pool=WIRING.library_pool,
            prediction_pool=WIRING.prediction_pool,
        )


@pytest.mark.parametrize("case", CCM_VALID.values(), ids=CCM_VALID.keys())
def test_ccm_valid(case: CCMCase) -> None:
    check_ccm(*case)


@pytest.mark.parametrize("case", CCM_INVALID.values(), ids=CCM_INVALID.keys())
def test_ccm_invalid(case: CCMCase) -> None:
    with pytest.raises(ValueError):
        ccm(
            WIRING.x,
            WIRING.y,
            WIRING.lib_sizes,
            signed_linear_predictor,
            library_pool=WIRING.library_pool,
            prediction_pool=WIRING.prediction_pool,
            n_samples=case.n_samples,
            sample_func=make_sample_func(case.seed),
            batch_size=case.batch_size,
            aggregate_func=case.aggregate_func,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("check", WITH_SIMPLEX_PROJECTION_VALID.values(), ids=WITH_SIMPLEX_PROJECTION_VALID.keys())
def test_with_simplex_projection_valid(check: Check) -> None:
    check()


@pytest.mark.parametrize("check", WITH_SMAP_VALID.values(), ids=WITH_SMAP_VALID.keys())
def test_with_smap_valid(check: Check) -> None:
    check()
