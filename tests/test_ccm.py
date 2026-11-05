from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import NamedTuple, cast
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.ccm import bootstrap
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap
from edmkit.types import PredictFunc
from edmkit.util import materialize


class BootstrapProblem(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc
    n_samples: int
    replace: bool
    seed: int | None
    mask: np.ndarray | None
    batch_size: int | None


class BootstrapCase(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    lib_sizes: np.ndarray
    predict_func: PredictFunc = simplex_projection
    n_samples: int = 5
    replace: bool = True
    seed: int | None = 42
    mask: np.ndarray | None = None
    batch_size: int | None = 2


def check_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    lib_sizes: np.ndarray,
    predict_func: PredictFunc,
    n_samples: int,
    replace: bool,
    seed: int | None,
    mask: np.ndarray | None,
    batch_size: int | None,
) -> None:
    # the dense `mask` is the oracle; bootstrap receives it as a rule
    rule = None if mask is None else (lambda *index: mask[index])
    run = partial(bootstrap, x, y, q, lib_sizes, n_samples=n_samples, replace=replace, seed=seed, mask=rule)
    predictor = Mock(wraps=predict_func)
    tasks = list(run(predict_func=cast(PredictFunc, predictor), batch_size=batch_size))
    assert predictor.call_count == 0

    X = x[:, None] if x.ndim == 1 else x
    Y = y[:, None] if y.ndim == 1 else y
    Q = q[:, None] if q.ndim == 1 else q
    batch_limit = n_samples if batch_size is None else min(batch_size, n_samples)
    sizes_order = [i for (i, _), _ in tasks]
    assert sizes_order == sorted(sizes_order)

    # Every task keeps its own sample even after the iterator has been exhausted.
    predictions = np.empty((len(lib_sizes), n_samples, len(Q), Y.shape[1]))
    covered = np.zeros((len(lib_sizes), n_samples), dtype=int)
    for index, samples in tasks:
        predictions[index] = samples()
        covered[index] += 1
        i, batch = index
        lib_X, lib_Y, queries = predictor.call_args.args
        B = batch.stop - batch.start
        assert 0 < B <= batch_limit
        assert lib_X.shape == (B, lib_sizes[i], X.shape[1])

        # The first input coordinate identifies the original row. X and Y must
        # stay paired, including repeated entries and per-query mask columns.
        indices = lib_X[..., 0].astype(int)
        assert np.all((indices >= 0) & (indices < len(X)))
        np.testing.assert_array_equal(lib_X, X[indices])
        np.testing.assert_array_equal(lib_Y, Y[indices])
        np.testing.assert_array_equal(queries, np.broadcast_to(Q, (B, *Q.shape)))
        assert np.shares_memory(queries, q)
        assert not queries.flags.writeable
        if mask is None:
            assert predictor.call_args.kwargs["mask"] is None
        else:
            sampled_mask = materialize(predictor.call_args.kwargs["mask"], (B, len(Q), lib_sizes[i]))
            for b, draw in enumerate(indices):
                np.testing.assert_array_equal(sampled_mask[b], mask[:, draw])
        if not replace:
            for draw in indices:
                assert len(np.unique(draw)) == lib_sizes[i]

    # The tasks fill every sample exactly once.
    np.testing.assert_array_equal(covered, 1)

    # Execution is independent of the iterator and of other tasks' execution.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [(index, executor.submit(samples)) for index, samples in reversed(tasks)]
        for index, future in futures:
            np.testing.assert_allclose(future.result(), predictions[index], rtol=1e-9, atol=1e-9)

    if seed is not None:
        # A fresh run with the same seed reproduces the predictions, and chunking changes neither the sampled libraries nor their predictions.
        alternate = np.empty_like(predictions)
        for index, samples in run(predict_func=predict_func, batch_size=1):
            alternate[index] = samples()
        np.testing.assert_allclose(alternate, predictions, rtol=1e-9, atol=1e-9)


@st.composite
def bootstrap_problems(draw) -> BootstrapProblem:
    n = draw(st.integers(5, 12))
    e = draw(st.integers(1, 3))
    d = draw(st.integers(1, 2))
    m = draw(st.integers(1, 4))
    elements = st.floats(-2, 2, allow_nan=False, allow_infinity=False)
    x = draw(hnp.arrays(np.float64, (n, e), elements=elements)).copy()
    x[:, 0] = np.arange(n)
    y = draw(hnp.arrays(np.float64, (n, d), elements=elements))
    q = draw(hnp.arrays(np.float64, (m, e), elements=elements))
    sizes = np.array(draw(st.lists(st.integers(e + 1, n), min_size=1, max_size=3)))
    n_samples = draw(st.integers(1, 4))
    batch_size = draw(st.one_of(st.none(), st.integers(1, n_samples + 2)))
    seed = draw(st.integers(0, 2**32 - 1))
    replace = draw(st.booleans())
    if e == 1 and draw(st.booleans()):
        x = x[:, 0]
    if e == 1 and draw(st.booleans()):
        q = q[:, 0]
    if d == 1 and draw(st.booleans()):
        y = y[:, 0]
    return BootstrapProblem(x, y, q, sizes, simplex_projection, n_samples, replace, seed, None, batch_size)


@given(problem=bootstrap_problems())
def test_bootstrap_compatibility(problem: BootstrapProblem) -> None:
    check_bootstrap(*problem)


X = np.arange(12, dtype=float)
Y = np.column_stack((X**2, np.sin(X)))
Q = np.array([0.13, 3.27, 8.41])
MASK = np.ones((len(Q), len(X)), dtype=bool)
MASK[np.arange(len(Q)), [0, 3, 8]] = False
TIED_LIBRARY = np.column_stack((np.arange(32, dtype=float), np.zeros(32)))
TIED_LIBRARY[[15, 16], 1] = [8, 20]

BOOTSTRAP_VALID = {
    "partial-batch-and-duplicate-sizes": BootstrapCase(X, Y, Q, np.array([5, 3, 5])),
    "all-samples-together": BootstrapCase(X, Y[:, 0], Q, np.array([3, 6]), batch_size=None),
    "oversized-batch": BootstrapCase(X, Y, Q, np.array([4]), batch_size=20),
    "single-sample": BootstrapCase(X[:, None], Y, Q[:, None], np.array([3]), n_samples=1),
    "with-replacement-exceeds-pool": BootstrapCase(X, Y, Q, np.array([20])),
    "partial-library-with-tied-neighbors": BootstrapCase(
        TIED_LIBRARY,
        (np.arange(32) == 17).astype(float),
        np.zeros((1, 2)),
        np.array([16]),
        n_samples=8,
        replace=False,
        seed=438,
        batch_size=8,
    ),
    "full-library-permutations-with-ties": BootstrapCase(
        np.column_stack((np.arange(64, dtype=float), np.zeros(64))),
        np.sin(np.arange(64, dtype=float)),
        np.tile([[2.5, 0.0], [4.03, 0.0], [31.11, 0.0], [63.0, 0.0]], (16, 1)),
        np.array([16, 36, 64]),
        n_samples=17,
        replace=False,
        batch_size=8,
    ),
    "full-library-with-mask": BootstrapCase(X, Y, Q, np.array([12]), replace=False, mask=MASK),
    "sampled-mask-with-replacement": BootstrapCase(X, Y, Q, np.array([20]), mask=MASK),
    "smap": BootstrapCase(X, Y, Q, np.array([5, 12]), predict_func=partial(smap, theta=2.0), replace=False),
    "smap-with-mask": BootstrapCase(X, Y, Q, np.array([12]), predict_func=partial(smap, theta=2.0), replace=False, mask=MASK),
    "fresh-entropy": BootstrapCase(X, Y, Q, np.array([5]), seed=None),
}

BOOTSTRAP_INVALID = {
    "scalar-input": BootstrapCase(np.array(1.0), Y, Q, np.array([3])),
    "batched-input": BootstrapCase(X[None, :, None], Y, Q, np.array([3])),
    "batched-target": BootstrapCase(X, Y[None], Q, np.array([3])),
    "batched-query": BootstrapCase(X, Y, Q[None, :, None], np.array([3])),
    "empty-input": BootstrapCase(X[:0], Y[:0], Q, np.array([3])),
    "empty-features": BootstrapCase(np.empty((12, 0)), Y, np.empty((3, 0)), np.array([3])),
    "empty-target-dimensions": BootstrapCase(X, np.empty((12, 0)), Q, np.array([3])),
    "empty-query": BootstrapCase(X, Y, Q[:0], np.array([3])),
    "mismatched-lengths": BootstrapCase(X[:-1], Y, Q, np.array([3])),
    "mismatched-dimensions": BootstrapCase(X, Y, np.zeros((3, 2)), np.array([3])),
    "empty-sizes": BootstrapCase(X, Y, Q, np.array([], dtype=int)),
    "scalar-size": BootstrapCase(X, Y, Q, np.array(3)),
    "matrix-sizes": BootstrapCase(X, Y, Q, np.array([[3]])),
    "float-sizes": BootstrapCase(X, Y, Q, np.array([3.0])),
    "boolean-sizes": BootstrapCase(X, Y, Q, np.array([True])),
    "zero-size": BootstrapCase(X, Y, Q, np.array([0])),
    "negative-size": BootstrapCase(X, Y, Q, np.array([-1])),
    "nonreplacement-exceeds-pool": BootstrapCase(X, Y, Q, np.array([13]), replace=False),
    "zero-samples": BootstrapCase(X, Y, Q, np.array([3]), n_samples=0),
    "negative-samples": BootstrapCase(X, Y, Q, np.array([3]), n_samples=-1),
    "zero-batch": BootstrapCase(X, Y, Q, np.array([3]), batch_size=0),
    "negative-batch": BootstrapCase(X, Y, Q, np.array([3]), batch_size=-1),
    "negative-seed": BootstrapCase(X, Y, Q, np.array([3]), seed=-1),
    "insufficient-neighbors-at-execution": BootstrapCase(X, Y, Q, np.array([1])),
    "masked-out-at-execution": BootstrapCase(X, Y, Q, np.array([12]), replace=False, mask=np.zeros_like(MASK)),
}


@pytest.mark.parametrize("case", BOOTSTRAP_VALID.values(), ids=BOOTSTRAP_VALID.keys())
def test_bootstrap_valid(case: BootstrapCase) -> None:
    check_bootstrap(*case)


@pytest.mark.parametrize("case", BOOTSTRAP_INVALID.values(), ids=BOOTSTRAP_INVALID.keys())
def test_bootstrap_invalid(case: BootstrapCase) -> None:
    mask = case.mask
    with pytest.raises(ValueError):
        for _, samples in bootstrap(
            case.x,
            case.y,
            case.q,
            case.lib_sizes,
            predict_func=case.predict_func,
            n_samples=case.n_samples,
            replace=case.replace,
            seed=case.seed,
            mask=None if mask is None else (lambda *index: mask[index]),
            batch_size=case.batch_size,
        ):
            samples()
