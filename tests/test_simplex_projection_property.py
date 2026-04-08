import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.simplex_projection import loo, simplex_projection
from tests.conftest import finite_floats

finite_float64 = finite_floats(10.0)


@st.composite
def simplex_inputs(draw):
    e = draw(st.integers(min_value=1, max_value=4))
    n = draw(st.integers(min_value=e + 2, max_value=20))
    m = draw(st.integers(min_value=1, max_value=6))
    x = draw(hnp.arrays(np.float64, (n, e), elements=finite_float64))
    y = draw(hnp.arrays(np.float64, n, elements=finite_float64))
    q = draw(hnp.arrays(np.float64, (m, e), elements=finite_float64))
    return x, y, q


class TestSimplexProjectionProperties:
    @given(data=simplex_inputs())  # ty: ignore[missing-argument]
    def test_permutation_invariance_of_library(self, data):
        x, y, q = data
        x = x.copy()
        x[:, 0] += np.arange(len(x)) * 1e-3
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(x))
        expected = simplex_projection(x, y, q)
        actual = simplex_projection(x[perm], y[perm], q)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    @given(data=simplex_inputs())  # ty: ignore[missing-argument]
    def test_scalar_predictions_stay_inside_target_range(self, data):
        x, y, q = data
        predictions = np.atleast_1d(simplex_projection(x, y, q))
        assert np.all(predictions >= y.min() - 1e-10)
        assert np.all(predictions <= y.max() + 1e-10)


@st.composite
def theiler_inputs(draw):
    """Generate valid inputs for theiler_window testing.

    X values are randomly perturbed to avoid kd-tree tie-breaking
    non-determinism between the full-tree and per-query naive paths.
    """
    E = draw(st.integers(min_value=1, max_value=4))
    W = draw(st.integers(min_value=1, max_value=10))
    k = E + 1
    min_n = 2 * W + 1 + k
    N = draw(st.integers(min_value=min_n, max_value=max(min_n, 40)))
    X = draw(hnp.arrays(np.float64, (N, E), elements=finite_float64))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    X = X + rng.uniform(-1e-6, 1e-6, X.shape)
    Y = draw(hnp.arrays(np.float64, N, elements=finite_float64))
    return X, Y, W


class TestSimplexProjectionTheilerWindowProperties:
    @given(data=theiler_inputs())  # ty: ignore[missing-argument]
    @settings(deadline=5000)
    def test_matches_naive_loop(self, data):
        """theiler_window must match per-sample simplex_projection with manual exclusion."""
        X, Y, W = data
        N = X.shape[0]

        expected = np.empty(N)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[max(0, i - W) : min(N, i + W + 1)] = False
            expected[i] = simplex_projection(X[mask], Y[mask], X[i : i + 1]).item()

        actual = loo(X, Y, theiler_window=W)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
