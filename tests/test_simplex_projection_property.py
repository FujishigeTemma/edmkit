import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.simplex_projection import simplex_projection
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
