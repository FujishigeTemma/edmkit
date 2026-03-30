import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.ccm import pearson_correlation
from tests.conftest import finite_floats

finite_float64 = finite_floats()


@st.composite
def nonconstant_pairs(draw):
    length = draw(st.integers(min_value=3, max_value=40))
    x = draw(hnp.arrays(np.float64, length, elements=finite_float64))
    y = draw(hnp.arrays(np.float64, length, elements=finite_float64))
    assume(np.std(x) > 1e-10)
    assume(np.std(y) > 1e-10)
    return x, y


class TestPearsonCorrelationProperties:
    @given(pair=nonconstant_pairs())  # ty: ignore[missing-argument]
    def test_is_symmetric(self, pair):
        x, y = pair
        np.testing.assert_allclose(pearson_correlation(x, y), pearson_correlation(y, x), atol=1e-12, rtol=1e-12)

    @given(pair=nonconstant_pairs())  # ty: ignore[missing-argument]
    def test_is_bounded_between_minus_one_and_one(self, pair):
        x, y = pair
        correlation = pearson_correlation(x, y)
        assert -1.0 - 1e-12 <= correlation <= 1.0 + 1e-12
