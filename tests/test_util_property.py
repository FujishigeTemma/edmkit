import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.util import autocorrelation, dtw, pairwise_distance_np
from tests.conftest import finite_floats

finite_float64 = finite_floats()


@st.composite
def point_clouds(draw):
    n = draw(st.integers(min_value=2, max_value=10))
    d = draw(st.integers(min_value=1, max_value=4))
    return draw(hnp.arrays(np.float64, (n, d), elements=finite_float64))


@st.composite
def translated_clouds(draw):
    cloud = draw(point_clouds())  # ty: ignore[missing-argument]
    shift = draw(hnp.arrays(np.float64, (1, cloud.shape[1]), elements=finite_float64))
    return cloud, shift


@st.composite
def paired_series(draw):
    n = draw(st.integers(min_value=2, max_value=15))
    d = draw(st.integers(min_value=1, max_value=3))
    a = draw(hnp.arrays(np.float64, (n, d), elements=finite_float64))
    b = draw(hnp.arrays(np.float64, (n, d), elements=finite_float64))
    return a, b


@st.composite
def nonconstant_series(draw):
    n = draw(st.integers(min_value=8, max_value=64))
    x = draw(hnp.arrays(np.float64, n, elements=finite_float64))
    assume(np.std(x) > 1e-10)
    return x


class TestDistanceProperties:
    @given(cloud=point_clouds())  # ty: ignore[missing-argument]
    def test_self_distance_diagonal_is_zero(self, cloud):
        distances = pairwise_distance_np(cloud)
        np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-10, rtol=1e-10)

    @given(data=translated_clouds())  # ty: ignore[missing-argument]
    def test_translation_does_not_change_distances(self, data):
        cloud, shift = data
        original = pairwise_distance_np(cloud)
        shifted = pairwise_distance_np(cloud + shift)
        np.testing.assert_allclose(shifted, original, atol=1e-10, rtol=1e-10)

    @given(pair=paired_series())  # ty: ignore[missing-argument]
    def test_dtw_is_symmetric(self, pair):
        a, b = pair
        np.testing.assert_allclose(dtw(a, b), dtw(b, a), atol=1e-12, rtol=1e-12)

    @given(x=nonconstant_series())  # ty: ignore[missing-argument]
    def test_autocorrelation_has_unit_lag_zero(self, x):
        result = autocorrelation(x, max_lag=5)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-10, rtol=1e-10)
