import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from edmkit.smap import smap
from tests.conftest import finite_floats

finite_float64 = finite_floats(10.0)


@st.composite
def smap_inputs(draw):
    e = draw(st.integers(min_value=1, max_value=3))
    n = draw(st.integers(min_value=max(12, e + 4), max_value=18))
    m = draw(st.integers(min_value=1, max_value=5))
    x = draw(hnp.arrays(np.float64, (n, e), elements=finite_float64))
    y = draw(hnp.arrays(np.float64, (n, 1), elements=finite_float64))
    q = draw(hnp.arrays(np.float64, (m, e), elements=finite_float64))
    # Keep the property focused on batching semantics rather than near-singular solves.
    x = x + np.linspace(0.0, 1.0, n)[:, None] * 1e-3
    return x, y, q


class TestSMapProperties:
    @given(data=smap_inputs())  # ty: ignore[missing-argument]
    def test_batch_path_matches_individual_calls(self, data):
        x, y, q = data
        x_batch = np.stack([x, x + 0.5], axis=0)
        y_batch = np.stack([y, y - 0.25], axis=0)
        q_batch = np.stack([q, q + 0.5], axis=0)

        actual = smap(x_batch, y_batch, q_batch, theta=2.0)

        for batch in range(x_batch.shape[0]):
            expected = smap(x_batch[batch], y_batch[batch], q_batch[batch], theta=2.0)
            np.testing.assert_allclose(actual[batch].squeeze(-1), np.asarray(expected), atol=1e-5, rtol=1e-5)
