import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from edmkit.embedding import scan


class TestScanProperties:
    def test_default_self_target_matches_explicit_target(self, logistic_map: np.ndarray):
        implicit = scan(logistic_map, E=[2, 3], tau=[1, 2])
        explicit = scan(logistic_map, logistic_map, E=[2, 3], tau=[1, 2])
        np.testing.assert_allclose(implicit, explicit, atol=1e-12, rtol=1e-12)

    @given(extra_e=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=4))
    def test_scores_for_same_e_are_independent_of_other_candidates(self, logistic_map: np.ndarray, extra_e: list[int]):
        target_e = 3
        single = scan(logistic_map, E=[target_e], tau=[1, 2])
        merged_e = sorted(set([target_e, *extra_e]))
        multi = scan(logistic_map, E=merged_e, tau=[1, 2])
        idx = merged_e.index(target_e)
        np.testing.assert_allclose(single[0], multi[idx], atol=1e-12, rtol=1e-12)
