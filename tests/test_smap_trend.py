import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.smap import smap


class TestSMapTrend:
    def test_nonlinear_series_benefits_from_locality(self, logistic_map: np.ndarray):
        embedded = lagged_embed(logistic_map, tau=1, e=2)
        x = embedded[:300]
        y = logistic_map[2:302]
        q = embedded[300:-1]
        actual = logistic_map[302 : 302 + len(q)]

        global_predictions = smap(x, y, q, theta=0.0)
        local_predictions = smap(x, y, q, theta=4.0)

        global_rho = np.corrcoef(global_predictions, actual)[0, 1]
        local_rho = np.corrcoef(local_predictions, actual)[0, 1]

        assert local_rho > global_rho + 0.2
