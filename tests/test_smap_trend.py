import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.smap import smap


class TestSMapTrend:
    def test_nonlinear_series_benefits_from_locality(self, logistic_map: np.ndarray):
        tau, e, n_ahead = 1, 2, 1
        shift = (e - 1) * tau + n_ahead
        embedded = lagged_embed(logistic_map, tau=tau, e=e)
        library_size = 300
        x = embedded[:library_size]
        y = logistic_map[shift : shift + library_size]
        q = embedded[library_size:-n_ahead]
        actual = logistic_map[shift + library_size : shift + library_size + len(q)]

        global_predictions = smap(x, y, q, theta=0.0)
        local_predictions = smap(x, y, q, theta=4.0)

        global_rho = np.corrcoef(global_predictions, actual)[0, 1]
        local_rho = np.corrcoef(local_predictions, actual)[0, 1]

        assert local_rho > global_rho + 0.2
