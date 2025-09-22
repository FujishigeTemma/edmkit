import numpy as np

from edmkit.embedding import lagged_embed


def test_lagged_embed():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    tau = 2
    e = 3

    X = lagged_embed(x, tau, e)
    assert (X == np.array([[5, 3, 1], [6, 4, 2], [7, 5, 3], [8, 6, 4], [9, 7, 5]])).all()
