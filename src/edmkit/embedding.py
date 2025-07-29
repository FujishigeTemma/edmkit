import numpy as np


def lagged_embed(x: np.ndarray, tau: int, e: int):
    """Lagged embedding of a time series `x`.

    Parameters
    ----------
        `x` : `np.ndarray` of shape `(N,)`
        `tau` : `int`
        `e` : `int`

    Returns
    -------
        `np.ndarray` of shape `(N - (e - 1) * tau, e)`

    Raises
    ------
    AssertionError
        - If `x` is not a 1D array.
        - If `tau` or `e` is not positive.
        - If `e * tau >= len(x)`.

    Notes
    -----
    - While open to interpretation, it's generally more intuitive to consider the embedding as starting from the `(e - 1) * tau`th element of the original time series and ending at the `len(x) - 1`th element (the last value), rather than starting from the 0th element and ending at `len(x) - 1 - (e - 1) * tau`.
    - This distinction reflects whether we think of "attaching past values to the present" or "attaching future values to the present". The information content of the result is the same either way.
    - The use of `reversed` in the implementation emphasizes this perspective.

    Examples
    --------
    ```
    import numpy as np
    from edm.embedding import lagged_embed

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tau = 2
    e = 3

    E = lagged_embed(x, tau, e)
    print(E)
    print(E.shape)
    # [[4 2 0]
    #  [5 3 1]
    #  [6 4 2]
    #  [7 5 3]
    #  [8 6 4]
    #  [9 7 5]]
    # (6, 3)
    ```
    """
    assert len(x.shape) == 1, f"X must be a 1D array, got x.shape={x.shape}"
    assert tau > 0 and e > 0, f"tau and e must be positive, got tau={tau}, e={e}"
    assert (e - 1) * tau <= x.shape[0], f"e and tau must satisfy `(e - 1) * tau < len(X)`, got e={e}, tau={tau}"

    return np.array([x[tau * (e - 1) :]] + [x[tau * i : -tau * ((e - 1) - i)] for i in reversed(range(e - 1))]).transpose()
