import numpy as np

__all__ = ["embed"]


def embed(coordinates: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Delay vectors for an arbitrary choice of variables and lags.

    Parameters
    ----------
    coordinates : np.ndarray
        Integer array of shape ``(E, 2)``. Row ``j`` is ``(variable, lag)``, selecting
        observation ``x[t - lag, variable]`` for the delay vector at time ``t``.
        A negative lag selects an observation taken after time ``t``.
    x : np.ndarray
        Observations of shape ``(T, d)``, or ``(T,)``.

    Returns
    -------
    X : np.ndarray
        Delay vectors of shape ``(N, E)``, one row per time in `t`.
    t : np.ndarray
        Times of shape ``(N,)`` at which every coordinate is available, that is
        ``lags.max() <= t <= T - 1 + lags.min()``. A time outside ``[0, T - 1]`` is
        included when every coordinate still reads an observation inside `x`.

    Raises
    ------
    ValueError
        - If `coordinates` is not an integer array of shape ``(E, 2)`` with ``E >= 1``.
        - If `x` is not a 1D or 2D array.
        - If a variable index lies outside `x`.
        - If no time has every coordinate available.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import embed

    x = np.arange(20).reshape(10, 2)  # ten times, two variables

    # Classical univariate delay coordinates.
    tau, E = 2, 3
    coordinates = np.array([[0, tau * j] for j in range(E)])
    X, t = embed(coordinates, x)
    print(X)
    print(t)
    # [[ 8  4  0]
    #  [10  6  2]
    #  [12  8  4]
    #  [14 10  6]
    #  [16 12  8]
    #  [18 14 10]]
    # [4 5 6 7 8 9]

    # One step ahead of the same coordinates, as iterated forecasting needs.
    Y, t_ahead = embed(coordinates - [0, 1], x)
    ```
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError(f"coordinates must have shape (E, 2), got coordinates.shape={coordinates.shape}")
    if coordinates.shape[0] < 1:
        raise ValueError("coordinates must select at least one delay coordinate")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError(f"coordinates must be an integer array, got coordinates.dtype={coordinates.dtype}")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"x must have shape (T,) or (T, d), got x.shape={x.shape}")

    variables, lags = coordinates[:, 0], coordinates[:, 1]
    if variables.min() < 0 or variables.max() >= x.shape[1]:
        raise ValueError(f"variable indices must lie in [0, {x.shape[1] - 1}], got {variables.min()} to {variables.max()}")

    # A delay vector at time t needs 0 <= t - lag <= T - 1 for every coordinate.
    first = int(lags.max())
    last = x.shape[0] - 1 + int(lags.min())
    if last < first:
        raise ValueError(f"no time has every coordinate available: lags span {int(lags.min())} to {int(lags.max())} for T={x.shape[0]}")

    t = np.arange(first, last + 1)
    return x[t[:, None] - lags[None, :], variables[None, :]], t
