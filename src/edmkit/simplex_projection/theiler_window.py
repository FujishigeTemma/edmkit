import numpy as np

__all__ = ["theiler_window"]


def theiler_window(
    t1: np.ndarray,
    t2: np.ndarray,
    width: int,
) -> np.ndarray:
    """
    Build a per-query mask that excludes temporally close library points.

    Passing the result to `simplex_projection(X, Y, Q, mask=...)` gives leave-one-out
    prediction with Theiler window exclusion when `Q` is `X` and `t1` is `t2`.

    Parameters
    ----------
    t1 : np.ndarray
        Time indices of the query points, shape (M,) or (B, M). One row of the mask per entry.
    t2 : np.ndarray
        Time indices of the library points, shape (N,) or (B, N). One column of the mask per entry.
    width : int
        Theiler window half-width. Library points ``j`` where ``|t1[i] - t2[j]| <= width``
        are excluded when predicting query ``i``. For lagged embedding, use ``(E - 1) * tau``.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (M, N) or (B, M, N), True where the library point lies outside the window.
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    return np.abs(t1[..., :, None] - t2[..., None, :]) > width
