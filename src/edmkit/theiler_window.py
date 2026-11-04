import numpy as np

from edmkit.types import Mask

__all__ = ["theiler_window"]


def theiler_window(
    t1: np.ndarray,
    t2: np.ndarray,
    width: int,
) -> Mask:
    """
    Build a per-query mask that excludes temporally close library points.

    Passing the result as `mask` to `simplex_projection`, `soft_simplex_projection`, or `smap`
    gives leave-one-out prediction with Theiler window exclusion when `Q` is `X` and `t1` is `t2`.

    Parameters
    ----------
    t1 : np.ndarray
        Time indices of the query points, shape (M,) or (B, M).
    t2 : np.ndarray
        Time indices of the library points, shape (N,) or (B, N).
    width : int
        Theiler window half-width. Library points ``j`` where ``|t1[i] - t2[j]| <= width``
        are excluded when predicting query ``i``. For lagged embedding, use ``(E - 1) * tau``.

    Returns
    -------
    mask : Mask
        ``mask(i, j)``, or ``mask(b, i, j)``, True where the library point lies outside the window.
    """
    t1, t2 = np.asarray(t1), np.asarray(t2)

    def mask(*index: np.ndarray) -> np.ndarray:
        *batch, i, j = index
        query, library = t1[(*batch, i)], t2[(*batch, j)]
        return (library < query - width) | (library > query + width)

    return mask
