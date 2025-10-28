import numpy as np
from scipy.spatial.distance import cdist
from tinygrad import Tensor


def pad(As: list[np.ndarray]):
    """Pad the `np.ndarray` in `Xs` to merge them into a single `np.ndarray`.

    Parameters
    ----------
        `As` : `list` of `np.ndarray` of shape `(L, D_i)`

    Returns
    -------
        Single `np.ndarray` of shape `(B, L, max(D))` where `B` is `len(As)`

    Raises
    ------
    ValueError
        - If any array in `As` is not 2D.
        - If the first dimension of all arrays in `As` are not equal.
    """
    if not all(A.ndim == 2 for A in As):
        raise ValueError(f"All arrays must be 2D, got {[A.ndim for A in As]}")
    if not all(A.shape[0] == As[0].shape[0] for A in As):
        raise ValueError(f"All arrays must have the same length, got {[A.shape[0] for A in As]}")

    B = len(As)
    L = As[0].shape[0]
    max_D = max(t.shape[-1] for t in As)  # type: ignore

    A = np.zeros((B, L, max_D), dtype=As[0].dtype)
    for i, x in enumerate(As):
        A[i, :, : x.shape[-1]] = x

    return np.ascontiguousarray(A)


def pairwise_distance(A: Tensor, B: Tensor | None = None) -> Tensor:
    """Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).

    Parameters
    ----------
    `A` : `Tensor` of shape `(L, D)` or `(B, L, D)`
        - `B`: batch size
        - `L`: number of points
        - `D`: dimension of each point
    `B` : `Tensor` of shape `(L', D)` or `(B, L', D)`
        - `B`: batch size
        - `L'`: number of points
        - `D`: dimension of each point

    Returns
    -------
    When `A` is of shape `(L, D)`:
        `Tensor` of shape `(L, L)` [or `(L, L')`] where the element at position `(i, j)` is the squared Euclidean distance between `A[i]` and `A[j]` [or between `A[i]` and `B[j]`].
    When `A` is of shape `(B, L, D)`:
        `Tensor` of shape `(B, L, L)` [or `(B, L, L')`] where the element at position `(b, i, j)` is the squared Euclidean distance between `A[b, i]` and `A[b, j]`.

    Raises
    ------
    ValueError
        - If `A` is not a 2D or 3D tensor.
        - If `B` is not `None` and `A` and `B` have different number of dimensions.
    """
    if A.ndim != 2 and A.ndim != 3:
        raise ValueError(f"A must be a 2D or 3D tensor, got A.ndim={A.ndim}")
    if B is not None and A.ndim != B.ndim:
        raise ValueError(f"A and B must have the same number of dimensions, got A.ndim={A.ndim}, B.ndim={B.ndim}")

    if B is None:
        B = A

    A_sq = A.pow(2).sum(-1, keepdim=True)
    B_sq = B.pow(2).sum(-1, keepdim=True).transpose(-1, -2)

    D: Tensor = A_sq + B_sq - 2 * A.matmul(B.transpose(-1, -2))  # type: ignore

    return D.clamp(min_=0)


def pairwise_distance_np(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Compute the pairwise squared Euclidean distance between points in `A` (or between points in `A` and `B`).

    Parameters
    ----------
    `A` : `np.ndarray` of shape `(L, D)` or `(B, L, D)`
        - `B`: batch size
        - `L`: number of points
        - `D`: dimension of each point
    `B` : `np.ndarray` of shape `(L', D)` or `(B, L', D)`
        - `B`: batch size
        - `L'`: number of points
        - `D`: dimension of each point

    Returns
    -------
    When `A` is of shape `(L, D)`:
        `np.ndarray` of shape `(L, L)` [or `(L, L')`] where the element at position `(i, j)` is the squared Euclidean distance between `A[i]` and `A[j]` [or between `A[i]` and `B[j]`].
    When `A` is of shape `(B, L, D)`:
        `np.ndarray` of shape `(B, L, L)` [or `(B, L, L')`] where the element at position `(b, i, j)` is the squared Euclidean distance between `A[b, i]` and `A[b, j]`.

    Raises
    ------
    ValueError
        - If `A` is not a 2D or 3D array.
        - If `B` is not `None` and `A` and `B` have different number of dimensions.
    """
    if A.ndim != 2 and A.ndim != 3:
        raise ValueError(f"A must be a 2D or 3D array, got A.ndim={A.ndim}")
    if B is not None and A.ndim != B.ndim:
        raise ValueError(f"A and B must have the same number of dimensions, got A.ndim={A.ndim}, B.ndim={B.ndim}")

    if B is None:
        B = A

    A_sq = np.sum(A**2, axis=-1, keepdims=True)
    B_sq = np.sum(B**2, axis=-1, keepdims=True).swapaxes(-1, -2)

    D: np.ndarray = A_sq + B_sq - 2 * np.matmul(A, B.swapaxes(-1, -2))

    return np.clip(D, a_min=0, a_max=None)


def dtw(A: np.ndarray, B: np.ndarray):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences `x` and `y`.

    Parameters
    ----------
        `A` : Tensor of shape `(N,D)`
        `B` : Tensor of shape `(M,D)`

    Returns
    -------
        distance : float
    """
    N: int = A.shape[0]
    M: int = B.shape[0]

    D = cdist(A, B, metric="euclidean")

    dp = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0  # left-top corner

    # Process the DP table along anti-diagonals
    # Grouping by anti-diagonals allows for vectorized computation: i + j = k
    for k in range(2, N + M + 1):
        i_start = max(1, k - M)
        i_end = min(N, k - 1)
        if i_start > i_end:
            continue

        # i and j are vectors of indices that satisfy i + j = k
        i = np.arange(i_start, i_end + 1)
        j = k - i

        # Vectorized version of dp[i,j] = D[i-1, j-1] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
        #                                               top            left       top-left
        min_prev = np.minimum(np.minimum(dp[i - 1, j], dp[i, j - 1]), dp[i - 1, j - 1])
        dp[i, j] = D[i - 1, j - 1] + min_prev

    return dp[N, M]


def autocorrelation(x: np.ndarray, max_lag: int, step: int = 1):
    """
    Computes the autocorrelation of a given 1D numpy array up to a specified maximum lag.

    Parameters
    ----------
        `x` : `np.ndarray` The input array for which to compute the autocorrelation.
        `max_lag` : `int` The maximum lag up to which the autocorrelation is computed.
        `step` : `int`, `optional` The step size for the lag. Default is 1.

    Returns
    -------
        `np.ndarray` of shape `(max_lag // step + 1,)` containing the autocorrelation values.
    """
    x = x - np.mean(x)

    n = len(x)
    # next_pow2 is the next power of 2 greater than or equal to 2 * n - 1 for efficient FFT computation
    next_pow2 = int(2 ** np.ceil(np.log2(2 * n - 1)))
    padded = np.zeros(next_pow2)
    padded[:n] = x

    f = np.fft.fft(padded)
    # Wiener-Khinchin theorem
    acf = np.fft.ifft(f * np.conjugate(f)).real
    acf = acf[:n]

    # Normalization
    acf = acf / (n * np.var(x))

    lags = np.arange(0, min(max_lag, n), step)
    return acf[lags]
