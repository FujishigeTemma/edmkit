import numpy as np

from edmkit.tensor import Tensor


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
    AssertionError
        - If any array in `As` is not 2D.
        - If the first dimension of all arrays in `As` are not equal.
    """
    assert all(A.ndim == 2 for A in As), f"All arrays must be 2D, got {[A.ndim for A in As]}"
    assert all(A.shape[0] == As[0].shape[0] for A in As), f"All arrays must have the same length, got {[A.shape[0] for A in As]}"

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
    AssertionError
        - If `A` is not a 2D or 3D tensor.
        - If `B` is not `None` and `A` and `B` have different number of dimensions.
    """
    assert A.ndim == 2 or A.ndim == 3, f"A must be a 2D or 3D tensor, got A.ndim={A.ndim}"
    assert B is None or (A.ndim == B.ndim), f"A and B must have the same number of dimensions, got A.ndim={A.ndim}, B.ndim={B.ndim}"

    if B is None:
        B = A

    A_sq = A.pow(2).sum(-1, keepdim=True)
    B_sq = B.pow(2).sum(-1, keepdim=True).transpose(-1, -2)

    D = A_sq + B_sq - 2 * A.matmul(B.transpose(-1, -2))

    return D.clamp(min_=0)


def dtw(A: Tensor, B: Tensor):
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
    N = A.shape[0]
    M = B.shape[0]

    D: np.ndarray = pairwise_distance(A, B).numpy()

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


def topk(x: np.ndarray, k: int, largest=True):
    """Find the `k` largest or smallest elements in `x`.

    Parameters
    ----------
    `x` : `np.ndarray` of shape (N,)
    `k` : `int`
    `largest` : `bool`

    Returns
    -------
    `indices` : `np.ndarray` of shape `(k,)`
    `values` : `np.ndarray` of shape `(k,)`

    Raises
    ------
    AssertionError
        - If `x` is not a 1D array.
        - If `k` is not in the range `(0, len(x)]`.

    Notes
    -----
    `values` are sorted in ascending order. (i.e. `values[0]` is the smallest value in `values`)
    """
    assert x.ndim == 1, f"x must be a 1D array, got x.ndim={x.ndim}"
    assert k > 0 and k <= len(x), f"k must satisfy 0 < k <= len(x), got k={k}, len(x)={len(x)}"

    if largest:
        indices = np.argpartition(-x, k - 1)[:k]
    else:
        indices = np.argpartition(x, k - 1)[:k]

    argsort = np.argsort(x[indices], kind="stable")

    indices = indices[argsort]
    values = x[indices]

    return indices, values


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
