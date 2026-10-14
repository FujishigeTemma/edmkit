from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from edmkit.smap.linear_operator import linear_operator

__all__ = ["smap"]

if TYPE_CHECKING:
    from tinygrad import Tensor


@overload
def smap(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def smap(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
) -> Tensor: ...


def smap(
    X,
    Y,
    Q,
    *,
    theta,
    alpha=1e-10,
    mask=None,
):
    """
    Perform S-Map (local linear regression) from `X` to `Y`.

    Parameters
    ----------
    X : np.ndarray or Tensor
        The input data of shape (N,) or (N, E) or (B, N, E)
    Y : np.ndarray or Tensor
        The target data of shape (N,) or (N, E') or (B, N, E')
    Q : np.ndarray or Tensor
        The query points of shape (M,) or (M, E) or (B, M, E) for which to make predictions.
    theta : float
        Locality parameter. (0: global linear, >0: local linear)
    alpha : float, default 1e-10
        Regularization parameter to stabilize the inversion.
    mask : np.ndarray or Tensor or None
        Boolean mask of shape (M, N) or (B, M, N) indicating, for each query in `Q`, which library points to include in the regression.

    Returns
    -------
    predictions : np.ndarray or Tensor
        The predicted values based on the weighted linear regression.
        Shape (M,) if `Y` is 1D, (M, E') if `Y` is 2D, or (B, M, E') if `Y` is 3D.

    Raises
    ------
    ValueError
        - If `theta` is negative.
        - If the input arrays `X` and `Y` do not have the same number of points.
        - If the batch size or dimension of `X` and `Q` do not match.
        - If `mask` does not have shape (M, N) or (B, M, N).
        - If fewer than E + 1 library points are unmasked for some query.

    Examples
    --------
    ```python
    import numpy as np

    from edmkit.embedding import embed
    from edmkit.smap import smap

    # Generate a simple time series (logistic map)
    N = 300
    x = np.zeros(N)
    x[0] = 0.4
    for i in range(1, N):
        x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])

    tau = 2
    E = 3

    coordinates = np.array([[0, tau * j] for j in range(E)])
    embedding, _ = embed(coordinates, x)
    shift = tau * (E - 1)

    lib_size = 200
    Tp = 1
    X = embedding[:lib_size - shift]
    Y = x[shift + Tp : lib_size + Tp]
    Q = embedding[lib_size - shift : -Tp]
    actual = x[lib_size + Tp :]

    # Local linear with theta=4.0
    predictions = smap(X, Y, Q, theta=4.0)
    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"Correlation (theta=4.0): {correlation:.3f}")

    # Global linear with theta=0.0
    predictions_global = smap(X, Y, Q, theta=0.0)
    correlation_global = np.corrcoef(predictions_global, actual)[0, 1]
    print(f"Correlation (theta=0.0): {correlation_global:.3f}")
    ```
    """
    if isinstance(X, np.ndarray):
        return _numpy(X, Y, Q, theta=theta, alpha=alpha, mask=mask)
    return _tensor(X, Y, Q, theta=theta, alpha=alpha, mask=mask)


def _numpy(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: np.ndarray | None = None,
):
    return_ndim = Y.ndim

    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.ndim != Y.ndim:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")

    S = linear_operator(X, Q, theta=theta, alpha=alpha, mask=mask)  # (M, N) or (B, M, N)

    predictions = np.matmul(S, Y)  # (M, E') or (B, M, E')

    if return_ndim == 1:
        return predictions[..., 0]  # (M,)
    return predictions


def _tensor(
    X: Tensor,
    Y: Tensor,
    Q: Tensor,
    *,
    theta: float,
    alpha: float = 1e-10,
    mask: Tensor | None = None,
):
    return_ndim = Y.ndim

    # ensure at least 2D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if X.ndim != Y.ndim:
        raise ValueError(f"X, Y, and Q must all be 2D or all be 3D tensors, got X.ndim={X.ndim}, Y.ndim={Y.ndim}, Q.ndim={Q.ndim}")
    if X.shape[:-1] != Y.shape[:-1]:
        raise ValueError(f"batch size and length of X and Y must match, got X.shape={X.shape} and Y.shape={Y.shape}")

    S = linear_operator(X, Q, theta=theta, alpha=alpha, mask=mask)  # (M, N) or (B, M, N)

    predictions: Tensor = S.matmul(Y)  # (M, E') or (B, M, E')

    if return_ndim == 1:
        return predictions[..., 0]  # (M,)
    return predictions


if TYPE_CHECKING:
    from functools import partial

    from edmkit.types import PredictFunc

    func: PredictFunc = partial(smap, theta=4.0)
