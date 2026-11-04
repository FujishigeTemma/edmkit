from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

__all__ = ["Mask", "PredictFunc"]

if TYPE_CHECKING:
    from tinygrad import Tensor

type Mask = Callable[..., np.ndarray]
"""``mask(i, j)`` (or ``mask(b, i, j)`` for batched inputs) is True where library point ``j`` may be used for query ``i``.

A dense boolean array `a` becomes a mask as ``lambda *index: a[index]``.
"""


class PredictFunc(Protocol):
    """Prediction function protocol.

    Accepts library X, target Y, query Q, and optional mask.
    """

    @overload
    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Q: np.ndarray,
        *,
        mask: Mask | None = None,
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self,
        X: Tensor,
        Y: Tensor,
        Q: Tensor,
        *,
        mask: Mask | None = None,
    ) -> Tensor: ...
