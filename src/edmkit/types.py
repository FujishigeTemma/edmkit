from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

if TYPE_CHECKING:
    from tinygrad import Tensor


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
        mask: np.ndarray | None = None,
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self,
        X: Tensor,
        Y: Tensor,
        Q: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor: ...
