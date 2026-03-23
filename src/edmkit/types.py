from typing import Protocol

import numpy as np


class PredictFunc(Protocol):
    """Prediction function protocol.

    Accepts library X, target Y, query Q, and optional mask.
    """

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Q: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray: ...
