from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

from edmkit.metrics.mae import mae
from edmkit.metrics.pearson_correlation import pearson_correlation
from edmkit.metrics.rmse import rmse

__all__ = ["MetricFunc", "pearson_correlation", "rmse", "mae"]

if TYPE_CHECKING:
    from tinygrad import Tensor


class MetricFunc(Protocol):
    """Metric function protocol.

    Accepts predictions and observations of the same shape and returns a metric value.
    """

    @overload
    def __call__(self, predictions: np.ndarray, observations: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, predictions: Tensor, observations: Tensor) -> Tensor: ...


if TYPE_CHECKING:
    func: MetricFunc

    func = pearson_correlation
    func = rmse
    func = mae
