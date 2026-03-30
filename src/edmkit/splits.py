from typing import TYPE_CHECKING, Callable, NamedTuple, TypeAlias

import numpy as np


class Fold(NamedTuple):
    """A single train/validation split."""

    train: np.ndarray  # index array
    validation: np.ndarray  # index array


SplitFunc: TypeAlias = Callable[[int], list[Fold]]


def temporal_fold(
    n: int,
    train_ratio: float,
    *,
    gap: int = 0,
) -> Fold:
    """Single temporal split.

    Layout::

        [===== Train =====][gap][== Val ==]

    Parameters
    ----------
    n : int
        Total number of samples.
    train_ratio : float
        Fraction of data for training, in ``(0, 1)``.
    gap : int
        Points to skip between train and validation.

    Returns
    -------
    Fold

    Raises
    ------
    ValueError
        If ratio is invalid or any split would be empty.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    train_end = int(n * train_ratio)
    validation_start = train_end + gap

    train_indices = np.arange(0, train_end)
    validation_indices = np.arange(validation_start, n)

    if len(train_indices) == 0:
        raise ValueError("Training set is empty")
    if len(validation_indices) == 0:
        raise ValueError("Validation set is empty")

    return Fold(train_indices, validation_indices)


def expanding_folds(
    n: int,
    *,
    initial_train_size: int,
    validation_size: int,
    stride: int | None = None,
    gap: int = 0,
) -> list[Fold]:
    """Generate folds for expanding-window time-series cross-validation.

    The training set starts at ``initial_train_size`` and grows with each
    fold while the validation window slides forward.

    Parameters
    ----------
    n : int
        Total number of samples.
    initial_train_size : int
        Minimum training samples (first fold).
    validation_size : int
        Validation samples per fold.
    stride : int or None
        Step size for the validation window. Defaults to ``validation_size``.
    gap : int
        Points to skip between train and validation.

    Returns
    -------
    list[Fold]
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if initial_train_size <= 0:
        raise ValueError(f"initial_train_size must be positive, got {initial_train_size}")
    if validation_size <= 0:
        raise ValueError(f"validation_size must be positive, got {validation_size}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    if stride is None:
        stride = validation_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    folds: list[Fold] = []
    validation_start = initial_train_size + gap
    while validation_start + validation_size <= n:
        train_end = validation_start - gap
        train_idx = np.arange(train_end)
        validation_idx = np.arange(validation_start, validation_start + validation_size)
        folds.append(Fold(train_idx, validation_idx))
        validation_start += stride
    return folds


def sliding_folds(
    n: int,
    *,
    train_size: int,
    validation_size: int,
    stride: int | None = None,
    gap: int = 0,
) -> list[Fold]:
    """Generate folds for sliding-window time-series cross-validation.

    A fixed-size training window slides forward together with the
    validation window, discarding the oldest observations at each step.

    Parameters
    ----------
    n : int
        Total number of samples.
    train_size : int
        Fixed training samples per fold.
    validation_size : int
        Validation samples per fold.
    stride : int or None
        Step size for the sliding window. Defaults to ``validation_size``.
    gap : int
        Points to skip between train and validation.

    Returns
    -------
    list[Fold]
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if validation_size <= 0:
        raise ValueError(f"validation_size must be positive, got {validation_size}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

    if stride is None:
        stride = validation_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    folds: list[Fold] = []
    validation_start = train_size + gap
    while validation_start + validation_size <= n:
        train_start = validation_start - gap - train_size
        train_idx = np.arange(train_start, train_start + train_size)
        validation_idx = np.arange(validation_start, validation_start + validation_size)
        folds.append(Fold(train_idx, validation_idx))
        validation_start += stride
    return folds


if TYPE_CHECKING:
    from functools import partial

    func: SplitFunc

    def temporal_folds(n: int) -> list[Fold]:
        return [temporal_fold(n, train_ratio=0.8, gap=10)]

    func = temporal_folds
    func = partial(
        expanding_folds,
        initial_train_size=100,
        validation_size=20,
        stride=20,
    )
    func = partial(
        sliding_folds,
        train_size=100,
        validation_size=20,
        stride=20,
    )
