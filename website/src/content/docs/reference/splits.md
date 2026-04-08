---
title: splits
description: Time-series cross-validation strategies.
sidebar:
  order: 6
---

## `splits`

**Classes:**

Name | Description
---- | -----------
[`Fold`](#edmkit.splits.Fold) | A single train/validation split.

**Functions:**

Name | Description
---- | -----------
[`temporal_fold`](#edmkit.splits.temporal_fold) | Single temporal split.
[`expanding_folds`](#edmkit.splits.expanding_folds) | Generate folds for expanding-window time-series cross-validation.
[`sliding_folds`](#edmkit.splits.sliding_folds) | Generate folds for sliding-window time-series cross-validation.

### `Fold`

Bases: <code>[NamedTuple](#typing.NamedTuple)</code>

A single train/validation split.

### `temporal_fold`

```python
temporal_fold(n: int, train_ratio: float, *, gap: int = 0) -> Fold
```

Single temporal split.

Layout::

    [===== Train =====][gap][== Val ==]

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n` | <code>[int](#int)</code> | Total number of samples. | *required*
`train_ratio` | <code>[float](#float)</code> | Fraction of data for training, in ``(0, 1)``. | *required*
`gap` | <code>[int](#int)</code> | Points to skip between train and validation. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Fold](#edmkit.splits.Fold)</code> | 

**Raises:**

Type | Description
---- | -----------
<code>[ValueError](#ValueError)</code> | If ratio is invalid or any split would be empty.

### `expanding_folds`

```python
expanding_folds(n: int, *, initial_train_size: int, validation_size: int, stride: int | None = None, gap: int = 0) -> list[Fold]
```

Generate folds for expanding-window time-series cross-validation.

The training set starts at ``initial_train_size`` and grows with each
fold while the validation window slides forward.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n` | <code>[int](#int)</code> | Total number of samples. | *required*
`initial_train_size` | <code>[int](#int)</code> | Minimum training samples (first fold). | *required*
`validation_size` | <code>[int](#int)</code> | Validation samples per fold. | *required*
`stride` | <code>[int](#int) or None</code> | Step size for the validation window. Defaults to ``validation_size``. | <code>None</code>
`gap` | <code>[int](#int)</code> | Points to skip between train and validation. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Fold](#edmkit.splits.Fold)]</code> | 

### `sliding_folds`

```python
sliding_folds(n: int, *, train_size: int, validation_size: int, stride: int | None = None, gap: int = 0) -> list[Fold]
```

Generate folds for sliding-window time-series cross-validation.

A fixed-size training window slides forward together with the
validation window, discarding the oldest observations at each step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n` | <code>[int](#int)</code> | Total number of samples. | *required*
`train_size` | <code>[int](#int)</code> | Fixed training samples per fold. | *required*
`validation_size` | <code>[int](#int)</code> | Validation samples per fold. | *required*
`stride` | <code>[int](#int) or None</code> | Step size for the sliding window. Defaults to ``validation_size``. | <code>None</code>
`gap` | <code>[int](#int)</code> | Points to skip between train and validation. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Fold](#edmkit.splits.Fold)]</code> | 

