"""Index-based sliding windows: `input_len` past steps -> `horizon` future steps.

A window is identified by (sensor, t0), where t0 is its first target step. Inputs are
steps [t0 - input_len, t0); targets are [t0, t0 + horizon). A window belongs to a split
only if all its targets lie inside that split. Inputs may reach into earlier splits
(past context), never later ones.
"""

import numpy as np

INPUT_LEN = 12
HORIZON = 12


def window_index(target_mask, splits, split, input_len=INPUT_LEN, horizon=HORIZON, min_observed_targets=1):
    """(N, 2) int array of [sensor, t0] for windows whose targets fall in `split`."""
    start, end = splits[split]
    first = max(start, input_len)
    last = end - horizon
    if last < first:
        return np.empty((0, 2), dtype=np.int64)
    t0 = np.arange(first, last + 1)
    # Observed-target count for every (sensor, t0) via a cumulative sum.
    csum = np.concatenate([np.zeros((len(target_mask), 1), np.int64),
                           np.cumsum(target_mask, axis=1, dtype=np.int64)], axis=1)
    counts = csum[:, t0 + horizon] - csum[:, t0]
    sensor, pos = np.nonzero(counts >= min_observed_targets)
    return np.stack([sensor, t0[pos]], axis=1).astype(np.int64)


def gather_windows(inputs, input_imputed, targets, target_mask, index, input_len=INPUT_LEN, horizon=HORIZON):
    """Materialise windows for an index.

    Returns X (N, input_len, 2) = [scaled filled input, observed flag], Y (N, horizon) with
    NaN where the target is missing, and M (N, horizon) bool target mask.
    """
    sensor, t0 = index[:, 0][:, None], index[:, 1][:, None]
    in_cols = t0 + np.arange(-input_len, 0)
    out_cols = t0 + np.arange(horizon)
    X = np.stack([inputs[sensor, in_cols], ~input_imputed[sensor, in_cols]], axis=-1).astype(np.float32)
    Y = targets[sensor, out_cols].astype(np.float32)
    M = target_mask[sensor, out_cols].astype(bool)
    if not np.array_equal(M, np.isfinite(Y)):
        raise ValueError("Target mask disagrees with missing targets")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in model inputs")
    return X, Y, M
