"""Chronological train/validation/test boundaries shared by every experiment arm.

Boundaries are half-open time-step index ranges [start, end) on the array's time axis.
Timestamps follow the upstream notebook convention: 2023-01-01 00:00, 5-minute steps,
timezone not proven.
"""

import json
from pathlib import Path

import pandas as pd

from aiori.config import STEPS_PER_DAY

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.7, 0.1, 0.2)
TIME_ORIGIN = "2023-01-01 00:00"
STEP_MINUTES = 5


def make_splits(n_steps, fractions=DEFAULT_FRACTIONS, steps_per_day=STEPS_PER_DAY):
    """Split whole days chronologically by the given fractions (train, val, test)."""
    if n_steps % steps_per_day:
        raise ValueError(f"{n_steps} steps is not a whole number of days of {steps_per_day} steps")
    days = n_steps // steps_per_day
    train_days = round(fractions[0] * days)
    val_days = round(fractions[1] * days)
    test_days = days - train_days - val_days
    if min(train_days, val_days, test_days) < 1:
        raise ValueError(f"Fractions {fractions} leave an empty split over {days} days")
    b1 = train_days * steps_per_day
    b2 = b1 + val_days * steps_per_day
    out = {"train": (0, b1), "val": (b1, b2), "test": (b2, n_steps)}
    validate_splits(out, n_steps)
    return out


def validate_splits(splits, n_steps):
    """Require ordered, contiguous, non-overlapping splits covering [0, n_steps)."""
    if tuple(splits) != SPLIT_NAMES:
        raise ValueError(f"Expected splits {SPLIT_NAMES}; got {tuple(splits)}")
    prev_end = 0
    for name in SPLIT_NAMES:
        start, end = splits[name]
        if start != prev_end or end <= start:
            raise ValueError(f"Split {name}={splits[name]} is not contiguous after {prev_end}")
        prev_end = end
    if prev_end != n_steps:
        raise ValueError(f"Splits end at {prev_end}, but the series has {n_steps} steps")


def _timestamp(step):
    return (pd.Timestamp(TIME_ORIGIN) + pd.Timedelta(minutes=STEP_MINUTES * int(step))).strftime("%Y-%m-%d %H:%M")


def split_config(splits, n_steps, fractions=DEFAULT_FRACTIONS, steps_per_day=STEPS_PER_DAY):
    validate_splits(splits, n_steps)
    return {
        "n_steps": n_steps,
        "steps_per_day": steps_per_day,
        "fractions_requested": list(fractions),
        "index_convention": "half-open [start, end) indices on the array time axis",
        "boundaries": {k: list(v) for k, v in splits.items()},
        "timestamps": {k: {"first": _timestamp(s), "last": _timestamp(e - 1)} for k, (s, e) in splits.items()},
        "time_convention": f"upstream notebook date_range from {TIME_ORIGIN}, {STEP_MINUTES}-min steps; "
                           "no time axis in the array and timezone not proven",
        "window_rule": "a window belongs to the split containing all of its target steps; "
                       "input context may come from earlier steps (including the previous split)",
    }


def save_split_config(path, cfg):
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_split_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {k: tuple(cfg["boundaries"][k]) for k in SPLIT_NAMES}
    validate_splits(splits, cfg["n_steps"])
    return splits
