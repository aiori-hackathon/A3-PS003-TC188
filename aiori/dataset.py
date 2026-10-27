"""Read-only access to a preprocessed run and PyTorch batches built from its window index.

A run is one output directory of `aiori.prepare_pilot` (primary pilot or the missing-data
validation set). Windows are materialised per batch with `aiori.windows.gather_windows`, so
preprocessing is reused, not duplicated. A `WindowDataset` can only read rows of
`windows[split]`; the training loop never sees validation or test indices.

Batch fields: x (B, 12, F) float32; y (B, 12) float32 z-scaled with missing targets set to
0.0; mask (B, 12) bool (True = observed target); sensor_pos, t0, row (B,) int64.
Features: [0] scaled filled flow, [1] observed flag (1 = observed, 0 = filled);
calendar=True adds [2] sin and [3] cos of the time-of-day slot of each input step.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aiori import config, windows

RUNS = {
    "primary": (config.PROCESSED_DIR, config.PROCESSED_DIR / "sensor_selection.csv"),
    "missing_validation": (config.PROCESSED_DIR / "missing_validation",
                           config.PROCESSED_DIR / "missing_validation_selection.csv"),
}
ARRAYS = ("inputs_scaled", "input_imputed", "targets_scaled", "observed_mask", "flow_raw", "tod_median",
          "array_index", "station_id")


@dataclass(frozen=True)
class Run:
    name: str
    inputs_scaled: np.ndarray
    input_imputed: np.ndarray
    targets_scaled: np.ndarray
    observed_mask: np.ndarray
    flow_raw: np.ndarray
    tod_median: np.ndarray
    array_index: np.ndarray
    station_id: np.ndarray       # provisional: node_order[array_index]
    scaler: dict                 # {"mean": [S], "std": [S], ...} fitted on training observations
    windows: dict                # split -> (N, 2) [sensor_pos, t0]
    client: np.ndarray           # (S,) proposed_client per sensor position
    steps_per_day: int

    @property
    def n_sensors(self):
        return len(self.array_index)


def _readonly(a):
    a = np.asarray(a)
    a.setflags(write=False)
    return a


def load_run(name="primary", run_dir=None, selection=None, splits=None):
    """`splits` limits which window indices are loaded (e.g. ("train", "val") keeps test windows out)."""
    default_dir, default_sel = RUNS.get(name, (None, None))
    run_dir, selection = Path(run_dir or default_dir), Path(selection or default_sel)
    with np.load(run_dir / "pilot_preprocessed.npz") as z:
        arrays = {k: _readonly(z[k]) for k in ARRAYS}
    with np.load(run_dir / "window_index.npz") as w:
        win = {k: _readonly(w[k]) for k in (splits or w.files)}
    scaler = json.loads((run_dir / "scaler_parameters.json").read_text(encoding="utf-8"))
    split_cfg = json.loads((run_dir / "split_config.json").read_text(encoding="utf-8"))
    sel = pd.read_csv(selection, usecols=["array_index", "proposed_client"])
    # Client membership is by sensor position, so the manifest order must equal the npz order.
    if not np.array_equal(sel["array_index"].to_numpy(), arrays["array_index"]):
        raise ValueError(f"{selection} order disagrees with {run_dir / 'pilot_preprocessed.npz'}")
    if not np.array_equal(scaler["array_index"], arrays["array_index"]):
        raise ValueError("Scaler parameters disagree with the preprocessed sensor order")
    return Run(name=name, scaler=scaler, windows=win, client=_readonly(sel["proposed_client"].to_numpy(str)),
               steps_per_day=int(split_cfg["steps_per_day"]), **arrays)


def client_positions(run):
    """{client: sorted sensor positions}."""
    return {c: np.flatnonzero(run.client == c) for c in np.unique(run.client)}


def calendar_features(t0, input_len=windows.INPUT_LEN, steps_per_day=config.STEPS_PER_DAY):
    """(B, input_len, 2) sin/cos of slot = t mod steps_per_day for each input step t."""
    slot = (np.asarray(t0)[:, None] + np.arange(-input_len, 0)) % steps_per_day
    angle = 2 * np.pi * slot / steps_per_day
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


class WindowDataset:
    def __init__(self, run, split, sensor_positions=None, calendar=False):
        if split not in run.windows:
            raise KeyError(f"Unknown split {split!r}")
        index = run.windows[split]
        if sensor_positions is not None:
            index = index[np.isin(index[:, 0], np.asarray(sensor_positions))]
        self.run, self.split, self.calendar = run, split, calendar
        self.index = _readonly(np.array(index))
        self.n_features = 4 if calendar else 2

    def __len__(self):
        return len(self.index)

    def get_batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        idx = self.index[rows]
        r = self.run
        x, y, m = windows.gather_windows(r.inputs_scaled, r.input_imputed, r.targets_scaled, r.observed_mask, idx)
        if self.calendar:
            x = np.concatenate([x, calendar_features(idx[:, 1], steps_per_day=r.steps_per_day)], axis=-1)
        y = np.where(m, y, 0.0).astype(np.float32)       # NaN never leaves this function
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Non-finite batch")
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(m),
                "sensor_pos": torch.from_numpy(idx[:, 0].copy()), "t0": torch.from_numpy(idx[:, 1].copy()),
                "row": torch.from_numpy(rows)}


def make_loader(dataset, batch_size, shuffle=False, seed=None):
    """DataLoader over row numbers; the default BatchSampler groups them and get_batch builds the batch.

    num_workers=0 (data are in memory; Windows spawns processes). Shuffling uses a seeded generator.
    """
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("A shuffled loader needs a seed")
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(range(len(dataset)), batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=dataset.get_batch, num_workers=0, drop_last=False)
