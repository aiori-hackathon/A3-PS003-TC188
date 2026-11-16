"""On-demand window loading and raw-unit metrics for the AIORI 2.0 three-year data (no full window materialisation)."""

import json
from pathlib import Path

import numpy as np

from aiori2 import core


class ThreeYearWindows:
    """Windows of one split for one policy/experiment, gathered on demand from memory-mapped arrays.

    get(i) returns X (B, input_len, C) scaled inputs (C = input channels; default flow only), Y (B, horizon) raw flow
    with NaN where the target is not valid, M (B, horizon) valid-target mask, and the sensor positions.
    """

    def __init__(self, root, policy, experiment, split, scaler_scope="per_sensor", input_channels=None, sensors=None, eval_mask=None):
        """eval_mask: optional path to a (200, T) bool evaluation-only mask (e.g. the Phase 3 constant-run sensitivity
        mask). It is AND-ed with the policy's valid-target mask for Y/M only; inputs, windows and scalers are unchanged."""
        root = Path(root)
        d = root / "experiments" / policy / experiment
        self.meta = json.loads((d / "experiment.json").read_text(encoding="utf-8"))
        sc = json.loads((d / "scalers.json").read_text(encoding="utf-8"))["scopes"][scaler_scope]
        self.mean, self.std = np.asarray(sc["mean"], np.float32), np.asarray(sc["std"], np.float32)
        self.scope = scaler_scope
        self.off = self.meta["timeline_offset"]
        self.L, self.H = self.meta["input_len"], self.meta["horizon"]
        self.channels = tuple(input_channels or self.meta["input_channels"])
        self.filled = np.load(d / "filled_inputs.npy", mmap_mode="r")
        self.imputed = np.load(d / "input_imputed.npy", mmap_mode="r")
        self.flow = np.load(root / "extracted" / "flow_raw.npy", mmap_mode="r")
        self.valid = np.load(root / "masks" / policy / "valid_target.npy", mmap_mode="r")
        self.extra = np.load(eval_mask, mmap_mode="r") if eval_mask is not None else None
        self.sensors = None if sensors is None else np.sort(np.asarray(sensors))
        idx = np.load(d / "windows" / f"{split}.npy", mmap_mode="r")
        if sensors is not None:                                    # a client's own sensors only
            idx = idx[np.isin(idx[:, 0], np.asarray(sensors))]
        self.index = idx

    def __len__(self):
        return len(self.index)

    def get(self, i):
        w = np.asarray(self.index[i]).reshape(-1, 2).astype(np.int64)
        s, t0 = w[:, 0], w[:, 1]
        cols_in = t0[:, None] + np.arange(-self.L, 0)
        cols_out = t0[:, None] + np.arange(self.H) + self.off            # targets live on the full timeline
        sc = s[:, None]
        feats = []
        for ch in self.channels:
            if ch == "flow":
                feats.append((self.filled[sc, cols_in] - self.mean[s][:, None]) / self.std[s][:, None])
            elif ch == "observed":
                feats.append(~self.imputed[sc, cols_in])
            else:
                raise ValueError(f"unknown input channel {ch!r}")
        X = np.stack(feats, -1).astype(np.float32)
        if self.sensors is not None and not np.isin(s, self.sensors).all():
            raise ValueError("window outside this stream's sensors")
        M = np.asarray(self.valid[sc, cols_out], bool)
        if self.extra is not None:
            M &= np.asarray(self.extra[sc, cols_out], bool)
        Y = np.where(M, self.flow[sc, cols_out], np.nan).astype(np.float32)
        if not np.isfinite(X).all():
            raise ValueError("non-finite model inputs")
        return X, Y, M, s

    def scale(self, y, s):
        """Raw flow -> this stream's scaled units (for training targets; NaN stays NaN)."""
        return (np.asarray(y, np.float32) - self.mean[s][:, None]) / self.std[s][:, None]

    def inverse(self, z, s):
        """Scaled predictions -> raw flow units for sensors `s`."""
        return np.asarray(z, np.float64) * self.std[s][:, None] + self.mean[s][:, None]

    def target_timestamps(self, i, timestamps):
        t0 = np.asarray(self.index[i]).reshape(-1, 2)[:, 1].astype(np.int64) + self.off
        return timestamps[t0]


def raw_metrics(y_hat, y_true, mask):
    """MAE, RMSE and WAPE on raw flow, over valid targets only (masked targets never contribute)."""
    m = np.asarray(mask, bool)
    e = np.asarray(y_hat, np.float64)[m] - np.asarray(y_true, np.float64)[m]
    y = np.asarray(y_true, np.float64)[m]
    if not np.isfinite(e).all():
        raise ValueError("non-finite values inside the valid-target mask")
    n = int(m.sum())
    return dict(n=n, mae=float(np.abs(e).mean()) if n else float("nan"), rmse=float(np.sqrt((e ** 2).mean())) if n else float("nan"),
                wape_pct=float(100 * np.abs(e).sum() / np.abs(y).sum()) if n and np.abs(y).sum() else float("nan"))


def load_root(config=None, base_dir=None, out_dir=None):
    return core.load_config(config, base_dir, out_dir)["_out"]
