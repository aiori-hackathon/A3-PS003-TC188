"""GPU-resident windows for the accuracy-optimisation experiments (training and validation splits only).

Reuses the frozen nine-month preprocessing (train-only scaler, causal fill) unchanged. Adds optional covariates that are
known at forecast time and fitted on the training period only:
  * longer past context (input_len up to 2016 steps reach back into earlier data, never forward);
  * calendar of the target steps (time of day, day of week);
  * seasonal profile of the target steps: the sensor's training-period median for that weekday and 5-minute slot;
  * same-time-yesterday and same-time-last-week observations for the target steps (causally filled, with flags).
The validation targets (windows, cells and masks) are exactly the Phase 9 validation set: 1,606,706 windows.
The test split and October-December are never loaded: arrays are cut at the validation end before use.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "nine_month"
SPD, WEEK, H = 288, 2016, 12
ORIGIN_WEEKDAY = 6
CLIENTS = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"]


@dataclass
class Store:
    z: torch.Tensor          # (S, T) scaled causally filled flow
    obs: torch.Tensor        # (S, T) 1 = observed input
    flow: torch.Tensor       # (S, T) raw flow, NaN = missing (targets)
    prof: torch.Tensor       # (S, 7, 288) scaled training weekday x slot median
    mean: torch.Tensor       # (S,)
    std: torch.Tensor        # (S,)
    client: torch.Tensor     # (S,) int client id
    train: torch.Tensor      # (N, 2) [sensor, t0]
    val: torch.Tensor        # (N, 2)
    val_cells: float         # sum |actual| over observed validation cells (WAPE denominator)
    device: torch.device
    n_sensors: int


def load(device, min_t0=WEEK):
    split = json.loads((DATA / "split_config.json").read_text())["boundaries"]
    tr_end, va_end = split["train"][1], split["val"][1]
    with np.load(DATA / "pilot_preprocessed.npz") as d:
        z = d["inputs_scaled"][:, :va_end].astype(np.float32)
        imp = d["input_imputed"][:, :va_end]
        flow = d["flow_raw"][:, :va_end].astype(np.float32)
        tod = d["tod_median"]
    sc = json.loads((DATA / "scaler_parameters.json").read_text())
    mean, std = np.asarray(sc["mean"], np.float64), np.asarray(sc["std"], np.float64)
    with np.load(DATA / "window_index.npz") as w:
        train, val = w["train"].astype(np.int64), w["val"].astype(np.int64)
    sel = pd.read_csv(DATA / "sensor_selection.csv", usecols=["proposed_client"])
    client = np.array([CLIENTS.index(c) for c in sel["proposed_client"]])
    assert val[:, 1].min() >= tr_end and (val[:, 1] + H).max() <= va_end
    assert (train[:, 1] + H).max() <= tr_end
    train = train[train[:, 1] >= min_t0]            # long-context / weekly-lag features need one week of history

    # Training-period weekday x slot median (scaled); falls back to the frozen training time-of-day median.
    S = z.shape[0]
    t = np.arange(tr_end)
    wd = (ORIGIN_WEEKDAY + t // SPD) % 7
    slot = t % SPD
    ftr = flow[:, :tr_end]
    prof = np.empty((S, 7, SPD), np.float32)
    for d_ in range(7):
        for s_ in range(SPD):
            cols = np.nonzero((wd == d_) & (slot == s_))[0]
            prof[:, d_, s_] = np.nanmedian(ftr[:, cols], axis=1)
    fallback = np.broadcast_to(tod[:, None, :], prof.shape)
    prof = np.where(np.isfinite(prof), prof, fallback)
    prof = ((prof - mean[:, None, None]) / std[:, None, None]).astype(np.float32)

    # Validation WAPE denominator over the exact Phase 9 cells.
    cols = val[:, 1][:, None] + np.arange(H)
    a = flow[val[:, 0][:, None], cols]
    val_cells = float(np.nansum(np.abs(a)))

    T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)  # noqa: E731
    return Store(z=T(z), obs=T(~imp), flow=T(flow), prof=T(prof), mean=T(mean), std=T(std),
                 client=T(client, torch.long), train=T(train, torch.long), val=T(val, torch.long),
                 val_cells=val_cells, device=device, n_sensors=S)


@dataclass(frozen=True)
class Features:
    input_len: int = 12
    calendar_in: bool = True       # sin/cos time of day of each input step (the frozen model's features)
    future_cal: bool = False       # time of day + weekday of the target steps
    profile: bool = False          # training weekday x slot median of the target steps
    lags: bool = False             # same time yesterday / last week (values + observed flags) for target steps

    @property
    def past_dim(self):
        return 2 + (2 if self.calendar_in else 0)

    @property
    def future_dim(self):
        return (4 if self.future_cal else 0) + (1 if self.profile else 0) + (4 if self.lags else 0)


def batch(store, idx, f: Features):
    """idx (B, 2) on device -> dict(x past (B, L, Fp), xf future (B, H, Ff) or None, y raw (B, H), m mask, s sensor)."""
    s, t0 = idx[:, 0], idx[:, 1]
    L = f.input_len
    cin = t0[:, None] + torch.arange(-L, 0, device=store.device)
    sr = s[:, None]
    parts = [store.z[sr, cin], store.obs[sr, cin]]
    if f.calendar_in:
        ang = (cin % SPD).float() * (2 * np.pi / SPD)
        parts += [torch.sin(ang), torch.cos(ang)]
    x = torch.stack(parts, -1)
    cout = t0[:, None] + torch.arange(H, device=store.device)
    fparts = []
    if f.future_cal:
        ang = (cout % SPD).float() * (2 * np.pi / SPD)
        wd = (ORIGIN_WEEKDAY + cout // SPD) % 7
        wang = wd.float() * (2 * np.pi / 7)
        fparts += [torch.sin(ang), torch.cos(ang), torch.sin(wang), torch.cos(wang)]
    if f.profile:
        wd = (ORIGIN_WEEKDAY + cout // SPD) % 7
        fparts.append(store.prof[sr, wd, cout % SPD])
    if f.lags:
        for lag in (SPD, WEEK):
            c = cout - lag
            fparts += [store.z[sr, c], store.obs[sr, c]]
    xf = torch.stack(fparts, -1) if fparts else None
    y = store.flow[sr, cout]
    m = torch.isfinite(y)
    return dict(x=x, xf=xf, y=torch.nan_to_num(y), m=m, s=s)
