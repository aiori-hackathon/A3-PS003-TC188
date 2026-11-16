"""Shared helpers for AIORI 2.0 preprocessing: configuration, atomic I/O, hashing, reconstructed calendar, run lengths.

Cross-platform (pathlib only); no Windows-specific code and no Claude Code dependency.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]                       # the repository's src/ directory
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "preprocess_v1.json"


def load_config(path=None, base_dir=None, out_dir=None):
    path = Path(path or DEFAULT_CONFIG)
    text = path.read_text(encoding="utf-8")
    cfg = json.loads(text)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cfg["_base"] = Path(base_dir or os.environ.get("AIORI2_BASE_DIR") or SRC)
    cfg["_out"] = Path(out_dir) if out_dir else cfg["_base"] / cfg["out_dir"]
    return cfg


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def save_npy(path, arr):
    """Write atomically: a partial file never sits at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


# ---- reconstructed calendar (no native timestamps exist in the raw arrays) ----

def year_steps(year, steps_per_day=288):
    days = (np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01")).astype(int)
    return int(days) * steps_per_day


def date_index(date, origin, steps_per_day=288):
    """Timeline index of 00:00 on `date` for a timeline starting at 00:00 on `origin`."""
    return int((np.datetime64(date, "D") - np.datetime64(origin, "D")).astype(int)) * steps_per_day


def weekday(days_since_epoch):
    return (np.asarray(days_since_epoch) + 3) % 7        # 1970-01-01 was a Thursday; Monday = 0


def us_spring_forward(year):
    """Second Sunday of March (US DST rule since 2007)."""
    d = np.datetime64(f"{year}-03-01")
    first_sunday = d + np.timedelta64((6 - int(weekday(d.astype(int)))) % 7, "D")
    return first_sunday + np.timedelta64(7, "D")


def calendar(years, steps_per_day=288, step_minutes=5):
    origin = np.datetime64(f"{years[0]}-01-01T00:00", "m")
    n = sum(year_steps(y, steps_per_day) for y in years)
    ts = origin + np.arange(n) * np.timedelta64(step_minutes, "m")
    day = ts.astype("datetime64[D]")
    days = day.astype(int)
    year = (day.astype("datetime64[Y]").astype(int) + 1970).astype(np.int16)
    month = day.astype("datetime64[M]").astype(int) % 12 + 1
    dom = (day - day.astype("datetime64[M]")).astype(int) + 1
    slot = np.tile(np.arange(steps_per_day, dtype=np.int16), n // steps_per_day)
    dst = np.zeros(n, bool)
    for y in years:
        i = date_index(us_spring_forward(y), origin, steps_per_day)
        dst[i + 2 * steps_per_day // 24:i + 3 * steps_per_day // 24] = True   # 02:00-02:59 local: does not exist that day
    return dict(timestamp=ts, year=year, month=month.astype(np.int8), day=dom.astype(np.int8),
                weekday=weekday(days).astype(np.int8), slot=slot, dst_nonexistent_hour=dst)


# ---- run lengths (vectorised per row; runs may cross month and year boundaries) ----

def run_lengths(mask):
    """For a (S, T) bool array: length of the True run each True element belongs to, 0 elsewhere."""
    mask = np.asarray(mask, bool)
    out = np.zeros(mask.shape, np.int32)
    for i, r in enumerate(mask):
        if not r.any():
            continue
        d = np.diff(np.concatenate(([0], r.view(np.int8), [0])))
        lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        out[i, r] = np.repeat(lengths, lengths)
    return out


def constant_runs(x):
    """Length of the run of identical consecutive finite non-zero values each element belongs to (1 = not repeated)."""
    x = np.asarray(x)
    eq = np.zeros(x.shape, bool)
    eq[:, 1:] = (x[:, 1:] == x[:, :-1]) & np.isfinite(x[:, 1:]) & (x[:, 1:] != 0)
    chain = run_lengths(eq)                                    # k identical values -> k-1 chained equalities
    length = np.where(eq, chain + 1, 0).astype(np.int32)
    head = np.zeros(x.shape, np.int32)
    head[:, :-1] = np.where(eq[:, 1:] & ~eq[:, :-1], length[:, 1:], 0)   # the first element of each run
    return np.maximum(length, head)
