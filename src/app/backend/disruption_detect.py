"""Historical Disruption Explorer (AIORI 2.0, Phase 9B): offline, retrospective anomaly detection on the real XTraffic flow
readings of the 200 study sensors. Read-only on every existing artifact; writes one new versioned folder.

    .venv\\Scripts\\python.exe src\\app\\backend\\disruption_detect.py build      # ~1-2 min on CPU, writes the catalog

What it does (all parameters are in CONFIG and are recorded, with their hash, in the catalog):
1. Baseline. For every sensor, day type (weekday-like / weekend-like; each US federal holiday and the day after
   Thanksgiving is assigned by its own 2023 traffic level) and five-minute slot: the median and the
   robust spread (1.4826 x MAD) of the valid 2023 readings (the model's training year), pooling the slot and its two
   neighbours. The spread is floored at sqrt(median) (count noise), so a sensor with a very regular history cannot produce
   enormous scores. Slots with fewer than `min_support` training readings have no baseline.
2. Score. z = (reading - median) / spread and the relative deviation (reading - median) / median, only for valid readings
   (the project's conservative mask) and only where the median is at least `min_baseline` vehicles (no ratios near zero).
3. Thresholds. For every sensor, the 0.1 % and 99.9 % quantiles of z over its scored 2023 readings without quality
   flags; the thresholds are the medians of these over the sensors (equal weight per sensor, so a few drifting sensors
   cannot set them). Calibrated on the training year only; never on 2024 or on forecast errors. A reading is anomalous when it passes a z threshold AND
   deviates by at least `min_relative` from the median (statistical and practical significance).
4. Segments. Per sensor and direction, anomalous readings separated by at most `max_gap` slots are joined; segments need
   at least `min_flagged` anomalous readings (15 minutes), so isolated one- or two-reading spikes are not reported.
5. Events. Segments of the same direction in the same dataset region (the study's geographic client; NOT road
   connectivity) that overlap in time (within `join_slots`) form one candidate event. Quality-dominated segments are never
   grouped across sensors; one sensor's quality-dominated segments within 24 h form one sensor-quality episode. A chance check (random whole-day shifts per sensor) is recorded for multi-sensor grouping.
6. Quality. Exact-zero readings, readings inside a run of >= 1 h of identical values, readings within 1 h of a missing/invalid reading, or in a
   sensor-month where the sensor is anomalous for more than `drift_share` of its readings (level shift, e.g. recalibration)
   carry quality flags. Events where at least half of the anomalous readings are quality-flagged are classed
   `sensor_quality`: a real traffic change and a sensor fault cannot be told apart from flow alone.

The catalog is descriptive. It is not a disaster, accident, closure or congestion detector, it uses complete days
retrospectively (not causal), and it never looks at model forecasts.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                       # src/
DATA = ROOT / "data" / "processed" / "aiori2_3yr"
OUT = ROOT / "results" / "aiori2" / "phase9b_disruptions"
VERSION = "disruption-detector-v1"
SCHEMA = 1

CONFIG = dict(
    baseline_period="train split of experiment conservative/A (2023-01-01 to 2023-12-31)",
    day_types=["weekday_like", "weekend_like"],
    holidays="US federal holidays (observed dates) and the day after Thanksgiving; each is weekday-like or weekend-like "
             "by its 2023 traffic level (median over sensors of daily mean flow / 2023 mean)",
    slot_pool=1,                    # pool slot-1, slot, slot+1 (15 minutes) per day when estimating the baseline
    min_support=30,                 # training readings needed for a baseline value
    spread="1.4826 x median absolute deviation, floored at sqrt(max(median, 1))",
    min_baseline=5.0,               # vehicles per 5 min; quieter slots are not scored
    tail_quantile=0.001,            # per tail, per sensor, over scored 2023 readings that carry no quality flag;
                                    # the threshold is the median of the 200 per-sensor quantiles
    min_relative=0.25,              # |reading - median| / median
    max_gap=2,                      # slots (10 min) allowed between anomalous readings inside a segment
    min_flagged=3,                  # anomalous readings per segment (>= 15 min of evidence)
    join_slots=3,                   # segments within 15 min of each other in the same region and direction form one event
    quality_join_slots=288,         # quality-dominated segments of one sensor within 24 h form one sensor-quality episode
    run_slots=12,                   # identical non-zero values for >= 1 h are a quality flag (exact zeros always are)
    near_missing_slots=12,          # within 1 h of a missing or invalid reading is a quality flag
    drift_share=0.05,               # sensor-month anomalous share above this marks a level shift (quality flag)
    quality_event_share=0.5,        # events with >= this share of quality-flagged anomalous readings -> sensor_quality
    coordinated_min_sensors=3,
    sustained_min_minutes=60,
    score="sum over anomalous readings of min(|z|, 10) x 5/60 (capped z-hours); a magnitude, not a probability",
)
Q_BITS = {"zero_reading": 1, "stuck_run": 2, "near_missing": 4, "level_shift": 8}
CODES = {"client_0_la_ventura": "LA", "client_1_bay_area": "BA", "client_2_coastal_south_central": "CS", "client_3_inland_central": "IC"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def config_hash(cfg=CONFIG):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------- calendar
def holiday_dates(days):
    """{date: name} for US federal holidays (observed dates) and the day after Thanksgiving."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    h = USFederalHolidayCalendar().holidays(days.min(), days.max(), return_name=True)
    out = {d.date(): str(n) for d, n in h.items()}
    for d, n in h.items():
        if "Thanksgiving" in str(n):
            out[(d + np.timedelta64(1, "D")).date()] = "Day after Thanksgiving"
    return out


def day_types(stamps, flow, valid, train, spd=288):
    """0 = weekday-like, 1 = weekend-like, one value per day; plus the holiday table.

    Weekends are weekend-like. Each holiday (by name) is weekday-like or weekend-like according to its 2023 traffic level:
    the median over sensors of the day's mean valid flow relative to the sensor's 2023 mean, compared with the median
    level of ordinary 2023 weekdays and weekends. Decided on the training year only and applied to every year."""
    import pandas as pd
    days = pd.DatetimeIndex(stamps[::spd].astype("datetime64[D]"))
    hol = holiday_dates(days)
    a, b = train
    nd = (b - a) // spd
    lv = []
    for s in range(flow.shape[0]):
        x = np.where(np.asarray(valid[s, a:b]), np.asarray(flow[s, a:b], np.float64), np.nan).reshape(nd, spd)
        with np.errstate(all="ignore"):
            lv.append(np.nanmean(x, 1) / np.nanmean(x))
    with np.errstate(all="ignore"):
        level = np.nanmedian(np.array(lv), 0)
    tdays = days[a // spd: a // spd + nd]
    is_hol = np.array([d.date() in hol for d in tdays])
    wkend = np.array([d.weekday() >= 5 for d in tdays])
    lw, le = np.nanmedian(level[~wkend & ~is_hol]), np.nanmedian(level[wkend & ~is_hol])
    rule = {}
    for d, lev in zip(tdays[is_hol], level[is_hol]):
        rule[hol[d.date()]] = dict(level_2023=round(float(lev), 3), weekend_like=bool(abs(lev - le) < abs(lev - lw)))
    table = {str(d): dict(name=n, weekend_like=rule.get(n, dict(weekend_like=True))["weekend_like"]) for d, n in sorted(hol.items())}
    dtype = np.array([int(d.weekday() >= 5 or (str(d.date()) in table and table[str(d.date())]["weekend_like"])) for d in days], np.int8)
    return dtype, dict(dates=table, rule_2023=rule, weekday_level=round(float(lw), 3), weekend_level=round(float(le), 3))


# ---------------------------------------------------------------------------------------------------- baseline
def build_baseline(flow, valid, dtype, a, b, cfg=CONFIG, spd=288):
    """Median, robust spread and support per (sensor, day type, slot) from readings in [a, b) (a, b multiples of spd)."""
    S = flow.shape[0]
    nd = (b - a) // spd
    med = np.full((S, 2, spd), np.nan, np.float32)
    spread = np.full((S, 2, spd), np.nan, np.float32)
    support = np.zeros((S, 2, spd), np.int32)
    dt = dtype[a // spd: a // spd + nd]
    k = cfg["slot_pool"]
    for s in range(S):
        x = np.where(np.asarray(valid[s, a:b]), np.asarray(flow[s, a:b], np.float64), np.nan).reshape(nd, spd)
        for t in (0, 1):
            xs = x[dt == t]
            pooled = np.concatenate([np.roll(xs, -o, axis=1) for o in range(-k, k + 1)], axis=0)   # circular within the day
            n = np.isfinite(pooled).sum(0)
            with np.errstate(all="ignore"):
                m = np.nanmedian(pooled, 0)
                mad = np.nanmedian(np.abs(pooled - m), 0)
            sp = np.maximum(1.4826 * mad, np.sqrt(np.maximum(m, 1.0)))
            good = n >= cfg["min_support"]
            med[s, t] = np.where(good, m, np.nan)
            spread[s, t] = np.where(good, sp, np.nan)
            support[s, t] = n
    return med, spread, support


def expected(med, spread, dtype, s, i0, i1, spd=288):
    """Baseline median and spread for sensor s over steps [i0, i1)."""
    idx = np.arange(i0, i1)
    t, slot = dtype[idx // spd], idx % spd
    return med[s, t, slot], spread[s, t, slot]


def score(x, ok, m, sp, cfg=CONFIG):
    """z and relative deviation; NaN where the reading is not valid or the slot is not scored."""
    scored = ok & np.isfinite(m) & (m >= cfg["min_baseline"]) & np.isfinite(x)
    with np.errstate(all="ignore"):
        z = np.where(scored, (x - m) / sp, np.nan)
        rel = np.where(scored, (x - m) / m, np.nan)
    return z, rel


# ---------------------------------------------------------------------------------------------------- runs and quality
def runs(mask):
    """(start, end_exclusive) of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def long_runs(mask, n):
    out = np.zeros(len(mask), bool)
    for a, b in zip(*runs(mask)):
        if b - a >= n:
            out[a:b] = True
    return out


def dilate(mask, n):
    """True within n steps of any True."""
    c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    i = np.arange(len(mask))
    return (c[np.minimum(i + n + 1, len(mask))] - c[np.maximum(i - n, 0)]) > 0


def slot_quality(x, ok, cfg=CONFIG):
    """Bitmask per step: exact-zero readings (a closure and a dropout look the same), identical-value runs, proximity to
    missing/invalid readings."""
    n = cfg["run_slots"]
    q = np.zeros(len(x), np.uint8)
    q[ok & (x == 0)] |= Q_BITS["zero_reading"]
    same = np.zeros(len(x), bool)
    same[1:] = ok[1:] & ok[:-1] & (x[1:] == x[:-1]) & (x[1:] != 0)
    stuck = long_runs(same, n - 1)                                   # n-1 equal steps = n identical readings
    stuck[:-1] |= stuck[1:]                                          # include the first reading of the run
    q[stuck] |= Q_BITS["stuck_run"]
    q[dilate(~ok, cfg["near_missing_slots"])] |= Q_BITS["near_missing"]
    return q


# ---------------------------------------------------------------------------------------------------- segments
def flag(z, rel, lo, hi, cfg=CONFIG):
    """+1 anomalous rise, -1 anomalous drop, 0 otherwise (NaN scores are never anomalous)."""
    r = cfg["min_relative"]
    f = np.zeros(len(z), np.int8)
    with np.errstate(invalid="ignore"):
        f[(z >= hi) & (rel >= r)] = 1
        f[(z <= lo) & (rel <= -r)] = -1
    return f


def segments(f, cfg=CONFIG):
    """Per direction: (direction, first, last) of anomalous readings joined across gaps <= max_gap with >= min_flagged."""
    out = []
    for d in (-1, 1):
        idx = np.flatnonzero(f == d)
        if not len(idx):
            continue
        cut = np.flatnonzero(np.diff(idx) > cfg["max_gap"] + 1)
        for g in np.split(idx, cut + 1):
            if len(g) >= cfg["min_flagged"]:
                out.append((d, int(g[0]), int(g[-1])))
    return out


def group(segs, join):
    """Cluster (start, end, payload) items whose intervals overlap or lie within `join` steps; returns lists of items."""
    out, cur, end = [], [], -10 ** 9
    for it in sorted(segs, key=lambda s: (s[0], s[1])):
        if cur and it[0] <= end + join:
            cur.append(it)
            end = max(end, it[1])
        else:
            if cur:
                out.append(cur)
            cur, end = [it], it[1]
    if cur:
        out.append(cur)
    return out


def chance_coordination(seg_items, n, cfg=CONFIG, spd=288, reps=5, seed=0):
    """Clean multi-sensor clusters (>= coordinated_min_sensors) actually found vs. after shifting every sensor's segments by
    a random whole number of days (time of day and day-to-day structure kept, cross-sensor timing destroyed)."""
    rng = np.random.default_rng(seed)
    k = cfg["coordinated_min_sensors"]

    def count(shift):
        c = 0
        for items in seg_items.values():
            clean = [((it[0] + shift[it[2]["pos"]]) % n, (it[1] + shift[it[2]["pos"]]) % n, it[2]) for it in items
                     if it[2]["n_quality"] / it[2]["n_anomalous"] < cfg["quality_event_share"]]
            c += sum(len({g[2]["pos"] for g in cl}) >= k for cl in group(clean, cfg["join_slots"]))
        return c

    nd = n // spd
    null = [count(rng.integers(1, nd, 200) * spd) for _ in range(reps)]
    return dict(observed=count(np.zeros(200, np.int64)), shifted_mean=float(np.mean(null)), shifted=null, reps=reps,
                method="each sensor's segments shifted by a random whole number of days (circular), seed 0")


def per_sensor(segs, ts):
    """One summary per affected sensor (its segments inside the event combined)."""
    out = []
    for sid in sorted({g["sensor"] for g in segs}):
        gs = [g for g in segs if g["sensor"] == sid]
        pk = max(gs, key=lambda g: abs(g["peak_z"]))
        bits = int(np.bitwise_or.reduce([g["quality_bits"] for g in gs]))
        out.append(dict(sensor=sid, start=ts(min(g["start"] for g in gs)), end=ts(max(g["end"] for g in gs)), n_segments=len(gs),
                        n_anomalous=sum(g["n_anomalous"] for g in gs), n_missing=sum(g["n_missing"] for g in gs),
                        n_quality=sum(g["n_quality"] for g in gs), quality_flags=[k for k, v in Q_BITS.items() if bits & v],
                        peak_z=round(pk["peak_z"], 2), peak_relative=round(pk["peak_rel"], 3), peak_time=ts(pk["peak_at"]),
                        peak_observed=round(pk["peak_obs"], 1), peak_expected=round(pk["peak_expected"], 1),
                        score=round(sum(g["z_hours"] for g in gs), 3)))
    return out


def event_id(code, direction, start_ts, sensors, start, end, chash):
    h = hashlib.sha256(f"{chash}|{code}|{direction}|{start}|{end}|{','.join(sensors)}".encode()).hexdigest()[:6]
    return f"{start_ts.replace('-', '').replace(':', '')}-{code}-{'D' if direction < 0 else 'U'}-{h}"


def categorize(direction, n_sensors, minutes, quality_share, cfg=CONFIG):
    if quality_share >= cfg["quality_event_share"]:
        return "sensor_quality"
    if n_sensors >= cfg["coordinated_min_sensors"]:
        return "coordinated_deviation"
    if minutes >= cfg["sustained_min_minutes"]:
        return "sustained_deviation"
    return "sudden_decrease" if direction < 0 else "sudden_increase"


def confidence(quality_share, missing_share):
    if quality_share >= 0.5 or missing_share >= 0.25:
        return "low"
    if quality_share == 0 and missing_share < 0.05:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------------------------------- pipeline
def detect(flow, valid, stamps, sensor_ids, client_geo, train, cfg=CONFIG, spd=288, log=print):
    """Full pipeline on arrays (sensors x steps). Returns (baseline dict, thresholds dict, events list, audit dict)."""
    S, n = flow.shape
    chash = config_hash(cfg)
    dtype, calendar = day_types(stamps, flow, valid, train, spd)
    holidays = calendar["dates"]
    a, b = train
    t0 = time.perf_counter()
    med, spread, support = build_baseline(flow, valid, dtype, a, b, cfg, spd)
    log(f"baseline {time.perf_counter() - t0:.1f}s")

    # thresholds: median over sensors of each sensor's tail quantiles of z (clean scored 2023 readings), so that a few
    # drifting or faulty sensors cannot set the threshold for everyone
    qs, n_cal = [], 0
    for s in range(S):
        m, sp = expected(med, spread, dtype, s, a, b, spd)
        x, ok = np.asarray(flow[s, a:b], np.float64), np.asarray(valid[s, a:b])
        z, _ = score(x, ok, m, sp, cfg)
        zc = z[np.isfinite(z) & (slot_quality(x, ok, cfg) == 0)]
        n_cal += len(zc)
        if len(zc) >= 1000:
            qs.append(np.quantile(zc, [cfg["tail_quantile"], 1 - cfg["tail_quantile"]]))
    qs = np.array(qs)
    lo, hi = (float(v) for v in np.median(qs, 0))
    thresholds = dict(z_low=round(lo, 4), z_high=round(hi, 4), min_relative=cfg["min_relative"], n_calibration_readings=int(n_cal),
                      n_calibration_sensors=len(qs),
                      per_sensor_iqr=dict(low=[round(float(v), 3) for v in np.percentile(qs[:, 0], [25, 75])],
                                          high=[round(float(v), 3) for v in np.percentile(qs[:, 1], [25, 75])]),
                      calibration="median over sensors of each sensor's 0.1 % / 99.9 % quantile of z over its scored 2023 "
                                  "(training-year) readings without quality flags")
    log(f"thresholds z<= {lo:.2f} / z>= {hi:.2f} ({time.perf_counter() - t0:.1f}s)")

    # per-sensor scores, quality and segments
    seg_items = {}
    month = (stamps.astype("datetime64[M]") - stamps[0].astype("datetime64[M]")).astype(int)
    audit = dict(flagged=np.zeros((S, 2), np.int64), scored=np.zeros(S, np.int64), level_shift_months=[])
    for s in range(S):
        x = np.asarray(flow[s], np.float64)
        ok = np.asarray(valid[s], bool)
        m, sp = expected(med, spread, dtype, s, 0, n, spd)
        z, rel = score(x, ok, m, sp, cfg)
        f = flag(z, rel, lo, hi, cfg)
        q = slot_quality(x, ok, cfg)
        scored = np.isfinite(z)
        n_sc = np.bincount(month[scored], minlength=month[-1] + 1)
        n_fl = np.bincount(month[f != 0], minlength=month[-1] + 1)
        with np.errstate(all="ignore"):
            share = np.where(n_sc > 0, n_fl / n_sc, 0)
        for mo in np.flatnonzero(share > cfg["drift_share"]):
            q[month == mo] |= Q_BITS["level_shift"]
            audit["level_shift_months"].append((s, int(mo)))
        audit["flagged"][s] = [(f < 0).sum(), (f > 0).sum()]
        audit["scored"][s] = scored.sum()
        for d, i0, i1 in segments(f, cfg):
            sl = slice(i0, i1 + 1)
            an = f[sl] == d
            za = np.abs(z[sl][an])
            k = int(np.nanargmax(np.abs(z[sl]) * an))
            seg_items.setdefault((client_geo[s], d), []).append((i0, i1, dict(
                sensor=sensor_ids[s], pos=s, start=i0, end=i1, n_anomalous=int(an.sum()),
                n_missing=int((~ok[sl]).sum()), n_quality=int((q[sl][an] > 0).sum()),
                quality_bits=int(np.bitwise_or.reduce(q[sl][an])), peak_z=float(z[i0 + k]), peak_rel=float(rel[i0 + k]),
                peak_at=i0 + k, peak_obs=float(x[i0 + k]), peak_expected=float(m[i0 + k]),
                z_hours=float(np.minimum(za, 10).sum() * 5 / 60))))
    log(f"segments {sum(len(v) for v in seg_items.values())} ({time.perf_counter() - t0:.1f}s)")

    ts = lambda i: str(stamps[int(i)])[:16]
    events = []
    for (geo, d), items in sorted(seg_items.items()):
        code = CODES.get(geo, geo)
        clean = [it for it in items if it[2]["n_quality"] / it[2]["n_anomalous"] < cfg["quality_event_share"]]
        noisy = {}
        for it in items:
            if it[2]["n_quality"] / it[2]["n_anomalous"] >= cfg["quality_event_share"]:
                noisy.setdefault(it[2]["pos"], []).append(it)
        noisy = [cl for its in noisy.values() for cl in group(its, cfg["quality_join_slots"])]
        for cl in group(clean, cfg["join_slots"]) + noisy:     # quality-dominated segments: per-sensor quality episodes
            segs = sorted((c[2] for c in cl), key=lambda g: (g["start"], g["sensor"]))
            start, end = min(g["start"] for g in segs), max(g["end"] for g in segs)
            n_an = sum(g["n_anomalous"] for g in segs)
            n_q = sum(g["n_quality"] for g in segs)
            span = sum(g["end"] - g["start"] + 1 for g in segs)
            q_share, miss_share = n_q / n_an, sum(g["n_missing"] for g in segs) / span
            sensors = sorted({g["sensor"] for g in segs})
            minutes = (end - start + 1) * 5
            peak = max(segs, key=lambda g: abs(g["peak_z"]))
            bits = int(np.bitwise_or.reduce([g["quality_bits"] for g in segs]))
            days = {ts(i)[:10] for i in (start, end)}
            events.append(dict(
                id=event_id(code, d, ts(start), sensors, start, end, chash), start=ts(start), end=ts(end), start_index=start,
                end_index=end, duration_min=minutes, direction="decrease" if d < 0 else "increase",
                category=categorize(d, len(sensors), minutes, q_share, cfg), region_code=code, client_geo=geo,
                n_sensors=len(sensors), sensors=sensors, n_anomalous=n_an,
                peak_z=round(peak["peak_z"], 2), peak_relative=round(peak["peak_rel"], 3), peak_sensor=peak["sensor"],
                peak_time=ts(peak["peak_at"]), score=round(sum(g["z_hours"] for g in segs), 3),
                quality_share=round(q_share, 3), missing_share=round(miss_share, 3),
                quality_flags=[k for k, v in Q_BITS.items() if bits & v], confidence=confidence(q_share, miss_share),
                day_type=CONFIG["day_types"][int(dtype[start // spd])], holiday=next((holidays[dd]["name"] for dd in sorted(days) if dd in holidays), None),
                sensor_details=per_sensor(segs, ts)))
    events.sort(key=lambda e: (e["start_index"], e["region_code"], e["direction"], e["id"]))
    if len({e["id"] for e in events}) != len(events):
        raise RuntimeError("event ids are not unique")
    baseline = dict(median=med, spread=spread, support=support, day_type=dtype)
    audit["calendar"] = calendar
    audit["chance_coordination"] = chance_coordination(seg_items, n, cfg, spd)
    return baseline, thresholds, events, audit


# ---------------------------------------------------------------------------------------------------- build / CLI
def _inputs(data=DATA):
    import pandas as pd
    tl = json.loads((data / "metadata" / "timeline.json").read_text(encoding="utf-8"))
    stamps = np.load(data / "metadata" / "calendar" / "timestamp.npy")
    flow = np.load(data / "extracted" / "flow_raw.npy", mmap_mode="r")
    valid = np.load(data / "masks" / "conservative" / "valid_target.npy", mmap_mode="r")
    exp = json.loads((data / "experiments" / "conservative" / "A" / "experiment.json").read_text(encoding="utf-8"))
    off, tr = exp["timeline_offset"], exp["splits_index"]["train"]
    s = pd.read_csv(data / "metadata" / "sensors.csv")
    counters, ids = {}, []
    for geo in s["client_geo"]:                                     # same id scheme as the replay service (LA-01 ...)
        c = CODES[geo]
        counters[c] = counters.get(c, 0) + 1
        ids.append(f"{c}-{counters[c]:02d}")
    files = {k: data / k for k in ("extracted/flow_raw.npy", "masks/conservative/valid_target.npy", "metadata/calendar/timestamp.npy",
                                   "metadata/sensors.csv", "experiments/conservative/A/experiment.json")}
    return tl, stamps, flow, valid, (off + tr[0], off + tr[1]), ids, s["client_geo"].to_numpy(), files


def events_hash(events):
    """Hash of the event list alone (the catalog file also carries its creation time)."""
    return hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize(events):
    by = lambda key: dict(sorted(((k, sum(1 for e in events if key(e) == k)) for k in {key(e) for e in events})))
    dur = np.array([e["duration_min"] for e in events]) if events else np.zeros(1)
    return dict(n_events=len(events), by_year=by(lambda e: e["start"][:4]), by_category=by(lambda e: e["category"]),
                by_direction=by(lambda e: e["direction"]), by_region=by(lambda e: e["region_code"]),
                by_confidence=by(lambda e: e["confidence"]),
                duration_min=dict(p50=float(np.percentile(dur, 50)), p90=float(np.percentile(dur, 90)), max=float(dur.max())),
                sensors_per_event=by(lambda e: min(e["n_sensors"], 10)),
                sensors_with_events=len({s for e in events for s in e["sensors"]}))


def build(out=OUT, data=DATA, log=print):
    t_start = time.perf_counter()
    tl, stamps, flow, valid, train, ids, geo, files = _inputs(data)
    baseline, thresholds, events, audit = detect(flow, valid, stamps, ids, geo, train, log=log)
    d = Path(out) / VERSION
    d.mkdir(parents=True, exist_ok=True)
    bl = d / "baseline.npz"
    tmp = d / "baseline.tmp.npz"
    np.savez_compressed(tmp, **baseline)
    tmp.replace(bl)
    catalog = dict(
        schema=SCHEMA, version=VERSION, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        config=CONFIG, config_sha256=config_hash(), thresholds=thresholds,
        baseline=dict(file=bl.name, sha256=sha256(bl), shape=list(baseline["median"].shape),
                      description="median and robust spread per sensor x day type x slot from valid 2023 readings"),
        provenance=dict(source="XTraffic historical observations, AIORI 2.0 Phase 2 preprocessing (conservative mask)",
                        timezone=tl["timezone"], step_minutes=tl["step_minutes"], first=str(stamps[0])[:16], last=str(stamps[-1])[:16],
                        n_sensors=int(flow.shape[0]), inputs={k: sha256(p) for k, p in files.items()}),
        calendar=audit["calendar"],
        audit=dict(level_shift_sensor_months=len(audit["level_shift_months"]), chance_coordination=audit["chance_coordination"],
                   anomalous_readings=dict(decrease=int(audit["flagged"][:, 0].sum()), increase=int(audit["flagged"][:, 1].sum())),
                   scored_readings=int(audit["scored"].sum())),
        summary=summarize(events), events_sha256=events_hash(events), events=events)
    path = d / "catalog.json"
    tmp = d / "catalog.tmp.json"
    tmp.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    (d / "catalog.sha256").write_text(f"{sha256(path)}  catalog.json\n", encoding="utf-8")
    log(f"{len(events)} events -> {path} ({time.perf_counter() - t_start:.1f}s)")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["build", "verify"])
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    if a.command == "build":
        build(a.out)
    else:
        d = Path(a.out) / VERSION
        want = (d / "catalog.sha256").read_text(encoding="utf-8").split()[0]
        ok = sha256(d / "catalog.json") == want
        print("catalog OK" if ok else "catalog hash MISMATCH")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
