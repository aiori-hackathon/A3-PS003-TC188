"""Phase 9: nine-month (Jan 1 - Sep 30, 2023) data audit, sensor selection, preprocessing and verification.

Only the monthly files 2023_p01 ... 2023_p09 are ever opened (October-December are refused). All statistics that
drive eligibility and selection use the training period only (Jan 1 - Jul 10). Arrays are read through memory maps
in blocks of sensor rows; no window is materialised (windows are stored as a (sensor, t0) index and gathered per
batch by aiori.dataset). Split: train Jan 1 - Jul 10 (191 days), val Jul 11 - Aug 7 (28), test Aug 8 - Sep 30 (54).
Windows are kept only if both their inputs and their targets lie inside one split.
Row-to-station mapping and geography are provisional (node_order[array_index], unconfirmed by provenance).
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aiori import config, preprocessing, splits as splits_mod, windows

MONTHS = tuple(range(1, 10))
DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}
SPD = 288
N_SENSORS = 16972
SPLIT_DAYS = {"train": 191, "val": 28, "test": 54}
EXPECTED_DATES = {"train": ("2023-01-01", "2023-07-10"), "val": ("2023-07-11", "2023-08-07"), "test": ("2023-08-08", "2023-09-30")}
MIN_COVERAGE_PCT, MAX_ZERO_PCT = 99.0, 50.0
SALT = "aiori-ninemonth-v1"
PER_CLIENT = 50


def month_paths(year_dir):
    paths = [Path(year_dir) / f"2023_p{m:02d}.npy" for m in MONTHS]
    for p in paths:
        if not p.name.startswith("2023_p0") or int(p.stem[-2:]) > 9:
            raise ValueError(f"Refusing a file outside January-September: {p.name}")
    return paths


def split_bounds():
    """Step boundaries and the calendar dates they cover, checked programmatically."""
    b, out = 0, {}
    for name in ("train", "val", "test"):
        out[name] = (b * SPD, (b + SPLIT_DAYS[name]) * SPD)
        b += SPLIT_DAYS[name]
    total = sum(DAYS.values())
    if b != total or total != 273:
        raise AssertionError(f"Split covers {b} days, data covers {total}")
    origin = pd.Timestamp(splits_mod.TIME_ORIGIN)
    for name, (s, e) in out.items():
        first = (origin + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%d")
        last = (origin + pd.Timedelta(minutes=5 * (e - 1))).strftime("%Y-%m-%d")
        if (first, last) != EXPECTED_DATES[name]:
            raise AssertionError(f"{name}: {first} - {last} != {EXPECTED_DATES[name]}")
    return out


def open_months(paths, n_sensors=N_SENSORS):
    arrays = []
    for m, p in zip(MONTHS, paths):
        a = np.load(p, mmap_mode="r")
        exp = (n_sensors, DAYS[m] * SPD, 3)
        if a.shape != exp or a.dtype != np.float32:
            raise ValueError(f"{p.name}: shape {a.shape} dtype {a.dtype}, expected {exp} float32")
        arrays.append(a)
    return arrays


def audit(paths, n_sensors=N_SENSORS, block=1024, dst_month=3):
    """Training-period statistics per sensor, monthly means (ordering check) and boundary values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _audit(paths, n_sensors, block, dst_month)


def _audit(paths, n_sensors, block, dst_month):
    arrays = open_months(paths, n_sensors)
    train_end = split_bounds()["train"][1]
    offs = np.cumsum([0] + [a.shape[1] for a in arrays])
    rows = []
    month_mean = np.full((n_sensors, len(arrays)), np.nan)
    first_val = np.full((n_sensors, len(arrays)), np.nan)
    last_val = np.full((n_sensors, len(arrays)), np.nan)
    step_diff_med = np.full(n_sensors, np.nan)
    dst_sum = np.zeros(arrays[dst_month - 1].shape[1])
    dst_cnt = np.zeros(arrays[dst_month - 1].shape[1])
    for r0 in range(0, n_sensors, block):
        r1 = min(n_sensors, r0 + block)
        k = r1 - r0
        fin = np.zeros(k, np.int64)
        zer = np.zeros(k, np.int64)
        s1 = np.zeros(k)
        s2 = np.zeros(k)
        mn = np.full(k, np.inf)
        mx = np.full(k, -np.inf)
        hashes = [hashlib.sha256() for _ in range(k)]
        diffs = []
        for mi, a in enumerate(arrays):
            x = np.asarray(a[r0:r1, :, config.FLOW])
            month_mean[r0:r1, mi] = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1)
            first_val[r0:r1, mi], last_val[r0:r1, mi] = x[:, 0], x[:, -1]
            if mi == dst_month - 1:
                f = np.isfinite(x)
                dst_sum += np.where(f, x, 0).sum(axis=0)
                dst_cnt += f.sum(axis=0)
            lo, hi = offs[mi], offs[mi + 1]
            if lo >= train_end:
                continue
            t = x[:, : min(hi, train_end) - lo]
            f = np.isfinite(t)
            v = np.where(f, t, 0.0).astype(np.float64)
            fin += f.sum(axis=1)
            zer += (f & (t == 0)).sum(axis=1)
            s1 += v.sum(axis=1)
            s2 += (v ** 2).sum(axis=1)
            mn = np.minimum(mn, np.where(f, t, np.inf).min(axis=1))
            mx = np.maximum(mx, np.where(f, t, -np.inf).max(axis=1))
            if mi == 0:
                d = np.abs(np.diff(t, axis=1))
                diffs = np.nanmedian(np.where(np.isfinite(d), d, np.nan), axis=1)
            tb = np.ascontiguousarray(t, dtype=np.float32)
            for i in range(k):
                hashes[i].update(tb[i].tobytes())
        n = train_end
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(fin > 0, s1 / np.maximum(fin, 1), np.nan)
            std = np.sqrt(np.maximum(s2 / np.maximum(fin, 1) - mean ** 2, 0))
        step_diff_med[r0:r1] = diffs
        for i in range(k):
            rows.append(dict(array_index=r0 + i, finite_count=int(fin[i]), coverage_pct=100.0 * fin[i] / n,
                             zero_pct=100.0 * zer[i] / max(fin[i], 1), mean=mean[i], std=std[i],
                             fully_missing=bool(fin[i] == 0), constant=bool(fin[i] > 0 and mn[i] == mx[i]),
                             series_sha256=hashes[i].hexdigest()))
    df = pd.DataFrame(rows)
    grp = df.groupby("series_sha256")["array_index"].transform("size")
    df["duplicate_group_size"] = grp
    df["duplicate_series"] = (grp > 1) & ~df["fully_missing"]
    return df, dict(month_mean=month_mean, first_val=first_val, last_val=last_val, step_diff_med=step_diff_med,
                    dst_profile=np.divide(dst_sum, dst_cnt, out=np.full_like(dst_sum, np.nan), where=dst_cnt > 0))


def eligibility(df):
    reason = np.select([df["fully_missing"], df["constant"], df["duplicate_series"],
                        df["coverage_pct"] < MIN_COVERAGE_PCT, df["zero_pct"] > MAX_ZERO_PCT],
                       ["fully_missing", "constant", "exact_copy", "low_coverage", "too_many_zeros"], "")
    out = df.copy()
    out["exclusion_reason"] = reason
    out["eligible"] = reason == ""
    return out


def ordering_checks(extra):
    """Sensor order consistency across months: rank correlation of monthly means and boundary-step continuity."""
    mm = extra["month_mean"]
    corr = []
    for i in range(mm.shape[1] - 1):
        a, b = mm[:, i], mm[:, i + 1]
        ok = np.isfinite(a) & np.isfinite(b)
        corr.append(float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank())) if ok.sum() > 10 else float("nan"))
    ratio = []
    for i in range(mm.shape[1] - 1):
        d = np.abs(extra["last_val"][:, i] - extra["first_val"][:, i + 1])
        ok = np.isfinite(d) & np.isfinite(extra["step_diff_med"])
        ratio.append(float(np.median(d[ok]) / max(np.median(extra["step_diff_med"][ok]), 1e-9)) if ok.sum() > 10 else float("nan"))
    return dict(month_mean_rank_corr=corr, boundary_jump_over_typical_step=ratio,
                passed=bool(all(c > 0.9 for c in corr) and all(r < 5 for r in ratio)))


def dst_check(profile, transition_day=11):
    """Lag (in 5-min steps) aligning the mean daily profile of the week after 12 Mar with the week before.

    `profile` is the March series pooled over sensors; day 11 (0-based) is 12 March, the DST transition.
    """
    p = np.asarray(profile, float)
    before = np.nanmean(p[(transition_day - 7) * SPD: transition_day * SPD].reshape(7, SPD), axis=0)
    after = np.nanmean(p[(transition_day + 1) * SPD: (transition_day + 8) * SPD].reshape(7, SPD), axis=0)
    lags = range(-24, 25)
    score = [np.corrcoef(before, np.roll(after, lag))[0, 1] for lag in lags]
    best = list(lags)[int(np.nanargmax(score))]
    return dict(best_lag_steps=int(best), corr_at_best=float(np.nanmax(score)), corr_at_zero=float(score[24]),
                interpretation=("no one-hour shift at the 12 Mar transition (best lag within 10 min): daily profiles are in "
                                "local clock time; step counts per month are complete, so no hour is missing or duplicated")
                if abs(best) <= 2 else f"daily profiles shift by {best} steps (~{best * 5} min) at 12 Mar: timestamps are not local clock time")


def _tie(i, salt=SALT):
    return hashlib.sha256(f"{salt}:{int(i)}".encode()).hexdigest()


def select(elig, meta_types, jan_manifest, pilot_idx, per_client=PER_CLIENT):
    """Retain January geo200 sensors that are still eligible; replace the others deterministically.

    A replacement comes from the same client and type group (tertile of nine-month training mean within that
    (client, type group) matched to the January tertile when possible), in salted-hash order, excluding the 48 pilot
    sensors, every January geo200 sensor and every sensor already chosen. Training-period statistics only.
    """
    from aiori.select_geo200 import TYPE_GROUP, flow_tertiles
    e = elig.set_index("array_index")
    jan = jan_manifest[["array_index", "proposed_client", "type_group", "flow_tertile", "District", "station_id"]].copy()
    jan["keep"] = e.loc[jan["array_index"], "eligible"].to_numpy()
    cand = elig[elig["eligible"] & ~elig["array_index"].isin(pilot_idx) & ~elig["array_index"].isin(jan["array_index"])].copy()
    cand["type_group"] = meta_types.reindex(cand["array_index"]).map(TYPE_GROUP).to_numpy()
    cand = cand[cand["type_group"].notna()]
    cand["tertile9"] = ""
    for _, g in cand.groupby(["proposed_client", "type_group"]):
        cand.loc[g.index, "tertile9"] = flow_tertiles(g["mean"]).to_numpy()
    cand["_tie"] = cand["array_index"].map(_tie)
    cand = cand.sort_values("_tie")
    used, out, log = set(), [], []
    for r in jan.itertuples(index=False):
        if r.keep:
            out.append(dict(array_index=r.array_index, proposed_client=r.proposed_client, type_group=r.type_group,
                            flow_tertile=r.flow_tertile, selection_basis="retained_january"))
            continue
        pool = cand[(cand["proposed_client"] == r.proposed_client) & ~cand["array_index"].isin(used)]
        for basis, sub in (("replacement_same_stratum", pool[(pool["type_group"] == r.type_group) & (pool["tertile9"] == r.flow_tertile)]),
                           ("replacement_same_type", pool[pool["type_group"] == r.type_group]),
                           ("replacement_same_client", pool)):
            if len(sub):
                new = sub.iloc[0]
                used.add(int(new["array_index"]))
                out.append(dict(array_index=int(new["array_index"]), proposed_client=r.proposed_client,
                                type_group=new["type_group"], flow_tertile=new["tertile9"], selection_basis=basis))
                log.append(dict(dropped_reason=e.at[r.array_index, "exclusion_reason"], basis=basis, client=r.proposed_client))
                break
        else:
            raise ValueError(f"No replacement available in {r.proposed_client}")
    sel = pd.DataFrame(out)
    if not sel["array_index"].is_unique or (sel["proposed_client"].value_counts() != per_client).any():
        raise AssertionError("Selection is not 50 unique sensors per client")
    return sel, log


def prepare(paths, sel, node_order, district, out_dir, max_ffill=6):
    """Build the nine-month checkpoint for the selected sensors (only their rows are read)."""
    arrays = open_months(paths)
    rows = sel["array_index"].to_numpy()
    order = np.argsort(rows)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(rows))
    flow = np.concatenate([np.asarray(a[rows[order], :, config.FLOW], dtype=np.float64)[inv] for a in arrays], axis=1)
    return build_checkpoint(flow, sel, node_order, district, out_dir, max_ffill)


def build_checkpoint(flow, sel, node_order, district, out_dir, max_ffill=6):
    """Preprocess a (sensors, 273 days) flow series and write the checkpoint (train-only scaler, causal fill, index)."""
    rows = sel["array_index"].to_numpy()
    sp = split_bounds()
    n_steps = flow.shape[1]
    splits_mod.validate_splits(sp, n_steps)
    observed = preprocessing.observed_mask(flow)
    scaler = preprocessing.fit_scaler(flow, sp)
    tod = preprocessing.fit_tod_median(flow, sp, SPD)
    filled, imputed = preprocessing.causal_fill(flow, sp, tod, max_ffill, SPD)
    inputs = preprocessing.transform(filled, scaler)
    targets = preprocessing.transform(flow, scaler)
    idx = {}
    for name, (s, e) in sp.items():
        w = windows.window_index(observed, sp, name)
        idx[name] = w[(w[:, 1] - windows.INPUT_LEN >= s) & (w[:, 1] + windows.HORIZON <= e)]   # inputs AND targets inside
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    station = np.asarray(node_order)[rows]
    cfg = splits_mod.split_config(sp, n_steps, (191 / 273, 28 / 273, 54 / 273), SPD)
    cfg.update(split_days=SPLIT_DAYS, window_rule="inputs and targets inside one split")
    (out / "split_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "scaler_parameters.json").write_text(json.dumps(dict(scaler, array_index=rows.tolist(), station_id=station.tolist(),
                                                                id_mapping_status=config.ID_MAPPING_STATUS), indent=2), encoding="utf-8")
    np.savez(out / "pilot_preprocessed.npz", array_index=rows, station_id=station, flow_raw=flow.astype(np.float32),
             observed_mask=observed, inputs_scaled=inputs.astype(np.float32), input_imputed=imputed,
             targets_scaled=targets.astype(np.float32), tod_median=tod)
    np.savez(out / "window_index.npz", **idx)
    man = sel.assign(station_id=station, District=np.asarray(district)[rows])
    man.to_csv(out / "sensor_selection.csv", index=False)
    return {k: int(len(v)) for k, v in idx.items()}


def verify(out_dir, paths):
    """Read-only checks of the nine-month checkpoint; returns a list of (check, ok, detail)."""
    out = Path(out_dir)
    sp = split_bounds()
    d = np.load(out / "pilot_preprocessed.npz")
    w = np.load(out / "window_index.npz")
    sc = json.loads((out / "scaler_parameters.json").read_text(encoding="utf-8"))
    man = pd.read_csv(out / "sensor_selection.csv")
    rows = d["array_index"]
    arrays = open_months(paths)
    raw = np.concatenate([np.asarray(a[np.sort(rows), :, config.FLOW], dtype=np.float64) for a in arrays], axis=1)
    raw = raw[np.argsort(np.argsort(rows))]
    obs = np.isfinite(raw)
    a, b = sp["train"]
    tr = raw[:, a:b]
    mean, std = np.nanmean(tr, axis=1), np.nanstd(tr, axis=1)
    std = np.where(std < preprocessing.MIN_STD, 1.0, std)
    res = [("200 sensors, 50 per client", len(rows) == 200 and (man["proposed_client"].value_counts() == 50).all(), len(rows)),
           ("manifest order == arrays", np.array_equal(man["array_index"].to_numpy(), rows), ""),
           ("flow_raw == source (9 months)", np.array_equal(d["flow_raw"], raw.astype(np.float32), equal_nan=True), raw.shape),
           ("273 days of steps", raw.shape[1] == 273 * SPD, raw.shape[1]),
           ("scaler fit on train only", sc["fit_range"] == [a, b] and np.allclose(sc["mean"], mean) and np.allclose(sc["std"], std), ""),
           ("observed mask == finite", np.array_equal(d["observed_mask"], obs), ""),
           ("targets NaN exactly where missing", np.array_equal(np.isnan(d["targets_scaled"]), ~obs), ""),
           ("inputs finite", bool(np.isfinite(d["inputs_scaled"]).all()), ""),
           ("tod median train-only", np.allclose(d["tod_median"], preprocessing.fit_tod_median(raw, sp, SPD), equal_nan=True), "")]
    for name, (s, e) in sp.items():
        t0 = w[name][:, 1]
        res.append((f"{name}: inputs and targets inside split", bool(((t0 - 12 >= s) & (t0 + 12 <= e)).all()), int(len(t0))))
        cs = np.cumsum(np.c_[np.zeros(len(obs)), obs], axis=1)
        cnt = cs[w[name][:, 0], t0 + 12] - cs[w[name][:, 0], t0]
        res.append((f"{name}: every window has >= 1 observed target", bool((cnt >= 1).all()), ""))
    res.append(("splits disjoint and ordered", sp["train"][1] == sp["val"][0] and sp["val"][1] == sp["test"][0], sp))
    return res
