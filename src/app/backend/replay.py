"""Historical Traffic Replay service (AIORI 2.0, Phase 9A): historical XTraffic observations for the 200 study sensors and
on-demand forecasts from the frozen Stage 3 winner (MLP + FedAvgM, history A). Read-only; nothing here trains or tunes.

Data (src/data/processed/aiori2_3yr, Phase 2 preprocessing, conservative missing-value policy):
- observations: extracted/flow_raw.npy (200 sensors x 315,648 five-minute steps, 2022-01-01 00:00 to 2024-12-31 23:55);
  a reading is shown only where masks/conservative/valid_target.npy is true (the same mask used to score the models).
- timestamps: metadata/calendar/timestamp.npy. Naive Pacific wall-clock time, inferred by the Phase 2 audit; DST slots are
  neither shifted nor removed, so every timestamp is passed through unchanged.
- model inputs: experiments/conservative/A/filled_inputs.npy (causal forward fill <= 6 steps, then the 2023 training
  time-of-day median) scaled with the run's own client scaler scope; exactly the training/evaluation window contract.

Forecast semantics: issued at replay time T with the 12 readings T-55min..T (window t0 = T + 5 min in the experiment's
convention: inputs [t0-12, t0), targets [t0, t0+12)). Future observations are read separately, only for comparison.

Model: the frozen winner configuration (final_manifest.json, SHA-256 checked against final_manifest.sha256). One seed is
served per partition: DEPLOY_SEED, the first seed of the pre-registered seed list (20261027), fixed before any Stage 3
or test result existed, so it is chosen neither by validation nor by test performance. Its checkpoint hash must equal
the manifest; otherwise the service refuses to start.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]                       # src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiori2.arms import ARMS  # noqa: E402
from aiori2.models.factory import build_model  # noqa: E402

DATA = Path(os.environ.get("AIORI_REPLAY_DATA", ROOT / "data" / "processed" / "aiori2_3yr"))
STAGE3 = Path(os.environ.get("AIORI_REPLAY_STAGE3", ROOT / "results" / "aiori2" / "phase8c_stage3"))
SENSOR_META = Path(os.environ.get("AIORI_REPLAY_SENSOR_META", ROOT / "sensor_meta_feature.csv"))
POLICY, HISTORY = "conservative", "A"
DEPLOY_SEED = 20261027
PARTITIONS = {"geo": "fedavg_geo", "iid": "fedavg_iid"}
PARTITION_LABEL = {"geo": "Geographic non-IID federated (4 regional clients)", "iid": "IID federated (4 random clients)"}
WINNER = dict(model="mlp", algorithm="fedavgm", history="A")
TEST_MEAN_WAPE = {"iid": 10.506, "geo": 10.546}                  # three-seed test means from PHASE8C_REPORT.md (aggregate)
UNITS = "vehicles per 5-minute interval"
TZ_NOTE = "Pacific local wall-clock time (naive; inferred from the dataset, not stated by the publisher)"
CLIENT_CODE = {"client_0_la_ventura": ("LA", "LA / Ventura"), "client_1_bay_area": ("BA", "Bay Area"),
               "client_2_coastal_south_central": ("CS", "Coastal / South Central"), "client_3_inland_central": ("IC", "Inland / Central")}
CA_BOX = (32.4, 42.1, -124.6, -114.0)                             # California bounding box: lat min/max, lng min/max
H = L = 12
MIN_TYPICAL = 1.0                                                 # a relative-flow ratio needs a typical flow of >= 1 vehicle


class ReplayError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _r(v, d=2):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


class _LRU:
    """Small bounded, thread-safe cache."""

    def __init__(self, n):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()

    def get(self, k, make):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        v = make()
        with self.lock:
            self.d[k] = v
            while len(self.d) > self.n:
                self.d.popitem(last=False)
        return v


class Replay:
    def __init__(self, data=DATA, stage3=STAGE3, sensor_meta=SENSOR_META, device=None):
        t_start = time.perf_counter()
        data, stage3 = Path(data), Path(stage3)
        self.device = torch.device(device or ("cuda" if os.environ.get("AIORI_REPLAY_DEVICE") == "cuda" and torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        tl = json.loads((data / "metadata" / "timeline.json").read_text(encoding="utf-8"))
        self.spd, self.n = tl["steps_per_day"], tl["n_steps"]
        self.stamps = np.load(data / "metadata" / "calendar" / "timestamp.npy")
        origin = np.datetime64(tl["origin"], "m")
        if len(self.stamps) != self.n or not (self.stamps == origin + np.arange(self.n) * np.timedelta64(5, "m")).all():
            raise ValueError("timestamp index is not a regular 5-minute grid from the recorded origin")
        self.origin = datetime.fromisoformat(tl["origin"])
        self.dst = np.load(data / "metadata" / "calendar" / "dst_nonexistent_hour.npy")
        self.flow = np.load(data / "extracted" / "flow_raw.npy", mmap_mode="r")
        self.valid = np.load(data / "masks" / POLICY / "valid_target.npy", mmap_mode="r")
        if self.flow.shape != (200, self.n) or self.valid.shape != (200, self.n):
            raise ValueError("observation arrays do not match the timeline")
        exp = data / "experiments" / POLICY / HISTORY
        self.exp = json.loads((exp / "experiment.json").read_text(encoding="utf-8"))
        if self.exp["input_len"] != L or self.exp["horizon"] != H or self.exp["input_channels"] != ["flow"]:
            raise ValueError("experiment contract is not 12 flow inputs -> 12 steps")
        self.off = self.exp["timeline_offset"]
        self.filled = np.load(exp / "filled_inputs.npy", mmap_mode="r")
        self.imputed = np.load(exp / "input_imputed.npy", mmap_mode="r")
        self.typical = np.load(exp / "tod_median_train.npy")      # (200, 288) 2023 training time-of-day median, raw flow
        self.scalers_file = exp / "scalers.json"
        scopes = json.loads(self.scalers_file.read_text(encoding="utf-8"))["scopes"]
        b = self.exp["splits_index"]
        self.periods = [(b[k][0] + self.off, b[k][1] + self.off, k) for k in ("train", "val", "test")]
        self._sensors(data, sensor_meta)
        self.models = {p: self._load_model(stage3, p, scopes) for p in PARTITIONS}
        self.valid_count = np.asarray(self.valid).sum(0, dtype=np.int16)          # sensors with a valid reading, per step
        self.frames, self.forecasts = _LRU(512), _LRU(2048)
        self.load_seconds = round(time.perf_counter() - t_start, 2)

    # ------------------------------------------------------------------ sensors
    def _sensors(self, data, sensor_meta):
        s = pd.read_csv(data / "metadata" / "sensors.csv")
        if list(s["position"]) != list(range(200)) or not s["station_id"].is_unique:
            raise ValueError("sensor manifest is not 200 unique sensors in position order")
        meta = pd.read_csv(sensor_meta, sep="\t", usecols=["station_id", "Lat", "Lng", "District", "County", "Fwy", "Direction", "Type"])
        if not meta["station_id"].is_unique:
            raise ValueError("sensor metadata has duplicate station ids")
        j = s.merge(meta, on="station_id", how="left", validate="one_to_one")
        counters, out = {}, []
        lat_lo, lat_hi, lng_lo, lng_hi = CA_BOX
        for r in j.itertuples(index=False):
            code, name = CLIENT_CODE[r.client_geo]
            counters[code] = counters.get(code, 0) + 1
            lat, lng = float(r.Lat), float(r.Lng)
            ok = (np.isfinite(lat) and np.isfinite(lng) and lat_lo <= lat <= lat_hi and lng_lo <= lng <= lng_hi
                  and r.District == r.district)
            out.append(dict(id=f"{code}-{counters[code]:02d}", client_geo=r.client_geo, region=name, client_iid=r.client_iid,
                            type_group=r.type_group, district=int(r.district),
                            county=None if pd.isna(r.County) else str(r.County),
                            freeway=None if pd.isna(r.Fwy) else f"{r.Fwy}{'' if pd.isna(r.Direction) else '-' + str(r.Direction)}",
                            lat=round(lat, 6) if ok else None, lng=round(lng, 6) if ok else None,
                            location_status="provisional" if ok else "excluded"))
        coords = [(x["lat"], x["lng"]) for x in out if x["lat"] is not None]
        for x in out:
            x["shares_location"] = x["lat"] is not None and coords.count((x["lat"], x["lng"])) > 1
        self.sensors = out
        self.by_id = {x["id"]: i for i, x in enumerate(out)}
        self.client_geo = np.array([x["client_geo"] for x in out])

    # ------------------------------------------------------------------ model
    def _load_model(self, stage3, part, scopes):
        man_path = stage3 / "analysis" / "final_manifest.json"
        expect = (stage3 / "analysis" / "final_manifest.sha256").read_text(encoding="utf-8").split()[0].strip()
        man_sha = sha256(man_path)
        if man_sha != expect:
            raise ValueError("final_manifest.json does not match its frozen SHA-256")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        arm = PARTITIONS[part]
        hits = [m for m in man["models"] if m["label"].startswith("winner:") and m["arm"] == arm and m["seed"] == DEPLOY_SEED]
        if len(hits) != 1:
            raise ValueError(f"manifest has no unique winner entry for {arm} seed {DEPLOY_SEED}")
        m = hits[0]
        if any(m[k] != v for k, v in WINNER.items()):
            raise ValueError("manifest winner is not the expected MLP + FedAvgM history A configuration")
        run = stage3 / m["run_id"]                                   # resolved inside the Stage 3 folder, never from the request
        if sha256(run / "best.pt") != m["best_pt_sha256"] or sha256(run / "run.json") != m["run_json_sha256"]:
            raise ValueError(f"{m['run_id']}: checkpoint does not match the frozen manifest")
        if sha256(self.scalers_file) != m["scalers_sha256"] or m["scaler_scope"] != ARMS[arm]["scaler"]:
            raise ValueError("scaler file or scope does not match the frozen manifest")
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if (cfg["arm"], cfg["history"], cfg["model"], cfg["policy"], cfg["seed"]) != (arm, HISTORY, "mlp", POLICY, DEPLOY_SEED):
            raise ValueError("run config does not match the manifest entry")
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        meta = json.loads(ck["meta_json"])
        if meta["run_id"] != m["run_id"] or set(ck["models"]) != {"global"}:
            raise ValueError("checkpoint is not the aggregated global model of this run")
        net = build_model(cfg["model"], cfg["preset"], cfg["in_channels"])
        net.load_state_dict(ck["models"]["global"], strict=True)
        net = net.to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        sc = scopes[m["scaler_scope"]]
        return dict(net=net, mean=np.asarray(sc["mean"], np.float32), std=np.asarray(sc["std"], np.float32),
                    provenance=dict(model_id=f"aiori2-mlp-fedavgm-A-{part}-s{DEPLOY_SEED}", partition=part,
                                    partition_label=PARTITION_LABEL[part], architecture="MLP (medium preset), 12 flow inputs -> 12 steps",
                                    algorithm="FedAvgM (server momentum 0.9)", training_history="A: trained on 2023",
                                    seed=DEPLOY_SEED, run_id=m["run_id"], checkpoint_epoch=meta["epoch"],
                                    checkpoint_sha256=m["best_pt_sha256"], manifest_sha256=man_sha,
                                    scaler_scope=m["scaler_scope"], validation_wape_pct_this_seed=round(m["val_full"]["wape_pct"], 3),
                                    test_wape_pct_three_seed_mean=TEST_MEAN_WAPE[part],
                                    seed_policy="first pre-registered seed; not chosen by validation or test results",
                                    performance_note="The test WAPE is the mean of three seeds over all sensors and windows; "
                                                     "it is not a guarantee for this seed, sensor or time.",
                                    negative_clipping=False, units=UNITS))

    # ------------------------------------------------------------------ time
    def ts(self, i):
        return str(self.stamps[int(i)])[:16]

    def parse_time(self, s):
        if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s):
            raise ReplayError(400, "time must look like 2024-03-12T07:30 (Pacific local wall-clock, 5-minute steps)")
        try:
            t = datetime.strptime(s, "%Y-%m-%dT%H:%M")
        except ValueError:
            raise ReplayError(400, "time is not a valid date and clock time")
        if t.minute % 5:
            raise ReplayError(400, "time must fall on a 5-minute boundary")
        i = (t - self.origin) // timedelta(minutes=5)
        if not 0 <= i < self.n:
            raise ReplayError(404, f"time is outside the dataset ({self.ts(0)} to {self.ts(self.n - 1)})")
        return int(i)

    def parse_date(self, s):
        if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ReplayError(400, "date must look like 2024-03-12")
        return self.parse_time(s + "T00:00") // self.spd

    def sensor_pos(self, sid):
        if not isinstance(sid, str) or sid not in self.by_id:
            raise ReplayError(404, "Unknown sensor. Use an id from /api/replay/sensors, for example LA-01.")
        return self.by_id[sid]

    def period(self, i):
        for a, b, k in self.periods:
            if a <= i < b:
                return k
        return "before_training"

    PERIOD_TEXT = {"before_training": "2022: before the model's training history (observations only, no forecasts)",
                   "train": "2023: the model's training period (in-sample)",
                   "val": "Jan-Jun 2024: validation period (used to select the model)",
                   "test": "Jul-Dec 2024: final test period (evaluated once; replay only, no tuning)"}

    # ------------------------------------------------------------------ views
    def meta(self):
        per_day = self.valid_count.reshape(-1, self.spd).sum(1) / (200 * self.spd)
        days = [str(self.stamps[d * self.spd])[:10] for d in range(len(per_day))]
        has = np.flatnonzero(self.valid_count > 0)
        mapped = sum(x["lat"] is not None for x in self.sensors)
        return dict(
            source="Historical XTraffic observations (2022-2024), replayed; not a live feed",
            units=UNITS, timezone=TZ_NOTE, step_minutes=5, n_sensors=200, mapped_sensors=mapped,
            first_timestamp=self.ts(has[0]), last_timestamp=self.ts(has[-1]),
            first_forecast_time=self.ts(self.off + L - 1),
            missing_value_policy="conservative: non-finite flow, and readings where flow, occupancy and speed are all zero "
                                 "(2022 and 2024), are treated as missing",
            dates=[dict(date=d, coverage=round(float(c), 4)) for d, c in zip(days, per_day)],
            periods=[dict(key=k, start=self.ts(a), end=self.ts(b - 1), label=self.PERIOD_TEXT[k]) for a, b, k in self.periods],
            location_note="Coordinates come from the dataset's station metadata through the study's row-to-station mapping, "
                          "which is supported but unconfirmed; all locations are provisional.",
            congestion_note="Congestion levels are not shown: the forecasting data contains flow only, and no validated "
                            "speed/occupancy thresholds exist in this project. Colours show flow relative to the sensor's "
                            "2023 typical flow for that time of day, which is not a congestion measure.",
            models=[m["provenance"] for m in self.models.values()], default_partition="geo")

    def day(self, date):
        d = self.parse_date(date)
        a = d * self.spd
        cnt = self.valid_count[a:a + self.spd]
        return dict(date=date, period=self.period(a), period_label=self.PERIOD_TEXT[self.period(a)],
                    valid_sensors=cnt.astype(int).tolist(), dst_nonexistent=np.flatnonzero(self.dst[a:a + self.spd]).tolist(),
                    available_slots=int((cnt > 0).sum()))

    def _obs(self, pos, i):
        """(value or None, status) for raw reading(s): ok | missing (no reading) | invalid (reading masked by the policy)."""
        v = np.asarray(self.flow[pos, i], np.float64)
        ok = np.asarray(self.valid[pos, i], bool)
        status = np.where(ok, "ok", np.where(np.isfinite(v), "invalid", "missing"))
        return np.where(ok, v, np.nan), status

    def frame(self, time_s, layer="observed", partition="geo", horizon=12):
        i = self.parse_time(time_s)
        if layer not in ("observed", "forecast"):
            raise ReplayError(400, "layer must be observed or forecast")
        if partition not in PARTITIONS:
            raise ReplayError(400, "partition must be geo or iid")
        if not isinstance(horizon, int) or not 1 <= horizon <= H:
            raise ReplayError(400, "horizon must be 1 to 12 steps")
        key = (i, layer, partition if layer == "forecast" else "", horizon if layer == "forecast" else 0)
        return self.frames.get(key, lambda: self._frame(i, layer, partition, horizon))

    def _frame(self, i, layer, partition, horizon):
        pos = np.arange(200)
        out = dict(time=self.ts(i), layer=layer, units=UNITS, period=self.period(i), dst_nonexistent=bool(self.dst[i]))
        if layer == "observed":
            v, st = self._obs(pos, i)
            typ = self.typical[:, i % self.spd]
            out.update(values=[_r(x) for x in v], status=st.tolist(),
                       relative=[_r(x / t, 3) if np.isfinite(x) and t >= MIN_TYPICAL else None for x, t in zip(v, typ)],
                       valid_sensors=int(np.isfinite(v).sum()))
            return out
        t0 = i + 1
        self._check_forecastable(t0)
        x, imp = self._inputs(pos, t0)
        pred = self._predict(partition, pos, x)[:, horizon - 1]
        target = t0 + horizon - 1
        usable = ~imp.all(1)
        pred = np.where(usable, pred, np.nan)
        typ = self.typical[:, target % self.spd]
        act = self._obs(pos, target)[0] if target < self.n else np.full(200, np.nan)
        out.update(target_time=self.ts(target) if target < self.n else None, horizon=horizon, partition=partition,
                   model_id=self.models[partition]["provenance"]["model_id"], values=[_r(x) for x in pred],
                   status=np.where(usable, "ok", "no_input").tolist(), actual=[_r(x) for x in act],
                   relative=[_r(p / t, 3) if np.isfinite(p) and t >= MIN_TYPICAL else None for p, t in zip(pred, typ)],
                   valid_sensors=int(usable.sum()))
        return out

    def series(self, sid, date):
        pos, d = self.sensor_pos(sid), self.parse_date(date)
        a = d * self.spd
        v, st = self._obs(pos, slice(a, a + self.spd))
        return dict(sensor=sid, date=date, units=UNITS, timestamps=[self.ts(a + k) for k in range(self.spd)],
                    values=[_r(x) for x in v], status=st.tolist(),
                    typical=[_r(x) for x in self.typical[pos]], typical_label="2023 training-period median for this time of day")

    # ------------------------------------------------------------------ inference
    def _check_forecastable(self, t0):
        if t0 - self.off < L:
            raise ReplayError(422, f"Forecasts start at {self.ts(self.off + L - 1)}: the model's inputs (history A "
                                   "preprocessing) begin on 2023-01-01, so 2022 is available as observations only.")

    def _inputs(self, pos, t0):
        e = t0 - self.off
        x = np.asarray(self.filled[pos, e - L:e], np.float32)
        imp = np.asarray(self.imputed[pos, e - L:e], bool)
        if not np.isfinite(x).all():
            raise ReplayError(500, "non-finite model inputs")
        return x, imp

    def _predict(self, part, pos, x):
        m = self.models[part]
        z = ((x - m["mean"][pos][:, None]) / m["std"][pos][:, None]).astype(np.float32)[..., None]     # (B, 12, 1)
        with self.lock, torch.no_grad():
            out = m["net"](torch.from_numpy(z).to(self.device)).float().cpu().numpy()
        return np.asarray(out, np.float64) * m["std"][pos][:, None] + m["mean"][pos][:, None]

    def forecast(self, sid, time_s, partition="geo"):
        pos, i = self.sensor_pos(sid), self.parse_time(time_s)
        if partition not in PARTITIONS:
            raise ReplayError(400, "partition must be geo or iid")
        return self.forecasts.get((pos, i, partition), lambda: self._forecast(pos, i, partition))

    def _forecast(self, pos, i, part):
        t0 = i + 1                                                  # first forecast step is the next 5-minute slot
        self._check_forecastable(t0)
        x, imp = self._inputs(np.array([pos]), t0)
        if imp[0].all():
            raise ReplayError(422, f"No valid reading at this sensor between {self.ts(t0 - L)} and {self.ts(i)}: the model "
                                   "input would be entirely imputed, so no forecast is shown.")
        tic = time.perf_counter()
        pred = self._predict(part, np.array([pos]), x)[0]
        ms = (time.perf_counter() - tic) * 1000
        steps = np.arange(t0, t0 + H)
        inside = steps < self.n
        act, st = self._obs(pos, steps[inside])
        actual = np.full(H, np.nan)
        actual[inside] = act
        status = np.array(["beyond_dataset"] * H, dtype=object)
        status[inside] = st
        err = np.where(np.isfinite(actual), pred - actual, np.nan)
        n = int(np.isfinite(err).sum())
        denom = float(np.nansum(np.where(np.isfinite(err), actual, np.nan))) if n else 0.0
        obs_in, _ = self._obs(pos, np.arange(t0 - L, t0))
        periods = {self.period(int(s)) for s in steps[inside]}
        warnings = []
        if imp[0].any():
            warnings.append(f"{int(imp[0].sum())} of 12 input readings were missing and filled by the original preprocessing "
                            "(forward fill up to 30 minutes, then the 2023 typical value).")
        if len(periods) > 1:
            warnings.append("The forecast hour crosses a dataset period boundary.")
        if not inside.all():
            warnings.append("Part of the forecast hour lies after the end of the dataset; no observations exist to compare.")
        if (pred < 0).any():
            warnings.append("Some predicted values are negative; they are shown unclipped, as evaluated in the study.")
        if self.dst[steps[inside]].any() or self.dst[t0 - L:t0].any():
            warnings.append("This hour includes the spring-forward DST hour, which does not exist on the wall clock.")
        p = self.period(t0)
        return dict(sensor=self.sensors[pos]["id"], issued_at=self.ts(i), units=UNITS, timezone=TZ_NOTE,
                    model=self.models[part]["provenance"], period=p, period_label=self.PERIOD_TEXT[p],
                    input=dict(timestamps=[self.ts(t0 - L + k) for k in range(L)], model_input=[_r(v) for v in x[0]],
                               observed=[_r(v) for v in obs_in], imputed=imp[0].tolist()),
                    forecast=dict(timestamps=[self.ts(s) if s < self.n else None for s in steps], values=[_r(v) for v in pred]),
                    actual=dict(label="observed afterwards (historical; never used as model input)",
                                values=[_r(v) for v in actual], status=status.tolist()),
                    comparison=dict(abs_error=[_r(abs(e)) if np.isfinite(e) else None for e in err], n_compared=n,
                                    mae=_r(np.nanmean(np.abs(err))) if n else None,
                                    wape_pct=_r(100 * np.nansum(np.abs(err)) / denom, 2) if n and denom > 0 else None,
                                    note="Errors are computed only for steps with a valid observation; WAPE needs a "
                                         "positive observed total."),
                    warnings=warnings, inference_ms=round(ms, 2), device=str(self.device))
