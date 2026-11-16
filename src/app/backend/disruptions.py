"""Historical Disruption Explorer service (AIORI 2.0, Phase 9B). Serves the offline candidate-event catalog written by
disruption_detect.py and reuses the Phase 9A Replay service for observations and frozen-model forecasts. Read-only.

The catalog is loaded once at start-up and checked (file hash, detector version, configuration hash, input-data hashes);
it is never rebuilt during a request. A missing or stale catalog makes only these endpoints answer 503 with the rebuild
command; the replay endpoints keep working.

    .venv\\Scripts\\python.exe src\\app\\backend\\disruptions.py robustness     # optional offline summary (~1 min, CPU)
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import disruption_detect as D  # noqa: E402
from replay import L, H, UNITS, TZ_NOTE, ReplayError, _LRU, _r  # noqa: E402

ID_RE = re.compile(r"\d{8}T\d{4}-(LA|BA|CS|IC)-[DU]-[0-9a-f]{6}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
CATEGORIES = {
    "sudden_decrease": "Sudden decrease: flow well below this sensor's typical 2023 level for this time and day type, for less than an hour",
    "sudden_increase": "Sudden increase: flow well above the typical level, for less than an hour",
    "sustained_deviation": "Sustained deviation: one or two sensors away from the typical level for an hour or more",
    "coordinated_deviation": "Coordinated deviation: three or more sensors in the same dataset region deviating in the same direction at overlapping times",
    "sensor_quality": "Sensor-quality anomaly: most anomalous readings carry quality flags, so a traffic change and a sensor fault cannot be told apart",
}
QUALITY_TEXT = {
    "zero_reading": "exact-zero readings (a closure and a detector dropout look the same)",
    "stuck_run": "the same non-zero value repeated for an hour or more",
    "near_missing": "within an hour of missing or invalid readings",
    "level_shift": "the sensor deviates from its 2023 pattern for more than 5% of that month's readings (recalibration, works, "
                   "seasonal or long-term change)",
}
LIMITATIONS = [
    "Historical traffic anomalies can be investigated here, but their underlying causes are not verified. These results do "
    "not establish forecasting accuracy during earthquakes or other disasters.",
    "Candidate events were identified retrospectively from complete days of data; this is not a real-time or live warning system.",
    "Flow alone cannot identify accidents, closures, weather, disasters or congestion; no external incident records were used.",
    "Regions are the study's four geographic data groups, not road connections; sensor coordinates are provisional.",
    "The typical-flow baseline comes from 2023 only (weekday-like / weekend-like days); seasonal change and long-term demand "
    "shifts in 2022 and 2024 can appear as deviations.",
    "Forecast comparisons are exploratory replays of the frozen model; they are separate from, and do not replace, the "
    "frozen final test evaluation, and no new test score is claimed.",
]
LIST_PARAMS = {"from", "to", "region", "category", "direction", "sensor", "min_sensors", "min_duration", "max_duration",
               "min_peak_z", "confidence", "quality", "holiday", "sort", "page", "page_size"}
SORTS = {"start": ("start_index", False), "-start": ("start_index", True), "score": ("score", False), "-score": ("score", True),
         "peak": ("abs_peak", False), "-peak": ("abs_peak", True), "duration": ("duration_min", False),
         "-duration": ("duration_min", True), "sensors": ("n_sensors", False), "-sensors": ("n_sensors", True)}
MAX_PAGE = 100
PAD = 24                      # series window: 2 h either side of the event
MAX_WINDOW = 3 * 288          # at most 3 days of readings per series response
MAX_ISSUES = 72               # forecast issuances per robustness request
COMPARE_MAX = 12


class Unavailable(Exception):
    pass


def _int(p, k, lo, hi, default=None):
    v = p.get(k)
    if v is None:
        return default
    if not re.fullmatch(r"\d{1,7}", v) or not lo <= int(v) <= hi:
        raise ReplayError(400, f"{k} must be a whole number from {lo} to {hi}")
    return int(v)


class Disruptions:
    def __init__(self, replay, root=D.OUT, verify_inputs=True):
        t0 = time.perf_counter()
        self.r = replay
        d = Path(root) / D.VERSION
        cat = d / "catalog.json"
        rebuild = "Build it with: .venv\\Scripts\\python.exe src\\app\\backend\\disruption_detect.py build"
        if not cat.is_file():
            raise Unavailable(f"The disruption catalog has not been built yet. {rebuild}")
        want = (d / "catalog.sha256").read_text(encoding="utf-8").split()[0]
        if D.sha256(cat) != want:
            raise Unavailable(f"The disruption catalog does not match its recorded hash. {rebuild}")
        c = json.loads(cat.read_text(encoding="utf-8"))
        if c.get("schema") != D.SCHEMA or c.get("version") != D.VERSION or c.get("config_sha256") != D.config_hash():
            raise Unavailable(f"The disruption catalog was made by a different detector version or configuration. {rebuild}")
        if D.events_hash(c["events"]) != c["events_sha256"]:
            raise Unavailable(f"The disruption catalog's events do not match their recorded hash. {rebuild}")
        if verify_inputs:
            data = D.DATA
            for k, h in c["provenance"]["inputs"].items():
                if D.sha256(data / k) != h:
                    raise Unavailable(f"The source data changed after the catalog was built ({k}). {rebuild}")
        bl = d / c["baseline"]["file"]
        if D.sha256(bl) != c["baseline"]["sha256"]:
            raise Unavailable(f"The baseline file does not match the catalog. {rebuild}")
        z = np.load(bl)
        self.med, self.spread, self.dtype = z["median"], z["spread"], z["day_type"]
        self.cat, self.events = c, c["events"]
        self.by_id = {e["id"]: e for e in self.events}
        ev = self.events
        self.ix = dict(start_index=np.array([e["start_index"] for e in ev]), end_index=np.array([e["end_index"] for e in ev]),
                       score=np.array([e["score"] for e in ev]), abs_peak=np.array([abs(e["peak_z"]) for e in ev]),
                       duration_min=np.array([e["duration_min"] for e in ev]), n_sensors=np.array([e["n_sensors"] for e in ev]),
                       category=np.array([e["category"] for e in ev]), region=np.array([e["region_code"] for e in ev]),
                       direction=np.array([e["direction"] for e in ev]), confidence=np.array([e["confidence"] for e in ev]),
                       holiday=np.array([e["holiday"] is not None for e in ev]))
        self.by_sensor = {}
        for k, e in enumerate(ev):
            for s in e["sensors"]:
                self.by_sensor.setdefault(s, []).append(k)
        self.sensor_events = {s: np.array(v) for s, v in self.by_sensor.items()}
        self.lo, self.hi, self.rmin = c["thresholds"]["z_low"], c["thresholds"]["z_high"], c["thresholds"]["min_relative"]
        rs = Path(root) / D.VERSION / "robustness_summary.json"
        self.summary = json.loads(rs.read_text(encoding="utf-8")) if rs.is_file() else None
        self.cache = _LRU(256)
        self.load_seconds = round(time.perf_counter() - t0, 2)

    # ------------------------------------------------------------------ meta / list
    def meta(self):
        c = self.cat
        return dict(
            title="Historical Disruption Explorer", version=c["version"], created_utc=c["created_utc"],
            config=c["config"], config_sha256=c["config_sha256"], thresholds=c["thresholds"], calendar=c["calendar"],
            baseline=dict(c["baseline"], method="Median and 1.4826 x MAD of valid 2023 readings per sensor, day type and "
                                                 "five-minute slot (pooled with the neighbouring slots), spread floored at "
                                                 "sqrt(median)", label="Typical 2023 flow (training year)"),
            provenance=dict(c["provenance"], inputs=sorted(c["provenance"]["inputs"])), summary=c["summary"],
            audit=c["audit"], events_sha256=c["events_sha256"], units=UNITS, timezone=TZ_NOTE,
            categories=CATEGORIES, quality_flags=QUALITY_TEXT, limitations=LIMITATIONS,
            regions={k: v for k, v in (("LA", "LA / Ventura"), ("BA", "Bay Area"), ("CS", "Coastal / South Central"),
                                       ("IC", "Inland / Central"))},
            first_forecast_time=self.r.ts(self.r.off + L - 1), robustness_summary=self.summary,
            score_note="Score = sum over the event's anomalous readings of min(|z|, 10) x 5/60 (capped z-hours): a magnitude, "
                       "not a probability or a likelihood of any cause.")

    def _date_index(self, s, k):
        if not isinstance(s, str) or not DATE_RE.fullmatch(s):
            raise ReplayError(400, f"{k} must look like 2024-03-12")
        return self.r.parse_date(s) * self.r.spd

    def list(self, p):
        extra = set(p) - LIST_PARAMS
        if extra:
            raise ReplayError(400, f"unknown parameter(s): {', '.join(sorted(extra))[:80]}")
        ix, keep = self.ix, np.ones(len(self.events), bool)
        if "from" in p:
            keep &= ix["end_index"] >= self._date_index(p["from"], "from")
        if "to" in p:
            keep &= ix["start_index"] < self._date_index(p["to"], "to") + self.r.spd
        for k, col, allowed in (("region", "region", {"LA", "BA", "CS", "IC"}), ("category", "category", set(CATEGORIES)),
                                ("direction", "direction", {"decrease", "increase"}), ("confidence", "confidence", {"high", "medium", "low"})):
            if k in p:
                vals = p[k].split(",")
                if not vals or not set(vals) <= allowed:
                    raise ReplayError(400, f"{k} must be one or more of {', '.join(sorted(allowed))}")
                keep &= np.isin(ix[col], vals)
        if "sensor" in p:
            self.r.sensor_pos(p["sensor"])
            m = np.zeros(len(self.events), bool)
            m[self.sensor_events.get(p["sensor"], np.array([], int))] = True
            keep &= m
        if (v := _int(p, "min_sensors", 1, 200)) is not None:
            keep &= ix["n_sensors"] >= v
        if (v := _int(p, "min_duration", 0, 10 ** 6)) is not None:
            keep &= ix["duration_min"] >= v
        if (v := _int(p, "max_duration", 0, 10 ** 6)) is not None:
            keep &= ix["duration_min"] <= v
        if "min_peak_z" in p:
            try:
                v = float(p["min_peak_z"])
            except ValueError:
                raise ReplayError(400, "min_peak_z must be a number")
            if not 0 <= v <= 1000:
                raise ReplayError(400, "min_peak_z must be from 0 to 1000")
            keep &= ix["abs_peak"] >= v
        for k, col in (("quality", ix["category"] == "sensor_quality"), ("holiday", ix["holiday"])):
            mode = p.get(k, "include")
            if mode not in ("include", "exclude", "only"):
                raise ReplayError(400, f"{k} must be include, exclude or only")
            keep &= ~col if mode == "exclude" else col if mode == "only" else True
        sort = p.get("sort", "-score")
        if sort not in SORTS:
            raise ReplayError(400, f"sort must be one of {', '.join(SORTS)}")
        size = _int(p, "page_size", 1, MAX_PAGE, 25)
        page = _int(p, "page", 1, 10 ** 6, 1)
        idx = np.flatnonzero(keep)
        col, desc = SORTS[sort]
        key = ix[col][idx]
        order = np.lexsort((ix["start_index"][idx], -key if desc else key))      # ties: earlier events first, then catalog order
        idx = idx[order]
        total = len(idx)
        sel = idx[(page - 1) * size: page * size]
        return dict(total=int(total), page=page, page_size=size, pages=max(1, -(-total // size)), sort=sort,
                    items=[self._brief(self.events[k]) for k in sel])

    @staticmethod
    def _brief(e):
        b = {k: e[k] for k in ("id", "start", "end", "duration_min", "direction", "category", "region_code", "n_sensors",
                               "peak_z", "peak_relative", "peak_sensor", "peak_time", "score", "quality_share", "missing_share",
                               "quality_flags", "confidence", "day_type", "holiday")}
        b["sensors"] = e["sensors"][:8]
        return b

    # ------------------------------------------------------------------ detail
    def event(self, eid):
        if not isinstance(eid, str) or not ID_RE.fullmatch(eid) or eid not in self.by_id:
            raise ReplayError(404, "Unknown event. Use an id from /api/replay/disruptions.")
        return self.by_id[eid]

    def detail(self, eid):
        e = self.event(eid)
        first = self.r.ts(self.r.off + L - 1)
        issue_onset = self.r.ts(max(e["start_index"] - 1, 0))
        forecastable = issue_onset >= first
        why = (f"{e['n_anomalous']} five-minute readings at {e['n_sensors']} sensor{'s' if e['n_sensors'] > 1 else ''} were "
               f"{'below' if e['direction'] == 'decrease' else 'above'} the sensor's typical 2023 flow for this time and "
               f"{e['day_type'].replace('_', '-')} day by more than the detection thresholds (z {'<=' if e['direction'] == 'decrease' else '>='} "
               f"{self.lo if e['direction'] == 'decrease' else self.hi:.2f} and at least {int(self.rmin * 100)}% from typical), "
               f"for at least 15 minutes per sensor.")
        return dict({k: v for k, v in e.items() if k not in ("start_index", "end_index")},
                    region=self.meta_regions()[e["region_code"]], category_text=CATEGORIES[e["category"]],
                    quality_text=[QUALITY_TEXT[q] for q in e["quality_flags"]], why_flagged=why,
                    period=self.r.period(e["start_index"]), period_label=self.r.PERIOD_TEXT[self.r.period(e["start_index"])],
                    replay=dict(date=e["peak_time"][:10], time=e["peak_time"], sensor=e["peak_sensor"]),
                    forecast=dict(available=forecastable, issue_at_onset=issue_onset if forecastable else None,
                                  issue_at_peak=self.r.ts(max(self.r.parse_time(e["peak_time"]) - 1, 0)) if forecastable else None,
                                  first_forecast_time=first,
                                  reason=None if forecastable else "This event is before the selected model's history A "
                                         "inputs begin (2023-01-01), so no forecast is available; observations only."),
                    detection_note="Identified retrospectively from complete historical days; not a real-time detection.",
                    units=UNITS, timezone=TZ_NOTE)

    @staticmethod
    def meta_regions():
        return {"LA": "LA / Ventura", "BA": "Bay Area", "CS": "Coastal / South Central", "IC": "Inland / Central"}

    def _window(self, e, sd=None):
        a, b = e["start_index"] - PAD, e["end_index"] + PAD + 1
        if b - a > MAX_WINDOW:                                     # long episode: centre on the sensor's peak
            c = self.r.parse_time((sd or e)["peak_time"])
            a, b = c - MAX_WINDOW // 2, c + MAX_WINDOW // 2
        return max(a, 0), min(b, self.r.n)

    def _expected(self, pos, a, b):
        m, sp = D.expected(self.med, self.spread, self.dtype, pos, a, b)
        return m.astype(np.float64), sp.astype(np.float64)

    def series(self, eid, sid):
        e = self.event(eid)
        pos = self.r.sensor_pos(sid)
        sd = next((s for s in e["sensor_details"] if s["sensor"] == sid), None)
        if sd is None:
            raise ReplayError(404, f"{sid} is not one of this event's affected sensors")
        return self.cache.get(("series", eid, sid), lambda: self._series(e, sd, pos))

    def _series(self, e, sd, pos):
        a, b = self._window(e, sd)
        x = np.asarray(self.r.flow[pos, a:b], np.float64)
        ok = np.asarray(self.r.valid[pos, a:b], bool)
        m, sp = self._expected(pos, a, b)
        z, rel = D.score(x, ok, m, sp)
        f = D.flag(z, rel, self.lo, self.hi)
        q = D.slot_quality(np.asarray(self.r.flow[pos, max(a - 12, 0):b + 12], np.float64),
                          np.asarray(self.r.valid[pos, max(a - 12, 0):b + 12], bool))[a - max(a - 12, 0):][:b - a]
        lo = np.where(np.isfinite(m), np.minimum(m + self.lo * sp, m * (1 - self.rmin)), np.nan)
        hi = np.where(np.isfinite(m), np.maximum(m + self.hi * sp, m * (1 + self.rmin)), np.nan)
        scored = np.isfinite(z)
        _, status = self.r._obs(pos, np.arange(a, b))
        return dict(event=e["id"], sensor=sd["sensor"], units=UNITS, timezone=TZ_NOTE,
                    window=dict(start=self.r.ts(a), end=self.r.ts(b - 1), truncated=(e["end_index"] + PAD + 1) - (e["start_index"] - PAD) > MAX_WINDOW),
                    sensor_interval=dict(start=sd["start"], end=sd["end"]), event_interval=dict(start=e["start"], end=e["end"]),
                    timestamps=[self.r.ts(i) for i in range(a, b)],
                    observed=[_r(v, 1) if k else None for v, k in zip(x, ok)], status=status.tolist(),
                    expected=[_r(v, 1) for v in m], threshold_low=[_r(v, 1) for v in lo], threshold_high=[_r(v, 1) for v in hi],
                    z=[_r(v, 2) for v in z], anomalous=f.astype(int).tolist(), scored=scored.tolist(),
                    quality=[[k for k, v in D.Q_BITS.items() if int(qq) & v] for qq in q],
                    dst_nonexistent=np.flatnonzero(self.r.dst[a:b]).tolist(),
                    expected_label="Typical 2023 flow for this time and day type (a reference, not a measurement)",
                    threshold_label="Detection thresholds: readings outside this range (and scored) are anomalous",
                    unscored_note=f"Readings are not scored where the typical flow is below {D.CONFIG['min_baseline']:g} "
                                  "vehicles per 5 minutes or no baseline exists.")

    def compare(self, eid):
        e = self.event(eid)
        return self.cache.get(("compare", eid), lambda: self._compare(e))

    def _compare(self, e):
        a, b = self._window(e)
        details = sorted(e["sensor_details"], key=lambda s: -abs(s["peak_z"]))[:COMPARE_MAX]
        rows = []
        for sd in details:
            pos = self.r.sensor_pos(sd["sensor"])
            x = np.asarray(self.r.flow[pos, a:b], np.float64)
            ok = np.asarray(self.r.valid[pos, a:b], bool)
            m, _ = self._expected(pos, a, b)
            good = ok & np.isfinite(m) & (m >= D.CONFIG["min_baseline"])
            with np.errstate(all="ignore"):
                ratio = np.where(good, x / m, np.nan)
            rows.append(dict(sensor=sd["sensor"], ratio=[_r(v, 3) for v in ratio], available=float(good.mean())))
        return dict(event=e["id"], timestamps=[self.r.ts(i) for i in range(a, b)], sensors=rows,
                    shown=len(rows), total_sensors=e["n_sensors"],
                    normalization="Each sensor's flow divided by its own typical 2023 flow for that time and day type "
                                  "(1.0 = typical), so sensors with different traffic volumes can share one axis. Blank "
                                  f"where the reading is missing or the typical flow is below {D.CONFIG['min_baseline']:g}.")

    # ------------------------------------------------------------------ forecast robustness (exploratory)
    def _forecast_block(self, part, pos, issues):
        """Frozen-model forecasts for issue indices T (inputs T-11..T, exactly the replay service's contract)."""
        r = self.r
        e = issues + 1 - r.off                                       # window t0 = T + 1 in experiment coordinates
        idx = e[:, None] + np.arange(-L, 0)[None, :]
        x = np.asarray(r.filled[pos][idx], np.float32)
        imp = np.asarray(r.imputed[pos][idx], bool)
        pred = r._predict(part, np.full(len(issues), pos), x)
        return pred, ~imp.all(1)

    def _errors(self, pos, issues, pred, usable, mask_fn=None):
        r = self.r
        tg = issues[:, None] + 1 + np.arange(H)[None, :]
        inside = tg < r.n
        tgc = np.minimum(tg, r.n - 1)
        act = np.where(np.asarray(r.valid[pos][tgc]) & inside, np.asarray(r.flow[pos][tgc], np.float64), np.nan)
        scope = inside & usable[:, None] & (mask_fn(tg) if mask_fn is not None else True)
        act = np.where(scope, act, np.nan)
        return pred - act, act, scope

    @staticmethod
    def _metrics(err, act, scope):
        ok = np.isfinite(err)
        n = int(ok.sum())
        if not n:
            return dict(n=0, mae=None, wape_pct=None, mae_by_horizon=[None] * H, missing_targets=int(scope.sum()))
        den = float(np.nansum(np.where(ok, act, np.nan)))
        with np.errstate(all="ignore"):
            byh = np.nanmean(np.where(ok, np.abs(err), np.nan), 0)
        return dict(n=n, mae=_r(np.nanmean(np.abs(err[ok]))), wape_pct=_r(100 * np.abs(err[ok]).sum() / den) if den > 0 else None,
                    mae_by_horizon=[_r(v) for v in byh], missing_targets=int(scope.sum() - n))

    def robustness(self, eid, sid, part):
        e = self.event(eid)
        pos = self.r.sensor_pos(sid)
        if part not in ("geo", "iid"):
            raise ReplayError(400, "partition must be geo or iid")
        sd = next((s for s in e["sensor_details"] if s["sensor"] == sid), None)
        if sd is None:
            raise ReplayError(404, f"{sid} is not one of this event's affected sensors")
        return self.cache.get(("rob", eid, sid, part), lambda: self._robustness(e, sd, pos, part))

    def _reference_days(self, pos, a, b, k=4, max_days=28):
        """Up to k days d (nearest first, same day type as the event day) for which [a+d, b+d) of this sensor overlaps no
        catalog event and has >= 80% valid readings."""
        spd, ev = self.r.spd, self.sensor_events.get(self.r.sensors[pos]["id"], np.array([], int))
        s_i, e_i = self.ix["start_index"][ev], self.ix["end_index"][ev]
        want = self.dtype[a // spd]
        out = []
        for d in sorted(range(-max_days, max_days + 1), key=lambda d: (abs(d), d)):
            if d == 0:
                continue
            aa, bb = a + d * spd, b + d * spd
            if aa < 0 or bb + H > self.r.n or self.dtype[aa // spd] != want:
                continue
            if ((s_i <= bb + H) & (e_i >= aa - L)).any():
                continue
            if np.asarray(self.r.valid[pos, aa:bb + H]).mean() < 0.8:
                continue
            out.append(d)
            if len(out) == k:
                break
        return out

    def _robustness(self, e, sd, pos, part):
        r = self.r
        first = r.off + L - 1
        s0, s1 = r.parse_time(sd["start"]), r.parse_time(sd["end"])
        a = max(s0 - L, first)                                       # issue from one hour before the sensor's anomaly
        b = s1 + 1
        if a >= b:
            return dict(event=e["id"], sensor=sd["sensor"], partition=part, available=False,
                        reason="The model's history A inputs begin on 2023-01-01; this event has observations only.")
        stride = max(1, -(-(b - a) // MAX_ISSUES))
        issues = np.arange(a, b, stride)
        in_event = lambda tg: (tg >= s0) & (tg <= s1)
        pred, usable = self._forecast_block(part, pos, issues)
        err, act, scope = self._errors(pos, issues, pred, usable, in_event)
        ev = self._metrics(err, act, scope)
        days = self._reference_days(pos, s0, s1 + 1)
        ref_err, ref_act, ref_scope = [], [], []
        for d in days:
            iss = issues + d * r.spd
            p2, u2 = self._forecast_block(part, pos, iss)
            e2, a2, s2 = self._errors(pos, iss, p2, u2, lambda tg, d=d: in_event(tg - d * r.spd))
            ref_err.append(e2)
            ref_act.append(a2)
            ref_scope.append(s2)
        ref = self._metrics(np.concatenate(ref_err), np.concatenate(ref_act), np.concatenate(ref_scope)) if days else dict(n=0, mae=None, wape_pct=None,
                                                                                                    mae_by_horizon=[None] * H, missing_targets=0)
        diff = (_r(ev["mae"] - ref["mae"]) if ev["mae"] is not None and ref["mae"] is not None else None)
        warnings = []
        if ev["n"] < 24:
            warnings.append(f"Only {ev['n']} forecast-target pairs fall inside the anomaly; treat the numbers as anecdotal.")
        if not days:
            warnings.append("No comparable reference day (same day type, same sensor, no catalog event, >= 80% valid "
                            "readings) was found within 4 weeks.")
        if e["category"] == "sensor_quality":
            warnings.append("This is a sensor-quality anomaly: the 'actual' readings may themselves be faulty.")
        return dict(event=e["id"], sensor=sd["sensor"], partition=part, available=True, model=r.models[part]["provenance"],
                    issues=dict(first=r.ts(issues[0]), last=r.ts(issues[-1]), count=int(len(issues)), stride_minutes=int(stride * 5),
                                usable=int(usable.sum())),
                    anomaly_interval=dict(start=sd["start"], end=sd["end"]),
                    event_window=ev, reference=dict(ref, days=[r.ts(s0 + d * r.spd)[:10] for d in days]), mae_difference=diff,
                    periods=sorted({r.period(int(i)) for i in issues}),
                    method="Forecasts are issued every few minutes from one hour before this sensor's anomaly to its end, "
                           "each from the 12 readings up to its issue time (strictly causal; the same inputs as the replay). "
                           "Only forecast steps whose target time lies inside the anomaly interval and has a valid reading are "
                           "scored. The reference repeats the same issue times on nearby days of the same day type on which "
                           "this sensor has no catalog event.",
                    caution="Exploratory and retrospective. Not a test score, not comparable with the study's three-seed "
                            "aggregate test WAPE, and not evidence that the event caused any error difference.",
                    warnings=warnings, units=UNITS)

    # ------------------------------------------------------------------ offline summary
    def robustness_summary(self, part="geo", max_issues=24, log=print):
        """All forecastable, non-quality events, at their peak sensor: pooled errors inside anomalies vs matched reference
        days. Bounded: <= max_issues issuances per event."""
        r = self.r
        first = r.off + L - 1
        acc = {}
        t0 = time.perf_counter()
        n_ev = 0
        for e in self.events:
            if e["category"] == "sensor_quality":
                continue
            sd = next(s for s in e["sensor_details"] if s["sensor"] == e["peak_sensor"])
            pos = r.sensor_pos(sd["sensor"])
            s0, s1 = r.parse_time(sd["start"]), r.parse_time(sd["end"])
            a, b = max(s0 - L, first), s1 + 1
            if a >= b:
                continue
            issues = np.arange(a, b, max(1, -(-(b - a) // max_issues)))
            in_event = lambda tg: (tg >= s0) & (tg <= s1)
            pred, usable = self._forecast_block(part, pos, issues)
            err, act, _ = self._errors(pos, issues, pred, usable, in_event)
            days = self._reference_days(pos, s0, s1 + 1, k=2)
            if not days:
                continue
            refs = []
            for d in days:
                iss = issues + d * r.spd
                p2, u2 = self._forecast_block(part, pos, iss)
                refs.append(self._errors(pos, iss, p2, u2, lambda tg, d=d: in_event(tg - d * r.spd))[:2])
            n_ev += 1
            for key in (("all", r.period(s0)), (e["category"], r.period(s0)), (e["direction"], r.period(s0))):
                g = acc.setdefault(key, dict(events=0, e_abs=0.0, e_act=0.0, e_n=0, r_abs=0.0, r_act=0.0, r_n=0))
                g["events"] += 1
                ok = np.isfinite(err)
                g["e_abs"] += float(np.abs(err[ok]).sum()); g["e_act"] += float(act[ok].sum()); g["e_n"] += int(ok.sum())
                for e2, a2 in refs:
                    ok2 = np.isfinite(e2)
                    g["r_abs"] += float(np.abs(e2[ok2]).sum()); g["r_act"] += float(a2[ok2].sum())
                    g["r_n"] += int(ok2.sum())
        rows = []
        for (grp, period), g in sorted(acc.items()):
            rows.append(dict(group=grp, period=period, events=g["events"], event_pairs=g["e_n"],
                             event_mae=_r(g["e_abs"] / g["e_n"]) if g["e_n"] else None,
                             event_wape_pct=_r(100 * g["e_abs"] / g["e_act"]) if g["e_act"] > 0 else None,
                             reference_pairs=g["r_n"],
                             reference_mae=_r(g["r_abs"] / g["r_n"]) if g["r_n"] else None,
                             reference_wape_pct=_r(100 * g["r_abs"] / g["r_act"]) if g["r_act"] > 0 else None))
        log(f"{n_ev} events evaluated in {time.perf_counter() - t0:.1f}s")
        return dict(created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), catalog_events_sha256=self.cat["events_sha256"],
                    partition=part, model=r.models[part]["provenance"]["model_id"], max_issues_per_event=max_issues,
                    reference_days_per_event=2, events_evaluated=n_ev, rows=rows,
                    method="For every non-quality candidate event from 2023-01-01 on, at its peak sensor: up to "
                           f"{max_issues} forecasts issued from one hour before the anomaly to its end (causal inputs), "
                           "scored only on valid targets inside the anomaly; the reference repeats the same issue times on "
                           "the two nearest same-day-type days without a catalog event at that sensor. WAPE is pooled "
                           "(sum of absolute errors / sum of observed flow).",
                    caution="Exploratory, retrospective and not a test score. Events were selected by unusual observed flow, "
                            "so larger errors are expected by construction; the difference is not evidence of causation and is "
                            "not comparable with the study's frozen three-seed test WAPE.")


def main(argv=None):
    import argparse
    from replay import Replay
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["robustness"])
    p.add_argument("--partition", default="geo", choices=["geo", "iid"])
    a = p.parse_args(argv)
    svc = Disruptions(Replay())
    out = svc.robustness_summary(a.partition)
    path = D.OUT / D.VERSION / "robustness_summary.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(path)
    print(path)


if __name__ == "__main__":
    main()
