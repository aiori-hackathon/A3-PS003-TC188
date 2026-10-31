"""AIORI Traffic Intelligence API: real inference with the frozen Phase 9 GRU checkpoints (standard library only).

    .venv\\Scripts\\python.exe app/backend/server.py            # http://127.0.0.1:8000

Endpoints: GET /health, /models, /sensors, /metrics, /demo; POST /predict (also under /api/). The front end is served at /:
the React build (app/web/dist) when it exists, otherwise the original static front end (app/frontend).
Configuration (environment variables): AIORI_HOST, AIORI_PORT, AIORI_DATA_DIR, AIORI_RUNS_DIR, AIORI_FINAL_DIR, AIORI_DEVICE,
AIORI_FRONTEND_DIR (folder to serve at /), AIORI_CORS_ORIGINS (comma-separated origins allowed to call the API, or *).

Models (selected on validation only, before any app work): the final model is the local-only GRU (validation-selected
arm) seed 1, the local-only seed with the lowest validation MAE; each sensor is forecast by its own client's model and
negative predictions are clipped at 0 (the validation-selected policy). The comparison model is the centralised GRU
seed 0 (lowest centralised validation MAE), unclipped. Historical requests use the frozen preprocessing exactly
(aiori.dataset batches); custom requests rebuild the same four input features from 12 supplied observations.
Public responses never contain station IDs, array rows or file paths; every observation is labelled historical.
"""

import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aiori import dataset, evaluation  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402

DATA = Path(os.environ.get("AIORI_DATA_DIR", ROOT / "data" / "processed" / "nine_month"))
RUNS = Path(os.environ.get("AIORI_RUNS_DIR", ROOT / "results" / "phase9" / "runs"))
FINAL = Path(os.environ.get("AIORI_FINAL_DIR", ROOT / "results" / "phase9" / "final"))
FRONTEND = Path(os.environ.get("AIORI_FRONTEND_DIR") or next(
    (d for d in (ROOT / "app" / "web" / "dist", ROOT / "app" / "frontend") if (d / "index.html").is_file()), ROOT / "app" / "frontend"))
CORS = [o.strip() for o in os.environ.get("AIORI_CORS_ORIGINS", "").split(",") if o.strip()]
PCT = ROOT / "app" / "data" / "percentage_error.json"
OPT = ROOT / "results" / "accuracy_optimization"   # accuracy-optimisation artifacts (aggregate metrics only are served)
ORIGIN = datetime(2023, 1, 1)
STEP = timedelta(minutes=5)
H = 12
CLIENTS = {"client_0_la_ventura": ("LA", "LA / Ventura"), "client_1_bay_area": ("BA", "Bay Area"),
           "client_2_coastal_south_central": ("CS", "Coastal / South Central"), "client_3_inland_central": ("IC", "Inland / Central")}
ARM_LABEL = {"local": "Local-only GRU", "centralised": "Centralised GRU", "fedavg_iid": "IID FedAvg", "fedavg_geo": "Geographic FedAvg"}
UNITS = "vehicles per 5-minute interval"
STUDY_END = "2023-09-30T23:55"  # last held-out (test) interval; nothing later was ever observed or evaluated
MAX_FLOW = 5000.0


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def ts(step):
    return (ORIGIN + STEP * int(step)).strftime("%Y-%m-%dT%H:%M")


def parse_time(s):
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:00)?", s):
        raise ApiError(400, "time must look like 2023-09-12T17:00 (local clock time, 5-minute steps)")
    try:
        t = datetime.strptime(s[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        raise ApiError(400, "time must look like 2023-09-12T17:00 (local clock time, 5-minute steps)")
    if t.minute % 5:
        raise ApiError(400, "time must fall on a 5-minute boundary")
    if not 2023 <= t.year <= 2099:
        raise ApiError(400, "time must fall between 2023 and 2099")
    return int((t - ORIGIN) / STEP)


class Engine:
    """Loads the frozen checkpoint data, the deployed models and the aggregate metrics once."""

    def __init__(self):
        torch.set_num_threads(int(os.environ.get("AIORI_THREADS", "2")))
        dev = os.environ.get("AIORI_DEVICE", "cpu")
        self.device = torch.device("cuda" if dev == "cuda" and torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.run = dataset.load_run("nine_month", run_dir=DATA, selection=DATA / "sensor_selection.csv", splits=("val", "test"))
        man = pd.read_csv(DATA / "sensor_selection.csv", usecols=["proposed_client", "type_group"])
        self.split_cfg = json.loads((DATA / "split_config.json").read_text(encoding="utf-8"))
        self.ds = {s: dataset.WindowDataset(self.run, s, calendar=True) for s in ("val", "test")}
        self.keys = {s: d.index[:, 0].astype(np.int64) * 10_000_000 + d.index[:, 1] for s, d in self.ds.items()}
        self.sensors, self.by_id = [], {}
        counters = {}
        for pos, (client, tg) in enumerate(zip(man["proposed_client"], man["type_group"])):
            code, name = CLIENTS[client]
            counters[code] = counters.get(code, 0) + 1
            sid = f"{code}-{counters[code]:02d}"
            flow = self.run.flow_raw[pos, self.split_cfg["boundaries"]["test"][0]:]
            med = float(np.nanmedian(flow)) if np.isfinite(flow).any() else None  # null, never a bare NaN token, in JSON
            s = dict(id=sid, client=client, client_name=name, type_group=str(tg), median_test_flow=None if med is None else round(med, 1))
            self.sensors.append(s)
            self.by_id[sid] = pos
        self.models = {"final": self._load("local", 1, clip=True), "centralised": self._load("centralised", 0, clip=False)}
        self.candidate = None
        if os.environ.get("AIORI_CANDIDATE") == "1":          # opt-in experimental model (app/backend/candidate.py)
            from candidate import Candidate
            sc = json.loads((DATA / "scaler_parameters.json").read_text(encoding="utf-8"))
            self.candidate = Candidate(self.run, np.asarray(sc["mean"]), np.asarray(sc["std"]), self.device)
        self.analysis = json.loads((FINAL / "analysis.json").read_text(encoding="utf-8"))
        self.frozen = json.loads((FINAL / "frozen.json").read_text(encoding="utf-8"))
        self.pct = json.loads(PCT.read_text(encoding="utf-8")) if PCT.exists() else None
        self.started = time.time()

    def _load(self, arm, seed, clip):
        d = RUNS / f"{arm}_seed{seed}"
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        files = sorted(d.glob("best_model*.pt"))
        if not files:
            raise FileNotFoundError(f"No best_model*.pt checkpoints in {d}")
        nets, sha = {}, {}
        for f in files:
            key = "all" if f.name == "best_model.pt" else f.stem.replace("best_model_", "")
            net = ForecastGRU(4, cfg["hidden_size"])
            net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
            nets[key] = net.to(self.device).eval()
            sha[key] = hashlib.sha256(f.read_bytes()).hexdigest()
        if "all" not in nets and set(CLIENTS) - set(nets):
            raise FileNotFoundError(f"{d} lacks checkpoints for {sorted(set(CLIENTS) - set(nets))}")
        val = json.loads((d / "COMPLETE.json").read_text(encoding="utf-8"))
        test = json.loads((FINAL / "test" / f"{arm}_seed{seed}" / "metrics_test.json").read_text(encoding="utf-8"))
        return dict(arm=arm, seed=seed, clip=clip, nets=nets, label=ARM_LABEL[arm], config=cfg, meta=meta,
                    provenance=dict(arm=ARM_LABEL[arm], seed=seed, architecture=f"GRU(4 inputs, {cfg['hidden_size']} hidden) + linear, 12 -> 12",
                                    selection=("local-only seed with the lowest validation MAE; the local-only arm was the pre-declared final "
                                               "model (lowest validation MAE, fixed before the held-out test)") if arm == "local" else
                                    "centralised seed with the lowest validation MAE (comparison model)",
                                    routing="one model for all sensors" if "all" in nets else "each sensor uses its own client's model",
                                    negative_clipping=clip, validation_mae=val["val_mae"], validation_rmse=val["val_rmse"],
                                    test_mae_this_seed=test["pooled"]["mae"], test_rmse_this_seed=test["pooled"]["rmse"],
                                    checkpoint_sha256={k: v[:16] for k, v in sha.items()}, units=UNITS))

    # ---------- inference ----------
    def _net(self, model, pos):
        nets = model["nets"]
        return nets["all"] if "all" in nets else nets[self.sensors[pos]["client"]]

    def _forecast(self, model, pos, x):
        with self.lock, torch.no_grad():
            z = self._net(model, pos)(x.to(self.device)).cpu().numpy()
        raw = evaluation.to_raw(z, self.run, np.array([pos]))[0].astype(np.float64)
        return np.maximum(raw, 0.0) if model["clip"] else raw, z[0]

    def historical(self, pos, t0):
        for split, keys in self.keys.items():
            row = np.searchsorted(keys, pos * 10_000_000 + t0)
            if row < len(keys) and keys[row] == pos * 10_000_000 + t0:
                return split, self.ds[split].get_batch([row])["x"]
        b = self.split_cfg["boundaries"]
        raise ApiError(404, f"No complete historical window starts at {ts(t0)} for this sensor. Choose a time between "
                            f"{ts(b['val'][0] + 12)} and {ts(b['test'][1] - 12)} (validation and held-out periods).")

    def custom_features(self, pos, obs, t0):
        """Same four features as training: scaled causal-filled flow, observed flag, time-of-day sin/cos."""
        sc = self.run.scaler
        mean, std = float(sc["mean"][pos]), float(sc["std"][pos])
        vals, flag, last, age = [], [], None, 0
        spd = self.run.steps_per_day
        for i, v in enumerate(obs):
            step = t0 - H + i
            if v is not None:
                last, age = v, 0
                vals.append(v)
                flag.append(1.0)
                continue
            age += 1
            if last is not None and age <= 6:
                vals.append(last)                                    # forward fill <= 6 steps (30 min)
            else:
                med = self.run.tod_median[pos, step % spd]
                vals.append(float(med) if np.isfinite(med) else mean)  # training time-of-day median, then training mean
            flag.append(0.0)
        z = (np.array(vals) - mean) / std
        cal = dataset.calendar_features(np.array([t0]), steps_per_day=spd)[0]
        x = np.concatenate([np.stack([z, flag], -1), cal], -1).astype(np.float32)
        return torch.from_numpy(x[None])

    def predict(self, body):
        if not isinstance(body, dict):
            raise ApiError(400, "Send a JSON object")
        sid = body.get("sensor")
        if not isinstance(sid, str) or sid not in self.by_id:
            raise ApiError(404, "Unknown sensor. Use an id from GET /sensors, for example LA-01.")
        pos = self.by_id[sid]
        wanted = body.get("models", ["final", "centralised"])
        allowed = sorted(self.models) + (["candidate"] if self.candidate else [])
        if not isinstance(wanted, list) or not wanted or any(m not in allowed for m in wanted):
            raise ApiError(400, f"models must be a non-empty list drawn from {allowed}")
        if "candidate" in wanted and body.get("observations") is not None:
            raise ApiError(400, "the candidate model needs one week of history, so it serves historical requests only")
        obs = body.get("observations")
        if obs is None:
            t0 = parse_time(body.get("time"))
            split, x = self.historical(pos, t0)
            hist = self.run.flow_raw[pos, t0 - H:t0].astype(float)
            actual = self.run.flow_raw[pos, t0:t0 + H].astype(float)
            mode = "historical"
        else:
            if not isinstance(obs, list) or len(obs) != H:
                raise ApiError(400, "observations must be a list of exactly 12 values (oldest first); use null for a missing reading")
            clean = []
            for v in obs:
                if v is None:
                    clean.append(None)
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= MAX_FLOW:  # also rejects NaN and huge ints
                    clean.append(float(v))
                else:
                    raise ApiError(400, f"each observation must be null or a number between 0 and {MAX_FLOW:g}")
            if all(v is None for v in clean):
                raise ApiError(400, "at least one observation is required")
            t0 = parse_time(body.get("time"))
            if t0 - H < 0:
                raise ApiError(400, "time must be at least one hour after 2023-01-01T00:00")
            x = self.custom_features(pos, clean, t0)
            hist, actual, split, mode = np.array([np.nan if v is None else v for v in clean]), None, None, "custom"
        t = time.perf_counter()
        forecasts = {}
        for m in wanted:
            if m == "candidate":
                with self.lock:
                    values = self.candidate.forecast(pos, t0).tolist()
                forecasts[m] = dict(label="Optimized MLP (1-day context, not federated)",
                                    values=[round(v, 2) for v in values], negative_clipping=True)
                continue
            raw, _ = self._forecast(self.models[m], pos, x)
            forecasts[m] = dict(label=self.models[m]["label"], values=[round(v, 2) for v in raw.tolist()],
                                negative_clipping=self.models[m]["clip"])
        ms = (time.perf_counter() - t) * 1000
        s = self.sensors[pos]
        beyond = ts(t0 + H - 1) > STUDY_END
        if mode == "historical" and "candidate" in wanted:
            note = ("Optimized MLP: validation WAPE 6.794%; independent October-December 2023 WAPE 7.669% (original GRU 8.834% on the same windows)."
                    + (" This window lies in the validation period used to select it." if split == "val" else
                       " It was not evaluated on this Aug-Sep window; the original GRU's measured WAPE here is 8.672%."))
        elif mode == "historical":
            note = ("Held-out test period: the original GRU's measured test WAPE over all 200 sensors is 8.672%." if split == "test"
                    else "Validation period: used to choose models and policies, not as an independent test.")
        elif beyond:
            note = ("Future forecast beyond the study period (after September 30, 2023). Its accuracy has not been validated: "
                    "the model was trained on January to July 2023 data and no later observations were evaluated.")
        else:
            note = "Forecast from user-supplied observations. Accuracy on user-supplied readings is not measured."
        return dict(sensor=dict(id=sid, client_name=s["client_name"], type_group=s["type_group"]), mode=mode,
                    data_label="historical XTraffic 2023 observations (not live)" if mode == "historical" else "user-supplied observations",
                    period=dict(val="validation period (Jul 11 - Aug 7, 2023)", test="held-out test period (Aug 8 - Sep 30, 2023)").get(split),
                    units=UNITS, forecast_start=ts(t0),
                    history=dict(timestamps=[ts(t0 - H + i) for i in range(H)],
                                 values=[None if not np.isfinite(v) else round(float(v), 2) for v in hist]),
                    forecast_timestamps=[ts(t0 + i) for i in range(H)], forecasts=forecasts,
                    actual=None if actual is None else dict(label="observed afterwards (historical, for comparison only)",
                                                            values=[None if not np.isfinite(v) else round(float(v), 2) for v in actual]),
                    beyond_study_period=beyond, validated=mode == "historical" and split == "test", accuracy_note=note,
                    inference_ms=round(ms, 2), device=str(self.device))

    # ---------- read-only views ----------
    def health(self):
        return dict(status="ok", device=str(self.device), cuda_available=torch.cuda.is_available(), torch=torch.__version__,
                    models_loaded=sorted(self.models), sensors=len(self.sensors), uptime_s=round(time.time() - self.started),
                    percentage_error_available=self.pct is not None, candidate_enabled=self.candidate is not None)

    def models_view(self):
        out = [dict(key=k, label=m["label"], **m["provenance"]) for k, m in self.models.items()]
        if self.candidate:
            out.append(dict(key="candidate", label="Optimized MLP (1-day context, not federated)", **self.candidate.provenance()))
        return out

    def metrics(self):
        a = self.analysis
        arms = {k: {f: v[f] for f in ("seeds", "mae", "rmse", "normalised_mae", "sensor_macro_mae", "negative_rate", "clipped_mae",
                                      "clipped_rmse", "per_seed", "client", "horizon", "month", "weekday_weekend", "validation_mae",
                                      "validation_rmse", "validation_to_test_change", "improvement_over_baselines_pct")}
                for k, v in a["arms"].items()}
        for k in arms:
            arms[k]["label"] = ARM_LABEL[k]
        runtime = {t: dict(minutes=r["minutes"], communication_mb=r["communication"].get("train_bytes_total", 0) / 1e6,
                           communication_to_best_mb=r["communication"].get("train_bytes_to_best", 0) / 1e6,
                           best_unit=r["summary"].get("best_unit"), units_run=r["summary"].get("units_run"))
                   for t, r in a["runtime"].items()}
        return dict(source="frozen Phase 9 held-out test (Aug 8 - Sep 30, 2023), evaluated once; three-seed means unless stated",
                    test_windows=a["test_windows"], n_sensors=a["n_sensors"], units=UNITS, arms=arms,
                    baselines={k: {f: v[f] for f in ("mae", "rmse", "horizon", "client")} for k, v in a["baselines"].items()},
                    contrasts=a["contrasts"], sesoi_mae=a["sesoi_mae"], runtime=runtime,
                    final_model=dict(arm=ARM_LABEL[self.frozen["selected_final_arm"]], variant=self.frozen["selected_final_variant"]),
                    percentage_error=self.pct, optimized=self.optimized(), deployed={k: m["provenance"] for k, m in self.models.items()},
                    split=dict(train="2023-01-01 to 2023-07-10 (191 days)", validation="2023-07-11 to 2023-08-07 (28 days)",
                               test="2023-08-08 to 2023-09-30 (54 days)"))

    def optimized(self):
        """Verified accuracy-optimisation results (aggregates only), or None when the artifacts are absent."""
        try:
            man = json.loads((OPT / "candidate_v1" / "manifest.json").read_text(encoding="utf-8"))
            ind = json.loads((OPT / "untouched_oct_dec" / "results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        v, c, o = man["validation_base_alone"], ind["candidate"], ind["frozen_local_seed1_clipped"]
        pick = lambda r: dict(wape_pct=r["wape_pct"], mae=r["mae"], rmse=r["rmse"], smape_pct=r["smape_pct"],  # noqa: E731
                              by_client_wape_pct={k: x["wape_pct"] for k, x in r["by_client"].items()},
                              by_horizon_wape_pct=r["by_horizon_wape_pct"])
        return dict(label="Optimized MLP (1-day context, local-only)", architecture=man["base_model"]["architecture"],
                    training=man["base_model"]["training"], federated=False, enabled_for_forecasts=self.candidate is not None,
                    validation=pick(v), validation_period="2023-07-11 to 2023-08-07",
                    independent=dict(period="2023-10-01 to 2023-12-31", windows=ind["n_windows"], optimized=pick(c), original_gru=pick(o),
                                     optimized_by_month_wape_pct=c["by_month_wape_pct"], original_by_month_wape_pct=o["by_month_wape_pct"],
                                     relative_reduction_pct=100 * (o["wape_pct"] - c["wape_pct"]) / o["wape_pct"],
                                     persistence=pick(ind["persistence"]),
                                     relative_reduction_vs_persistence_pct=100 * (ind["persistence"]["wape_pct"] - c["wape_pct"]) / ind["persistence"]["wape_pct"]),
                    residual_correction=dict(validation_oof_wape_pct=man["validation_with_residual_oof"]["wape_pct"],
                                             independent_wape_pct=ind["candidate_plus_residual"]["wape_pct"], deployed=False),
                    target=dict(wape_target_pct=6.0, achieved=False))

    def demo(self):
        """Curated historical examples (fixed choices, not picked by forecast quality)."""
        picks = [("Weekday morning peak", "2023-09-12T07:30"), ("Weekday evening peak", "2023-08-17T17:00"),
                 ("Saturday midday", "2023-09-02T12:00"), ("Overnight low flow", "2023-08-23T02:00")]
        out = []
        for code in ("LA", "BA", "CS", "IC"):
            sid = next(s["id"] for s in self.sensors if s["id"].startswith(code) and s["type_group"] == "mainline")
            for label, t in picks:
                try:
                    self.historical(self.by_id[sid], parse_time(t))
                except ApiError:
                    continue
                out.append(dict(sensor=sid, client_name=self.sensors[self.by_id[sid]]["client_name"], time=t, label=label))
        return dict(note="All examples are historical windows from the held-out test period; nothing is live.", examples=out)


ENGINE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "AIORI/1.0"
    sys_version = ""      # no Python version in the Server header
    timeout = 10          # seconds; a client that stalls mid-request releases its thread

    def log_message(self, fmt, *args):  # quieter console; no request bodies are logged
        sys.stderr.write(f"{self.command} {urlparse(self.path).path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin and ("*" in CORS or origin in CORS):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(data)

    def _err(self, status, message):
        self._send(status, dict(error=dict(status=status, message=message)))

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            routes = {"/health": ENGINE.health, "/models": ENGINE.models_view, "/sensors": lambda: ENGINE.sensors,
                      "/metrics": ENGINE.metrics, "/demo": ENGINE.demo}
            if path in routes:
                return self._send(200, routes[path]())
            if path.startswith("/api/") and path[4:] in routes:
                return self._send(200, routes[path[4:]]())
            return self._static(path)
        except ApiError as e:
            return self._err(e.status, e.message)
        except Exception:  # noqa: BLE001 - never leak internals
            return self._err(500, "Internal error")

    def do_OPTIONS(self):  # CORS preflight, only for origins listed in AIORI_CORS_ORIGINS
        origin = self.headers.get("Origin")
        self.send_response(204 if origin and ("*" in CORS or origin in CORS) else 403)
        if origin and ("*" in CORS or origin in CORS):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        if path not in ("/predict", "/api/predict"):
            return self._err(404, "Not found")
        try:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > 20_000:
                raise ApiError(400, "Send a JSON body under 20 kB")
            try:
                body = json.loads(self.rfile.read(n))
            except ValueError:  # JSONDecodeError and UnicodeDecodeError
                raise ApiError(400, "Body is not valid JSON")
            return self._send(200, ENGINE.predict(body))
        except ApiError as e:
            return self._err(e.status, e.message)
        except TimeoutError:
            self.close_connection = True
            return self._err(408, "Request body not received in time")
        except Exception:  # noqa: BLE001
            return self._err(500, "Internal error")

    def _static(self, path):
        rel = "index.html" if path == "/" else "favicon.svg" if path == "/favicon.ico" else path.lstrip("/")
        f = (FRONTEND / rel).resolve()
        if FRONTEND.resolve() not in f.parents or not f.is_file():
            # client-side routes (extensionless, outside /api) get the app shell; missing files stay 404
            if path == "/api" or path.startswith("/api/") or "." in path.rsplit("/", 1)[-1]:
                return self._err(404, "Not found")
            f = FRONTEND / "index.html"
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return self._send(200, f.read_bytes(), ctype)


def make_server(host=None, port=None):
    global ENGINE
    ENGINE = ENGINE or Engine()
    return ThreadingHTTPServer((host or os.environ.get("AIORI_HOST", "127.0.0.1"), int(port if port is not None else os.environ.get("AIORI_PORT", "8000"))), Handler)


def main():
    srv = make_server()
    print(f"AIORI Traffic Intelligence on http://{srv.server_address[0]}:{srv.server_address[1]} (device {ENGINE.device})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
