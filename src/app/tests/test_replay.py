"""Historical Traffic Replay (Phase 9A): data indexing, checkpoint provenance, inference equivalence with the frozen
evaluation path, forecast alignment, no future leakage, and the HTTP API next to the unchanged original routes."""

import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "backend"))
DATA = ROOT / "data" / "processed" / "aiori2_3yr"
STAGE3 = ROOT / "results" / "aiori2" / "phase8c_stage3"
pytestmark = pytest.mark.skipif(not (DATA / "extracted" / "flow_raw.npy").exists() or not (STAGE3 / "analysis" / "final_manifest.json").exists(),
                                reason="local AIORI 2.0 data or Stage 3 checkpoints not present")


@pytest.fixture(scope="module")
def R():
    from replay import Replay
    return Replay()


def err(fn):
    from replay import ReplayError
    with pytest.raises(ReplayError) as e:
        fn()
    return e.value.status, e.value.message


# ---------------------------------------------------------------- historical data and time indexing

def test_timeline_and_boundaries(R):
    m = R.meta()
    assert (m["first_timestamp"], m["last_timestamp"]) == ("2022-01-01T00:00", "2024-12-31T23:55")
    assert len(m["dates"]) == 1096 and m["dates"][0]["date"] == "2022-01-01" and m["dates"][-1]["date"] == "2024-12-31"
    assert m["first_forecast_time"] == "2023-01-01T00:55"
    assert R.ts(R.parse_time("2024-07-01T00:00")) == "2024-07-01T00:00"          # no silent shift
    assert R.period(R.parse_time("2024-07-01T00:00")) == "test" and R.period(R.parse_time("2024-06-30T23:55")) == "val"
    assert R.period(R.parse_time("2023-05-01T00:00")) == "train" and R.period(R.parse_time("2022-05-01T00:00")) == "before_training"


@pytest.mark.parametrize("bad,status", [("2024-02-30", 400), ("2021-12-31", 404), ("2025-01-01", 404), ("24-01-01", 400), ("", 400)])
def test_invalid_dates(R, bad, status):
    assert err(lambda: R.day(bad))[0] == status


@pytest.mark.parametrize("bad,status", [("2024-01-01T00:03", 400), ("2024-01-01 00:00", 400), ("2025-01-01T00:00", 404),
                                        ("2024-01-01T24:00", 400), (None, 400)])
def test_invalid_times(R, bad, status):
    assert err(lambda: R.frame(bad))[0] == status


def test_day_counts_match_mask_and_dst_gap(R):
    d = R.day("2024-03-10")                                                       # spring-forward day
    a = R.parse_time("2024-03-10T00:00")
    assert d["valid_sensors"] == np.asarray(R.valid[:, a:a + 288]).sum(0).tolist()
    assert d["dst_nonexistent"] == list(range(24, 36))
    assert all(d["valid_sensors"][k] == 0 for k in range(24, 36))                 # the missing hour has no readings
    assert d["available_slots"] == 276


def test_missing_observations_are_null_not_zero(R):
    i = R.parse_time("2024-08-14T08:00")
    f = R.frame("2024-08-14T08:00")
    v, ok = np.asarray(R.flow[:, i], float), np.asarray(R.valid[:, i])
    for k in range(200):
        if ok[k]:
            assert f["values"][k] == round(float(v[k]), 2) and f["status"][k] == "ok"
        else:
            assert f["values"][k] is None and f["status"][k] in ("missing", "invalid")
    assert f["valid_sensors"] == int(ok.sum())
    assert "congestion" not in json.dumps(f)                                     # no fabricated congestion classes


def test_sensor_series_is_raw_flow(R):
    s = R.series("BA-05", "2024-08-14")
    pos, a = R.by_id["BA-05"], R.parse_time("2024-08-14T00:00")
    raw = np.asarray(R.flow[pos, a:a + 288], float)
    ok = np.asarray(R.valid[pos, a:a + 288])
    assert len(s["values"]) == 288 and s["timestamps"][0] == "2024-08-14T00:00" and s["timestamps"][-1] == "2024-08-14T23:55"
    assert all((x is None) if not o else x == round(float(r), 2) for x, r, o in zip(s["values"], raw, ok))
    assert s["units"] == "vehicles per 5-minute interval"


# ---------------------------------------------------------------- sensors and coordinates

def test_sensor_ids_and_coordinates(R):
    sens = pd.read_csv(DATA / "metadata" / "sensors.csv")
    meta = pd.read_csv(ROOT / "sensor_meta_feature.csv", sep="\t").set_index("station_id")
    ids = [s["id"] for s in R.sensors]
    assert len(set(ids)) == 200 and ids[0] == "LA-01" and ids[50] == "BA-01" and ids[199] == "IC-50"
    for s, (_, row) in zip(R.sensors, sens.iterrows()):
        m = meta.loc[row.station_id]
        assert s["client_geo"] == row.client_geo and s["client_iid"] == row.client_iid
        assert s["lat"] == round(float(m.Lat), 6) and s["lng"] == round(float(m.Lng), 6)   # metadata, never invented
        assert 32.4 <= s["lat"] <= 42.1 and -124.6 <= s["lng"] <= -114.0 and m.District == row.district
        assert s["location_status"] == "provisional"
        assert "station_id" not in s and "array_index" not in s


def test_ids_match_original_app_order():
    sel = pd.read_csv(ROOT / "data" / "processed" / "nine_month" / "sensor_selection.csv")
    sens = pd.read_csv(DATA / "metadata" / "sensors.csv")
    assert (sel["station_id"].to_numpy() == sens["station_id"].to_numpy()).all()
    assert (sel["proposed_client"].to_numpy() == sens["client_geo"].to_numpy()).all()


# ---------------------------------------------------------------- model provenance

def test_checkpoint_provenance(R):
    man = json.loads((STAGE3 / "analysis" / "final_manifest.json").read_text(encoding="utf-8"))
    for part, arm, scope in (("geo", "fedavg_geo", "client_geo"), ("iid", "fedavg_iid", "client_iid")):
        p = R.models[part]["provenance"]
        e = next(m for m in man["models"] if m["label"].startswith("winner:") and m["arm"] == arm and m["seed"] == 20261027)
        assert p["run_id"] == e["run_id"] and p["checkpoint_sha256"] == e["best_pt_sha256"] and p["scaler_scope"] == scope
        assert p["seed"] == 20261027 and p["checkpoint_epoch"] == e["best_epoch"]
        assert R.models[part]["net"].training is False
        assert not any(q.requires_grad for q in R.models[part]["net"].parameters())
    assert not np.allclose(R.models["geo"]["mean"], R.models["iid"]["mean"])          # partition-specific scalers


def test_scalers_fitted_on_training_period_only():
    sc = json.loads((DATA / "experiments" / "conservative" / "A" / "scalers.json").read_text(encoding="utf-8"))
    assert sc["train_dates"] == ["2023-01-01", "2024-01-01"]


def _copy_stage3(tmp):
    man = json.loads((STAGE3 / "analysis" / "final_manifest.json").read_text(encoding="utf-8"))
    (tmp / "analysis").mkdir(parents=True)
    for f in ("final_manifest.json", "final_manifest.sha256"):
        shutil.copy2(STAGE3 / "analysis" / f, tmp / "analysis" / f)
    for m in man["models"]:
        if m["label"].startswith("winner:") and m["seed"] == 20261027:
            (tmp / m["run_id"]).mkdir()
            for f in ("best.pt", "run.json", "config.json"):
                shutil.copy2(STAGE3 / m["run_id"] / f, tmp / m["run_id"] / f)
            yield tmp / m["run_id"]


def test_checkpoint_mismatch_is_rejected(tmp_path):
    from replay import Replay
    runs = list(_copy_stage3(tmp_path))
    Replay(stage3=tmp_path)                                                      # untouched copy loads
    b = runs[0] / "best.pt"
    data = bytearray(b.read_bytes())
    data[-10] ^= 0xFF
    b.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        Replay(stage3=tmp_path)


def test_tampered_manifest_is_rejected(tmp_path):
    from replay import Replay
    list(_copy_stage3(tmp_path))
    p = tmp_path / "analysis" / "final_manifest.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen SHA-256"):
        Replay(stage3=tmp_path)


# ---------------------------------------------------------------- inference

@pytest.mark.parametrize("part,arm", [("geo", "fedavg_geo"), ("iid", "fedavg_iid")])
def test_inference_equals_frozen_evaluation_path(R, part, arm):
    """Same predictions as aiori2.run.Runner (the code that produced the frozen validation/test numbers), on
    validation windows only."""
    from aiori2 import core
    from aiori2.data import ThreeYearWindows
    from aiori2.run import Runner, load_config
    d = STAGE3 / R.models[part]["provenance"]["run_id"]
    cfg = load_config(None, dict(core.load_json(d / "config.json"), device="cpu"))
    r = Runner(cfg, root=DATA, out_root=d.parent)
    from aiori2.models.factory import build_model
    r.models = {"global": build_model(cfg["model"], cfg["preset"], cfg["in_channels"])}   # Runner._setup minus the training loaders
    r.opts, r.fl = {}, None
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    r.models["global"].load_state_dict(ck["models"]["global"])
    r.owner = np.full(200, "global", object)
    ld = ThreeYearWindows(DATA, "conservative", "A", "val", R.models[part]["provenance"]["scaler_scope"])
    idx = np.sort(np.random.default_rng(7).choice(len(ld), 256, replace=False))
    X, Y, M, s = ld.get(idx)
    with torch.no_grad():
        ref = ld.inverse(r.predict(torch.as_tensor(X), s).numpy(), s)
    t0 = np.asarray(ld.index[idx])[:, 1].astype(int) + R.off
    got = np.stack([R._predict(part, np.array([p]), R._inputs(np.array([p]), t)[0])[0] for p, t in zip(s, t0)])
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-3)   # float32 batch-of-256 vs batch-of-1 rounding (~1e-4 vehicles)
    # API-level forecast for one of these windows: issued at t0 - 5 min, actuals equal the evaluation targets
    k = int(np.flatnonzero(M.all(1))[0])
    f = R.forecast(R.sensors[s[k]]["id"], R.ts(t0[k] - 1), part)
    np.testing.assert_allclose(f["forecast"]["values"], np.round(ref[k], 2), atol=0.011)
    np.testing.assert_allclose(f["actual"]["values"], np.round(Y[k], 2), atol=1e-6)


def test_forecast_contract_and_alignment(R):
    f = R.forecast("CS-07", "2024-02-06T17:30", "geo")
    assert f["issued_at"] == "2024-02-06T17:30"
    assert f["input"]["timestamps"][0] == "2024-02-06T16:35" and f["input"]["timestamps"][-1] == "2024-02-06T17:30"
    assert f["forecast"]["timestamps"][0] == "2024-02-06T17:35" and f["forecast"]["timestamps"][-1] == "2024-02-06T18:30"
    assert len(f["input"]["model_input"]) == 12 and len(f["forecast"]["values"]) == 12 and len(f["actual"]["values"]) == 12
    assert all(np.isfinite(v) for v in f["forecast"]["values"])
    pos, i = R.by_id["CS-07"], R.parse_time("2024-02-06T17:30")
    for k in range(12):
        ok = bool(R.valid[pos, i + 1 + k])
        assert (f["actual"]["values"][k] is not None) == ok
        if ok:
            assert f["actual"]["values"][k] == round(float(R.flow[pos, i + 1 + k]), 2)
            assert f["comparison"]["abs_error"][k] == pytest.approx(abs(f["forecast"]["values"][k] - f["actual"]["values"][k]), abs=0.011)
    assert f["model"]["model_id"] == "aiori2-mlp-fedavgm-A-geo-s20261027" and f["period"] == "val"


def test_no_future_leakage(R):
    """The forecast reads model inputs only from steps before the first target; future steps are read only as actuals."""

    class Spy:
        def __init__(self, a):
            self.a, self.cols = a, []

        def __getitem__(self, key):
            c = key[1]
            self.cols.append(c.stop - 1 if isinstance(c, slice) else int(np.max(c)))
            return self.a[key]

    from replay import Replay
    r = Replay.__new__(Replay)
    r.__dict__.update(R.__dict__)
    r.filled, r.imputed = Spy(R.filled), Spy(R.imputed)
    r.forecasts = type(R.forecasts)(4)
    i = R.parse_time("2024-09-03T07:00")
    f = r.forecast("LA-10", "2024-09-03T07:00", "iid")
    e_last = i - R.off                                                             # experiment index of time T itself
    assert max(r.filled.cols) == e_last and max(r.imputed.cols) == e_last
    # perturbing everything after T leaves the forecast unchanged
    r2 = Replay.__new__(Replay)
    r2.__dict__.update(R.__dict__)
    fut = np.array(R.filled[:, :e_last + 300])
    fut[:, e_last + 1:] = 9999.0
    r2.filled, r2.forecasts = fut, type(R.forecasts)(4)
    assert r2.forecast("LA-10", "2024-09-03T07:00", "iid")["forecast"]["values"] == f["forecast"]["values"]


def test_missing_future_actuals_and_dataset_end(R):
    f = R.forecast("LA-01", "2024-12-31T23:30", "geo")
    assert f["actual"]["status"][-6:] == ["beyond_dataset"] * 6 and f["forecast"]["timestamps"][-1] is None
    assert f["comparison"]["abs_error"][-6:] == [None] * 6 and f["comparison"]["n_compared"] <= 6
    # a window with an invalid target inside the test period
    a = R.parse_time("2024-07-01T00:00")
    v = np.asarray(R.valid[:, a:a + 288 * 60])
    pos, k = map(int, np.argwhere(~v[:, 3:] & v[:, :-3])[0])                   # target invalid, reading at T valid
    k += 3
    t = R.ts(a + k - 3)                                                         # T = k - 3, so k is forecast step 3
    f = R.forecast(R.sensors[pos]["id"], t, "geo")
    assert f["actual"]["values"][2] is None and f["actual"]["status"][2] in ("missing", "invalid") and f["comparison"]["abs_error"][2] is None


def test_fully_imputed_input_is_unavailable(R):
    imp = np.asarray(R.imputed[:, 288 * 190:288 * 400])
    run = np.lib.stride_tricks.sliding_window_view(imp, 12, axis=1).all(-1)
    s, e = map(int, np.argwhere(run)[0])
    t = R.ts(R.off + 288 * 190 + e + 11)                                          # T = last of 12 imputed inputs
    status, msg = err(lambda: R.forecast(R.sensors[s]["id"], t, "geo"))
    assert status == 422 and "entirely imputed" in msg
    fr = R.frame(t, "forecast", "geo", 12)
    assert fr["values"][s] is None and fr["status"][s] == "no_input"


def test_no_forecast_before_model_history(R):
    assert err(lambda: R.forecast("LA-01", "2022-06-01T08:00"))[0] == 422
    assert err(lambda: R.frame("2023-01-01T00:50", "forecast"))[0] == 422
    assert R.forecast("LA-01", "2023-01-01T00:55")["forecast"]["timestamps"][0] == "2023-01-01T01:00"


def test_forecast_frame_matches_single_forecasts(R):
    fr = R.frame("2024-04-02T08:00", "forecast", "iid", 6)
    assert fr["target_time"] == "2024-04-02T08:30"
    for sid in ("LA-03", "BA-20", "IC-44"):
        f = R.forecast(sid, "2024-04-02T08:00", "iid")
        assert fr["values"][R.by_id[sid]] == pytest.approx(f["forecast"]["values"][5], abs=0.011)


# ---------------------------------------------------------------- HTTP API alongside the original routes

@pytest.fixture(scope="module")
def api():
    import replay_server as RS
    srv = RS.make_server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_replay_endpoints(api):
    assert get(api, "/api/replay/meta")[0] == 200
    st, sens = get(api, "/api/replay/sensors")
    assert st == 200 and len(sens) == 200
    assert get(api, "/api/replay/day?date=2024-03-10")[1]["available_slots"] == 276
    st, fr = get(api, "/api/replay/frame?time=2024-08-14T08:00")
    assert st == 200 and len(fr["values"]) == 200
    st, fc = get(api, "/api/replay/forecast?sensor=LA-01&time=2024-08-14T08:00&partition=iid")
    assert st == 200 and fc["model"]["partition"] == "iid"
    for bad, code in (("/api/replay/frame?time=2024-08-14T08:00&x=1", 400), ("/api/replay/frame?time=a&time=b", 400),
                      ("/api/replay/forecast?sensor=../../etc&time=2024-08-14T08:00", 404), ("/api/replay/nope", 404),
                      ("/api/replay/frame?time=2024-08-14T08:00&horizon=13", 400), ("/api/replay/frame?time=2024-08-14T08:00&horizon=-1", 400)):
        s, b = get(api, bad)
        assert s == code and "error" in b
    text = json.dumps(get(api, "/api/replay/meta")[1]) + json.dumps(sens)
    assert "Desktop" not in text and "station_id" not in text and ".pt" not in text   # no paths or station ids


def test_original_routes_unchanged(api):
    assert get(api, "/api/health")[1]["status"] == "ok"
    st, orig = get(api, "/api/sensors")
    _, rep = get(api, "/api/replay/sensors")
    assert st == 200 and [s["id"] for s in orig] == [s["id"] for s in rep]
    assert [s["client"] for s in orig] == [s["client_geo"] for s in rep]
    st, p = get(api, "/api/predict", {"sensor": "LA-10", "time": "2023-09-12T07:30", "models": ["final"]})
    assert st == 200 and len(p["forecasts"]["final"]["values"]) == 12
