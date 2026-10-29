"""AIORI app backend: forecasts equal the frozen Phase 9 predictions; API validation and privacy."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "backend"))
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "nine_month" / "pilot_preprocessed.npz").exists(),
                                reason="local nine-month checkpoint not present")


@pytest.fixture(scope="module")
def api():
    import server as S
    srv = S.make_server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield S, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _frozen(tag):
    with np.load(ROOT / "results" / "phase9" / "final" / "test" / tag / "predictions_test.npz") as z:
        return z["sensor_pos"], z["t0"], z["y_hat_z"]


@pytest.mark.parametrize("model,tag", [("final", "local_seed1"), ("centralised", "centralised_seed0")])
def test_forecasts_equal_frozen_test_predictions(api, model, tag):
    S, _ = api
    e = S.ENGINE
    sp, t0, yz = _frozen(tag)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(t0), 40, replace=False):
        _, x = e.historical(int(sp[i]), int(t0[i]))
        raw, z = e._forecast(e.models[model], int(sp[i]), x)
        assert np.allclose(z, yz[i], atol=1e-5)
        if e.models[model]["clip"]:
            assert (raw >= 0).all()


def test_custom_observations_reproduce_historical_features(api):
    S, _ = api
    e = S.ENGINE
    sp, t0, _ = _frozen("local_seed1")
    for i in range(0, len(t0), 400_000):
        pos, t = int(sp[i]), int(t0[i])
        hist = e.run.flow_raw[pos, t - 12:t]
        if not np.isfinite(hist).all():
            continue
        _, xh = e.historical(pos, t)
        xc = e.custom_features(pos, [float(v) for v in hist], t)
        assert np.allclose(xh.numpy(), xc.numpy(), atol=1e-5)


def test_endpoints_and_privacy(api):
    _, base = api
    for path in ("/health", "/models", "/sensors", "/metrics", "/demo", "/api/metrics"):
        status, body = call(base, path)
        assert status == 200
        text = json.dumps(body)
        assert "station_id" not in text and "array_index" not in text and "by_sensor_pos" not in text
        assert ":\\\\" not in text and "results/phase9" not in text and "Users" not in text
    _, sensors = call(base, "/sensors")
    assert len(sensors) == 200 and set(sensors[0]) == {"id", "client", "client_name", "type_group", "median_test_flow"}


def test_predict_historical_and_custom(api):
    _, base = api
    status, r = call(base, "/predict", {"sensor": "LA-10", "time": "2023-09-12T07:30"})
    assert status == 200 and r["mode"] == "historical" and "not live" in r["data_label"]
    assert len(r["forecast_timestamps"]) == 12 and len(r["forecasts"]["final"]["values"]) == 12
    assert r["forecast_timestamps"][0] == "2023-09-12T07:30" and r["history"]["timestamps"][-1] == "2023-09-12T07:25"
    assert min(r["forecasts"]["final"]["values"]) >= 0
    status, c = call(base, "/predict", {"sensor": "LA-10", "time": "2023-09-12T07:30", "observations": [100, None, 110, 120, None, None, 130, 125, 128, 131, 135, 140]})
    assert status == 200 and c["mode"] == "custom" and c["actual"] is None
    _, demo = call(base, "/demo")
    for ex in demo["examples"]:
        assert call(base, "/predict", {"sensor": ex["sensor"], "time": ex["time"]})[0] == 200


@pytest.mark.parametrize("body,status", [
    ({"sensor": "XX-99", "time": "2023-09-12T07:30"}, 404),
    ({"sensor": "LA-10", "time": "yesterday"}, 400),
    ({"sensor": "LA-10", "time": "2023-09-12T07:31"}, 400),
    ({"sensor": "LA-10", "time": "2023-03-01T08:00"}, 404),
    ({"sensor": "LA-10", "time": "2023-09-12T07:30", "models": ["gpt"]}, 400),
    ({"sensor": "LA-10", "time": "2023-09-12T07:30", "observations": [1, 2, 3]}, 400),
    ({"sensor": "LA-10", "time": "2023-09-12T07:30", "observations": [None] * 12}, 400),
    ({"sensor": "LA-10", "time": "2023-09-12T07:30", "observations": [-5] + [1] * 11}, 400),
    ({"sensor": "LA-10", "time": "2023-09-12T07:30", "observations": ["a"] * 12}, 400),
    ([1, 2], 400)])
def test_invalid_requests(api, body, status):
    _, base = api
    s, r = call(base, "/predict", body)
    assert s == status and r["error"]["message"]


def test_bad_json_unknown_route_and_traversal(api):
    _, base = api
    assert call(base, "/predict", raw=b"{not json")[0] == 400
    assert call(base, "/nope")[0] == 404
    assert call(base, "/../../data/processed/nine_month/sensor_selection.csv")[0] == 404


def _page(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        assert r.status == 200
        return r.read().decode("utf-8")


def test_frontend_is_served_with_first_party_scripts_only(api):
    import re
    S, base = api
    index = _page(base, "/")
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', index)
    styles = re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"', index)
    assert scripts and all(not u.startswith(("http:", "https:", "//")) for u in scripts + styles)
    bundle = "".join(_page(base, "/" + u.lstrip("/")) for u in scripts)
    assert "/predict" in bundle or "predict" in bundle
    for route in sorted(S.SPA_ROUTES):  # client-side routes return the app shell, unknown paths stay 404
        assert _page(base, route) == index


def test_frontend_source_invents_no_data():
    """Our own front-end code (not third-party libraries) has no random values, stand-in data or leftover notes."""
    files = [p for p in (ROOT / "app" / "web" / "src").rglob("*") if p.suffix in (".ts", ".tsx", ".css") and ".test." not in p.name]
    files += [ROOT / "app" / "frontend" / n for n in ("index.html", "app.js", "styles.css")]
    assert len(files) > 10
    for f in files:
        text = f.read_text(encoding="utf-8").lower()
        for banned in ("lorem", "todo", "placeholder data", "math.random", "fake", "mock data", "dummy"):
            assert banned not in text, f"{banned} in {f.name}"


def test_future_forecast_is_labelled_unvalidated(api):
    _, base = api
    obs = [210, 220, None, 236, 241, 250, 262, 270, None, 281, 290, 300]
    s, r = call(base, "/api/predict", {"sensor": "BA-03", "time": "2024-03-05T08:00", "observations": obs})
    assert s == 200 and r["mode"] == "custom" and r["actual"] is None
    assert r["beyond_study_period"] is True and r["validated"] is False and "not been validated" in r["accuracy_note"]
    assert r["forecast_timestamps"][0] == "2024-03-05T08:00" and r["forecast_timestamps"][-1] == "2024-03-05T08:55"
    assert r["history"]["timestamps"][0] == "2024-03-05T07:00" and r["history"]["values"][2] is None
    assert min(r["forecasts"]["final"]["values"]) >= 0
    s, edge = call(base, "/predict", {"sensor": "BA-03", "time": "2023-09-30T23:30", "observations": obs})
    assert s == 200 and edge["beyond_study_period"] is True
    s, inside = call(base, "/predict", {"sensor": "BA-03", "time": "2023-09-12T08:00", "observations": obs})
    assert s == 200 and inside["beyond_study_period"] is False and inside["validated"] is False
    s, hist = call(base, "/predict", {"sensor": "LA-10", "time": "2023-09-12T07:30"})
    assert hist["validated"] is True and hist["beyond_study_period"] is False and "8.672%" in hist["accuracy_note"]
    assert call(base, "/predict", {"sensor": "BA-03", "time": "2100-01-01T00:00", "observations": obs})[0] == 400


def test_custom_forecast_is_time_of_day_only(api):
    """The model has no date input beyond time of day, so the same readings at the same clock time give the same forecast."""
    _, base = api
    obs = [120.0] * 12
    a = call(base, "/predict", {"sensor": "IC-02", "time": "2023-09-12T17:00", "observations": obs})[1]
    b = call(base, "/predict", {"sensor": "IC-02", "time": "2025-06-01T17:00", "observations": obs})[1]
    assert a["forecasts"]["final"]["values"] == b["forecasts"]["final"]["values"]


def test_cors_is_closed_by_default(api):
    _, base = api
    req = urllib.request.Request(base + "/predict", method="OPTIONS", headers={"Origin": "https://example.org"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 403
