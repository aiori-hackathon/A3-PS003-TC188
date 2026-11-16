"""Historical Disruption Explorer (Phase 9B): detector behaviour on synthetic fixtures (software tests only; never shown
in the app), the real catalog's integrity and time alignment, causal forecast reuse, and the HTTP API."""

import copy
import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app" / "backend"))
import disruption_detect as D  # noqa: E402

DATA = ROOT / "data" / "processed" / "aiori2_3yr"
STAGE3 = ROOT / "results" / "aiori2" / "phase8c_stage3"
CATALOG = D.OUT / D.VERSION / "catalog.json"
real = pytest.mark.skipif(not (DATA / "extracted" / "flow_raw.npy").exists() or not CATALOG.exists()
                          or not (STAGE3 / "analysis" / "final_manifest.json").exists(),
                          reason="local AIORI 2.0 data, Stage 3 checkpoints or the disruption catalog not present")
SPD = 288


# ---------------------------------------------------------------- synthetic fixture (unit tests only)

def day_slot(day, slot):
    return day * SPD + slot


@pytest.fixture(scope="module")
def synth():
    """5 sensors x 70 days from 2023-02-01 with a smooth daily profile and Poisson noise; the first 35 days are the
    'training' period. Injected: a short drop, a 2-reading spike, a missing block, a run of exact zeros and a sustained
    drop (sensor 0, BA); a simultaneous rise at sensors 1-3 (LA); sensor 4 (IC) has a typical flow below 5."""
    rng = np.random.default_rng(7)
    days = 70
    slot = np.arange(SPD)
    prof = 50 + 150 * np.exp(-((slot - 96) / 30) ** 2) + 120 * np.exp(-((slot - 204) / 30) ** 2)
    lam = np.tile(prof, days)
    flow = rng.poisson(np.stack([lam] * 4 + [np.full(days * SPD, 1.0)])).astype(np.float32)
    valid = np.ones_like(flow, bool)
    flow[0, day_slot(45, 60):day_slot(45, 64)] = np.round(0.3 * lam[day_slot(45, 60):day_slot(45, 64)])     # 20-min drop
    flow[0, day_slot(48, 130):day_slot(48, 132)] *= 3                                                    # 2-reading spike
    flow[0, day_slot(52, 100):day_slot(52, 130)] = np.nan                                                # missing block
    valid[0, day_slot(52, 100):day_slot(52, 130)] = False
    flow[0, day_slot(58, 90):day_slot(58, 101)] = 0                                                      # exact zeros (valid)
    flow[0, day_slot(61, 120):day_slot(61, 160)] = np.round(0.4 * lam[day_slot(61, 120):day_slot(61, 160)])  # 200-min drop
    for s in (1, 2, 3):
        flow[s, day_slot(55, 150):day_slot(55, 170)] = np.round(2.5 * lam[day_slot(55, 150):day_slot(55, 170)])
    flow[4, day_slot(50, 10):day_slot(50, 20)] = 40                                                      # low-baseline sensor
    stamps = np.datetime64("2023-02-01T00:00") + np.arange(days * SPD) * np.timedelta64(5, "m")
    ids = ["BA-01", "LA-01", "LA-02", "LA-03", "IC-01"]
    geo = np.array(["client_1_bay_area"] + ["client_0_la_ventura"] * 3 + ["client_3_inland_central"])
    with np.errstate(all="raise"):                                   # no division warnings anywhere in the pipeline
        out = D.detect(flow, valid, stamps, ids, geo, (0, 35 * SPD), log=lambda *_: None)
    return dict(flow=flow, valid=valid, stamps=stamps, ids=ids, geo=geo, out=out)


def ts(stamps, day, s):
    return str(stamps[day_slot(day, s)])[:16]


def test_injected_drops_and_rises_are_found(synth):
    ev, st = synth["out"][2], synth["stamps"]
    drop = [e for e in ev if e["sensors"] == ["BA-01"] and e["start"][:10] == ts(st, 45, 0)[:10]]
    assert len(drop) == 1 and drop[0]["direction"] == "decrease" and drop[0]["category"] == "sudden_decrease"
    assert drop[0]["start"] == ts(st, 45, 60) and drop[0]["end"] == ts(st, 45, 63) and drop[0]["duration_min"] == 20
    sus = [e for e in ev if e["sensors"] == ["BA-01"] and e["start"][:10] == ts(st, 61, 0)[:10]]
    assert len(sus) == 1 and sus[0]["category"] == "sustained_deviation" and sus[0]["duration_min"] >= 60
    co = [e for e in ev if e["region_code"] == "LA"]
    assert len(co) == 1 and co[0]["n_sensors"] == 3 and co[0]["category"] == "coordinated_deviation"
    assert co[0]["direction"] == "increase" and co[0]["start"] == ts(st, 55, 150)


def test_normal_days_spikes_and_missing_blocks_are_not_events(synth):
    ev, st = synth["out"][2], synth["stamps"]
    days = {e["start"][:10] for e in ev}
    assert days == {ts(st, d, 0)[:10] for d in (45, 55, 58, 61)}                 # nothing on the 66 ordinary days
    assert not any(e["start"][:10] in (ts(st, 48, 0)[:10], ts(st, 52, 0)[:10]) for e in ev)
    assert not any("IC-01" in e["sensors"] for e in ev)                         # typical flow < 5: never scored


def test_zero_readings_are_quality_flagged_not_missing(synth):
    ev, st = synth["out"][2], synth["stamps"]
    z = [e for e in ev if e["start"][:10] == ts(st, 58, 0)[:10]]
    assert len(z) == 1 and z[0]["category"] == "sensor_quality" and "zero_reading" in z[0]["quality_flags"]
    assert z[0]["confidence"] == "low"
    # zeros are valid readings, so they are scored (not dropped as missing)
    assert synth["valid"][0, day_slot(58, 95)] and synth["flow"][0, day_slot(58, 95)] == 0


def test_event_ids_and_output_are_deterministic(synth):
    again = D.detect(synth["flow"], synth["valid"], synth["stamps"], synth["ids"], synth["geo"], (0, 35 * SPD), log=lambda *_: None)
    assert D.events_hash(again[2]) == D.events_hash(synth["out"][2])
    assert [e["id"] for e in again[2]] == [e["id"] for e in synth["out"][2]]
    cfg = dict(D.CONFIG, min_relative=0.3)
    assert D.event_id("LA", -1, "2023-01-01T00:00", ["LA-01"], 1, 2, D.config_hash(cfg)) != \
        D.event_id("LA", -1, "2023-01-01T00:00", ["LA-01"], 1, 2, D.config_hash())


def test_thresholds_come_from_the_training_window_only(synth):
    flow = synth["flow"].copy()
    flow[:4, 40 * SPD:] *= 5                                         # change everything after the training window
    out = D.detect(flow, synth["valid"], synth["stamps"], synth["ids"], synth["geo"], (0, 35 * SPD), log=lambda *_: None)
    assert out[1]["z_low"] == synth["out"][1]["z_low"] and out[1]["z_high"] == synth["out"][1]["z_high"]
    assert np.array_equal(out[0]["median"], synth["out"][0]["median"], equal_nan=True)


def test_segments_gap_rule_and_minimum_evidence():
    f = np.zeros(40, np.int8)
    f[[2, 3, 6]] = -1            # gap of 2 readings joins
    f[[10, 11]] = 1              # only 2 readings: dropped
    f[[20, 21, 25, 26]] = 1      # gap of 3 splits into two segments of 2: both dropped
    f[[30, 31, 32]] = 1
    assert D.segments(f) == [(-1, 2, 6), (1, 30, 32)]


def test_score_ignores_invalid_and_low_baseline_and_quality_bits():
    x = np.array([0, 10, np.nan, 3, 100.0])
    ok = np.array([True, True, False, True, True])
    m = np.array([50, 50, 50, 2, np.nan])
    z, rel = D.score(x, ok, m, np.full(5, 5.0))
    assert np.isfinite(z[:2]).all() and np.isnan(z[2:]).all()
    q = D.slot_quality(np.array([5.0] * 14 + [0, 0, 7, 8]), np.array([True] * 17 + [False]))
    assert all(q[:14] & D.Q_BITS["stuck_run"]) and q[14] & D.Q_BITS["zero_reading"] and q[16] & D.Q_BITS["near_missing"]


def test_grouping_is_by_overlap_within_join_window():
    items = [(0, 5, "a"), (8, 9, "b"), (20, 25, "c"), (3, 4, "d")]
    assert [[i[2] for i in g] for g in D.group(items, 3)] == [["a", "d", "b"], ["c"]]


# ---------------------------------------------------------------- the real catalog

@pytest.fixture(scope="module")
def DS():
    from disruptions import Disruptions
    from replay import Replay
    return Disruptions(Replay())


@real
def test_catalog_integrity_and_alignment(DS):
    r, ev = DS.r, DS.events
    from disruptions import ID_RE
    assert DS.cat["version"] == D.VERSION and DS.cat["config_sha256"] == D.config_hash()
    assert len({e["id"] for e in ev}) == len(ev) and all(ID_RE.fullmatch(e["id"]) for e in ev)
    assert [e["start_index"] for e in ev] == sorted(e["start_index"] for e in ev)
    for e in ev[:: max(1, len(ev) // 500)]:
        assert r.ts(e["start_index"]) == e["start"] and r.ts(e["end_index"]) == e["end"]
        assert e["duration_min"] == (e["end_index"] - e["start_index"] + 1) * 5
        assert all(s in r.by_id for s in e["sensors"]) and e["n_sensors"] == len(e["sensors"])
        assert e["category"] in ("sensor_quality",) or e["quality_share"] < 0.5
    assert {e["region_code"] for e in ev} <= {"LA", "BA", "CS", "IC"}
    th = DS.cat["thresholds"]
    assert th["z_low"] < 0 < th["z_high"] and "2023" in th["calibration"]
    assert DS.cat["summary"]["n_events"] == len(ev)


@real
def test_missing_and_dst_readings_are_never_anomalous(DS):
    r = DS.r
    assert not np.asarray(r.valid[:, np.flatnonzero(r.dst)]).any()           # the skipped DST hour has no valid reading
    picks = [e for e in DS.events if e["category"] != "sensor_quality"][::400][:25]
    for e in picks:
        s = DS.series(e["id"], e["peak_sensor"])
        an = np.array(s["anomalous"]) != 0
        assert all(st == "ok" for st, a in zip(s["status"], an) if a)
        assert all(s["observed"][i] is not None for i in np.flatnonzero(an))
        assert all(s["observed"][i] is None for i, st in enumerate(s["status"]) if st != "ok")    # null, never 0
        assert s["timestamps"][0] <= e["start"] and s["timestamps"][-1] >= e["end"] or s["window"]["truncated"]


@real
def test_baseline_is_the_2023_median(DS):
    r = DS.r
    pos, slot = r.by_id["LA-05"], 100
    days = np.arange(365, 730)
    dt = DS.dtype[days]
    xs = []
    for o in (-1, 0, 1):
        idx = days[dt == 0] * SPD + slot + o
        v = np.asarray(r.flow[pos, idx], float)
        xs.append(np.where(np.asarray(r.valid[pos, idx]), v, np.nan))
    assert DS.med[pos, 0, slot] == pytest.approx(np.nanmedian(np.concatenate(xs)), abs=1e-3)


@real
def test_robustness_uses_the_replay_forecast_contract(DS):
    r = DS.r
    e = next(e for e in DS.events if e["category"] == "sudden_decrease" and e["start"] >= "2024-02-01")
    pos = r.by_id[e["peak_sensor"]]
    T = r.parse_time(e["start"]) - 1
    pred, usable = DS._forecast_block("geo", pos, np.array([T]))
    f = r.forecast(e["peak_sensor"], r.ts(T), "geo")
    assert usable[0] and np.allclose(pred[0], f["forecast"]["values"], atol=0.011)
    x, _ = r._inputs(np.array([pos]), T + 1)                        # same 12 causal inputs as the replay service
    assert f["input"]["timestamps"][-1] == r.ts(T)
    rb = DS.robustness(e["id"], e["peak_sensor"], "geo")
    assert rb["available"] and rb["event_window"]["n"] > 0
    assert rb["issues"]["count"] <= 72 and rb["issues"]["first"] >= r.ts(r.parse_time(e["start"]) - 12)


@real
def test_robustness_targets_are_valid_readings_only(DS):
    r = DS.r
    e = next(e for e in DS.events if e["category"] == "sustained_deviation" and e["start"] >= "2024-07-01")
    pos = r.by_id[e["peak_sensor"]]
    sd = next(s for s in e["sensor_details"] if s["sensor"] == e["peak_sensor"])
    s0, s1 = r.parse_time(sd["start"]), r.parse_time(sd["end"])
    issues = np.arange(s0 - 12, s1 + 1)
    pred, usable = DS._forecast_block("geo", pos, issues)
    err, act, scope = DS._errors(pos, issues, pred, usable, lambda tg: (tg >= s0) & (tg <= s1))
    tg = issues[:, None] + 1 + np.arange(12)[None, :]
    assert not np.isfinite(err[~scope]).any()
    ok = np.isfinite(err)
    assert np.asarray(r.valid[pos][tg[ok]]).all() and ((tg[ok] >= s0) & (tg[ok] <= s1)).all()


@real
def test_2022_events_have_no_forecast(DS):
    e = next(e for e in DS.events if e["start"] < "2022-12-01" and e["category"] != "sensor_quality")
    d = DS.detail(e["id"])
    assert d["forecast"]["available"] is False and "2023" in d["forecast"]["reason"]
    assert DS.robustness(e["id"], e["peak_sensor"], "geo")["available"] is False


@real
def test_stale_or_missing_catalog_is_rejected(DS, tmp_path):
    from disruptions import Disruptions, Unavailable
    with pytest.raises(Unavailable, match="has not been built"):
        Disruptions(DS.r, root=tmp_path / "none")
    d = tmp_path / D.VERSION
    shutil.copytree(D.OUT / D.VERSION, d)
    c = json.loads((d / "catalog.json").read_text(encoding="utf-8"))
    c["config_sha256"] = "0" * 64
    (d / "catalog.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(Unavailable, match="hash"):
        Disruptions(DS.r, root=tmp_path, verify_inputs=False)
    (d / "catalog.sha256").write_text(D.sha256(d / "catalog.json") + "  catalog.json\n", encoding="utf-8")
    with pytest.raises(Unavailable, match="different detector version or configuration"):
        Disruptions(DS.r, root=tmp_path, verify_inputs=False)


# ---------------------------------------------------------------- HTTP API

@pytest.fixture(scope="module")
def api():
    import replay_server as RS
    srv = RS.make_server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@real
def test_http_list_filters_sorting_and_pagination(api):
    st, m = get(api, "/api/replay/disruptions/meta")
    assert st == 200 and m["version"] == D.VERSION and len(m["limitations"]) >= 5
    assert "causes are not verified" in m["limitations"][0]
    text = json.dumps(m)
    assert "Desktop" not in text and "\\\\" not in text and "station_id" not in text
    st, a = get(api, "/api/replay/disruptions?page_size=100")
    assert st == 200 and a["total"] == m["summary"]["n_events"] and len(a["items"]) == 100
    scores = [i["score"] for i in a["items"]]
    assert scores == sorted(scores, reverse=True)
    st, b = get(api, "/api/replay/disruptions?from=2024-03-01&to=2024-03-31&sort=start&quality=exclude&page_size=50")
    assert all("2024-02-29" <= i["end"][:10] and i["start"][:10] <= "2024-03-31" and i["category"] != "sensor_quality" for i in b["items"])
    assert [i["start"] for i in b["items"]] == sorted(i["start"] for i in b["items"])
    st, p2 = get(api, "/api/replay/disruptions?from=2024-03-01&to=2024-03-31&sort=start&quality=exclude&page_size=50&page=2")
    assert not {i["id"] for i in b["items"]} & {i["id"] for i in p2["items"]}
    st, s = get(api, "/api/replay/disruptions?sensor=LA-05&category=sudden_decrease,sudden_increase&min_sensors=1")
    assert st == 200 and all(i["category"] in ("sudden_decrease", "sudden_increase") for i in s["items"])
    for i in s["items"]:
        assert "LA-05" in get(api, f"/api/replay/disruptions/{i['id']}")[1]["sensors"]
    st, c = get(api, "/api/replay/disruptions?min_sensors=3&region=BA&direction=increase")
    assert all(i["n_sensors"] >= 3 and i["region_code"] == "BA" and i["direction"] == "increase" for i in c["items"])
    st, empty = get(api, "/api/replay/disruptions?from=2022-01-01&to=2022-01-01&min_sensors=200")
    assert st == 200 and empty["total"] == 0 and empty["items"] == []


@real
def test_http_event_detail_series_compare_robustness(api):
    st, a = get(api, "/api/replay/disruptions?category=coordinated_deviation&from=2024-01-01&page_size=1")
    e = a["items"][0]
    st, d = get(api, f"/api/replay/disruptions/{e['id']}")
    assert st == 200 and d["replay"]["sensor"] in d["sensors"] and d["replay"]["time"].startswith(d["replay"]["date"])
    assert d["forecast"]["available"] and d["why_flagged"] and d["sensor_details"]
    st, s = get(api, f"/api/replay/disruptions/{e['id']}/series?sensor={d['peak_sensor']}")
    assert st == 200 and len(s["timestamps"]) == len(s["observed"]) == len(s["expected"]) == len(s["anomalous"])
    assert len(s["timestamps"]) <= 3 * 288
    st, c = get(api, f"/api/replay/disruptions/{e['id']}/compare")
    assert st == 200 and c["shown"] == min(12, d["n_sensors"]) and all(len(x["ratio"]) == len(c["timestamps"]) for x in c["sensors"])
    st, rb = get(api, f"/api/replay/disruptions/{e['id']}/robustness?sensor={d['peak_sensor']}&partition=iid")
    assert st == 200 and rb["available"] and rb["model"]["partition"] == "iid" and "Not a test score" in rb["caution"]
    other = next(x for x in get(api, "/api/replay/sensors")[1] if x["id"] not in d["sensors"])["id"]
    assert get(api, f"/api/replay/disruptions/{e['id']}/series?sensor={other}")[0] == 404


@real
def test_http_rejects_bad_requests(api):
    for path, code in (("/api/replay/disruptions?page_size=101", 400), ("/api/replay/disruptions?page_size=0", 400),
                       ("/api/replay/disruptions?bogus=1", 400), ("/api/replay/disruptions?sort=random", 400),
                       ("/api/replay/disruptions?from=2024-13-01", 400), ("/api/replay/disruptions?from=yesterday", 400),
                       ("/api/replay/disruptions?region=XX", 400), ("/api/replay/disruptions?sensor=ZZ-99", 404),
                       ("/api/replay/disruptions?min_peak_z=abc", 400), ("/api/replay/disruptions?quality=maybe", 400),
                       ("/api/replay/disruptions?page=1&page=2", 400), ("/api/replay/disruptions/20240101T0000-LA-D-000000", 404),
                       ("/api/replay/disruptions/..%2F..%2Fetc", 404), ("/api/replay/disruptions/not-an-id", 404),
                       ("/api/replay/disruptions/meta?x=1", 400)):
        st, b = get(api, path)
        assert st == code and "error" in b, path
    st, a = get(api, "/api/replay/disruptions?page_size=1")
    eid = a["items"][0]["id"]
    assert get(api, f"/api/replay/disruptions/{eid}/series")[0] == 400
    assert get(api, f"/api/replay/disruptions/{eid}/nope")[0] == 404
    assert get(api, f"/api/replay/disruptions/{eid}/robustness?sensor={a['items'][0]['peak_sensor']}&partition=x")[0] == 400


@real
def test_http_missing_catalog_returns_503_and_replay_keeps_working(api):
    import replay_server as RS
    saved = RS.DISRUPTIONS
    RS.DISRUPTIONS = "The disruption catalog has not been built yet. Build it with: ..."
    try:
        st, b = get(api, "/api/replay/disruptions")
        assert st == 503 and "not been built" in b["error"]["message"]
        assert get(api, "/api/replay/meta")[0] == 200
        assert get(api, "/api/replay/forecast?sensor=LA-01&time=2024-08-14T08:00")[0] == 200
    finally:
        RS.DISRUPTIONS = saved
