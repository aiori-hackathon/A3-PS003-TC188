"""End-to-end test of the pilot preprocessing CLI on small synthetic files."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from aiori import data_io, prepare_pilot
from aiori.config import FLOW

SPD, DAYS, N_SENSORS = 24, 10, 6     # 240 steps: train 168, val 24, test 48


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixture(tmp_path, array=None, station_ids=None):
    rng = np.random.default_rng(1)
    if array is None:
        array = rng.uniform(0, 100, size=(N_SENSORS + 2, SPD * DAYS, 3)).astype(np.float32)
        array[1, [5, 6, 200, 201], FLOW] = np.nan      # missing in train and test
        array[2, 30:40, FLOW] = 0.0                    # zeros are observations
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    node_order = np.arange(500, 500 + len(array), dtype=np.int64)
    np.save(raw / "arr.npy", array)
    np.save(raw / "node_order.npy", node_order)
    sel_idx = np.arange(N_SENSORS)
    pd.DataFrame(dict(
        array_index=sel_idx,
        station_id=node_order[sel_idx] if station_ids is None else station_ids,
        proposed_client=["a", "a", "b", "b", "c", "c"],
        eligible=True,
    )).to_csv(raw / "sel.csv", index=False)
    return raw, array


def _run(raw, out):
    return prepare_pilot.main([
        "--array", str(raw / "arr.npy"), "--node-order", str(raw / "node_order.npy"),
        "--selection", str(raw / "sel.csv"), "--out-dir", str(out),
        "--steps-per-day", str(SPD), "--min-sensors", "2", "--no-shape-check",
    ])


def test_pipeline_outputs_and_raw_untouched(tmp_path):
    raw, array = _fixture(tmp_path)
    before = {p: (_sha(p), data_io.file_fingerprint(p)) for p in raw.iterdir()}
    out = tmp_path / "processed"
    assert _run(raw, out) == 0
    assert {p: (_sha(p), data_io.file_fingerprint(p)) for p in raw.iterdir()} == before

    split = json.loads((out / "split_config.json").read_text())
    assert split["boundaries"] == {"train": [0, 168], "val": [168, 192], "test": [192, 240]}

    scaler = json.loads((out / "scaler_parameters.json").read_text())
    assert scaler["mode"] == "per_sensor" and scaler["fit_range"] == [0, 168]
    assert scaler["n_obs"][1] == 168 - 2
    np.testing.assert_allclose(scaler["mean"][0], array[0, :168, FLOW].astype(np.float64).mean())
    assert "unconfirmed" in scaler["id_mapping_status"]

    data = np.load(out / "pilot_preprocessed.npz")
    assert data["flow_raw"].shape == (N_SENSORS, SPD * DAYS)
    assert np.isnan(data["targets_scaled"][1, [5, 6, 200, 201]]).all()       # targets never filled
    assert not data["observed_mask"][1, [5, 6, 200, 201]].any()
    assert data["input_imputed"][1, [5, 6, 200, 201]].all()
    assert np.isfinite(data["inputs_scaled"]).all()
    assert (data["flow_raw"][2, 30:40] == 0).all() and data["observed_mask"][2, 30:40].all()

    idx = np.load(out / "window_index.npz")
    for name, (start, end) in {"train": (0, 168), "val": (168, 192), "test": (192, 240)}.items():
        t0 = idx[name][:, 1]
        assert len(t0) and (t0 >= start).all() and (t0 + 12 <= end).all()
    assert (out / "preprocessing_report.md").is_file()


def test_future_values_do_not_change_fitted_statistics(tmp_path):
    raw, array = _fixture(tmp_path)
    _run(raw, tmp_path / "a")
    perturbed = array.copy()
    perturbed[:, 168:, FLOW] = 9999.0                     # change val + test only
    (tmp_path / "second").mkdir()
    raw2, _ = _fixture(tmp_path / "second", array=perturbed)
    _run(raw2, tmp_path / "b")
    a = json.loads((tmp_path / "a" / "scaler_parameters.json").read_text())
    b = json.loads((tmp_path / "b" / "scaler_parameters.json").read_text())
    assert a["mean"] == b["mean"] and a["std"] == b["std"]
    da, db = np.load(tmp_path / "a" / "pilot_preprocessed.npz"), np.load(tmp_path / "b" / "pilot_preprocessed.npz")
    np.testing.assert_array_equal(da["inputs_scaled"][:, :168], db["inputs_scaled"][:, :168])
    np.testing.assert_array_equal(da["tod_median"], db["tod_median"])


def test_manifest_station_id_mismatch_rejected(tmp_path):
    raw, _ = _fixture(tmp_path, station_ids=np.arange(N_SENSORS) + 9000)
    with pytest.raises(ValueError, match="station_id"):
        _run(raw, tmp_path / "out")


def test_sensor_count_bounds_enforced(tmp_path):
    raw, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="sensors"):
        prepare_pilot.main([
            "--array", str(raw / "arr.npy"), "--node-order", str(raw / "node_order.npy"),
            "--selection", str(raw / "sel.csv"), "--out-dir", str(tmp_path / "o"),
            "--steps-per-day", str(SPD), "--no-shape-check",
        ])
