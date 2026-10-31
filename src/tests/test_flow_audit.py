"""Unit tests for the Channel 0 flow audit. All arrays are small and synthetic."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from aiori import data_io, flow_audit
from aiori.config import FLOW


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_array():
    """4 sensors x 10 steps x 3 channels with known Channel 0 patterns."""
    a = np.ones((4, 10, 3), dtype=np.float32) * 7.0
    a[0, :, FLOW] = np.arange(10)                 # fully observed, includes one zero
    a[1, :, FLOW] = np.nan                        # fully missing
    a[2, :, FLOW] = 0.0                           # constant (all zero) but observed
    a[3, :, FLOW] = [1, np.nan, np.nan, np.nan, 5, 6, np.nan, 8, 9, 10]
    # Other channels carry different missingness to prove only Ch0 is read.
    a[0, :, 1] = np.nan
    a[3, :, 2] = np.nan
    return a


# --- array loading and channel selection -------------------------------------

def test_load_traffic_array_checks_shape_and_dtype(tmp_path):
    path = tmp_path / "arr.npy"
    np.save(path, _toy_array())
    arr = data_io.load_traffic_array(path, expected_shape=(4, 10, 3))
    assert isinstance(arr, np.memmap)
    assert arr.shape == (4, 10, 3) and arr.dtype == np.float32

    with pytest.raises(ValueError, match="shape"):
        data_io.load_traffic_array(path, expected_shape=(4, 11, 3))

    np.save(tmp_path / "f64.npy", _toy_array().astype(np.float64))
    with pytest.raises(ValueError, match="dtype"):
        data_io.load_traffic_array(tmp_path / "f64.npy")

    np.save(tmp_path / "flat.npy", np.zeros((4, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="3-D"):
        data_io.load_traffic_array(tmp_path / "flat.npy")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_io.load_traffic_array(tmp_path / "nope.npy")


def test_audit_reads_only_channel_zero():
    stats = flow_audit.audit_flow(_toy_array(), batch_size=3)
    # Sensor 0 has NaNs in Ch1 only; sensor 3 has NaNs in Ch2 only.
    assert stats.loc[0, "missing_count"] == 0
    assert stats.loc[3, "missing_count"] == 4


# --- per-sensor statistics ----------------------------------------------------

def test_finite_and_missing_counts():
    stats = flow_audit.audit_flow(_toy_array(), batch_size=3)
    assert stats["array_index"].tolist() == [0, 1, 2, 3]
    assert stats["finite_count"].tolist() == [10, 0, 10, 6]
    assert stats["missing_count"].tolist() == [0, 10, 0, 4]
    assert stats.loc[3, "missing_pct"] == pytest.approx(40.0)
    assert stats.loc[3, "max_gap_steps"] == 3


def test_batch_size_does_not_change_results():
    a = _toy_array()
    pd.testing.assert_frame_equal(
        flow_audit.audit_flow(a, batch_size=1), flow_audit.audit_flow(a, batch_size=100)
    )


def test_zero_is_a_valid_observation():
    stats = flow_audit.audit_flow(_toy_array())
    assert stats.loc[0, "zero_count"] == 1
    assert stats.loc[0, "finite_count"] == 10
    assert stats.loc[2, "zero_count"] == 10
    assert not stats.loc[2, "fully_missing"]


def test_infinity_counted_as_missing():
    a = _toy_array()
    a[0, 4, FLOW] = np.inf
    stats = flow_audit.audit_flow(a)
    assert stats.loc[0, "inf_count"] == 1
    assert stats.loc[0, "missing_count"] == 1


def test_fully_missing_and_constant_flags():
    stats = flow_audit.audit_flow(_toy_array())
    assert stats["fully_missing"].tolist() == [False, True, False, False]
    assert stats["constant"].tolist() == [False, False, True, False]
    assert np.isnan(stats.loc[1, "std"])
    assert stats.loc[2, "std"] == 0.0


def test_identical_series_flagged_as_duplicates():
    a = _toy_array()
    a = np.concatenate([a, a[[0, 0, 3]]])          # rows 4, 5 copy row 0; row 6 copies row 3
    a[6, 1, FLOW] = 0.0                            # ...but row 6 now differs by one value
    stats = flow_audit.audit_flow(a, batch_size=2)
    assert stats["duplicate_series"].tolist() == [True, False, False, False, True, True, False]
    assert stats.loc[0, "duplicate_group_size"] == 3
    # All-NaN sensors share a hash but are fully missing, not duplicates.
    b = np.concatenate([_toy_array(), _toy_array()[[1]]])
    assert not flow_audit.audit_flow(b).loc[[1, 4], "duplicate_series"].any()


# --- eligibility and selection ------------------------------------------------

def _audit_frame(n_per_group=6, groups=("a", "b")):
    rows = []
    idx = 0
    for g in groups:
        for k in range(n_per_group):
            rows.append(dict(array_index=idx, proposed_client=g, missing_count=k % 3,
                             coverage_pct=100.0 - (k % 3), zero_pct=1.0,
                             fully_missing=False, constant=False, duplicate_series=False))
            idx += 1
    return pd.DataFrame(rows)


def test_eligibility_reasons():
    df = pd.DataFrame(dict(
        array_index=[0, 1, 2, 3, 4, 5],
        fully_missing=[True, False, False, False, False, False],
        constant=[False, True, False, False, False, False],
        duplicate_series=[False, False, True, False, False, False],
        coverage_pct=[0.0, 100.0, 100.0, 90.0, 100.0, 100.0],
        zero_pct=[np.nan, 100.0, 0.0, 0.0, 80.0, 2.0],
    ))
    out = flow_audit.assign_eligibility(df, min_coverage_pct=99.0, max_zero_pct=50.0)
    assert out["exclusion_reason"].tolist() == [
        "fully_missing", "constant", "duplicate_series", "low_coverage", "high_zero_fraction", ""]
    assert out["eligible"].tolist() == [False, False, False, False, False, True]


def test_selection_is_deterministic_and_balanced():
    df = flow_audit.assign_eligibility(_audit_frame(), 95.0, 50.0)
    first = flow_audit.select_pilot(df, per_group=3)
    second = flow_audit.select_pilot(df.sample(frac=1, random_state=1), per_group=3)
    assert first["array_index"].tolist() == second["array_index"].tolist()
    assert first.groupby("proposed_client").size().tolist() == [3, 3]
    # Best coverage first: only missing_count == 0 or 1 sensors fit into 3 slots per group.
    assert first["missing_count"].max() <= 1


def test_salt_never_overrides_coverage_ranking():
    df = flow_audit.assign_eligibility(_audit_frame(n_per_group=9), 95.0, 50.0)
    a = flow_audit.select_pilot(df, per_group=3, salt="x")
    b = flow_audit.select_pilot(df, per_group=3, salt="y")
    assert (a["missing_count"] == 0).all() and (b["missing_count"] == 0).all()


def test_selection_fails_when_group_too_small():
    df = flow_audit.assign_eligibility(_audit_frame(n_per_group=2), 95.0, 50.0)
    with pytest.raises(ValueError, match="eligible"):
        flow_audit.select_pilot(df, per_group=3)


# --- IDs and metadata ---------------------------------------------------------

def test_duplicate_node_ids_rejected(tmp_path):
    np.save(tmp_path / "n.npy", np.array([5, 6, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        data_io.load_node_order(tmp_path / "n.npy")


def test_metadata_alignment_and_id_set_check():
    node_order = np.array([30, 10, 20])
    meta = pd.DataFrame(dict(station_id=[10, 20, 30], District=[4, 7, 12]))
    aligned = data_io.align_metadata(meta, node_order)
    assert aligned["station_id"].tolist() == [30, 10, 20]
    assert aligned["District"].tolist() == [12, 4, 7]
    with pytest.raises(ValueError, match="ID sets differ"):
        data_io.align_metadata(meta, np.array([10, 20, 99]))


# --- end-to-end on synthetic files; raw inputs must not change ---------------

def _write_fixture(tmp_path, n_per_district=4):
    districts = [7, 4, 12, 3]
    n = n_per_district * len(districts)
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, size=(n, 24, 3)).astype(np.float32)
    a[0, :, FLOW] = np.nan
    ids = np.arange(1000, 1000 + n, dtype=np.int64)[::-1].copy()
    meta = pd.DataFrame(dict(
        station_id=np.sort(ids),
        District=np.repeat(districts, n_per_district),
        County=["c"] * n,
    ))
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "arr.npy", a)
    np.save(raw / "node_order.npy", ids)
    meta.to_csv(raw / "meta.csv", sep="\t", index=False)
    return raw


def test_main_writes_outputs_and_leaves_raw_files_untouched(tmp_path):
    raw = _write_fixture(tmp_path)
    raw_files = sorted(raw.iterdir())
    before = {p: (_sha256(p), data_io.file_fingerprint(p)) for p in raw_files}
    out = tmp_path / "processed"

    rc = flow_audit.main([
        "--array", str(raw / "arr.npy"), "--node-order", str(raw / "node_order.npy"),
        "--metadata", str(raw / "meta.csv"), "--out-dir", str(out),
        "--per-client", "2", "--no-shape-check", "--steps-per-day", "4",
        "--validation-per-client", "0",
    ])

    assert rc == 0
    assert {p: (_sha256(p), data_io.file_fingerprint(p)) for p in raw_files} == before
    assert sorted(raw.iterdir()) == raw_files
    sel = pd.read_csv(out / "sensor_selection.csv")
    audit = pd.read_csv(out / "flow_sensor_audit.csv")
    assert len(sel) == 8 and sel["station_id"].is_unique
    assert len(audit) == 16 and audit["station_id"].is_unique
    assert 0 not in sel["array_index"].tolist()           # fully missing sensor excluded
    assert audit.loc[audit.array_index == 0, "exclusion_reason"].item() == "fully_missing"
    assert (out / "flow_quality_report.md").is_file()


def test_refuses_to_write_into_raw_paths(tmp_path):
    raw = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="raw"):
        data_io.check_output_dir(raw, raw_paths=[raw / "arr.npy"])
