"""Synthetic tests for training-period eligibility, District floors and the
missing-data validation selection."""

import numpy as np
import pandas as pd
import pytest

from aiori import flow_audit
from aiori.config import FLOW

SPD, DAYS = 4, 10                 # 40 steps: train [0, 28), val [28, 32), test [32, 40)
TRAIN = (0, 28)


def _array(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 100, size=(n, SPD * DAYS, 3)).astype(np.float32)


def _aligned(districts):
    n = len(districts)
    return pd.DataFrame(dict(station_id=np.arange(100, 100 + n), District=districts, County=["c"] * n))


# --- training-period eligibility ---------------------------------------------

def test_eligibility_ignores_validation_and_test_values():
    a = _array(6)
    a[0, 35, FLOW] = np.nan                # missing only in test -> still eligible
    a[1, 3, FLOW] = np.nan                 # missing in train -> low coverage (27/28)
    a[2, 28:, FLOW] = 0.0                  # all-zero val/test -> zero share ignored
    a[4] = a[3]                            # identical in train ...
    a[4, 30, FLOW] += 1.0                  # ... but diverges in val
    audit = flow_audit.build_audit(a, _aligned([7] * 6), TRAIN, 99.0, 50.0)

    assert audit["exclusion_reason"].tolist() == ["", "low_coverage", "", "duplicate_series", "duplicate_series", ""]
    assert audit.loc[0, "coverage_pct"] == 100.0 and audit.loc[0, "month_coverage_pct"] < 100.0
    assert audit.loc[0, "eval_missing_count"] == 1
    assert audit.loc[2, "zero_pct"] == 0.0 and audit.loc[2, "month_zero_pct"] > 0
    # Whole-month duplicate flag is informational: rows 3/4 differ over the month.
    assert not audit.loc[[3, 4], "month_duplicate_series"].any()
    assert (audit["stats_period"] == "train steps [0, 28)").all()


def test_primary_selection_unchanged_when_future_values_change():
    a = _array(12, seed=3)
    a[[1, 5], 10, FLOW] = np.nan
    districts = [7] * 6 + [4] * 6
    first = flow_audit.select_pilot(flow_audit.build_audit(a, _aligned(districts), TRAIN, 99.0, 50.0), per_group=3)
    b = a.copy()
    b[:, 28:, FLOW] = np.nan               # wipe validation + test entirely
    b[0, 28:, FLOW] = 0.0
    second = flow_audit.select_pilot(flow_audit.build_audit(b, _aligned(districts), TRAIN, 99.0, 50.0), per_group=3)
    assert first["array_index"].tolist() == second["array_index"].tolist()


# --- District floors -----------------------------------------------------------

def _ranked_frame():
    # client "c": District 12 sensors rank best (missing 0); District 5 sensors rank worst.
    rows = [dict(array_index=i, proposed_client="c", District=12, missing_count=0) for i in range(6)]
    rows += [dict(array_index=10 + i, proposed_client="c", District=5, missing_count=2) for i in range(3)]
    rows += [dict(array_index=20 + i, proposed_client="d", District=7, missing_count=0) for i in range(5)]
    df = pd.DataFrame(rows)
    df["eligible"] = True
    return df


def test_district_floor_guarantees_every_district():
    df = _ranked_frame()
    plain = flow_audit.select_pilot(df, per_group=4)
    assert 5 not in plain.loc[plain.proposed_client == "c", "District"].tolist()

    floored = flow_audit.select_pilot(df, per_group=4, floor_col="District")
    c = floored[floored.proposed_client == "c"]
    assert len(c) == 4 and (c.District == 5).sum() == 1 and (c.District == 12).sum() == 3
    assert c.loc[c.District == 5, "selection_basis"].item() == "District_floor"
    assert len(floored[floored.proposed_client == "d"]) == 4


def test_district_floor_is_deterministic_and_order_invariant():
    df = _ranked_frame()
    a = flow_audit.select_pilot(df, per_group=4, floor_col="District")
    b = flow_audit.select_pilot(df.sample(frac=1, random_state=7), per_group=4, floor_col="District")
    assert a["array_index"].tolist() == b["array_index"].tolist()
    # The floor picks the best-ranked District 5 sensor, chosen by the salted tie-break.
    best5 = min(range(10, 13), key=lambda i: flow_audit._tie_key(i, flow_audit.DEFAULT_SALT))
    assert best5 in a["array_index"].tolist()


def test_district_floor_fails_without_eligible_sensor():
    df = _ranked_frame()
    df.loc[df.District == 5, "eligible"] = False
    with pytest.raises(ValueError, match="No eligible sensors for District"):
        flow_audit.select_pilot(df, per_group=4, floor_col="District")


def test_district_floor_fails_when_more_districts_than_slots():
    df = _ranked_frame()
    with pytest.raises(ValueError, match="slots"):
        flow_audit.select_pilot(df, per_group=1, floor_col="District")


# --- missing-data validation selection -----------------------------------------

def _validation_frame():
    rows = []
    for g in ("a", "b"):
        for k in range(6):
            i = len(rows)
            rows.append(dict(array_index=i, proposed_client=g, eligible=True,
                             month_coverage_pct=[100.0, 99.5, 99.2, 98.0, 99.9, 99.0][k],
                             eval_missing_count=[0, 2, 0, 5, 1, 1][k]))
    return pd.DataFrame(rows)


def test_validation_selection_rules():
    df = _validation_frame()
    sel = flow_audit.select_missing_validation(df, per_group=2, exclude=[5])
    # Candidates per group: k=1 (99.5, 2 missing), k=4 (99.9, 1), k=5 (99.0, 1) minus excluded index 5.
    assert set(sel.loc[sel.proposed_client == "a", "array_index"]) == {1, 4}
    assert ((sel.month_coverage_pct >= 99.0) & (sel.month_coverage_pct < 100.0)).all()
    assert (sel.eval_missing_count >= 1).all() and 5 not in sel.array_index.tolist()
    assert (sel.selection_basis == "missing_validation").all()


def test_validation_selection_deterministic_and_fails_when_short():
    df = _validation_frame()
    a = flow_audit.select_missing_validation(df, per_group=2)
    b = flow_audit.select_missing_validation(df.sample(frac=1, random_state=2), per_group=2)
    assert a["array_index"].tolist() == b["array_index"].tolist()
    with pytest.raises(ValueError, match="validation candidates"):
        flow_audit.select_missing_validation(df, per_group=4)


def test_main_writes_disjoint_validation_manifest(tmp_path):
    n_per, districts = 6, [7, 4, 12, 3]
    a = _array(n_per * len(districts), seed=5)
    # Four sensors per client have one test-period gap. They stay primary-eligible
    # (training-only rules), so the primary pilot may take some; at least two remain.
    for s in range(0, len(a), n_per):
        a[s:s + 4, 36, FLOW] = np.nan
    ids = np.arange(1000, 1000 + len(a), dtype=np.int64)
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "arr.npy", a)
    np.save(raw / "node_order.npy", ids)
    pd.DataFrame(dict(station_id=ids, District=np.repeat(districts, n_per), County="c")).to_csv(
        raw / "meta.csv", sep="\t", index=False)
    out = tmp_path / "out"
    rc = flow_audit.main([
        "--array", str(raw / "arr.npy"), "--node-order", str(raw / "node_order.npy"),
        "--metadata", str(raw / "meta.csv"), "--out-dir", str(out), "--per-client", "2",
        "--validation-per-client", "1", "--validation-coverage", "90", "100",
        "--steps-per-day", str(SPD), "--no-shape-check",
    ])
    assert rc == 0
    primary = pd.read_csv(out / "sensor_selection.csv")
    val = pd.read_csv(out / "missing_validation_selection.csv")
    assert len(val) == 4 and set(val.array_index).isdisjoint(primary.array_index)
    assert (val.eval_missing_count >= 1).all()
    assert "Missing-data validation selection" in (out / "flow_quality_report.md").read_text()
