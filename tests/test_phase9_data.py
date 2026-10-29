"""Phase 9 nine-month data module on synthetic data only."""

import json

import numpy as np
import pandas as pd
import pytest

from aiori import nine_month as nm


def test_split_bounds_match_the_required_calendar():
    sp = nm.split_bounds()
    assert sp == {"train": (0, 55008), "val": (55008, 63072), "test": (63072, 78624)}
    assert [p.name for p in nm.month_paths("y")] == [f"2023_p{m:02d}.npy" for m in range(1, 10)]


def test_eligibility_reasons():
    df = pd.DataFrame(dict(fully_missing=[False, True, False, False, False, False], constant=[False, False, True, False, False, False],
                           duplicate_series=[False, False, False, True, False, False], coverage_pct=[100, 0, 100, 100, 98.9, 100],
                           zero_pct=[0, 0, 0, 0, 0, 51]))
    e = nm.eligibility(df)
    assert e["exclusion_reason"].tolist() == ["", "fully_missing", "constant", "exact_copy", "low_coverage", "too_many_zeros"]
    assert e["eligible"].tolist() == [True, False, False, False, False, False]


def test_ordering_check_detects_a_shuffled_month():
    rng = np.random.default_rng(0)
    base = rng.gamma(2, 50, 500)
    mm = np.stack([base * rng.uniform(0.95, 1.05, 500) for _ in range(9)], axis=1)
    extra = dict(month_mean=mm, first_val=mm.copy(), last_val=mm.copy(), step_diff_med=np.full(500, 5.0))
    assert nm.ordering_checks(extra)["passed"]
    mm2 = mm.copy()
    mm2[:, 5] = rng.permutation(mm2[:, 5])
    assert not nm.ordering_checks(dict(extra, month_mean=mm2))["passed"]


def test_dst_check_finds_a_one_hour_shift_and_none():
    day = np.sin(2 * np.pi * np.arange(288) / 288) + 0.3 * np.cos(4 * np.pi * np.arange(288) / 288)
    month = np.tile(day, 31)
    assert nm.dst_check(month)["best_lag_steps"] == 0
    shifted = month.copy()
    shifted[12 * 288:] = np.roll(month, 12)[12 * 288:]
    assert nm.dst_check(shifted)["best_lag_steps"] == -12


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(1)
    n, t = 8, 273 * 288
    flow = np.maximum(50 + 20 * np.sin(2 * np.pi * np.arange(t) / 288)[None] + rng.normal(0, 3, (n, t)), 0)
    flow[2, 100:110] = np.nan
    flow[4, 60000:60005] = np.nan
    flow[6, 64000:64010] = np.nan
    flow[:, 55008 - 3: 55008 + 3] *= 1.0
    sel = pd.DataFrame(dict(array_index=np.arange(n), proposed_client=np.repeat(["a", "b", "c", "d"], 2)))
    counts = nm.build_checkpoint(flow, sel, np.arange(100, 100 + n), np.repeat([7, 4, 12, 8], 2), tmp_path / "ck")
    return tmp_path / "ck", flow, counts


def test_checkpoint_windows_stay_inside_their_split_and_scaler_is_train_only(checkpoint):
    out, flow, counts = checkpoint
    sp = nm.split_bounds()
    w = np.load(out / "window_index.npz")
    for name, (s, e) in sp.items():
        t0 = w[name][:, 1]
        assert ((t0 - 12 >= s) & (t0 + 12 <= e)).all() and len(t0) == counts[name]
    sc = json.loads((out / "scaler_parameters.json").read_text())
    assert sc["fit_range"] == [0, 55008]
    assert np.allclose(sc["mean"], np.nanmean(flow[:, :55008], axis=1))
    d = np.load(out / "pilot_preprocessed.npz")
    assert np.array_equal(np.isnan(d["targets_scaled"]), np.isnan(flow)) and np.isfinite(d["inputs_scaled"]).all()
    changed = flow.copy()
    changed[:, 55008:] += 500                                      # val/test values cannot move the scaler
    nm.build_checkpoint(changed, pd.read_csv(out / "sensor_selection.csv")[["array_index", "proposed_client"]],
                        np.arange(100, 108), np.repeat([7, 4, 12, 8], 2), out.parent / "ck2")
    assert json.loads((out.parent / "ck2" / "scaler_parameters.json").read_text())["mean"] == sc["mean"]


def test_selection_retains_eligible_and_replaces_deterministically():
    rng = np.random.default_rng(2)
    n = 400
    clients = np.array(["c0", "c1", "c2", "c3"])[np.arange(n) % 4]
    elig = pd.DataFrame(dict(array_index=np.arange(n), proposed_client=clients, eligible=rng.random(n) > 0.2,
                             exclusion_reason="", mean=rng.gamma(2, 50, n)))
    elig.loc[~elig["eligible"], "exclusion_reason"] = "low_coverage"
    types = pd.Series(rng.choice(["Mainline", "HOV", "On Ramp"], n))
    jan_idx = np.concatenate([np.flatnonzero(clients == c)[:50] for c in ("c0", "c1", "c2", "c3")])
    jan = pd.DataFrame(dict(array_index=jan_idx, proposed_client=clients[jan_idx], type_group="mainline", flow_tertile="mid",
                            District=7, station_id=jan_idx))
    pilot = list(range(300, 320))
    sel, log = nm.select(elig, types, jan, pilot)
    sel2, _ = nm.select(elig, types, jan, pilot)
    pd.testing.assert_frame_equal(sel, sel2)
    kept = set(jan_idx[elig.set_index("array_index").loc[jan_idx, "eligible"].to_numpy()])
    assert kept <= set(sel["array_index"]) and len(log) == 200 - len(kept)
    new = set(sel["array_index"]) - kept
    assert not new & set(pilot) and not new & set(jan_idx) and elig.set_index("array_index").loc[list(new), "eligible"].all()
    assert (sel["proposed_client"].value_counts() == 50).all()
    assert all(clients[i] == c for i, c in zip(sel["array_index"], sel["proposed_client"]))
