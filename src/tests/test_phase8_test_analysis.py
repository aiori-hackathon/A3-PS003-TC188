"""Phase 8 E4: test-analysis script (scripts/phase8_test_analysis.py) on a synthetic project only."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from aiori import phase8_analysis as pa  # noqa: E402
from test_phase8_test_eval import ROOT, _freeze, _load_script, base  # noqa: E402,F401  (base is a fixture)

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
ARMS = ("centralised", "local", "fedavg_iid", "fedavg_geo")


def _validation_s5(root, sesoi=0.1638):
    an = root / "results/phase8/analysis"
    (an / "sesoi_log.json").write_text(json.dumps({"sesoi_mae": sesoi}))
    verdicts = ["detectable but below the practical threshold", "not distinguishable on this validation set",
                "not distinguishable on this validation set", "supported on January validation data"]
    (an / "analysis.json").write_text(json.dumps({
        "primary": {c: dict(diff=0.1, reported_ci95=[0.05, 0.15], p_holm=0.01, verdict=v) for c, v in zip(pa.CONTRASTS, verdicts)},
        "descriptive": {"pooled_mae_seed_avg": {a: 10.0 + i for i, a in enumerate(ARMS)}}}))


def _refreeze(root):
    """Freeze manifest with the S5 analysis hash, matching E1 record; returns evaluator argv."""
    argv = _freeze(root, None)
    tp = root / "results/phase8/test_protocol"
    fz = json.loads((tp / "frozen_checkpoints.json").read_text())
    fz["analysis_json_sha256"] = sha(root / "results/phase8/analysis/analysis.json")
    (tp / "frozen_checkpoints.json").write_text(json.dumps(fz))
    e1 = json.loads((tp / "e1_record.json").read_text())
    e1["freeze_sha256"] = sha(tp / "frozen_checkpoints.json")
    (tp / "e1_record.json").write_text(json.dumps(e1))
    argv[3] = e1["freeze_sha256"]
    return argv


def _e4_record(root, argv):
    (root / "results/phase8/test_protocol/e4_record.json").write_text(json.dumps(dict(
        script_sha256=sha(ROOT / "scripts/phase8_test_analysis.py"), protocol_sha256=argv[1], freeze_sha256=argv[3],
        dependency_sha256={"aiori/frozen_code.py": sha(root / "aiori/frozen_code.py")})))


def _args(root, argv):
    return ["--protocol-sha256", argv[1], "--freeze-sha256", argv[3], "--root", str(root), "--data-dir", str(root / "data/geo200"),
            "--metadata", str(root / "data/meta_fwy.csv")]


@pytest.fixture(scope="module")
def completed(base, tmp_path_factory):
    """Synthetic project after a real E2 + E3 pass of the frozen evaluator."""
    root = tmp_path_factory.mktemp("p8an") / "p"
    shutil.copytree(base, root)
    _validation_s5(root)
    rng = np.random.default_rng(0)
    pd.DataFrame({"Fwy": rng.choice(["5", "10", "405"], 80), "Direction": rng.choice(["N", "S"], 80)}).to_csv(root / "data/meta_fwy.csv", index=False)
    argv = _refreeze(root)
    ev = _load_script("phase8_test_eval")
    assert ev.main(["--mode", "precheck", *argv]) == 0 and ev.main(["--mode", "test", *argv]) == 0
    _e4_record(root, argv)
    return root, argv


@pytest.fixture
def proj(completed, tmp_path):
    root = tmp_path / "q"
    shutil.copytree(completed[0], root)
    return root, completed[1]


@pytest.fixture
def an():
    return _load_script("phase8_test_analysis")


def test_full_analysis_on_synthetic_test_records(proj, an, monkeypatch):
    root, argv = proj
    monkeypatch.setattr(an.splits, "make_splits", lambda *a, **k: pytest.fail("no data re-split"))
    import aiori.dataset as ds
    monkeypatch.setattr(ds, "load_run", lambda *a, **k: pytest.fail("analysis must not load any dataset split"))
    assert an.main(_args(root, argv)) == 0
    r = json.loads((root / "results/phase8/test/analysis_test.json").read_text())
    assert set(r["primary"]) == set(pa.CONTRASTS) and r["sesoi"] == pytest.approx(0.1638)
    for cid, p in r["primary"].items():
        assert p["verdict"] in an.WORDING.values() and p["verdict"].endswith("January test data")
        assert p["reported_ci95"] == p[p["reported_interval"]]["ci95"]
        assert p["p_conservative"] == max(p["sensor"]["p"], p["freeway"]["p"])
        assert p["consistent_with_validation"] == (p["category"] == p["validation"]["category"])
    assert np.allclose([r["primary"][c]["p_holm"] for c in pa.CONTRASTS],
                       pa.holm([r["primary"][c]["p_conservative"] for c in pa.CONTRASTS]))
    d = r["descriptive"]
    assert [v["date"] for v in d["days"].values()] == [f"2023-01-{x}" for x in range(26, 32)]
    assert [v["weekend"] for v in d["days"].values()] == [False, False, True, True, False, False]
    assert set(r["leave_one_day_out"]) == {f"2023-01-{x}" for x in range(26, 32)}
    assert set(d["weekday_weekend_mae"]["centralised"]) == {"weekday", "weekend"}
    assert set(d["reference_forecasts"]) == {"persistence", "seasonal_naive_daily", "historical_average"}
    assert d["validation_vs_test"]["local"]["validation_mae"] == 11.0
    assert (root / "results/phase8/test/summary_test.md").read_text().count("January test data") >= 4


def test_point_estimates_equal_direct_recomputation(proj, an):
    root, argv = proj
    assert an.main(_args(root, argv)) == 0
    r = json.loads((root / "results/phase8/test/analysis_test.json").read_text())
    per_arm = {}
    for a in ARMS:
        e = []
        for s in (0, 1, 2):
            with np.load(next((root / f"results/phase8/test/{a}_b1_geo200_seed{s}").glob("predictions_*"))) as z:
                e.append(np.abs(z["y_hat"].astype(float) - z["y_true"].astype(float))[z["mask"]])
        per_arm[a] = np.mean(e, axis=0).mean()
        assert r["descriptive"]["pooled_mae"][a] == pytest.approx(per_arm[a], rel=1e-9)
    for cid, (a, b) in pa.CONTRASTS.items():
        assert r["primary"][cid]["diff"] == pytest.approx(per_arm[a] - per_arm[b], rel=1e-9, abs=1e-12)
    wk = r["descriptive"]["weekday_weekend_mae"]["centralised"]
    days = r["descriptive"]["day_mae"]["centralised"]
    assert min(days.values()) <= wk["weekend"] <= max(days.values())


@pytest.mark.parametrize("tamper", ["no_e3", "sesoi_log", "s5_analysis", "script_record", "dependency", "e3_log", "protocol"])
def test_gate_fails_closed(proj, an, tamper):
    root, argv = proj
    args = _args(root, argv)
    if tamper == "no_e3":
        (root / "results/phase8/test/RUN_COMPLETED").unlink()
    elif tamper == "sesoi_log":
        (root / "results/phase8/analysis/sesoi_log.json").write_text('{"sesoi_mae": 0.2}')
    elif tamper == "s5_analysis":
        p = root / "results/phase8/analysis/analysis.json"
        p.write_text(p.read_text() + " ")
    elif tamper == "script_record":
        p = root / "results/phase8/test_protocol/e4_record.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), script_sha256="0" * 64)))
    elif tamper == "dependency":
        (root / "aiori/frozen_code.py").write_text("# edited\n")
    elif tamper == "e3_log":
        p = root / "results/phase8/test/run_log.json"
        p.write_text(json.dumps(dict(json.loads(p.read_text()), script_sha256="0" * 64)))
    else:
        args[1] = "0" * 64
    with pytest.raises(SystemExit, match="Refusing"):
        an.main(args)
    assert not (root / "results/phase8/test/analysis_test.json").exists()


def test_sesoi_must_equal_the_protocol_value(base, tmp_path, an):
    root = tmp_path / "s"
    shutil.copytree(base, root)
    _validation_s5(root, sesoi=0.2)
    pd.DataFrame({"Fwy": ["5"] * 80, "Direction": ["N"] * 80}).to_csv(root / "data/meta_fwy.csv", index=False)
    argv = _refreeze(root)
    ev = _load_script("phase8_test_eval")
    assert ev.main(["--mode", "precheck", *argv]) == 0 and ev.main(["--mode", "test", *argv]) == 0
    _e4_record(root, argv)
    with pytest.raises(SystemExit, match="SESOI"):
        an.main(_args(root, argv))


def test_runs_once(proj, an):
    root, argv = proj
    assert an.main(_args(root, argv)) == 0
    with pytest.raises(SystemExit, match="overwrite"):
        an.main(_args(root, argv))


def test_wording_covers_every_s5_verdict_and_day_labels(an):
    for ci, p, diff in (([0.1, 0.5], 0.001, 0.3), ([0.05, 0.1], 0.001, 0.07), ([-0.1, 0.5], 0.5, 0.2)):
        v = pa.verdict(ci, p, diff, 0.1638)
        assert v in an.WORDING and v in an.CATEGORY
    lab = an.day_labels(range(25, 31), 288)
    assert [lab[d]["weekday"] for d in range(25, 31)] == ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue"]
