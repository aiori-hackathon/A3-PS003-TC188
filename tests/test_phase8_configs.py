"""Phase 8 configs, the geo200 run entry and the Phase 6 runner's overwrite guard."""

import importlib.util
import json
from pathlib import Path

import pytest

from aiori import config, dataset

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "experiments" / "configs"
PHASE8 = {"phase8_centralised_b1.json": "phase6_b1.json", "phase8_local_b1.json": "phase7_local_b1.json",
          "phase8_fedavg_iid_b1.json": "phase7_fedavg_iid_b1.json",
          "phase8_fedavg_geo_b1.json": "phase7_fedavg_geo_b1.json"}
CHANGED = {"arm", "description", "run", "derived_from", "results_subdir"}


def _cfg(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,pilot", sorted(PHASE8.items()))
def test_phase8_configs_equal_pilot_settings(name, pilot):
    cfg, ref = _cfg(name), _cfg(pilot)
    assert cfg["derived_from"] == pilot and cfg["run"] == "geo200" and cfg["results_subdir"] == "phase8"
    assert cfg["arm"] == ref["arm"] + "_geo200"
    assert {k: v for k, v in cfg.items() if k not in CHANGED} == {k: v for k, v in ref.items() if k not in CHANGED}


def test_phase8_fedavg_settings_match_d8_7():
    for name in ("phase8_fedavg_iid_b1.json", "phase8_fedavg_geo_b1.json"):
        cfg = _cfg(name)
        assert cfg["optimizer_state"] == "reset_each_round" and cfg["client_fraction"] == 1.0
        assert cfg["local_epochs"] == 1 and cfg["rounds_max"] == 50 and cfg["threads"] == 8
    assert _cfg("phase8_fedavg_iid_b1.json")["partition_seed"] == 20261027
    assert len({_cfg(n)["arm"] for n in PHASE8}) == 4


def test_geo200_run_entry_is_additive_and_absent_data_is_not_created():
    assert dataset.RUNS["geo200"] == (config.PROCESSED_DIR / "geo200", config.PROCESSED_DIR / "geo200" / "sensor_selection.csv")
    assert dataset.RUNS["primary"][0] == config.PROCESSED_DIR
    assert dataset.RUNS["missing_validation"][0] == config.PROCESSED_DIR / "missing_validation"
    if not (config.PROCESSED_DIR / "geo200").exists():
        with pytest.raises(FileNotFoundError):
            dataset.load_run("geo200", splits=("train", "val"))
        assert not (config.PROCESSED_DIR / "geo200").exists()


def _runner():
    spec = importlib.util.spec_from_file_location("run_phase6", ROOT / "experiments" / "run_phase6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("cfg,run,folder", [("phase8_centralised_b1.json", "geo200", "phase8"),
                                            ("phase6_b1.json", "primary", "phase6")])
def test_phase6_runner_default_root_and_overwrite_guard(tmp_path, monkeypatch, cfg, run, folder):
    pytest.importorskip("torch")
    runner = _runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.dataset, "load_run", lambda *a, **k: pytest.fail("data loaded"))
    existing = tmp_path / "results" / folder / run / f"{_cfg(cfg)['arm']}_seed0_smoke"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text("{}")
    with pytest.raises(SystemExit, match="overwrite"):
        runner.main(["--config", str(CONFIGS / cfg), "--run", run, "--smoke"])


def test_phase6_runner_guard_with_explicit_out_dir(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    runner = _runner()
    monkeypatch.setattr(runner.dataset, "load_run", lambda *a, **k: pytest.fail("data loaded"))
    existing = tmp_path / "primary" / "centralised_b1_seed1"
    existing.mkdir(parents=True)
    (existing / "best_model.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="overwrite"):
        runner.main(["--config", str(CONFIGS / "phase6_b1.json"), "--seed", "1", "--out-dir", str(tmp_path)])
