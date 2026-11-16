"""AIORI 2.0 Phase 7: persistence baseline scored with the runner's Evaluator (synthetic data)."""

import json

import numpy as np
import pytest
import torch

from aiori2 import run as R
from aiori2.baseline import persistence
from aiori2.data import ThreeYearWindows, raw_metrics
from test_aiori2 import synth  # noqa: F401


def test_stage2_configs_carry_each_algorithms_own_hyperparameters(tmp_path):
    from aiori2 import plan
    base = R.load_config(None, dict(out_root=str(tmp_path), batch_size=128))
    s2 = plan.planned(2, base, top=["nhits", "mlp"])
    eff = {c["algorithm"]: R.fl_config(c) for c in s2}                 # what the FL engine actually trains with
    assert eff["fedprox"]["mu"] == 0.01 and eff["fedavgm"]["server_momentum"] == 0.9
    assert all(eff[a]["server_lr"] == 0.01 for a in ("fedadam", "fedyogi", "fedadagrad"))
    assert eff["fedavg"]["mu"] == 0 and eff["fedavg"]["server_lr"] == 1.0 and eff["fedavg"]["server_momentum"] == 0
    s1 = {R.run_id(c) for c in plan.planned(1, base)}
    assert {R.run_id(c) for c in s2 if c["algorithm"] == "fedavg"} <= s1   # FedAvg ids unchanged -> Stage-1 reuse holds
    assert len({R.run_id(c) for c in s2}) == 64 and len({R.run_id(c) for c in s2} & s1) == 8


def test_stage3_retrains_recorded_finalists_with_new_seeds_and_budget_in_a_separate_root(tmp_path):
    from aiori2 import plan
    src, s3 = tmp_path / "s2", tmp_path / "s3"
    fin = plan.planned(2, R.load_config(None, dict(out_root=str(src), batch_size=128)), top=["nhits"])
    fin = next(c for c in fin if c["algorithm"] == "fedavgm" and c["history"] == "B")
    (src / R.run_id(fin)).mkdir(parents=True)
    (src / R.run_id(fin) / "run.json").write_text(json.dumps(dict(run_id=R.run_id(fin), status="completed", config=fin)))
    base = R.load_config(None, dict(out_root=str(s3), batch_size=128, device="cpu", budget=dict(epochs=20)))
    cfgs = plan.planned(3, base, finalists=[R.run_id(fin)], seeds=[20261027, 7], root=str(src))
    ids = [R.run_id(c) for c in cfgs]
    assert [c["seed"] for c in cfgs] == [20261027, 7] and len(set(ids)) == 2 and R.run_id(fin) not in ids
    assert all(c["budget"] == dict(epochs=20, windows_per_epoch=2_000_000) and c["out_root"] == str(s3) for c in cfgs)
    assert all(R.fl_config(c)["server_momentum"] == 0.9 and c["fl"] == fin["fl"] for c in cfgs)   # finalist settings kept


def test_test_split_evaluator_and_frozen_final_eval_reproduce_validation(synth, tmp_path):
    from aiori2 import final_eval
    from test_aiori2_phase6 import cfg as tiny
    with pytest.raises(ValueError):
        R.Evaluator(synth.out, tiny(), None, "train")
    ev = R.Evaluator(synth.out, tiny(), None, "test")
    assert np.array_equal(np.asarray(ev.ld.index), np.load(synth.out / "experiments" / "conservative" / "A" / "windows" / "test.npy"))
    c = tiny(arm="fedavg_geo", algorithm="fedavgm", validation=dict(max_windows=1500, final_full=True))
    info = R.Runner(c, root=synth.out, out_root=tmp_path / "runs").run()
    d = tmp_path / "runs" / info["run_id"]
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(dict(models=[dict(label="m", run_dir=str(d))], persistence=["A"])))
    sha = final_eval.freeze(sel, tmp_path / "man.json", root=synth.out)
    with pytest.raises(FileExistsError):
        final_eval.freeze(sel, tmp_path / "man.json", root=synth.out)
    with pytest.raises(ValueError):
        final_eval.evaluate(tmp_path / "man.json", "0" * 64, tmp_path / "res.json", "val", "cpu", synth.out)
    res = final_eval.evaluate(tmp_path / "man.json", sha, tmp_path / "res.json", "val", "cpu", synth.out)
    m = res["models"][0]
    assert m["checkpoint_epoch"] == info["best"]["epoch"] and m["n_valid"] == info["best"]["val_full"]["n_valid"]
    assert m["wape_pct"] == pytest.approx(info["best"]["val_full"]["wape_pct"], rel=1e-9) and res["same_target_set"]
    h = m["by_horizon"]
    assert sum(x["n_valid"] for x in h) == m["n_valid"]
    assert 100 * sum(x["mae"] * x["n_valid"] for x in h) / sum(x["wape_denominator"] for x in h) == pytest.approx(m["wape_pct"], rel=1e-9)
    with pytest.raises(FileExistsError):                                                # one evaluation per manifest
        final_eval.evaluate(tmp_path / "man.json", sha, tmp_path / "res.json", "val", "cpu", synth.out)


def test_persistence_repeats_the_last_input():
    X = torch.arange(24, dtype=torch.float32).reshape(2, 12, 1)
    assert torch.equal(persistence(X), torch.tensor([[11.0] * 12, [23.0] * 12]))


def test_persistence_scored_like_direct_masked_metrics(synth):
    cfg = R.load_config(None, dict(arm="centralised", history="B", device="cpu", validation=dict(batch_size=500)))
    ev = R.Evaluator(synth.out, cfg, None)
    out = ev.evaluate(persistence, torch.device("cpu"))["primary"]
    ld = ThreeYearWindows(synth.out, "conservative", "B", "val", "pooled")
    X, Y, M, s = ld.get(np.arange(len(ld)))
    pred = ld.inverse(np.repeat(X[:, -1:, 0], 12, 1), s)
    ref = raw_metrics(pred, np.where(M, Y, 0), M)
    assert out["n_valid"] == ref["n"] and abs(out["wape_pct"] - ref["wape_pct"]) < 1e-9
    # the scaler cancels: persistence equals the raw causally filled last input
    filled = np.load(synth.out / "experiments" / "conservative" / "B" / "filled_inputs.npy", mmap_mode="r")
    w = np.asarray(ld.index[:5]).astype(np.int64)
    assert np.allclose(pred[:5, 0], filled[w[:, 0], w[:, 1] - 1], atol=1e-3)
