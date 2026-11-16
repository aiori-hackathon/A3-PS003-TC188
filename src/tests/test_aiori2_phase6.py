"""AIORI 2.0 Phase 6: experiment runner, tracking and planner (tiny synthetic runs; no real training)."""

import json
import re
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from aiori2 import arms, core, plan, track
from aiori2 import run as R
from aiori2.data import ThreeYearWindows, raw_metrics
from test_aiori2 import synth  # noqa: F401

TINY = dict(budget=dict(epochs=3, windows_per_epoch=512), batch_size=32, device="cpu",
            validation=dict(every_epochs=1, max_windows=1500, batch_size=700, final_full=False, sensitivity_mask=False))


def cfg(**kw):
    base = R._merge(R.DEFAULTS, TINY)
    base.pop("note", None)
    if kw.get("arm") in ("fedavg_iid", "fedavg_geo"):
        base = R._merge(base, dict(fl=dict(local_steps=2)))
    return R.load_config(None, R._merge(base, kw))


def runner(synth, tmp_path, **kw):
    return R.Runner(cfg(**kw), root=synth.out, out_root=tmp_path / "runs")


def states(r):
    return {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in r.models.items()}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[n][k], b[n][k]) for n in a for k in a[n])


# ---- configuration and identity ----

@pytest.mark.parametrize("bad", [dict(arm="centralised", algorithm="fedavg"), dict(arm="local", algorithm="fedprox"),
                                 dict(arm="fedavg_iid", algorithm=None), dict(arm="fedavg_geo", algorithm="scaffold", fl=dict(client_optimizer="adam")),
                                 dict(arm="fedavg_iid", algorithm="fednova", fl=dict(mu=0.1)), dict(arm="fedavg_iid", algorithm="fedavg", weight_decay=0.01),
                                 dict(budget=dict(epochs=0)), dict(device="tpu"), dict(model="stgcn"), dict(optimizer="lbfgs"), dict(arm="everything")])
def test_invalid_configurations_rejected(bad):
    with pytest.raises((ValueError, KeyError)):
        cfg(**bad)


def test_run_ids_are_unique_and_stable():
    a, b = cfg(), cfg(lr=0.002)
    assert R.run_id(a) == R.run_id(cfg()) and R.run_id(a) != R.run_id(b)
    assert R.run_id(a) == R.run_id(cfg(device="cuda", out_root="elsewhere"))      # device/output do not define the run
    assert R.run_id(cfg(seed=1)) != R.run_id(a)
    assert re.fullmatch(r"[A-Za-z0-9_.\-]+", R.run_id(cfg(arm="fedavg_geo", algorithm="scaffold")))


# ---- training/evaluation correctness ----

def test_streaming_validation_equals_direct_masked_metrics(synth, tmp_path):
    r = runner(synth, tmp_path)
    r._setup()
    ev = R.Evaluator(synth.out, r.cfg, None)
    model = r.models["all"]
    out = ev.evaluate(lambda X, s: r.predict(X, s), r.device)
    ld = ThreeYearWindows(synth.out, "conservative", "A", "val", "pooled")
    X, Y, M, s = ld.get(np.arange(len(ld)))
    with torch.no_grad():
        pred = ld.inverse(model.eval()(torch.as_tensor(X)).numpy(), s)
    ref = raw_metrics(pred, np.where(M, Y, 0), M)
    p = out["primary"]
    assert p["n_valid"] == ref["n"] and p["mae"] == pytest.approx(ref["mae"], rel=1e-9)
    assert p["rmse"] == pytest.approx(ref["rmse"], rel=1e-9) and p["wape_pct"] == pytest.approx(ref["wape_pct"], rel=1e-9)
    assert p["wape_denominator"] == pytest.approx(float(np.abs(Y[M].astype(np.float64)).sum()), rel=1e-9)
    n_clients = sum(x["n_valid"] for x in p["by_client_geo"].values())
    assert n_clients == p["n_valid"]


def test_validation_population_is_shared_by_arms_and_histories(synth, tmp_path):
    pos = {}
    for arm, alg in (("centralised", None), ("local", None), ("fedavg_iid", "fedavg"), ("fedavg_geo", "fedavg")):
        for h in ("A", "B"):
            c = cfg(arm=arm, algorithm=alg, history=h)
            ev = R.Evaluator(synth.out, c, 1500)
            w = np.asarray(ev.ld.index[ev.positions]).astype(np.int64)
            pos[(arm, h)] = np.stack([w[:, 0], w[:, 1] + ev.ld.off], 1)
    ref = next(iter(pos.values()))
    assert all(np.array_equal(ref, v) for v in pos.values())


def test_training_never_touches_test_data(synth, tmp_path, monkeypatch):
    seen = []
    orig = ThreeYearWindows.__init__

    def spy(self, root, policy, experiment, split, *a, **k):
        seen.append(split)
        orig(self, root, policy, experiment, split, *a, **k)
    monkeypatch.setattr(ThreeYearWindows, "__init__", spy)
    for arm, alg in (("centralised", None), ("local", None), ("fedavg_geo", "scaffold")):
        runner(synth, tmp_path, arm=arm, algorithm=alg, budget=dict(epochs=1)).run()
    assert "test" not in seen and {"train", "val"} <= set(seen)


@pytest.mark.parametrize("model", ["mlp", "gru", "tcn", "patchtst", "nhits"])
def test_every_model_runs_through_the_runner(synth, tmp_path, model):
    info = runner(synth, tmp_path, model=model, budget=dict(epochs=1)).run()
    assert info["status"] == "completed" and np.isfinite(info["best"]["val"]["wape_pct"])
    assert info["counters"]["windows"] == 512 and info["counters"]["steps"] == 16


def test_local_models_are_independent(synth, tmp_path):
    full = runner(synth, tmp_path, arm="local", budget=dict(epochs=1))
    full.run()
    keep = sorted(full.models)[2]
    solo = runner(synth, tmp_path / "solo", arm="local", budget=dict(epochs=1))
    solo._setup()
    solo.train = {keep: solo.train[keep]}
    solo.models = {keep: solo.models[keep]}
    solo.opts = {keep: solo.opts[keep]}
    solo.cfg["budget"]["windows_per_epoch"] = 512 // 4                 # same per-client budget as in the full run
    solo._epoch_central_local(1)
    assert all(torch.equal(full.models[keep].state_dict()[k], solo.models[keep].state_dict()[k]) for k in solo.models[keep].state_dict())
    assert len({tuple(m.state_dict()["net.head.net.weight"].flatten()[:3].tolist()) for m in full.models.values()}) == 4


def test_local_and_fl_predictions_use_each_sensors_scaler_and_model(synth, tmp_path):
    r = runner(synth, tmp_path, arm="local")
    r._setup()
    X = torch.randn(6, 12, 1)
    s = np.array([r.train[n].sensors[0] for n in sorted(r.train)] + [r.train[sorted(r.train)[0]].sensors[1]] * 2)
    out = r.predict(X, s)
    for i, si in enumerate(s):
        with torch.no_grad():
            assert torch.allclose(out[i], r.models[r.owner[si]].eval()(X[i:i + 1])[0], atol=1e-5)   # batched vs single-row GEMM
    ev = R.Evaluator(synth.out, r.cfg, 10)
    sc = json.loads((synth.out / "experiments" / "conservative" / "A" / "scalers.json").read_text())["scopes"]["client_geo"]
    assert np.allclose(ev.ld.mean, sc["mean"])


# ---- determinism, resume and checkpoint safety ----

def test_same_seed_is_repeatable_on_cpu(synth, tmp_path):
    a = runner(synth, tmp_path / "a", budget=dict(epochs=2)); a.run()
    b = runner(synth, tmp_path / "b", budget=dict(epochs=2)); b.run()
    assert same(states(a), states(b))
    c = runner(synth, tmp_path / "c", budget=dict(epochs=2), seed=7); c.run()
    assert not same(states(a), states(c))


@pytest.mark.parametrize("arm, alg", [("centralised", None), ("local", None), ("fedavg_iid", "fedadam"), ("fedavg_geo", "scaffold"),
                                      ("fedavg_geo", "fednova")])
def test_interrupted_and_resumed_equals_uninterrupted(synth, tmp_path, arm, alg):
    kw = dict(arm=arm, algorithm=alg, fl=dict(server_lr=0.01) if alg == "fedadam" else {})
    full = runner(synth, tmp_path / "full", **kw)
    info_full = full.run()
    part = runner(synth, tmp_path / "part", **kw)
    assert part.run(stop_after_epochs=1)["status"] == "interrupted"
    with pytest.raises(FileExistsError):
        runner(synth, tmp_path / "part", **kw).run()                   # no silent restart
    res = runner(synth, tmp_path / "part", **kw)
    info_res = res.run(resume=True)
    assert info_res["status"] == info_full["status"] and len(info_res["resumes"]) == 1
    assert same(states(full), states(res))
    if alg:
        for k, v in full.fl.alg.state.items():
            assert all(torch.equal(v[n], res.fl.alg.state[k][n]) for n in v)
    for a, b in zip(info_full["history"], info_res["history"]):
        assert a["val"]["wape_pct"] == b["val"]["wape_pct"] and a["steps"] == b["steps"]
    assert info_full["counters"]["steps"] == info_res["counters"]["steps"]


def test_completed_runs_are_skipped_and_foreign_dirs_refused(synth, tmp_path):
    r = runner(synth, tmp_path, budget=dict(epochs=1))
    info = r.run()
    again = runner(synth, tmp_path, budget=dict(epochs=1)).run()
    assert again["status"] == "completed" and again["started"] == info["started"]
    j = core.load_json(r.dir / "run.json")
    j["config_sha256"] = "0" * 64
    core.save_json(r.dir / "run.json", j)
    with pytest.raises(ValueError):
        runner(synth, tmp_path, budget=dict(epochs=1)).run(resume=True)
    with pytest.raises(FileNotFoundError):
        runner(synth, tmp_path / "none").run(resume=True)


def test_checkpoints_are_atomic_and_safe_to_load(synth, tmp_path):
    r = runner(synth, tmp_path, arm="fedavg_geo", algorithm="scaffold", budget=dict(epochs=2))
    r.run()
    assert not list(r.dir.glob("*.partial"))
    for name in ("latest.pt", "best.pt"):
        ck = torch.load(r.dir / name, map_location="cpu", weights_only=True)
        meta = json.loads(ck["meta_json"])
        assert meta["config_sha256"] == r.hash and "c" in ck["fl_server"] and ck["fl_round"] >= 1
    other = runner(synth, tmp_path, arm="fedavg_geo", algorithm="scaffold", budget=dict(epochs=2), seed=3)
    other._setup()
    with pytest.raises(ValueError):
        other._load_state(r.dir / "latest.pt")


def test_early_stopping(synth, tmp_path):
    info = runner(synth, tmp_path, budget=dict(epochs=6), early_stopping=dict(patience=1, min_delta=1e9)).run()
    assert info["status"] == "early_stopped" and len(info["history"]) == 2 and info["best"]["epoch"] == 1


def test_fl_accounting_is_recorded(synth, tmp_path):
    info = runner(synth, tmp_path, arm="fedavg_iid", algorithm="fedavg", budget=dict(epochs=1)).run()
    c = info["counters"]
    assert c["rounds"] == 2 and c["steps"] == 4 * 2 * 2 and c["upload_bytes"] > 0 and c["download_bytes"] > 0   # ceil(512/(4*2*32)) = 2 rounds


# ---- tracking and planner ----

def test_tracking_summary_and_stale_status(synth, tmp_path):
    runner(synth, tmp_path, budget=dict(epochs=1)).run()
    runner(synth, tmp_path, arm="local", budget=dict(epochs=1)).run()
    df = track.summary(tmp_path / "runs")
    assert len(df) == 2 and set(df.status) == {"completed"} and df.val_wape_pct.notna().all()
    lines = (tmp_path / "runs" / "registry.jsonl").read_text().strip().splitlines()
    assert sum(json.loads(l)["event"] == "start" for l in lines) == 2
    old = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    assert track.status(dict(status="running", heartbeat=old)) == "interrupted"
    assert track.status(dict(status="running", heartbeat=datetime.now().isoformat(timespec="seconds"))) == "running"


def test_plan_sizes_and_deduplication(tmp_path):
    base = R.load_config(None, dict(out_root=str(tmp_path)))
    s1 = plan.planned(1, base)
    s2 = plan.planned(2, base, top=["mlp", "gru"])
    ids1, ids2 = {R.run_id(c) for c in s1}, {R.run_id(c) for c in s2}
    assert len(s1) == len(ids1) == 40 and len(s2) == len(ids2) == 64
    assert len(ids1 & ids2) == 8                                       # Stage-2 FedAvg configs reuse Stage-1 runs
    assert all(c["algorithm"] is None for c in s1 if c["arm"] in ("centralised", "local"))
    with pytest.raises(RuntimeError):
        plan.top_models(base, root=tmp_path)                          # Stage 1 not run: no selection possible
    assert len(plan._only(s1, "arm=local,history=B")) == 5


def test_new_code_is_portable():
    for name in ("run.py", "plan.py", "track.py"):
        t = (core.SRC / "aiori2" / name).read_text(encoding="utf-8")
        assert "wintypes" not in t and "powershell" not in t.lower() and not re.search(r"[A-Za-z]:\\\\", t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this environment")
def test_cuda_run(synth, tmp_path):
    info = runner(synth, tmp_path, device="cuda", arm="fedavg_iid", algorithm="fedavg", budget=dict(epochs=1)).run()
    assert info["status"] == "completed" and info["env"]["device"].startswith("cuda") and info["peak_gpu_mb"] > 0
