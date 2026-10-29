"""Phase 9 resumable trainers: identical to the Phase 8 loops, and interrupt + resume == uninterrupted."""

import time

import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, federated, resumable, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402
from conftest import make_run  # noqa: E402

CFG = {"calendar": True, "hidden_size": 16, "lr": 1e-3, "batch_size": 16, "grad_clip": 1.0, "local_epochs": 1,
       "rounds_max": 6, "max_epochs": 6, "patience": 2, "optimizer_state": "reset_each_round"}


@pytest.fixture(autouse=True)
def _torch():
    training.configure_torch(2)


@pytest.fixture
def run4():
    return make_run(n_sensors=4, clients=["n", "n", "s", "s"])


def _eq(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _hist(h):
    if isinstance(h, dict):
        return {k: _hist(v) for k, v in h.items() if "seconds" not in k}
    if isinstance(h, list):
        return [_hist(v) for v in h]
    return h


def _central(run, resum=None, **kw):
    training.seed_everything(3)
    m = ForecastGRU(4, 16)
    tr, va = dataset.WindowDataset(run, "train", calendar=True), dataset.WindowDataset(run, "val", calendar=True)
    if resum is None:
        r = training.fit(m, tr, va, run, seed=3, lr=1e-3, batch_size=16, max_epochs=6, patience=2, grad_clip=1.0)
    else:
        r = resumable.fit(m, tr, va, run, seed=3, ckpt=resum, lr=1e-3, batch_size=16, max_epochs=6, patience=2, grad_clip=1.0, **kw)
    return m.state_dict(), r


def test_fit_matches_training_fit(run4, tmp_path):
    a, ra = _central(run4)
    b, rb = _central(run4, tmp_path / "c.pt")
    assert _eq(a, b) and _hist(ra) == _hist(rb)


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_fit_interrupt_and_resume_is_exact(run4, tmp_path, cut):
    a, ra = _central(run4, tmp_path / "full.pt")
    with pytest.raises(resumable.Interrupted):
        _central(run4, tmp_path / "part.pt", stop_after=cut)
    b, rb = _central(run4, tmp_path / "part.pt")                 # fresh process-like restart: new model, same checkpoint
    assert _eq(a, b) and _hist(ra) == _hist(rb)
    c, rc = _central(run4, tmp_path / "part.pt")                 # resuming a finished run changes nothing
    assert _eq(a, c) and _hist(ra) == _hist(rc)


def test_fedavg_matches_and_resumes_exactly(run4, tmp_path):
    part = federated.geographic_partition(run4)
    ref = federated.run_fedavg(run4, part, CFG, seed=2)
    full = resumable.fedavg(run4, part, CFG, 2, ckpt=tmp_path / "f.pt")
    assert _eq(ref["state"], full["state"]) and _hist(ref["history"]) == _hist(full["history"]) and ref["byte_log"] == full["byte_log"]
    with pytest.raises(resumable.Interrupted):
        resumable.fedavg(run4, part, CFG, 2, ckpt=tmp_path / "p.pt", stop_after=2)
    res = resumable.fedavg(run4, part, CFG, 2, ckpt=tmp_path / "p.pt")
    assert _eq(ref["state"], res["state"]) and _hist(ref["history"]) == _hist(res["history"]) and ref["byte_log"] == res["byte_log"]
    for k in ("y_hat_z", "mask", "sensor_pos", "t0"):
        assert (ref["predictions"][k] == res["predictions"][k]).all()


def test_local_only_matches_and_resumes_across_clients(run4, tmp_path):
    part = federated.geographic_partition(run4)
    ref = federated.run_local_only(run4, part, CFG, seed=1)
    with pytest.raises(resumable.Interrupted):
        resumable.local_only(run4, part, CFG, 1, ckpt_dir=tmp_path, stop_after=4)   # interrupts inside the second client
    res = resumable.local_only(run4, part, CFG, 1, ckpt_dir=tmp_path)
    assert all(_eq(ref["states"][c], res["states"][c]) for c in part)
    assert _hist({c: v["history"] for c, v in ref["clients"].items()}) == _hist({c: v["history"] for c, v in res["clients"].items()})


def test_deadline_stops_before_training_and_atomic_save(run4, tmp_path):
    with pytest.raises(resumable.Interrupted, match="deadline"):
        _central(run4, tmp_path / "d.pt", deadline=time.time() - 1)
    assert not (tmp_path / "d.pt").exists()
    resumable.atomic_save({"x": 1}, tmp_path / "a" / "s.pt")
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["s.pt"]
