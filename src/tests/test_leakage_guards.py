import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori import dataset, training  # noqa: E402
from aiori.model import ForecastGRU  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "experiments" / "configs" / "phase6_b0.json"


def _runner():
    spec = importlib.util.spec_from_file_location("run_phase6", ROOT / "experiments" / "run_phase6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("train_split, val_split", [("val", "val"), ("test", "val"), ("train", "test")])
def test_fit_rejects_non_train_updates_and_non_val_stopping(synthetic_run, train_split, val_split):
    with pytest.raises(ValueError):
        training.fit(ForecastGRU(2), dataset.WindowDataset(synthetic_run, train_split),
                     dataset.WindowDataset(synthetic_run, val_split), synthetic_run, seed=0, max_epochs=1)


def test_only_train_windows_are_built_with_gradients_enabled(synthetic_run, monkeypatch):
    calls = []
    real = dataset.WindowDataset.get_batch

    def spy(self, rows):
        calls.append((self.split, torch.is_grad_enabled(), real(self, rows)["t0"].numpy()))
        return real(self, rows)

    monkeypatch.setattr(dataset.WindowDataset, "get_batch", spy)
    training.configure_torch(2)
    training.fit(ForecastGRU(2), dataset.WindowDataset(synthetic_run, "train"),
                 dataset.WindowDataset(synthetic_run, "val"), synthetic_run,
                 seed=0, batch_size=16, max_epochs=1, max_batches=3)
    grad_calls = [c for c in calls if c[1]]
    assert grad_calls and all(split == "train" for split, _, _ in grad_calls)
    train_end = synthetic_run.windows["train"][:, 1].max() + 12
    assert all((t0 + 12 <= train_end).all() for _, _, t0 in grad_calls)
    assert {split for split, _, _ in calls} == {"train", "val"}     # test never read


@pytest.mark.parametrize("extra", [[], ["--final"], ["--final", "--approved-config-sha256", "0" * 64],
                                   ["--smoke", "--final", "--approved-config-sha256", "MATCH"]])
def test_runner_refuses_test_split_without_approved_final(extra, monkeypatch):
    sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    extra = [sha if e == "MATCH" else e for e in extra]
    runner = _runner()
    monkeypatch.setattr(runner.dataset, "load_run", lambda *a, **k: pytest.fail("data loaded before the guard"))
    with pytest.raises(SystemExit, match="Refusing"):
        runner.main(["--config", str(CONFIG), "--evaluate", "test", *extra])


def test_runner_guard_allows_validation():
    runner = _runner()
    args = runner.parse_args(["--config", str(CONFIG)])
    runner.check_test_access(args, "irrelevant")       # no exception


def test_window_sets_of_splits_are_disjoint_in_time(synthetic_run):
    t = {k: synthetic_run.windows[k][:, 1] for k in ("train", "val", "test")}
    assert t["train"].max() + 12 <= t["val"].min() and t["val"].max() + 12 <= t["test"].min()
    assert np.isin(dataset.WindowDataset(synthetic_run, "train").index[:, 1], t["val"]).sum() == 0
