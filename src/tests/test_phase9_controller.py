"""Phase 9 controller and watchdog building blocks: lock, atomic state, bounded retries, integrity stop."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("msvcrt")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "phase9"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / "phase9" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    import common as C
    monkeypatch.setattr(C, "STATE", tmp_path / "state")
    monkeypatch.setattr(C, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(C, "PY", Path(sys.executable))
    mod = _load("run_nine_month_experiment")
    monkeypatch.setattr(mod, "BACKOFF", (0, 0, 0))
    return mod, C, tmp_path


def test_single_instance_lock(ctl):
    mod, C, tmp = ctl
    a = mod.acquire_lock(tmp / "x.lock")
    assert a is not None and mod.acquire_lock(tmp / "x.lock") is None
    wd = _load("watchdog")
    assert not wd.lock_free(tmp / "x.lock")
    a.close()
    assert wd.lock_free(tmp / "x.lock")


def test_atomic_json_roundtrip(ctl):
    _, C, tmp = ctl
    C.write_json(tmp / "s.json", {"a": 1})
    C.write_json(tmp / "s.json", {"a": 2})
    assert C.read_json(tmp / "s.json") == {"a": 2} and not list(tmp.glob("*.tmp"))
    assert C.read_json(tmp / "missing.json", "d") == "d"


def _script(tmp, code):
    p = tmp / "task.py"
    p.write_text(f"import sys\nsys.exit({code})\n")
    return p


def test_bounded_retries_then_failed(ctl):
    mod, C, tmp = ctl
    c = mod.Controller("2099-01-01T00:00:00Z")
    assert not c.run_task("smoke", [_script(tmp, 1)], "t.log")
    st = c.state["stages"]["smoke"]
    assert st["status"] == "failed" and st["attempts"] == 4
    assert "attempt 4 failed" in (C.LOGS / "recovery.log").read_text()


def test_integrity_failure_is_not_retried(ctl):
    mod, C, tmp = ctl
    c = mod.Controller("2099-01-01T00:00:00Z")
    assert not c.run_task("verify", [_script(tmp, 3)], "t.log")
    assert c.state["stages"]["verify"]["attempts"] == 1


def test_success_and_state_survives_restart(ctl):
    mod, C, tmp = ctl
    c = mod.Controller("2099-01-01T00:00:00Z")
    assert c.run_task("smoke", [_script(tmp, 0)], "t.log")
    c2 = mod.Controller(None)                                     # a restarted controller reloads the durable state
    assert c2.state["stages"]["smoke"]["status"] == "done" and c2.state["deadline_utc"] == "2099-01-01T00:00:00Z"
    assert c2.cutoff == c2.deadline - mod.RESERVE_S
