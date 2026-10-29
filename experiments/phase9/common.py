"""Shared paths and helpers for the Phase 9 autonomous runner (no Claude Code, network or user input needed)."""

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = ROOT / ".venv" / "Scripts" / "python.exe"
R = Path(os.environ.get("PHASE9_RESULTS", ROOT / "results" / "phase9"))
DATA = Path(os.environ.get("PHASE9_DATA", ROOT / "data" / "processed" / "nine_month"))
YEAR = Path(os.environ.get("PHASE9_YEAR", ROOT / "year_2023"))
DOCS = Path(os.environ.get("PHASE9_DOCS", ROOT / "docs"))
RESERVE_S = int(os.environ.get("PHASE9_RESERVE_S", 90 * 60))       # final-stage reserve before the deadline
STATE = R / "state"
LOGS = R / "logs"
RUNS = R / "runs"
ARMS = ("centralised", "fedavg_geo", "local", "fedavg_iid")
QUEUE = [(a, s) for s in (0, 1, 2) for a in ARMS]          # priority: all four arms at seed 0, then seeds 1 and 2


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tag(arm, seed):
    return f"{arm}_seed{seed}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    """Atomic JSON write (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def log(msg, name="execution.log"):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / name, "a", encoding="utf-8") as f:
        f.write(f"{now()} [{os.getpid()}] {msg}\n")


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]


def peak_mb():
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wt.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2 ** 20)
    except Exception:  # noqa: BLE001 - non-Windows or API unavailable
        return None


def pid_alive(pid):
    """True if a process with this PID exists (Windows OpenProcess; POSIX kill 0)."""
    if not pid:
        return False
    if os.name == "nt":
        k32 = ctypes.WinDLL("kernel32")
        h = k32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = wt.DWORD()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259                                # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def keep_awake(on=True):
    """Ask Windows not to sleep while the runner works (no setting is changed; released when the process exits)."""
    try:
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:  # noqa: BLE001
        pass
