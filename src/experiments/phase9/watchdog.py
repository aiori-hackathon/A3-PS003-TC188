"""Phase 9 watchdog: keeps the controller alive until the experiment is complete (no Claude Code needed).

Started by the per-user scheduled task "AIORI_PS003_Phase9" (at logon and every 5 minutes; a second copy exits at once
because of the watchdog lock). Every 60 s it checks the controller:
  - experiment complete            -> unregisters the scheduled task and exits;
  - controller lock free           -> starts the controller (bounded: at most 6 starts per hour, 40 in total);
  - lock held but heartbeat stale  -> kills the hung controller tree (20 min without heartbeat), then restarts it.
Everything is logged to results/phase9/logs/recovery.log. It stops relaunching 2 hours after the deadline.
"""

import msvcrt
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TASK = "AIORI_PS003_Phase9"
STALE_S = 20 * 60
NO_WINDOW = 0x08000000
DETACHED = 0x00000008


def lock_free(path):
    try:
        f = open(path, "a+")
    except OSError:
        return False
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return True
    except OSError:
        return False
    finally:
        f.close()


def unregister():
    subprocess.run(["powershell", "-NoProfile", "-Command", f"Unregister-ScheduledTask -TaskName {TASK} -Confirm:$false"],
                   capture_output=True, creationflags=NO_WINDOW)


def main():
    C.STATE.mkdir(parents=True, exist_ok=True)
    wlock = open(C.STATE / "watchdog.lock", "a+")
    try:
        msvcrt.locking(wlock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return 0                                                  # another watchdog is running
    C.log(f"watchdog start (pid {os.getpid()})", "recovery.log")
    starts = []
    while True:
        st = C.read_json(C.STATE / "experiment_status.json", {})
        if st.get("status") == "complete":
            C.log("experiment complete; unregistering the scheduled task and exiting", "recovery.log")
            unregister()
            return 0
        deadline = st.get("deadline_utc")
        if deadline and time.time() > C_parse(deadline) + 2 * 3600:
            C.log("2 hours past the deadline without completion; watchdog stops", "recovery.log")
            unregister()
            return 1
        ctl_lock = C.STATE / "controller.lock"
        hb = C.read_json(C.STATE / "heartbeat.json", {})
        if lock_free(ctl_lock):
            starts = [t for t in starts if time.time() - t < 3600]
            wd = C.read_json(C.STATE / "watchdog.json", {})
            total = wd.get("controller_starts", 0)
            if len(starts) < 6 and total < 40:
                p = subprocess.Popen([str(C.PY), str(Path(__file__).with_name("run_nine_month_experiment.py"))], cwd=C.ROOT,
                                     stdout=subprocess.DEVNULL, stderr=open(C.LOGS / "controller_stderr.log", "a"),
                                     creationflags=NO_WINDOW)
                starts.append(time.time())
                C.log(f"controller not running: started pid {p.pid} (start {total + 1})", "recovery.log")
                C.write_json(C.STATE / "watchdog.json", dict(controller_starts=total + 1, last_start_utc=C.now(), last_pid=p.pid))
            else:
                C.log("restart budget exhausted; not starting the controller", "recovery.log")
        elif hb and time.time() - C_parse(hb["utc"]) > STALE_S and C.pid_alive(hb.get("pid")):
            C.log(f"heartbeat stale ({hb['utc']}); killing hung controller tree {hb['pid']}", "recovery.log")
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(hb["pid"])], capture_output=True, creationflags=NO_WINDOW)
        time.sleep(60)


def C_parse(s):
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


if __name__ == "__main__":
    sys.exit(main())
