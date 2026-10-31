"""Phase 9 autonomous controller: runs every stage to completion without Claude Code, network or user input.

    .venv\\Scripts\\python.exe experiments/phase9/run_nine_month_experiment.py [--deadline-utc 2026-10-29T03:38:10Z]

Durable state in results/phase9/state/experiment_status.json (atomic writes); heartbeat every 20 s; single instance
via an OS file lock (released automatically if the process dies); every task runs in its own subprocess with bounded
retries and exponential backoff; completed stages and runs are skipped on restart; training resumes from per-epoch
checkpoints. The training scheduler launches runs in priority order only when their estimated duration fits before
the training cutoff (deadline - 90 min) and stops them at the cutoff (an interrupted run is never reported complete).
The final stages (freeze, validation pre-check, one-time test, analysis, report) always run before the deadline.
"""

import argparse
import msvcrt
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGES = ["audit", "prepare", "verify", "smoke", "benchmark", "train", "freeze", "precheck", "test", "analysis", "report"]
CRITICAL = {"audit", "prepare", "verify", "smoke"}
FINAL = {"freeze", "precheck", "test", "analysis", "report"}
RESERVE_S = C.RESERVE_S
EXPECTED_UNITS = {"centralised": 25, "local": 25, "fedavg_geo": 40, "fedavg_iid": 40}   # planning only; budgets unchanged
BACKOFF = (30, 120, 300)
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


class Controller:
    def __init__(self, deadline_utc=None):
        C.STATE.mkdir(parents=True, exist_ok=True)
        self.status_path = C.STATE / "experiment_status.json"
        self.state = C.read_json(self.status_path) or {}
        if "deadline_utc" not in self.state:
            self.state.update(created_utc=C.now(), deadline_utc=deadline_utc, stages={s: {"status": "pending", "attempts": 0} for s in STAGES},
                              runs={}, decisions=[], tasks=[])
        self.deadline = parse_utc(self.state["deadline_utc"])
        self.cutoff = self.deadline - RESERVE_S
        self.state["training_cutoff_utc"] = datetime.fromtimestamp(self.cutoff, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.current = "start"
        self.save()

    # ---- state ----
    def save(self):
        self.state["updated_utc"] = C.now()
        C.write_json(self.status_path, self.state)

    def decide(self, msg):
        self.state["decisions"].append(f"{C.now()} {msg}")
        C.log(f"DECISION {msg}")
        self.save()

    def task_log(self, name, outcome, detail=""):
        self.state["tasks"].append(dict(utc=C.now(), task=name, outcome=outcome, detail=str(detail)))
        self.save()

    def heartbeat(self):
        while True:
            running = {t: r.get("pid") for t, r in self.state["runs"].items() if r.get("status") == "running"}
            prog = {t: C.read_json(C.RUNS / t / "progress.json") for t in running}
            C.write_json(C.STATE / "heartbeat.json", dict(utc=C.now(), pid=os.getpid(), stage=self.current, running=running,
                                                         progress=prog, seconds_to_deadline=round(self.deadline - time.time())))
            time.sleep(20)

    # ---- subprocesses ----
    def spawn(self, args, logname):
        C.LOGS.mkdir(parents=True, exist_ok=True)
        f = open(C.LOGS / logname, "a", encoding="utf-8")
        f.write(f"\n==== {C.now()} {' '.join(map(str, args))}\n")
        f.flush()
        return subprocess.Popen([str(C.PY), *map(str, args)], cwd=C.ROOT, stdout=f, stderr=subprocess.STDOUT,
                                creationflags=NO_WINDOW)

    def run_task(self, stage, args, logname, timeout_s=None):
        """Run a task subprocess with bounded retries and exponential backoff. Returns True on success."""
        st = self.state["stages"][stage]
        while st["attempts"] < len(BACKOFF) + 1:
            st["attempts"] += 1
            st.update(status="running", started_utc=C.now())
            self.save()
            p = self.spawn(args, logname)
            try:
                rc = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                kill_tree(p.pid)
                rc = "timeout"
            if rc == 0:
                st.update(status="done", finished_utc=C.now())
                self.task_log(stage, "completed")
                return True
            C.log(f"{stage} attempt {st['attempts']} failed (rc={rc})", "recovery.log")
            if rc == 3:                                                      # integrity failure: do not retry blindly
                break
            if st["attempts"] <= len(BACKOFF) and time.time() + BACKOFF[st["attempts"] - 1] < self.deadline - 600:
                time.sleep(BACKOFF[st["attempts"] - 1])
            else:
                break
        st.update(status="failed", finished_utc=C.now())
        self.task_log(stage, "failed", f"attempts={st['attempts']}")
        return False

    # ---- benchmark ----
    def benchmark(self):
        st = self.state["stages"]["benchmark"]
        res = {}
        for k in (1, 2, 3):
            outs = [C.STATE / f"bench_k{k}_{i}.json" for i in range(k)]
            procs = [self.spawn([HERE / "worker.py", "benchstep", "--out", o], "benchmark.log") for o in outs]
            if any(p.wait(timeout=900) != 0 for p in procs):
                break
            ms = [C.read_json(o)["ms_per_step"] for o in outs]
            res[k] = dict(ms_per_step=sum(ms) / k, throughput_steps_per_s=sum(1000 / m for m in ms))
        if not res:
            self.decide("benchmark failed; defaulting to 1 process")
            res = {1: dict(ms_per_step=10.0, throughput_steps_per_s=100.0)}
        best = max(res, key=lambda k: res[k]["throughput_steps_per_s"])
        k = best if best == 1 or res[best]["throughput_steps_per_s"] >= 1.15 * res[1]["throughput_steps_per_s"] else 1
        train_windows = C.read_json(C.STATE / "bench_k1_0.json", {}).get("train_windows", 11_000_000)
        st.update(status="done", results={str(a): b for a, b in res.items()}, parallel=k, train_windows=train_windows)
        self.decide(f"parallel training processes = {k} (measured throughput {res})")
        self.task_log("benchmark", "completed", res)

    def est_seconds(self, arm):
        b = self.state["stages"]["benchmark"]
        k = b.get("parallel", 1)
        ms = b["results"][str(k)]["ms_per_step"]
        unit = ms / 1000 * b.get("train_windows", 11e6) / 512 + 25                 # one full-data epoch/round + validation
        done = [r for t, r in self.state["runs"].items() if t.startswith(arm) and r.get("status") == "complete"]
        if done:
            unit = max(r["seconds_per_unit"] for r in done)
        return unit * EXPECTED_UNITS[arm]

    # ---- training ----
    def train(self):
        st = self.state["stages"]["train"]
        st.update(status="running", started_utc=C.now())
        k = self.state["stages"]["benchmark"].get("parallel", 1)
        procs = {}
        for t, r in self.state["runs"].items():                                  # restart after a controller crash
            if r.get("status") == "running":
                if C.pid_alive(r.get("pid")):                                   # orphaned worker: stop it before resuming
                    kill_tree(r["pid"])
                    C.log(f"killed orphaned worker {t} (pid {r['pid']}) before resuming from its checkpoint", "recovery.log")
                r["status"] = "pending"
        while True:
            now = time.time()
            for t, (p, arm, seed) in list(procs.items()):
                rc = p.poll()
                if now > self.cutoff + 300 and rc is None:
                    kill_tree(p.pid)
                    rc = "cutoff"
                if rc is None:
                    continue
                r = self.state["runs"][t]
                del procs[t]
                marker = C.read_json(C.RUNS / t / "COMPLETE.json")
                if rc == 0 and marker:
                    meta = C.read_json(C.RUNS / t / "run_meta.json", {})
                    s = meta.get("summary", {})
                    units = s.get("units_run") or sum(c["epochs_run"] for c in s.get("clients", {}).values()) / 4 or 1
                    r.update(status="complete", finished_utc=C.now(), val_mae=marker.get("val_mae"),
                             seconds_per_unit=meta.get("train_seconds_total", 0) / max(units, 1))
                    self.task_log(f"train {t}", "completed", f"val MAE {marker.get('val_mae')}")
                elif rc in (4, "cutoff") or time.time() > self.cutoff:
                    r.update(status="incomplete_deadline", finished_utc=C.now())
                    self.task_log(f"train {t}", "incomplete (training cutoff reached; not reported as complete)")
                else:
                    r["attempts"] = r.get("attempts", 0) + 1
                    C.log(f"train {t} crashed (rc={rc}); attempt {r['attempts']}", "recovery.log")
                    r["status"] = "pending" if r["attempts"] < 3 else "failed"
                    if r["status"] == "failed":
                        self.task_log(f"train {t}", "failed", f"rc={rc}")
                    else:
                        r["retry_after"] = time.time() + BACKOFF[min(r["attempts"], 3) - 1]
                self.save()
            if time.time() < self.cutoff:
                for arm, seed in C.QUEUE:
                    if len(procs) >= k:
                        break
                    t = C.tag(arm, seed)
                    r = self.state["runs"].setdefault(t, {"status": "pending", "attempts": 0})
                    if r["status"] != "pending" or t in procs or r.get("retry_after", 0) > time.time():
                        continue
                    est = self.est_seconds(arm)
                    if time.time() + est > self.cutoff:
                        if not any(x.get("status") == "running" for x in self.state["runs"].values()) or arm in ("centralised", "fedavg_geo") and seed == 0:
                            pass                                                 # primary runs always get their chance
                        else:
                            r["status"] = "skipped_deadline"
                            self.task_log(f"train {t}", "skipped", f"estimated {est / 60:.0f} min does not fit before the training cutoff")
                            continue
                    p = self.spawn([HERE / "worker.py", "train", arm, seed, "--deadline", self.cutoff], f"train_{t}.log")
                    procs[t] = (p, arm, seed)
                    r.update(status="running", pid=p.pid, started_utc=C.now(), estimated_minutes=round(est / 60))
                    C.log(f"launched {t} (pid {p.pid}, est {est / 60:.0f} min)")
                    self.save()
            pending = [t for t, r in self.state["runs"].items() if r["status"] == "pending"]
            unstarted = [C.tag(a, s) for a, s in C.QUEUE if C.tag(a, s) not in self.state["runs"]]
            if not procs and (time.time() >= self.cutoff or (not pending and not unstarted)):
                break
            time.sleep(15)
        for a, s in C.QUEUE:
            t = C.tag(a, s)
            r = self.state["runs"].setdefault(t, {"status": "skipped_deadline"})
            if r["status"] == "pending":
                r["status"] = "skipped_deadline"
                self.task_log(f"train {t}", "skipped", "not started before the training cutoff")
        st.update(status="done", finished_utc=C.now(),
                  complete=[t for t, r in self.state["runs"].items() if r["status"] == "complete"])
        self.save()

    # ---- main ----
    def run(self):
        threading.Thread(target=self.heartbeat, daemon=True).start()
        C.keep_awake(True)
        C.log(f"controller start (pid {os.getpid()}), deadline {self.state['deadline_utc']}")
        data_ok = True
        for stage in STAGES:
            st = self.state["stages"][stage]
            self.current = stage
            if st["status"] in ("done", "skipped"):
                continue
            if stage == "train" and not data_ok:
                st["status"] = "skipped"
                self.task_log(stage, "skipped", "dataset not validated")
                continue
            if stage in CRITICAL:
                ok = data_ok and self.run_task(stage, [HERE / "worker.py", stage], f"{stage}.log")
                if not ok:
                    data_ok = False
                    if st["status"] != "failed":
                        st["status"] = "skipped"
                    self.decide(f"{stage} failed or skipped: nine-month dataset not validated; partial-delivery report only")
            elif stage == "benchmark":
                self.benchmark() if data_ok else st.update(status="skipped")
            elif stage == "train":
                self.train()
            elif stage in FINAL:
                if stage != "report" and (not data_ok or not self.state["stages"]["train"].get("complete")):
                    st["status"] = "skipped"
                    self.task_log(stage, "skipped", "no complete training run")
                    continue
                self.run_task(stage, [HERE / "final.py", stage], f"final_{stage}.log", timeout_s=max(600, self.deadline - time.time()))
            self.save()
        self.state["status"] = "complete"
        self.state["completed_utc"] = C.now()
        self.save()
        C.log("controller finished all stages")
        C.keep_awake(False)


def kill_tree(pid):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW)


def acquire_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    return f


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--deadline-utc", default=None)
    p.add_argument("--init-only", action="store_true", help="write the initial state (deadline) and exit")
    a = p.parse_args(argv)
    lock = acquire_lock(C.STATE / "controller.lock")
    if lock is None:
        print("another controller instance holds the lock; exiting")
        return 0
    state = C.read_json(C.STATE / "experiment_status.json", {})
    if state.get("status") == "complete":
        print("experiment already complete")
        return 0
    if not state.get("deadline_utc") and not a.deadline_utc:
        raise SystemExit("--deadline-utc required on first start")
    ctl = Controller(a.deadline_utc)
    if a.init_only:
        print(f"state initialised; deadline {ctl.state['deadline_utc']}, training cutoff {ctl.state['training_cutoff_utc']}")
        return 0
    ctl.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
