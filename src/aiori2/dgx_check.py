"""DGX (or any Linux/Windows host) readiness check and code manifest. Nothing is sent anywhere.

    python -m aiori2.dgx_check                       # GPU model, allocated/free memory, driver, CUDA, PyTorch, tiny GPU op
    python -m aiori2.dgx_check --verify              # re-hash transferred data (dgx_transfer_manifest) and code (dgx_code_manifest)
    python -m aiori2.dgx_check --write-code-manifest # (on the source machine) list + hash every code/config file needed

Run from the src/ directory, or set AIORI2_BASE_DIR=<path to src>.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys

import torch

from aiori2 import core

CODE_GLOBS = ["aiori2/**/*.py", "aiori2/**/*.json", "aiori2/**/*.md", "aiori/*.py", "experiments/accuracy_optimization/aopt/models.py",
              "tests/conftest.py", "tests/test_aiori2*.py"]
EXTRA = ["../requirements.txt", "../pytest.ini"]


def code_files(base):
    files = set()
    for g in CODE_GLOBS:
        files |= {p for p in base.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    files |= {(base / e).resolve() for e in EXTRA if (base / e).is_file()}
    return sorted(files)


def write_code_manifest(cfg):
    base = cfg["_base"]
    m = {}
    for p in code_files(base):
        rel = p.relative_to(base.parent).as_posix()                       # relative to the repository root
        m[rel] = dict(size=p.stat().st_size, sha256=core.sha256_file(p))
    out = cfg["_out"] / "manifests" / "dgx_code_manifest.json"
    core.save_json(out, dict(root="repository root (the parent of src/)", files=m))
    return out, len(m)


def verify(cfg):
    problems, n = [], 0
    data = core.load_json(cfg["_out"] / "manifests" / "dgx_transfer_manifest.json", {})
    for rel, x in data.get("data_files", {}).items():
        p = cfg["_out"] / rel
        n += 1
        if not p.is_file() or core.sha256_file(p) != x["sha256"]:
            problems.append(f"data: {rel}")
    code = core.load_json(cfg["_out"] / "manifests" / "dgx_code_manifest.json", {})
    for rel, x in code.get("files", {}).items():
        p = cfg["_base"].parent / rel
        n += 1
        if not p.is_file() or core.sha256_file(p) != x["sha256"]:
            problems.append(f"code: {rel}")
    return dict(checked=n, problems=problems, ok=not problems and n > 0)


def gpu_report():
    r = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__, torch_cuda_build=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        r["nvidia_smi"] = q.stdout.strip().splitlines()
        v = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        r["driver_cuda_version"] = next((l.split("CUDA Version:")[1].split()[0] for l in v.splitlines() if "CUDA Version:" in l), None)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        p = torch.cuda.get_device_properties(0)
        r.update(device_count=torch.cuda.device_count(), gpu=p.name, capability=list(torch.cuda.get_device_capability(0)),
                 total_mb=round(total / 2**20), free_mb=round(free / 2**20),
                 note="free_mb is what this process can allocate now; a MIG slice or quota shows up here, not in the GPU name")
        a = torch.randn(1024, 1024, device="cuda")
        r["matmul_ok"] = bool(torch.isfinite(a @ a).all())
        try:
            from aiori2.models import build_model
            m = build_model("patchtst", seed=0).cuda()
            y = m(torch.randn(64, 12, 1, device="cuda"))
            y.sum().backward()
            r["aiori2_model_ok"] = bool(torch.isfinite(y).all())
        except Exception as e:                                             # report, do not hide
            r["aiori2_model_error"] = repr(e)
    return r


def main():
    ap = argparse.ArgumentParser(prog="python -m aiori2.dgx_check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write-code-manifest", action="store_true")
    a = ap.parse_args()
    cfg = core.load_config()
    if a.write_code_manifest:
        out, n = write_code_manifest(cfg)
        print(f"{n} code files -> {out}")
    elif a.verify:
        res = verify(cfg)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["ok"] else 1)
    else:
        print(json.dumps(gpu_report(), indent=1))


if __name__ == "__main__":
    main()
