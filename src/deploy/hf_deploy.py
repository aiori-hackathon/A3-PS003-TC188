"""Create the Hugging Face backend: a PRIVATE dataset with the runtime artifacts and a public Docker Space with code only.

    hf auth login                      # once, with a write token (used only here, on this machine; never sent to the Space)
    set AIORI_SPACE_TOKEN=...          # a separate READ-only token (fine-grained: read access to the artifact dataset only)
    python deploy/hf_deploy.py [--owner NAME] [--space aiori-traffic] [--artifacts aiori-traffic-artifacts]

Uploads only the files the API reads at start-up (about 490 MB): the nine-month processed checkpoint, the two deployed
GRU runs, the frozen aggregate results, and the optimized-MLP candidate. Nothing else from data/ or results/ is sent.
The Space builds the exact pushed commit checked out here (its Dockerfile REF is set to it). No token is ever printed.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "deploy" / "hf-space"
FILES = [
    *[f"data/processed/nine_month/{n}" for n in ("pilot_preprocessed.npz", "window_index.npz", "scaler_parameters.json",
                                                  "sensor_selection.csv", "split_config.json")],
    *[f"results/phase9/runs/{r}/{n}" for r in ("local_seed1", "centralised_seed0") for n in ("run_meta.json", "config.json", "COMPLETE.json")],
    *[f"results/phase9/runs/local_seed1/best_model_{c}.pt" for c in ("client_0_la_ventura", "client_1_bay_area",
                                                                     "client_2_coastal_south_central", "client_3_inland_central")],
    "results/phase9/runs/centralised_seed0/best_model.pt",
    "results/phase9/final/analysis.json", "results/phase9/final/frozen.json",
    "results/phase9/final/test/local_seed1/metrics_test.json", "results/phase9/final/test/centralised_seed0/metrics_test.json",
    *[f"results/accuracy_optimization/candidate_v1/{n}" for n in ("manifest.json", "mlp_config.json", "mlp_models.pt", "profile_scaled.npy")],
    "results/accuracy_optimization/untouched_oct_dec/results.json",
]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def pushed_head():
    """The commit the Space will build: HEAD, which must already be on a remote branch (the Space clones it from GitHub)."""
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"HEAD {head[:12]} is not on any remote branch; push it before deploying.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked files have uncommitted changes; the Space would not build them. Commit and push first.")
    return head


def can_write(info):
    """True when a token (from whoami) can write or manage anything."""
    tok = info.get("auth", {}).get("accessToken", {})
    if tok.get("role") == "write":
        return True
    scoped = tok.get("fineGrained", {}).get("scoped", [])
    perms = [p for s in scoped for p in s.get("permissions", [])] + tok.get("fineGrained", {}).get("global", [])
    return any("write" in p or "manage" in p for p in perms)


def space_files(ref):
    """Space repository contents, with the Dockerfile pinned to `ref`."""
    docker = (SPACE_DIR / "Dockerfile").read_text(encoding="utf-8")
    pinned, n = re.subn(r"(?m)^ARG REF=\S+$", f"ARG REF={ref}", docker)
    if n != 1:
        raise SystemExit("deploy/hf-space/Dockerfile needs exactly one 'ARG REF=<commit>' line.")
    ops = [CommitOperationAdd("Dockerfile", pinned.encode("utf-8"))]
    ops += [CommitOperationAdd(n, str(SPACE_DIR / n)) for n in ("README.md", "start.sh")]
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--space", default="aiori-traffic")
    ap.add_argument("--artifacts", default="aiori-traffic-artifacts")
    args = ap.parse_args()
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"Missing local artifacts: {missing}")
    ref = pushed_head()
    token = get_token()
    if not token:
        raise SystemExit("Run `hf auth login` first.")
    space_token = os.environ.get("AIORI_SPACE_TOKEN", "").strip()
    if not space_token:
        raise SystemExit("Set AIORI_SPACE_TOKEN to a read-only token for the artifact dataset (it becomes the Space secret).")
    if space_token == token:
        raise SystemExit("AIORI_SPACE_TOKEN must be a separate read-only token, not your login (write) token.")
    api = HfApi(token=token)
    if can_write(api.whoami(token=space_token)):
        raise SystemExit("AIORI_SPACE_TOKEN can write; the public Space needs a read-only token.")
    owner = args.owner or api.whoami()["name"]
    data_id, space_id = f"{owner}/{args.artifacts}", f"{owner}/{args.space}"

    api.create_repo(data_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(data_id, repo_type="dataset").private:
        raise SystemExit(f"{data_id} is not private; refusing to upload.")
    api.create_commit(data_id, repo_type="dataset", commit_message="Add runtime artifacts",
                      operations=[CommitOperationAdd(f, str(ROOT / f)) for f in FILES])
    print(f"uploaded {len(FILES)} artifact files", flush=True)

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space_id, "HF_TOKEN", space_token)
    api.add_space_variable(space_id, "AIORI_ARTIFACTS_REPO", data_id)
    api.create_commit(space_id, repo_type="space", operations=space_files(ref),
                      commit_message=f"AIORI Traffic Intelligence API at {ref[:12]}")
    print(f"Space: https://huggingface.co/spaces/{space_id}  API: https://{space_id.replace('/', '-').lower()}.hf.space/api/health")


if __name__ == "__main__":
    main()
