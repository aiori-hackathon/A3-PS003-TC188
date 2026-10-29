"""Create the Hugging Face backend: a PRIVATE dataset with the runtime artifacts and a public Docker Space with code only.

    hf auth login                      # once, with a write token (the token is cached locally; this script never prints it)
    python deploy/hf_deploy.py [--owner NAME] [--space aiori-traffic] [--artifacts aiori-traffic-artifacts]

Uploads only the files the API reads at start-up (about 490 MB): the nine-month processed checkpoint, the two deployed
GRU runs, the frozen aggregate results, and the optimized-MLP candidate. Nothing else from data/ or results/ is sent.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--space", default="aiori-traffic")
    ap.add_argument("--artifacts", default="aiori-traffic-artifacts")
    args = ap.parse_args()
    missing = [f for f in FILES if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"Missing local artifacts: {missing}")
    token = get_token()
    if not token:
        raise SystemExit("Run `hf auth login` first.")
    api = HfApi(token=token)
    owner = args.owner or api.whoami()["name"]
    data_id, space_id = f"{owner}/{args.artifacts}", f"{owner}/{args.space}"

    api.create_repo(data_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(data_id, repo_type="dataset").private:
        raise SystemExit(f"{data_id} is not private; refusing to upload.")
    for f in FILES:
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=data_id, repo_type="dataset",
                        commit_message=f"Add {Path(f).name}")
        print("uploaded", f, flush=True)

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space_id, "HF_TOKEN", token)
    api.add_space_variable(space_id, "AIORI_ARTIFACTS_REPO", data_id)
    api.upload_folder(folder_path=str(ROOT / "deploy" / "hf-space"), repo_id=space_id, repo_type="space",
                      commit_message="AIORI Traffic Intelligence API")
    print(f"Space: https://huggingface.co/spaces/{space_id}  API: https://{space_id.replace('/', '-').lower()}.hf.space/api/health")


if __name__ == "__main__":
    main()
