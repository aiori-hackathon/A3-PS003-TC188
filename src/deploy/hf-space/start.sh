#!/bin/sh
# Download the private runtime artifacts into the code checkout (same relative paths as the repository), then serve.
set -e
cd /home/user/app/src
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["AIORI_ARTIFACTS_REPO"], repo_type="dataset", local_dir=".", token=os.environ["HF_TOKEN"])
print("artifacts ready", flush=True)
PY
exec python app/backend/server.py
