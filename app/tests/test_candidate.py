"""Opt-in candidate model: the backend's CPU inference equals the experiment's validated predictions."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "results" / "accuracy_optimization" / "candidate_v1"
PRED = ROOT / "results" / "accuracy_optimization" / "validation_predictions" / "s2_mlp512_L288_emb.npy"
pytestmark = pytest.mark.skipif(not (CAND / "manifest.json").exists() or not PRED.exists(), reason="candidate artifacts not present")
sys.path.insert(0, str(ROOT / "app" / "backend"))


def test_candidate_matches_validated_predictions():
    import json
    import torch
    from aiori import dataset
    from candidate import Candidate
    data = ROOT / "data" / "processed" / "nine_month"
    run = dataset.load_run("nine_month", run_dir=data, selection=data / "sensor_selection.csv", splits=("val",))
    sc = json.loads((data / "scaler_parameters.json").read_text())
    c = Candidate(run, np.asarray(sc["mean"]), np.asarray(sc["std"]), torch.device("cpu"))
    with np.load(data / "window_index.npz") as w:
        val = w["val"]
    saved = np.load(PRED, mmap_mode="r")
    for row in (0, 123_456, 800_000, len(val) - 1):
        pos, t0 = map(int, val[row])
        assert np.allclose(c.forecast(pos, t0), saved[row], atol=0.05), row
    with pytest.raises(ValueError):
        c.forecast(0, 100)


def test_candidate_is_off_by_default():
    assert os.environ.get("AIORI_CANDIDATE") != "1"
