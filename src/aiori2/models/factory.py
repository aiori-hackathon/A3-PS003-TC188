"""One construction path for the five forecasters, plus checkpoint and parameter-exchange helpers.

Contract: model(x) with x (B, input_len, C) float32 scaled inputs (C = 1 flow, or 2 with the observed flag)
-> (B, horizon) forecasts in the same scaled units. Models hold no sensor embeddings, scalers or client data,
and have no buffers (no BatchNorm), so every state_dict entry is a shareable parameter.
"""

import copy
import json
from pathlib import Path

import torch

from aiori2.models.nhits import NHiTS
from aiori2.models.patchtst import PatchTST
from aiori2.models.reuse import GRU, MLP
from aiori2.models.tcn import TCN

MODELS = {"mlp": MLP, "gru": GRU, "tcn": TCN, "patchtst": PatchTST, "nhits": NHiTS}
PRESETS = Path(__file__).resolve().parents[1] / "configs" / "models" / "presets.json"
DATA_CONTRACT = "aiori2_3yr_v1/phase3"


def presets():
    return json.loads(PRESETS.read_text(encoding="utf-8"))


def model_config(name, preset=None, in_channels=1, input_len=12, horizon=12, **overrides):
    p = presets()
    preset = preset or p["default_preset"]
    cfg = dict(p[name][preset])
    cfg.update(overrides)
    return dict(name=name, preset=preset, in_channels=in_channels, input_len=input_len, horizon=horizon, params=cfg)


def build_model(name, preset=None, in_channels=1, input_len=12, horizon=12, seed=None, **overrides):
    """Build a model; with `seed`, initialisation is deterministic and the global RNG state is left untouched."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    cfg = model_config(name, preset, in_channels, input_len, horizon, **overrides)
    kw = dict(cfg["params"])
    for k in ("pool", "freq"):
        if k in kw:
            kw[k] = tuple(kw[k])
    if seed is None:
        model = MODELS[name](in_channels=in_channels, input_len=input_len, horizon=horizon, **kw)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = MODELS[name](in_channels=in_channels, input_len=input_len, horizon=horizon, **kw)
    model.config = dict(cfg, seed=seed)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---- checkpoints: tensors + JSON-able metadata only, loadable with weights_only=True ----

def save_checkpoint(model, path, extra=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(model.config, data_contract=DATA_CONTRACT, **(extra or {}))
    tmp = path.with_name(path.name + ".partial")
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, meta_json=json.dumps(meta)), tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = json.loads(ck["meta_json"])
    model = build_model(meta["name"], meta["preset"], meta["in_channels"], meta["input_len"], meta["horizon"], **meta["params"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.config = {k: meta[k] for k in ("name", "preset", "in_channels", "input_len", "horizon", "params", "seed")}
    return model.to(device), meta


# ---- parameter exchange for future federated algorithms (no FL algorithm here) ----

def average_states(states, weights):
    """Weighted average of floating-point entries (accumulated in float64); non-floating entries must agree."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        vals = [s[k] for s in states]
        if any(v.shape != vals[0].shape for v in vals):
            raise ValueError(f"shape mismatch for {k}")
        if vals[0].is_floating_point():
            out[k] = sum(wi * v.double() for wi, v in zip(w, vals)).to(vals[0].dtype)
        else:
            if any(not torch.equal(v, vals[0]) for v in vals):
                raise ValueError(f"non-floating entry {k} differs between clients")
            out[k] = vals[0].clone()
    return out


def clone_model(model):
    m = copy.deepcopy(model)
    m.config = dict(model.config)
    return m
