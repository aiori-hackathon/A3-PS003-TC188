"""Training engine: centralised, local-only and simulated federated algorithms on four geographic clients.

Every algorithm sees only training windows for optimiser updates and validation windows for early stopping and
selection. Validation WAPE (raw units, negative forecasts clipped at 0, the frozen final model's policy) is the
selection criterion; it is a validation estimate, not a held-out test result. Federated clients are simulated in
one process: "communication" is counted as parameters exchanged, never measured on a network.

Federated algorithms (update rules implemented explicitly):
  fedavg     w <- sum_k p_k w_k                                       (server lr 1, plain averaging)
  fedavgm    v <- beta v + d;  w <- w - eta v,  d = w - sum_k p_k w_k (server momentum, Hsu et al. 2019)
  fedadam    m, v Adam moments of d; w <- w - eta m_hat / (sqrt(v_hat) + tau)  (Reddi et al. 2021)
  fedyogi    as fedadam with v <- v - (1 - b2) d^2 sign(v - d^2)
  fedadagrad v <- v + d^2
  fedprox    client loss + mu/2 ||w - w_global||^2                    (Li et al. 2020)
  fedper     only the body is averaged; heads and sensor embeddings stay on each client (Arivazhagan et al. 2019)
  ditto      global model by FedAvg; personal model v_k trained with loss + lam/2 ||v_k - w_global||^2 (Li et al. 2021)
  fedavg_ft  FedAvg, then each client fine-tunes the global model on its own data with early stopping
"""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .models import Forecaster

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "accuracy_optimization"
EVAL_BATCH = 32768


@dataclass
class Config:
    exp_id: str
    algorithm: str = "local"            # central | local | fedavg | fedavgm | fedadam | fedyogi | fedadagrad | fedprox | fedper | ditto | fedavg_ft
    kind: str = "gru"
    hidden: int = 64
    layers: int = 1
    head_hidden: int = 0
    emb_dim: int = 0
    dropout: float = 0.0
    input_len: int = 12
    future_cal: bool = False
    profile: bool = False
    lags: bool = False
    calendar_in: bool = True
    loss: str = "mse_z"                 # mse_z (frozen study) | l1_raw | huber_raw | l1_z
    huber_delta: float = 10.0
    horizon_weight: float = 0.0         # w_h = 1 + horizon_weight * h / 11
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch: int = 512
    max_steps: int = 60000              # optimiser steps per model (local) or per client (federated, summed over rounds)
    eval_every: int = 2000
    patience: int = 5                   # evaluations without improvement
    grad_clip: float = 1.0
    lr_decay: bool = False              # cosine decay over max_steps
    seed: int = 0
    # federated
    local_steps: int = 500              # client steps per round
    server_lr: float = 1.0
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 1e-3
    mu: float = 0.0                     # FedProx / Ditto strength
    ft_steps: int = 20000
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def features(self):
        return D.Features(self.input_len, self.calendar_in, self.future_cal, self.profile, self.lags)


def make_model(cfg, store):
    f = cfg.features()
    return Forecaster(cfg.kind, f.past_dim, f.future_dim, cfg.input_len, store.n_sensors, cfg.emb_dim, cfg.hidden,
                      cfg.layers, cfg.head_hidden, cfg.dropout).to(store.device)


def loss_fn(cfg, store, pred_z, b):
    mean, std = store.mean[b["s"]][:, None], store.std[b["s"]][:, None]
    m = b["m"]
    w = 1 + cfg.horizon_weight * torch.arange(12, device=pred_z.device) / 11
    if cfg.loss == "mse_z":
        y = (b["y"] - mean) / std
        r = (pred_z - y) ** 2
    elif cfg.loss == "l1_z":
        r = (pred_z - (b["y"] - mean) / std).abs()
    elif cfg.loss == "l1_raw":                       # raw-unit absolute error: the quantity WAPE sums
        r = (pred_z * std + mean - b["y"]).abs()
    elif cfg.loss == "huber_raw":
        r = torch.nn.functional.huber_loss(pred_z * std + mean, b["y"], reduction="none", delta=cfg.huber_delta)
    else:
        raise ValueError(cfg.loss)
    r = torch.where(m, r * w, torch.zeros_like(r))
    return r.sum() / m.sum().clamp(min=1)


class Sampler:
    """Seeded uniform sampling of training windows (with replacement across steps, without within a pass)."""

    def __init__(self, idx, seed, device):
        self.idx, self.g = idx, torch.Generator(device=device).manual_seed(seed)
        self.perm, self.pos = None, 0

    def next(self, n):
        if self.perm is None or self.pos + n > len(self.perm):
            self.perm = torch.randperm(len(self.idx), generator=self.g, device=self.idx.device)
            self.pos = 0
        out = self.idx[self.perm[self.pos:self.pos + n]]
        self.pos += n
        return out


@torch.no_grad()
def predict_raw(models_by_client, store, f, idx):
    """Raw-unit forecasts (clipped at 0) for windows idx; each window uses its sensor's client model."""
    out = torch.empty(len(idx), 12, device=store.device)
    cl = store.client[idx[:, 0]]
    for c, model in models_by_client.items():
        model.eval()
        rows = torch.nonzero(cl == c).flatten() if c is not None else torch.arange(len(idx), device=store.device)
        for i in range(0, len(rows), EVAL_BATCH):
            r = rows[i:i + EVAL_BATCH]
            b = D.batch(store, idx[r], f)
            z = model(b["x"], b["xf"], b["s"])
            out[r] = (z * store.std[b["s"]][:, None] + store.mean[b["s"]][:, None]).clamp(min=0)
    return out


@torch.no_grad()
def val_error(models_by_client, store, f, clients=None):
    """(total |e|, total |a|) over validation cells, optionally only for some clients."""
    idx = store.val
    if clients is not None:
        idx = idx[torch.isin(store.client[idx[:, 0]], torch.as_tensor(clients, device=store.device))]
    p = predict_raw(models_by_client, store, f, idx)
    b_y = store.flow[idx[:, 0][:, None], idx[:, 1][:, None] + torch.arange(12, device=store.device)]
    m = torch.isfinite(b_y)
    a = torch.nan_to_num(b_y)
    return float(torch.where(m, (p - a).abs(), 0).sum()), float(a.sum())


def step(cfg, store, f, model, opt, sampler, prox=None):
    model.train()
    b = D.batch(store, sampler.next(cfg.batch), f)
    loss = loss_fn(cfg, store, model(b["x"], b["xf"], b["s"]), b)
    if prox is not None:
        anchor, mu, keys = prox
        params = dict(model.named_parameters())
        loss = loss + mu / 2 * sum(((params[k] - anchor[k]) ** 2).sum() for k in keys)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    return float(loss.detach())


def make_opt(cfg, model, lr=None):
    return torch.optim.AdamW(model.parameters(), lr=lr or cfg.lr, weight_decay=cfg.weight_decay) if cfg.weight_decay \
        else torch.optim.Adam(model.parameters(), lr=lr or cfg.lr)


def client_idx(store, c):
    return store.train[store.client[store.train[:, 0]] == c]


# ----------------------------------------------------------------- centralised / local
def train_single(cfg, store, f, idx, val_clients, log, tag):
    """Adam on idx with early stopping on validation WAPE of val_clients; returns best model and history."""
    torch.manual_seed(cfg.seed)
    model = make_model(cfg, store)
    opt = make_opt(cfg, model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.max_steps) if cfg.lr_decay else None
    sampler = Sampler(idx, cfg.seed, store.device)
    best, best_state, bad, hist, losses = math.inf, None, 0, [], []
    for s in range(1, cfg.max_steps + 1):
        losses.append(step(cfg, store, f, model, opt, sampler))
        if sched:
            sched.step()
        if s % cfg.eval_every == 0 or s == cfg.max_steps:
            e, a = val_error({None: model}, store, f, val_clients)
            w = 100 * e / a
            hist.append(dict(step=s, train_loss=float(np.mean(losses)), val_wape=w))
            log(f"{tag} step {s} loss {np.mean(losses):.4f} val WAPE {w:.3f}%")
            losses = []
            if w < best - 1e-4:
                best, best_state, bad = w, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    return model, dict(history=hist, best_val_wape=best, steps=s)


# ----------------------------------------------------------------- federated
class Server:
    """Server optimiser on the pseudo-gradient d = w_global - weighted mean of client models."""

    def __init__(self, cfg, keys):
        self.cfg, self.keys, self.m, self.v, self.t = cfg, keys, {}, {}, 0

    def apply(self, gstate, avg):
        cfg, alg = self.cfg, self.cfg.algorithm
        self.t += 1
        for k in self.keys:
            d = gstate[k] - avg[k]
            if alg in ("fedavg", "fedprox", "fedper", "ditto", "fedavg_ft"):
                gstate[k] = avg[k].clone()
            elif alg == "fedavgm":
                self.m[k] = cfg.server_beta1 * self.m.get(k, torch.zeros_like(d)) + d
                gstate[k] = gstate[k] - cfg.server_lr * self.m[k]
            elif alg in ("fedadam", "fedyogi", "fedadagrad"):
                m = self.m[k] = cfg.server_beta1 * self.m.get(k, torch.zeros_like(d)) + (1 - cfg.server_beta1) * d
                v = self.v.get(k, torch.full_like(d, cfg.server_tau ** 2))
                if alg == "fedadam":
                    v = cfg.server_beta2 * v + (1 - cfg.server_beta2) * d * d
                elif alg == "fedyogi":
                    v = v - (1 - cfg.server_beta2) * d * d * torch.sign(v - d * d)
                else:
                    v = v + d * d
                self.v[k] = v
                gstate[k] = gstate[k] - cfg.server_lr * m / (v.sqrt() + cfg.server_tau)
            else:
                raise ValueError(alg)
        return gstate


def train_federated(cfg, store, f, log):
    torch.manual_seed(cfg.seed)
    gmodel = make_model(cfg, store)
    scope = "body" if cfg.algorithm == "fedper" else "all"
    keys = gmodel.shared_keys(scope)
    float_keys = [k for k in keys if gmodel.state_dict()[k].is_floating_point()]
    server = Server(cfg, float_keys)
    clients = list(range(4))
    idx = {c: client_idx(store, c) for c in clients}
    n = np.array([len(idx[c]) for c in clients], float)
    p = n / n.sum()
    samplers = {c: Sampler(idx[c], cfg.seed * 100 + c, store.device) for c in clients}
    # Per-client persistent state: full model (embeddings, and heads for FedPer), Adam state, Ditto personal model.
    cmodels = {c: copy.deepcopy(gmodel) for c in clients}
    copts = {c: make_opt(cfg, cmodels[c]) for c in clients}
    pmodels = {c: copy.deepcopy(gmodel) for c in clients} if cfg.algorithm == "ditto" else None
    popts = {c: make_opt(cfg, pmodels[c]) for c in clients} if pmodels else None
    rounds = max(1, cfg.max_steps // cfg.local_steps)
    eval_rounds = max(1, cfg.eval_every // cfg.local_steps)
    params_per_round = sum(gmodel.state_dict()[k].numel() for k in float_keys) * 2 * len(clients)
    best, best_states, bad, hist = math.inf, None, 0, []
    gstate = copy.deepcopy(gmodel.state_dict())
    for r in range(1, rounds + 1):
        for c in clients:
            sd = cmodels[c].state_dict()
            for k in keys:
                sd[k].copy_(gstate[k])
            prox = (gstate, cfg.mu, float_keys) if cfg.algorithm == "fedprox" and cfg.mu else None
            for _ in range(cfg.local_steps):
                step(cfg, store, f, cmodels[c], copts[c], samplers[c], prox)
            if pmodels:
                for _ in range(cfg.local_steps):
                    step(cfg, store, f, pmodels[c], popts[c], samplers[c], (gstate, cfg.mu, float_keys))
        avg = {k: sum(p[c] * cmodels[c].state_dict()[k] for c in clients) for k in float_keys}
        gstate = server.apply(gstate, avg)
        if r % eval_rounds == 0 or r == rounds:
            ev = eval_models(cfg, cmodels, pmodels, gstate, keys)
            e, a = val_error(ev, store, f)
            w = 100 * e / a
            hist.append(dict(round=r, val_wape=w, communicated_params=params_per_round * r))
            log(f"round {r}/{rounds} val WAPE {w:.3f}%")
            if w < best - 1e-4:
                best, best_states, bad = w, {c: copy.deepcopy(m.state_dict()) for c, m in ev.items()}, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    final = {c: copy.deepcopy(gmodel) for c in clients}
    for c in clients:
        final[c].load_state_dict(best_states[c])
    info = dict(history=hist, best_val_wape=best, rounds=r, communicated_params_per_round=params_per_round)
    if cfg.algorithm == "fedavg_ft":
        for c in clients:
            ft_cfg = copy.copy(cfg)
            ft_cfg.max_steps = cfg.ft_steps
            m, fi = finetune(ft_cfg, store, f, final[c], idx[c], [c], log, f"ft client {c}")
            final[c] = m
            info[f"finetune_client_{c}"] = fi
        e, a = val_error(final, store, f)
        info["best_val_wape"] = 100 * e / a
    return final, info


def eval_models(cfg, cmodels, pmodels, gstate, keys):
    """Models to evaluate per client: personal (Ditto), client model with the global shared part (FedPer), or global."""
    if pmodels:
        return pmodels
    out = {}
    for c, m in cmodels.items():
        mm = copy.deepcopy(m)
        sd = mm.state_dict()
        for k in keys:
            sd[k].copy_(gstate[k])
        out[c] = mm
    return out


def finetune(cfg, store, f, model, idx, val_clients, log, tag):
    opt = make_opt(cfg, model, cfg.lr * cfg.extra.get("ft_lr_scale", 0.3))
    sampler = Sampler(idx, cfg.seed + 7, store.device)
    best_e, a = val_error({None: model}, store, f, val_clients)
    best, best_state, bad, hist = 100 * best_e / a, copy.deepcopy(model.state_dict()), 0, []
    for s in range(1, cfg.max_steps + 1):
        step(cfg, store, f, model, opt, sampler)
        if s % cfg.eval_every == 0:
            e, a = val_error({None: model}, store, f, val_clients)
            w = 100 * e / a
            hist.append(dict(step=s, val_wape=w))
            log(f"{tag} step {s} val WAPE {w:.3f}%")
            if w < best - 1e-4:
                best, best_state, bad = w, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    return model, dict(history=hist, best_val_wape=best)


# ----------------------------------------------------------------- entry point
def run(cfg: Config, store=None, save_predictions=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False   # full fp32: validation numbers must match the frozen CPU study
    torch.backends.cudnn.allow_tf32 = False
    store = store or D.load(device)
    f = cfg.features()
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lf = open(logs / f"{cfg.exp_id}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()

    log(f"config {json.dumps(asdict(cfg))}")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if cfg.algorithm == "central":
        model, info = train_single(cfg, store, f, store.train, None, log, "central")
        models = {None: model}
    elif cfg.algorithm == "local":
        models, info = {}, {"clients": {}}
        for c in range(4):
            models[c], info["clients"][c] = train_single(cfg, store, f, client_idx(store, c), [c], log, f"client {c}")
    else:
        models, info = train_federated(cfg, store, f, log)
    runtime = time.time() - t0
    rep = evaluate(models, store, f)
    lat = latency(models, store, f)
    ck = OUT / "checkpoints" / cfg.exp_id
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({str(c): m.state_dict() for c, m in models.items()}, ck / "models.pt")
    (ck / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    if save_predictions:
        vp = OUT / "validation_predictions"
        vp.mkdir(parents=True, exist_ok=True)
        p = predict_raw(models, store, f, store.val).cpu().numpy().astype(np.float32)
        np.save(vp / f"{cfg.exp_id}.npy", p)
    entry = dict(exp_id=cfg.exp_id, algorithm=cfg.algorithm, architecture=cfg.kind, config=asdict(cfg), seed=cfg.seed,
                 runtime_s=round(runtime, 1), status="completed", checkpoint=str(ck.relative_to(ROOT)),
                 n_params=sum(p.numel() for p in next(iter(models.values())).parameters()),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
                 inference_ms_per_window_batch1=lat, train_info=info, validation=rep, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(OUT / "registry.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=float) + "\n")
    log(f"DONE {cfg.exp_id}: val WAPE {rep['wape_pct']:.3f}% MAE {rep['mae']:.3f} RMSE {rep['rmse']:.3f} sMAPE {rep['smape_pct']:.3f}% ({runtime / 60:.1f} min)")
    lf.close()
    return entry


def evaluate(models, store, f):
    from .metrics import full_report
    p = predict_raw(models, store, f, store.val).double().cpu().numpy()
    v = store.val.cpu().numpy()
    flow = store.flow.cpu().numpy()
    y = flow[v[:, 0][:, None], v[:, 1][:, None] + np.arange(12)].astype(np.float64)
    m = np.isfinite(y)
    rep = full_report(p, np.nan_to_num(y), m, v[:, 0], v[:, 1], store.client.cpu().numpy()[v[:, 0]], D.CLIENTS)
    rep.pop("by_hour_share_of_abs_error", None)
    return rep


@torch.no_grad()
def latency(models, store, f):
    m = next(iter(models.values()))
    m.eval()
    b = D.batch(store, store.val[:1], f)
    for _ in range(3):
        m(b["x"], b["xf"], b["s"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        m(b["x"], b["xf"], b["s"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return round((time.perf_counter() - t) / 20 * 1000, 3)
