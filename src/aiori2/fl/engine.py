"""Single-process federated simulation on the AIORI 2.0 three-year data (no networking, no raw-data exchange).

Round t: deterministically select participants -> each starts from the same global state, trains `local_steps`
optimizer steps on its OWN training stream (fresh client optimizer every round, never shared) -> returns
Delta_i = w_i - w plus algorithm extras (step count, SCAFFOLD dc_i) -> the algorithm's server rule updates w and its
persistent state. Validation/test streams are never passed to local training. Randomness is stateless: every
(round, client) derives its own seed, so a resumed run repeats an uninterrupted one exactly (CPU).
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from aiori2 import arms, core
from aiori2.fl import algorithms
from aiori2.losses import batch_tensors, masked_loss
from aiori2.models import build_model

FL_ARMS = ("fedavg_iid", "fedavg_geo")
DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "fl" / "defaults.json").read_text(encoding="utf-8"))
WEIGHTINGS = ("train_windows", "valid_targets_used", "uniform")


def make_config(**kw):
    cfg = copy.deepcopy(DEFAULTS["common"])
    alg = kw.get("algorithm", cfg["algorithm"])
    cfg.update(DEFAULTS["algorithms"].get(alg, {}))
    cfg.update(kw)
    validate(cfg)
    return cfg


def validate(cfg):
    a = cfg["algorithm"]
    if a not in algorithms.REGISTRY:
        raise ValueError(f"unknown algorithm {a!r}")
    if cfg["arm"] not in FL_ARMS:
        raise ValueError(f"federated algorithms run only on {FL_ARMS}, not {cfg['arm']!r}")
    cls = algorithms.REGISTRY[a]
    if cfg["client_optimizer"] not in cls.client_optimizers:
        raise ValueError(f"{a} supports client optimizers {cls.client_optimizers}")
    if cls.scaffold and cfg["client_momentum"]:
        raise ValueError("SCAFFOLD (Option II) is implemented for plain SGD only")
    if (cls.scaffold or cls.nova) and cfg["mu"]:
        raise ValueError(f"{a} does not support a proximal term")
    if cfg["mu"] and not cls.uses_prox:
        raise ValueError("mu > 0 is only meaningful for fedprox")
    if cfg["client_momentum"] and (cfg["client_optimizer"] != "sgd" or not 0 <= cfg["client_momentum"] < 1):
        raise ValueError("client_momentum needs SGD and 0 <= rho < 1")
    if cfg["weighting"] not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if not 0 < cfg["participation"] <= 1 or cfg["server_lr"] <= 0 or cfg["client_lr"] <= 0:
        raise ValueError("need 0 < participation <= 1 and positive learning rates")
    if not (0 <= cfg["beta1"] < 1 and 0 <= cfg["beta2"] < 1 and cfg["tau"] > 0 and 0 <= cfg["server_momentum"] < 1):
        raise ValueError("invalid server optimizer hyper-parameters")
    if cfg["history"] not in arms.HISTORIES:
        raise ValueError("history must be A or B")
    return cfg


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "device"}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def _seed(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def payload_bytes(tensors, like=None):
    """Logical payload; with `like`, each tensor counts at the model parameter's dtype (server state is kept in
    float64 for accuracy but would be transmitted at parameter precision)."""
    return int(sum(t.numel() * (like[k].element_size() if like is not None else t.element_size()) for k, t in tensors.items()))


# ---- one client's local training (pure: model, its own batches, its own optimizer) ----

def local_train(model, batches, cfg, anchor=None, correction=None, loss_fn=None):
    """Run the local optimizer over `batches`; returns steps taken, skipped batches, valid targets, mean loss.

    anchor: round-start parameters (FedProx reference, never updated here); correction: SCAFFOLD c - c_i per name.
    Parameters with requires_grad=False are untouched. A parameter that the loss does not use gets grad None; its true
    gradient is zero, so plain/momentum SGD leaves it unchanged and SCAFFOLD applies only the correction term.
    """
    params = dict(model.named_parameters())
    train = [p for p in params.values() if p.requires_grad]
    if cfg["client_optimizer"] == "sgd":
        opt = torch.optim.SGD(train, lr=cfg["client_lr"], momentum=cfg["client_momentum"])
    else:
        opt = torch.optim.Adam(train, lr=cfg["client_lr"])
    loss_fn = loss_fn or (lambda m, b: masked_loss(m(b[0]), b[1], b[2], b[3], b[4], cfg["loss"]))
    steps = skipped = 0
    valid = 0
    losses = []
    model.train()
    for b in batches:
        loss = loss_fn(model, b)
        if loss is None:
            skipped += 1
            continue
        if anchor is not None and cfg["mu"]:
            loss = loss + cfg["mu"] / 2 * sum(((params[k] - anchor[k]) ** 2).sum() for k in anchor if params[k].requires_grad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(train, cfg["grad_clip"])
        if correction is not None:
            for k, p in params.items():
                if p.requires_grad and k in correction:
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    p.grad = g + correction[k].to(p.dtype)
        opt.step()
        steps += 1
        valid += int(b[2].sum()) if isinstance(b, (tuple, list)) and len(b) > 2 and torch.is_tensor(b[2]) else 0
        losses.append(float(loss.detach()))
    return dict(steps=steps, skipped=skipped, valid_targets=valid, loss=float(np.mean(losses)) if losses else None)


# ---- the federated run ----

class FederatedRun:
    def __init__(self, cfg, root=None, loaders=None):
        """loaders: optional prebuilt arms.loaders(root, arm, history, "train", policy) to reuse across runs."""
        self.cfg = validate(dict(cfg))
        self.root = Path(root) if root else core.load_config()["_out"]
        self.device = torch.device(cfg.get("device") or "cpu")
        self.loaders = loaders or arms.loaders(self.root, cfg["arm"], cfg["history"], "train", cfg["policy"])   # training only
        if sorted(self.loaders) != sorted(s.name for s in arms.streams(self.root, cfg["arm"])):
            raise ValueError("loaders do not match the arm's clients")
        self.clients = sorted(self.loaders)
        self.model = build_model(cfg["model"], cfg["preset"], cfg["in_channels"], seed=cfg["seed"]).to(self.device)
        template = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        self.alg = algorithms.make(cfg["algorithm"], cfg, template, self.clients)
        self.round = 0
        self.log = []

    # deterministic, stateless choices
    def participants(self, r):
        n = len(self.clients)
        k = max(1, int(round(self.cfg["participation"] * n)))
        rng = np.random.default_rng(_seed(self.cfg["seed"], 1, r))
        return sorted(self.clients[i] for i in rng.choice(n, size=k, replace=False))

    def steps_for(self, client):
        s = self.cfg["local_steps"]
        return int(s[client]) if isinstance(s, dict) else int(s)

    def client_batches(self, client, r):
        ld = self.loaders[client]
        ci = self.clients.index(client)
        k, bs = self.steps_for(client), self.cfg["batch_size"]
        rng = np.random.default_rng(_seed(self.cfg["seed"], 2, r, ci))
        idx = rng.choice(len(ld), size=min(len(ld), k * bs), replace=False)
        for j in range(k):
            part = idx[j * bs:(j + 1) * bs]
            if len(part):
                yield batch_tensors(ld, part, self.device, self.cfg["in_channels"])

    def run_round(self):
        r = self.round + 1
        t0 = time.perf_counter()
        gstate = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        results, acct = [], dict(round=r, clients={})
        for c in self.participants(r):
            torch.manual_seed(_seed(self.cfg["seed"], 3, r, self.clients.index(c)))           # dropout etc.
            m = copy.deepcopy(self.model)
            m.load_state_dict(gstate)                                                         # identical start
            anchor = {k: v.detach().clone() for k, v in m.named_parameters()} if self.alg.uses_prox else None
            corr = None
            if self.alg.scaffold:
                corr = {k: v.to(self.device) for k, v in self.alg.correction(c).items()}
            tc = time.perf_counter()
            st = local_train(m, self.client_batches(c, r), self.cfg, anchor, corr)
            dt = time.perf_counter() - tc
            new = m.state_dict()
            delta = {k: (new[k].detach() - gstate[k]).cpu() for k in self.alg.keys}
            res = dict(client=c, delta=delta, steps=st["steps"], valid_targets=st["valid_targets"])
            down = payload_bytes({k: gstate[k] for k in self.alg.keys})
            up = payload_bytes(delta) + 8                                                     # + weight (one float64)
            if self.alg.scaffold:
                down += payload_bytes(self.alg.state["c"], gstate)
                if st["steps"] > 0:
                    res["dc"] = self.alg.client_control(c, delta, st["steps"], self.cfg["client_lr"])
                    up += payload_bytes(res["dc"], gstate)
            if self.alg.nova:
                up += 8                                                                           # ||a_i||_1
            weight = dict(train_windows=len(self.loaders[c]), valid_targets_used=st["valid_targets"], uniform=1)[self.cfg["weighting"]]
            acct["clients"][c] = dict(steps=st["steps"], skipped_batches=st["skipped"], valid_targets=st["valid_targets"],
                                      weight=weight, loss=st["loss"], download_bytes=down, upload_bytes=up, local_seconds=round(dt, 4))
            if st["steps"] > 0:
                res["weight"] = weight
                results.append(res)
        ts = time.perf_counter()
        if results:
            # server arithmetic always on CPU (float64), so its state is device-independent and checkpoint-portable
            newg = self.alg.update({k: v.cpu() for k, v in gstate.items()}, results, [x["weight"] for x in results])
            self.model.load_state_dict({k: v.to(self.device) for k, v in newg.items()})
        acct.update(participants=list(acct["clients"]), aggregated=[x["client"] for x in results],
                    server_seconds=round(time.perf_counter() - ts, 4), round_seconds=round(time.perf_counter() - t0, 4),
                    upload_bytes=sum(v["upload_bytes"] for v in acct["clients"].values()),
                    download_bytes=sum(v["download_bytes"] for v in acct["clients"].values()),
                    payload_note="logical payload of the exchanged tensors; single-process simulation, no network measured")
        acct["total_bytes"] = acct["upload_bytes"] + acct["download_bytes"]
        if self.alg.nova and results:
            acct["tau_eff"] = self.alg.last_tau_eff
        self.round = r
        self.log.append(acct)
        return acct

    def run(self, rounds):
        for _ in range(rounds):
            self.run_round()
        return self.log

    # ---- checkpoints: tensors + JSON metadata, weights_only loadable, atomic, never overwritten ----
    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(algorithm=self.cfg["algorithm"], algorithm_version=self.alg.version, config=self.cfg,
                    config_sha256=config_hash(self.cfg), round=self.round, clients=self.clients, model=self.model.config,
                    data_manifest_sha256=core.sha256_file(self.root / "manifests" / "dgx_transfer_manifest.json")
                    if (self.root / "manifests" / "dgx_transfer_manifest.json").is_file() else None,
                    rng="stateless: every (round, client) seeds from the config seed", log=self.log)
        tmp = path.with_name(path.name + ".partial")
        torch.save(dict(model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                        server=self.alg.state_dict(), meta_json=json.dumps(meta, default=str)), tmp)
        tmp.replace(path)

    def resume(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = json.loads(ck["meta_json"])
        if meta["config_sha256"] != config_hash(self.cfg) or meta["clients"] != self.clients:
            raise ValueError("checkpoint was written by a different configuration or client set")
        self.model.load_state_dict({k: v.to(self.device) for k, v in ck["model"].items()})
        self.alg.load_state_dict(ck["server"])
        self.round, self.log = meta["round"], meta["log"]
        return meta
