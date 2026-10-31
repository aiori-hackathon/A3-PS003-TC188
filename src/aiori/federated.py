"""FedAvg simulation for the B1 GRU (docs/phase7_technical_spec.md).

Clients run sequentially in one process. Isolation is by construction:
- a `Client` holds a sliced, read-only copy of its own sensors' arrays (`subset_run`) and no
  reference to the full run;
- the `Server` holds only the global state_dict, aggregation weights, metric sums and a byte log,
  and its methods reject anything else.
Local training reuses the Phase 6 harness (`training.train_one_epoch`, `training.fit`). Only
train and val windows are used; the test split is never touched.
"""

import copy
import dataclasses
import io
import logging
import time

import numpy as np
import torch

from aiori import dataset, evaluation, metrics, training
from aiori.model import ForecastGRU

log = logging.getLogger(__name__)

PER_SENSOR_SCALER_KEYS = ("mean", "std", "n_obs", "array_index", "station_id")


# ---- partitions and client data ----

def geographic_partition(run):
    return {str(c): p for c, p in dataset.client_positions(run).items()}


def iid_partition(run, partition_seed, n_clients=4):
    """Seeded balanced sensor-level partition: shuffle sensors, deal round-robin."""
    if run.n_sensors % n_clients:
        raise ValueError(f"{run.n_sensors} sensors cannot be split evenly into {n_clients} clients")
    perm = np.random.default_rng(partition_seed).permutation(run.n_sensors)
    return {f"iid_{k}": np.sort(perm[k::n_clients]) for k in range(n_clients)}


def check_partition(partition, n_sensors):
    allpos = np.concatenate(list(partition.values()))
    if len(allpos) != n_sensors or not np.array_equal(np.sort(allpos), np.arange(n_sensors)):
        raise ValueError("Partition is not a disjoint cover of all sensors")


def subset_run(run, positions, name=None):
    """A physically separate, read-only copy of `run` restricted to `positions` (re-indexed 0..k-1)."""
    positions = np.sort(np.asarray(positions, dtype=np.int64))
    remap = np.full(run.n_sensors, -1, dtype=np.int64)
    remap[positions] = np.arange(len(positions))
    win = {}
    for split, w in run.windows.items():
        keep = np.isin(w[:, 0], positions)
        win[split] = dataset._readonly(np.stack([remap[w[keep, 0]], w[keep, 1]], axis=1))
    scaler = {k: (np.asarray(v)[positions].tolist() if k in PER_SENSOR_SCALER_KEYS else copy.deepcopy(v))
              for k, v in run.scaler.items()}
    arrays = {k: dataset._readonly(np.array(getattr(run, k)[positions]))
              for k in ("inputs_scaled", "input_imputed", "targets_scaled", "observed_mask", "flow_raw",
                        "tod_median", "array_index", "station_id", "client")}
    return dataclasses.replace(run, name=name or run.name, scaler=scaler, windows=win, **arrays)


def client_seed(seed, round_idx, client_idx):
    """Independent shuffle stream per (training seed, round, client)."""
    return int(np.random.SeedSequence([seed, round_idx, client_idx]).generate_state(1)[0])


def payload_bytes(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return len(buf.getvalue())


def raw_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


def aggregate(states, weights):
    """Weighted mean of every tensor, accumulated in float64 and cast back to the original dtype."""
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(states) or (w <= 0).any():
        raise ValueError("Need one positive weight per state")
    w = w / w.sum()
    out = {}
    for k, ref in states[0].items():
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for wi, s in zip(w, states):
            acc += float(wi) * s[k].to(torch.float64)
        out[k] = acc.to(ref.dtype)
    return out


# ---- client ----

OPTIMIZER_STATES = ("reset_each_round", "persist_per_client")


class Client:
    def __init__(self, name, run, positions, cfg):
        self.name = name
        self.global_positions = np.sort(np.asarray(positions, dtype=np.int64))
        self.run = subset_run(run, self.global_positions, name=f"{run.name}:{name}")
        self.cfg = cfg
        self.optimizer_state = cfg.get("optimizer_state", "reset_each_round")
        if self.optimizer_state not in OPTIMIZER_STATES:
            raise ValueError(f"Unknown optimizer_state {self.optimizer_state!r}")
        self.train_ds = dataset.WindowDataset(self.run, "train", calendar=cfg["calendar"])
        self.val_ds = dataset.WindowDataset(self.run, "val", calendar=cfg["calendar"])
        self.n_train = len(self.train_ds)
        self._local_model = self._optimizer = None      # used only by persist_per_client; never sent

    def _model(self, state):
        model = ForecastGRU(self.train_ds.n_features, self.cfg["hidden_size"])
        model.load_state_dict(state)
        return model

    def _model_and_optimizer(self, global_state):
        """reset_each_round: fresh model and Adam every call.
        persist_per_client: one model object and one Adam per client for the whole run; the global
        weights are copied in place (load_state_dict keeps parameter identity, so Adam's moments and
        step counts carry over while the parameters start from the global model)."""
        if self.optimizer_state == "reset_each_round":
            model = self._model(global_state)
            return model, torch.optim.Adam(model.parameters(), lr=self.cfg["lr"])
        if self._local_model is None:
            self._local_model = self._model(global_state)
            self._optimizer = torch.optim.Adam(self._local_model.parameters(), lr=self.cfg["lr"])
        else:
            self._local_model.load_state_dict(global_state)
        return self._local_model, self._optimizer

    def adam_step_count(self):
        """Local Adam step counter (persist mode; 0 before training). For tests and logs only."""
        if self._optimizer is None:
            return 0
        steps = {int(s["step"]) for s in self._optimizer.state.values()}
        return steps.pop() if len(steps) == 1 else max(steps, default=0)

    def local_train(self, global_state, shuffle_seed, epochs=1, max_batches=None):
        """Train from the global weights; optimiser state never leaves the client. Returns (state, n_k, stats)."""
        model, optimizer = self._model_and_optimizer(global_state)
        loader = dataset.make_loader(self.train_ds, self.cfg["batch_size"], shuffle=True, seed=shuffle_seed)
        stats = [training.train_one_epoch(model, loader, optimizer, self.cfg["grad_clip"], max_batches)
                 for _ in range(epochs)]
        if self.optimizer_state == "persist_per_client":
            stats[-1]["adam_step_total"] = self.adam_step_count()
        return {k: v.detach().clone() for k, v in model.state_dict().items()}, self.n_train, stats

    def predict_val(self, state):
        """Own validation predictions with global sensor positions (stays with the client/harness)."""
        pred = training.predict(self._model(state), self.val_ds)
        if not np.isfinite(pred["y_hat_z"]).all():
            raise FloatingPointError(f"Non-finite predictions at client {self.name}")
        pred["local_pos"] = pred["sensor_pos"]
        pred["sensor_pos"] = self.global_positions[pred["sensor_pos"]]
        return pred

    def validation_sums(self, state):
        """Additive error sums in raw units (the only validation information sent to the server)."""
        pred = self.predict_val(state)
        index = np.stack([pred["local_pos"], pred["t0"]], axis=1)
        y_hat = evaluation.to_raw(pred["y_hat_z"], self.run, pred["local_pos"])
        y_true = evaluation.target_truth(self.run, index)
        a, s, n = metrics.metric_sums(y_hat, y_true, np.isfinite(y_true))
        ah, sh, nh = metrics.metric_sums(y_hat, y_true, np.isfinite(y_true), axis=0)
        return {"abs": float(a), "sq": float(s), "n": int(n),
                "abs_h": ah.tolist(), "sq_h": sh.tolist(), "n_h": nh.astype(int).tolist()}


# ---- server ----

SUM_KEYS = {"abs", "sq", "n", "abs_h", "sq_h", "n_h"}


class Server:
    """Holds only the global model, weights, metric sums and the byte log."""

    def __init__(self, initial_state):
        self._check_state(initial_state, reference=None)
        self.state = {k: v.detach().clone() for k, v in initial_state.items()}
        self.byte_log = []          # dicts: round, client, direction, purpose, bytes
        self._inbox = []

    def _check_state(self, state, reference):
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise TypeError("Server accepts only state_dicts of tensors")
        ref = reference if reference is not None else state
        if set(state) != set(ref) or any(state[k].shape != ref[k].shape for k in ref):
            raise TypeError("state_dict does not match the global model")

    def send(self, round_idx, client, purpose="train"):
        state = {k: v.clone() for k, v in self.state.items()}
        self.byte_log.append({"round": round_idx, "client": client, "direction": "down", "purpose": purpose,
                              "bytes": payload_bytes(state)})
        return state

    def receive(self, round_idx, client, state, n_k):
        self._check_state(state, self.state)
        if isinstance(n_k, bool) or not isinstance(n_k, (int, np.integer)) or n_k <= 0:
            raise TypeError("n_k must be a positive integer")
        self.byte_log.append({"round": round_idx, "client": client, "direction": "up", "purpose": "train",
                              "bytes": payload_bytes(state)})
        self._inbox.append((client, {k: v.detach().clone() for k, v in state.items()}, int(n_k)))

    def aggregate_round(self):
        if not self._inbox:
            raise RuntimeError("No client updates received")
        names, states, weights = zip(*sorted(self._inbox, key=lambda x: x[0]))
        self.state = aggregate(list(states), list(weights))
        self._inbox = []
        return dict(zip(names, (w / sum(weights) for w in weights)))

    @staticmethod
    def pool(sums):
        """Pooled metrics from client error sums (list of dicts with SUM_KEYS only)."""
        for s in sums:
            if not isinstance(s, dict) or set(s) != SUM_KEYS:
                raise TypeError("Server accepts only metric-sum dictionaries")
        tot = {k: np.sum([s[k] for s in sums], axis=0) for k in SUM_KEYS}
        out = metrics.from_sums(tot["abs"], tot["sq"], tot["n"])
        out["by_horizon"] = [dict(metrics.from_sums(a, q, n), horizon=h + 1)
                             for h, (a, q, n) in enumerate(zip(tot["abs_h"], tot["sq_h"], tot["n_h"]))]
        return out


# ---- training loops ----

def merge_predictions(preds):
    """Concatenate client predictions and order them like the full run's window index (sensor, t0)."""
    merged = {k: np.concatenate([p[k] for p in preds]) for k in ("y_hat_z", "mask", "sensor_pos", "t0")}
    order = np.lexsort((merged["t0"], merged["sensor_pos"]))
    return {k: v[order] for k, v in merged.items()}


def run_fedavg(run, partition, cfg, seed, rounds_max=None, max_batches=None):
    """Synchronous FedAvg with full participation; early stopping on pooled validation MAE."""
    check_partition(partition, run.n_sensors)
    clients = [Client(name, run, pos, cfg) for name, pos in sorted(partition.items())]
    training.seed_everything(seed)
    model = ForecastGRU(clients[0].train_ds.n_features, cfg["hidden_size"])   # same init as centralised
    server = Server(model.state_dict())
    rounds_max = rounds_max or cfg["rounds_max"]
    best = {"mae": float("inf"), "round": 0, "state": copy.deepcopy(server.state)}
    history, early_stopped = [], False
    for r in range(1, rounds_max + 1):
        t = time.perf_counter()
        client_stats = {}
        for idx, c in enumerate(clients):
            state, n_k, stats = c.local_train(server.send(r, c.name), client_seed(seed, r, idx),
                                              cfg["local_epochs"], max_batches)
            server.receive(r, c.name, state, n_k)
            client_stats[c.name] = stats
        weights = server.aggregate_round()
        t_train = time.perf_counter() - t
        val = server.pool([c.validation_sums(server.send(r, c.name, purpose="eval")) for c in clients])
        flat = [s for st in client_stats.values() for s in st]
        n_cells = sum(s["observed_cells"] for s in flat)
        history.append({
            "round": r, "val_mae": val["mae"], "val_rmse": val["rmse"], "val_n": val["n"],
            "train_loss": sum(s["train_loss"] * s["observed_cells"] for s in flat) / n_cells,
            "steps": sum(s["steps"] for s in flat), "skipped_batches": sum(s["skipped_batches"] for s in flat),
            "clipped_steps": sum(s["clipped_steps"] for s in flat),
            "grad_norm_max": max(s["grad_norm_max"] for s in flat), "weights": weights,
            "train_bytes": sum(b["bytes"] for b in server.byte_log if b["round"] == r and b["purpose"] == "train"),
            "train_seconds": t_train, "round_seconds": time.perf_counter() - t,
            "client_stats": client_stats,
        })
        log.info("round %d: val_mae %.3f (%.1fs)", r, val["mae"], history[-1]["round_seconds"])
        if val["mae"] < best["mae"]:
            best = {"mae": val["mae"], "round": r, "state": copy.deepcopy(server.state), "pooled": val}
        elif r - best["round"] >= cfg["patience"]:
            early_stopped = True
            break
    server.state = best["state"]
    pred = merge_predictions([c.predict_val(best["state"]) for c in clients])
    return {"state": best["state"], "best_round": best["round"], "best_val_mae": best["mae"],
            "best_pooled_from_sums": best.get("pooled"), "early_stopped": early_stopped, "history": history,
            "byte_log": server.byte_log, "predictions": pred, "n_k": {c.name: c.n_train for c in clients}}


def run_local_only(run, partition, cfg, seed, max_epochs=None, max_batches=None):
    """Each client trains alone from the same initial weights; early stopping on its own validation."""
    check_partition(partition, run.n_sensors)
    out = {"clients": {}, "states": {}}
    preds = []
    for idx, (name, pos) in enumerate(sorted(partition.items())):
        c = Client(name, run, pos, cfg)
        training.seed_everything(seed)
        model = ForecastGRU(c.train_ds.n_features, cfg["hidden_size"])
        fit = training.fit(model, c.train_ds, c.val_ds, c.run, seed=client_seed(seed, 0, idx), lr=cfg["lr"],
                           batch_size=cfg["batch_size"], max_epochs=max_epochs or cfg["max_epochs"],
                           patience=cfg["patience"], grad_clip=cfg["grad_clip"], max_batches=max_batches)
        out["clients"][name] = fit
        out["states"][name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        preds.append(c.predict_val(out["states"][name]))
    out["predictions"] = merge_predictions(preds)
    return out
