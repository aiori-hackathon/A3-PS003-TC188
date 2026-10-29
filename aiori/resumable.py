"""Resumable versions of training.fit, federated.run_fedavg and federated.run_local_only (Phase 9).

Each mirrors the original loop operation for operation (same seeding, loaders, optimiser, early stopping), and adds
an atomic checkpoint after every epoch / round holding model and optimiser state, the shuffle generator state, the
RNG states, the epoch / round, the best-validation state, the early-stopping state and the history. Resuming from
any checkpoint reproduces the uninterrupted run exactly (tests/test_phase9_resumable.py).
"""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from aiori import dataset, evaluation, training
from aiori.federated import Client, Server, check_partition, client_seed, merge_predictions
from aiori.model import ForecastGRU


class Interrupted(RuntimeError):
    """Raised by the `stop_after` test hook or the deadline guard; the checkpoint stays valid."""


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _guard(done_now, stop_after, deadline):
    if stop_after is not None and done_now >= stop_after:
        raise Interrupted(f"stopped after {done_now} unit(s) (test hook)")
    if deadline is not None and time.time() > deadline:
        raise Interrupted("deadline reached")


def fit(model, train_ds, val_ds, run, *, seed, ckpt, lr=1e-3, batch_size=512, max_epochs=50, patience=5,
        grad_clip=1.0, max_batches=None, stop_after=None, deadline=None, on_epoch=None):
    """training.fit with a checkpoint after every epoch; restores the best weights at the end."""
    if train_ds.split != "train" or val_ds.split != "val":
        raise ValueError("Optimiser updates require train, early stopping requires val")
    ckpt = Path(ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = dataset.make_loader(train_ds, batch_size, shuffle=True, seed=seed)
    best = {"mae": float("inf"), "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    history, early_stopped, start, finished = [], False, 1, False
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        optimizer.load_state_dict(c["optimizer"])
        loader.generator.set_state(c["generator"])
        _set_rng(c["rng"])
        best, history, early_stopped, finished = c["best"], c["history"], c["early_stopped"], c["finished"]
        start = c["epoch"] + 1
    done_now = 0
    for epoch in range(start, max_epochs + 1):
        if finished:
            break
        _guard(done_now, stop_after, deadline)
        t = time.perf_counter()
        stats = training.train_one_epoch(model, loader, optimizer, grad_clip, max_batches)
        t_train = time.perf_counter() - t
        val = evaluation.evaluate_predictions(training.predict(model, val_ds), run)["pooled"]
        stats.update(epoch=epoch, val_mae=val["mae"], val_rmse=val["rmse"], val_n=val["n"],
                     train_seconds=t_train, epoch_seconds=time.perf_counter() - t)
        history.append(stats)
        if val["mae"] < best["mae"]:
            best = {"mae": val["mae"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= patience:
            early_stopped = True
        finished = early_stopped or epoch == max_epochs
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), generator=loader.generator.get_state(),
                         rng=_rng_state(), best=best, history=history, early_stopped=early_stopped, finished=finished,
                         epoch=epoch), ckpt)
        done_now += 1
        if on_epoch:
            on_epoch(stats)
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_mae": best["mae"], "early_stopped": early_stopped}


def fedavg(run, partition, cfg, seed, *, ckpt, rounds_max=None, max_batches=None, stop_after=None, deadline=None,
           on_round=None):
    """federated.run_fedavg with a checkpoint after every round (clients reset Adam each round, so no client state)."""
    check_partition(partition, run.n_sensors)
    if cfg.get("optimizer_state", "reset_each_round") != "reset_each_round":
        raise ValueError("Resumable FedAvg supports reset_each_round only")
    clients = [Client(name, run, pos, cfg) for name, pos in sorted(partition.items())]
    training.seed_everything(seed)
    model = ForecastGRU(clients[0].train_ds.n_features, cfg["hidden_size"])
    server = Server(model.state_dict())
    rounds_max = rounds_max or cfg["rounds_max"]
    best = {"mae": float("inf"), "round": 0, "state": copy.deepcopy(server.state)}
    history, early_stopped, start, finished = [], False, 1, False
    ckpt = Path(ckpt)
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        server.state, server.byte_log = c["global"], c["byte_log"]
        _set_rng(c["rng"])
        best, history, early_stopped, finished = c["best"], c["history"], c["early_stopped"], c["finished"]
        start = c["round"] + 1
    done_now = 0
    for r in range(start, rounds_max + 1):
        if finished:
            break
        _guard(done_now, stop_after, deadline)
        t = time.perf_counter()
        client_stats = {}
        for idx, cl in enumerate(clients):
            state, n_k, stats = cl.local_train(server.send(r, cl.name), client_seed(seed, r, idx), cfg["local_epochs"], max_batches)
            server.receive(r, cl.name, state, n_k)
            client_stats[cl.name] = stats
        weights = server.aggregate_round()
        t_train = time.perf_counter() - t
        val = server.pool([cl.validation_sums(server.send(r, cl.name, purpose="eval")) for cl in clients])
        flat = [s for st in client_stats.values() for s in st]
        n_cells = sum(s["observed_cells"] for s in flat)
        history.append({
            "round": r, "val_mae": val["mae"], "val_rmse": val["rmse"], "val_n": val["n"],
            "train_loss": sum(s["train_loss"] * s["observed_cells"] for s in flat) / n_cells,
            "steps": sum(s["steps"] for s in flat), "skipped_batches": sum(s["skipped_batches"] for s in flat),
            "clipped_steps": sum(s["clipped_steps"] for s in flat), "grad_norm_max": max(s["grad_norm_max"] for s in flat),
            "weights": weights,
            "train_bytes": sum(b["bytes"] for b in server.byte_log if b["round"] == r and b["purpose"] == "train"),
            "train_seconds": t_train, "round_seconds": time.perf_counter() - t, "client_stats": client_stats})
        if val["mae"] < best["mae"]:
            best = {"mae": val["mae"], "round": r, "state": copy.deepcopy(server.state), "pooled": val}
        elif r - best["round"] >= cfg["patience"]:
            early_stopped = True
        finished = early_stopped or r == rounds_max
        atomic_save(dict(**{"global": server.state}, byte_log=server.byte_log, rng=_rng_state(), best=best, history=history,
                         early_stopped=early_stopped, finished=finished, round=r), ckpt)
        done_now += 1
        if on_round:
            on_round(history[-1])
    server.state = best["state"]
    pred = merge_predictions([cl.predict_val(best["state"]) for cl in clients])
    return {"state": best["state"], "best_round": best["round"], "best_val_mae": best["mae"],
            "best_pooled_from_sums": best.get("pooled"), "early_stopped": early_stopped, "history": history,
            "byte_log": server.byte_log, "predictions": pred, "n_k": {cl.name: cl.n_train for cl in clients}}


def local_only(run, partition, cfg, seed, *, ckpt_dir, max_epochs=None, max_batches=None, stop_after=None, deadline=None,
               on_epoch=None):
    """federated.run_local_only with per-client resumable training (one checkpoint per client)."""
    check_partition(partition, run.n_sensors)
    out, preds, budget = {"clients": {}, "states": {}}, [], [stop_after]
    for idx, (name, pos) in enumerate(sorted(partition.items())):
        c = Client(name, run, pos, cfg)
        training.seed_everything(seed)
        model = ForecastGRU(c.train_ds.n_features, cfg["hidden_size"])
        ck = Path(ckpt_dir) / f"client_{name}.pt"
        before = len(torch.load(ck, weights_only=False)["history"]) if ck.exists() else 0
        try:
            res = fit(model, c.train_ds, c.val_ds, c.run, seed=client_seed(seed, 0, idx), ckpt=ck, lr=cfg["lr"],
                      batch_size=cfg["batch_size"], max_epochs=max_epochs or cfg["max_epochs"], patience=cfg["patience"],
                      grad_clip=cfg["grad_clip"], max_batches=max_batches, stop_after=budget[0], deadline=deadline,
                      on_epoch=on_epoch)
        finally:
            if budget[0] is not None and ck.exists():
                budget[0] = max(0, budget[0] - (len(torch.load(ck, weights_only=False)["history"]) - before))
        out["clients"][name] = res
        out["states"][name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        preds.append(c.predict_val(out["states"][name]))
    out["predictions"] = merge_predictions(preds)
    return out
