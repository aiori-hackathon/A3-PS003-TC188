"""Centralised training harness: seeded, CPU, masked loss, validation-only early stopping.

Leakage guards: `fit` accepts only a `train` dataset for optimiser updates and a `val`
dataset for early stopping; evaluation runs under `torch.no_grad()` with no optimiser.
The test split is never touched here.
"""

import copy
import logging
import random
import time

import numpy as np
import torch

from aiori import evaluation
from aiori.dataset import make_loader
from aiori.losses import masked_mse

log = logging.getLogger(__name__)

EVAL_BATCH = 4096


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch(threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_one_epoch(model, loader, optimizer, grad_clip=1.0, max_batches=None):
    """One pass over `loader`. Batches with no observed target take no step and are counted.

    Raises if a loss or gradient norm is non-finite; reports pre-clipping gradient norms.
    """
    model.train()
    loss_sum, n_sum, steps, skipped, clipped, norms = 0.0, 0, 0, 0, 0, []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, n = masked_mse(model(batch["x"]), batch["y"], batch["mask"])
        if loss is None:
            skipped += 1
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {i}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf"),
                                              error_if_nonfinite=True).item()
        norms.append(norm)
        clipped += bool(grad_clip) and norm > grad_clip
        optimizer.step()
        loss_sum += loss.item() * n
        n_sum += n
        steps += 1
    return {"train_loss": loss_sum / n_sum if n_sum else float("nan"), "steps": steps,
            "skipped_batches": skipped, "observed_cells": n_sum, "clipped_steps": int(clipped),
            "grad_norm_median": float(np.median(norms)) if norms else float("nan"),
            "grad_norm_max": float(np.max(norms)) if norms else float("nan")}


@torch.no_grad()
def predict(model, dataset, batch_size=EVAL_BATCH):
    """z-scaled predictions for every window of `dataset`, in index order (numpy arrays)."""
    model.eval()
    parts = {k: [] for k in ("y_hat_z", "mask", "sensor_pos", "t0")}
    for batch in make_loader(dataset, batch_size):
        parts["y_hat_z"].append(model(batch["x"]).numpy())
        for k in ("mask", "sensor_pos", "t0"):
            parts[k].append(batch[k].numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def fit(model, train_ds, val_ds, run, *, seed, lr=1e-3, batch_size=512, max_epochs=50, patience=5,
        grad_clip=1.0, max_batches=None):
    """Adam training with early stopping on pooled validation MAE (raw units); restores the best weights."""
    if train_ds.split != "train":
        raise ValueError(f"Optimiser updates require the train split, got {train_ds.split!r}")
    if val_ds.split != "val":
        raise ValueError(f"Early stopping requires the val split, got {val_ds.split!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed)
    best = {"mae": float("inf"), "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    history, early_stopped = [], False
    for epoch in range(1, max_epochs + 1):
        t = time.perf_counter()
        stats = train_one_epoch(model, loader, optimizer, grad_clip, max_batches)
        t_train = time.perf_counter() - t
        val = evaluation.evaluate_predictions(predict(model, val_ds), run)["pooled"]
        stats.update(epoch=epoch, val_mae=val["mae"], val_rmse=val["rmse"], val_n=val["n"],
                     train_seconds=t_train, epoch_seconds=time.perf_counter() - t)
        history.append(stats)
        log.info("epoch %d: train_loss %.4f val_mae %.3f (%.1fs)", epoch, stats["train_loss"], val["mae"],
                 stats["epoch_seconds"])
        if val["mae"] < best["mae"]:
            best = {"mae": val["mae"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= patience:
            early_stopped = True
            break
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_mae": best["mae"],
            "early_stopped": early_stopped}
