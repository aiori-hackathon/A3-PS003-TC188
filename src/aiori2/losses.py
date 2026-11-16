"""The one masked training loss for every architecture and arm.

Targets arrive as raw flow with NaN where invalid. They are scaled with the stream's own training scaler AFTER
replacing invalid entries by 0 (torch.where, so no NaN ever enters arithmetic), and the loss is taken over the
valid elements only (boolean selection), normalised by their count. Invalid targets therefore contribute exactly
zero to the loss and to every gradient. A batch with no valid target returns None: the caller skips the update.
"""

import numpy as np
import torch


def scaled_targets(y_raw, mask, mean, std):
    """Raw flow (B, H) -> scaled units with the per-sample scaler (mean/std shape (B,)); invalid entries become 0-derived."""
    y = torch.where(mask, y_raw, torch.zeros_like(y_raw))
    return (y - mean[:, None]) / std[:, None]


def masked_loss(pred, y_raw, mask, mean, std, kind="mae"):
    if kind not in ("mae", "mse"):
        raise ValueError(f"unknown loss {kind!r}")
    if not bool(mask.any()):
        return None
    d = pred[mask] - scaled_targets(y_raw, mask, mean, std)[mask]
    return d.abs().mean() if kind == "mae" else (d * d).mean()


def batch_tensors(loader, idx, device="cpu", channels=None):
    """One loader batch as tensors: X, raw Y (NaN kept), mask, per-sample scaler mean/std, sensor positions."""
    X, Y, M, s = loader.get(idx)
    if channels is not None and X.shape[-1] != channels:
        raise ValueError(f"loader gives {X.shape[-1]} input channels, model expects {channels}")
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    return t(X), t(Y), t(M, torch.bool), t(loader.mean[s]), t(loader.std[s]), t(s, torch.long)


def train_step(model, optimizer, batch, kind="mae", clip=1.0):
    """One masked update. Returns the loss value, or None when the batch has no valid target (no update)."""
    X, Y, M, mean, std, _ = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = masked_loss(model(X), Y, M, mean, std, kind)
    if loss is None:
        return None
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return float(loss.detach())
