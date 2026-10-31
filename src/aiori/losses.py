"""NaN-safe masked MSE.

`torch.where(mask, y_hat - y, 0)` selects instead of multiplying, so a NaN target on a masked
cell can affect neither the loss (NaN * 0 = NaN) nor the gradient.
"""

import torch


def masked_mse(y_hat, y, mask):
    """(loss, n_valid); loss is None when no target is observed (the caller skips the step)."""
    n = int(mask.sum())
    if n == 0:
        return None, 0
    diff = torch.where(mask, y_hat - y, torch.zeros_like(y_hat))
    return (diff ** 2).sum() / n, n
