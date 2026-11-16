"""Causal TCN: stacked dilated 1-D convolutions with left-only padding, residual connections, direct 12-step head.

Generalises the study's TCN (aopt `Forecaster(kind="tcn")`): with kernel_size=3, dilation_base=2 and dropout=0
it computes exactly the same function (verified by a weight-copy test). Block i uses dilation base**i and pads
(kernel_size-1)*dilation zeros on the left only, so output step t sees inputs <= t. Receptive field
1 + (k-1) * sum(base**i); the defaults (k=3, 3 layers, base 2) cover 15 >= 12 steps.
"""

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0.0)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return self.drop(self.act(self.conv(self.pad(h))))


class TCN(nn.Module):
    def __init__(self, in_channels=1, input_len=12, horizon=12, channels=64, layers=3, kernel_size=3, dilation_base=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(CausalBlock(in_channels if i == 0 else channels, channels, kernel_size, dilation_base ** i, dropout)
                                    for i in range(layers))
        self.head = nn.Linear(channels, horizon)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilation_base ** i for i in range(layers))

    def features(self, x):
        """(B, L, C) -> (B, channels, L) causal features; position t depends only on inputs 0..t."""
        h = x.transpose(1, 2)
        for i, b in enumerate(self.blocks):
            o = b(h)
            h = o + h if i and o.shape == h.shape else o
        return h

    def forward(self, x):
        return self.head(self.features(x)[:, :, -1])
