"""Compact N-HiTS (Challu et al., 2023) for direct 12-step forecasting.

Stacks of blocks; block b of stack s:
  * multi-rate input sampling: MaxPool1d(pool[s]) over the current flow residual (and the optional observed
    channel, which is used as an exogenous input and never residualised),
  * an MLP produces backcast coefficients (input_len values) and forecast coefficients at a coarse rate
    ceil(horizon / freq[s]),
  * hierarchical interpolation: the coarse forecast is linearly interpolated to the full horizon,
  * doubly residual: residual <- residual - backcast; forecast <- forecast + block forecast.
Defaults pool=(2, 2, 1), freq=(4, 2, 1): the stacks forecast 3, 6 and 12 points (coarse to fine).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, input_len, horizon, pool, freq, hidden, mlp_layers, dropout):
        super().__init__()
        self.pool = nn.MaxPool1d(pool, pool, ceil_mode=True) if pool > 1 else nn.Identity()
        pooled = math.ceil(input_len / pool)
        dims = [pooled * in_channels] + [hidden] * mlp_layers
        mlp = []
        for a, b in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*mlp)
        self.horizon = horizon
        self.n_forecast = math.ceil(horizon / freq)
        self.backcast = nn.Linear(hidden, input_len)
        self.forecast = nn.Linear(hidden, self.n_forecast)

    def forward(self, residual, exo):
        z = torch.cat([residual[:, None, :], exo.transpose(1, 2)], 1)          # (B, C, L)
        h = self.mlp(self.pool(z).flatten(1))
        coarse = self.forecast(h)
        fc = coarse if self.n_forecast == self.horizon else F.interpolate(coarse[:, None, :], size=self.horizon, mode="linear",
                                                                           align_corners=True)[:, 0]
        return self.backcast(h), fc


class NHiTS(nn.Module):
    def __init__(self, in_channels=1, input_len=12, horizon=12, hidden=128, mlp_layers=2, pool=(2, 2, 1), freq=(4, 2, 1),
                 blocks_per_stack=1, dropout=0.0):
        super().__init__()
        if len(pool) != len(freq):
            raise ValueError("pool and freq need one entry per stack")
        self.blocks = nn.ModuleList(Block(in_channels, input_len, horizon, p, f, hidden, mlp_layers, dropout)
                                    for p, f in zip(pool, freq) for _ in range(blocks_per_stack))

    def parts(self, x):
        """Forecast plus every block's backcast/forecast and the final residual (for inspection and tests)."""
        residual, exo = x[..., 0], x[..., 1:]
        total = torch.zeros(x.shape[0], self.blocks[0].horizon, device=x.device, dtype=x.dtype)
        backcasts, forecasts = [], []
        for b in self.blocks:
            bc, fc = b(residual, exo)
            residual = residual - bc
            total = total + fc
            backcasts.append(bc); forecasts.append(fc)
        return dict(forecast=total, backcasts=backcasts, forecasts=forecasts, residual=residual)

    def forward(self, x):
        return self.parts(x)["forecast"]
