"""MLP and GRU reused from the accuracy-optimisation study (`experiments/accuracy_optimization/aopt/models.py`).

The file is loaded directly (no sys.path change, no edit). `Forecaster` with future_dim=0 and emb_dim=0 maps
(B, L, C) -> (B, 12): GRU = nn.GRU + linear head on the last hidden state; MLP = N-BEATS-style residual MLP
(input linear, `layers` x [LayerNorm, Linear, GELU, Dropout, Linear] with skip connections, linear head).
No sensor embeddings and no future covariates, so one parameter set serves any client.
"""

import importlib.util
from pathlib import Path

from torch import nn

_AOPT = Path(__file__).resolve().parents[2] / "experiments" / "accuracy_optimization" / "aopt" / "models.py"


def _load_aopt():
    spec = importlib.util.spec_from_file_location("aiori2_reused_aopt_models", _AOPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aopt_models = _load_aopt()


class Reused(nn.Module):
    """Thin adapter around aopt.Forecaster(kind) with the shared (B, L, C) -> (B, 12) contract."""

    def __init__(self, kind, in_channels=1, input_len=12, horizon=12, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        if horizon != aopt_models.H:
            raise ValueError(f"the reused aopt models forecast exactly {aopt_models.H} steps")
        self.net = aopt_models.Forecaster(kind, past_dim=in_channels, future_dim=0, input_len=input_len, emb_dim=0,
                                          hidden=hidden, layers=layers, head_hidden=0, dropout=dropout)

    def forward(self, x):
        return self.net(x)


def MLP(in_channels=1, input_len=12, horizon=12, hidden=128, layers=2, dropout=0.0):
    return Reused("mlp", in_channels, input_len, horizon, hidden, layers, dropout)


def GRU(in_channels=1, input_len=12, horizon=12, hidden=64, layers=1, dropout=0.0):
    return Reused("gru", in_channels, input_len, horizon, hidden, layers, dropout)
