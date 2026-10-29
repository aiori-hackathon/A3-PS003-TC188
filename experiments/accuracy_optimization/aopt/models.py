"""Forecasters: (past (B, L, Fp), future covariates (B, H, Ff) or None, sensor id) -> (B, H) z-scaled flow.

Parameters are split into a shared `body` and a `head` (plus optional per-sensor embeddings, which are always
client-local because every sensor belongs to exactly one client) so personalised federated methods can choose
which part to share.
"""

import torch
from torch import nn

H = 12


class Head(nn.Module):
    """Combines the body's representation with future covariates and the sensor embedding."""

    def __init__(self, rep, future_dim, emb_dim, hidden):
        super().__init__()
        d = rep + future_dim * H + emb_dim
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, H)) if hidden else nn.Linear(d, H)

    def forward(self, r, xf, e):
        parts = [r] + ([xf.flatten(1)] if xf is not None else []) + ([e] if e is not None else [])
        return self.net(torch.cat(parts, 1))


class Forecaster(nn.Module):
    def __init__(self, kind, past_dim, future_dim, input_len, n_sensors=200, emb_dim=0, hidden=64, layers=1,
                 head_hidden=0, dropout=0.0):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.body = nn.GRU(past_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0)
            rep = hidden
        elif kind == "mlp":
            # N-BEATS-style residual MLP over the flattened past window.
            d = past_dim * input_len
            blocks = [nn.Linear(d, hidden)]
            self.body = nn.ModuleDict(dict(inp=nn.Linear(d, hidden), blocks=nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
                 for _ in range(layers)])))
            del blocks
            rep = hidden
        elif kind == "tcn":
            chans, k = hidden, 3
            convs, dil = [], 1
            c_in = past_dim
            for _ in range(layers):
                convs.append(nn.Sequential(nn.ConstantPad1d((2 * dil, 0), 0.0), nn.Conv1d(c_in, chans, k, dilation=dil), nn.GELU()))
                c_in, dil = chans, dil * 2
            self.body = nn.ModuleList(convs)
            rep = hidden
        else:
            raise ValueError(kind)
        self.emb = nn.Embedding(n_sensors, emb_dim) if emb_dim else None
        self.head = Head(rep, future_dim, emb_dim, head_hidden)

    def represent(self, x):
        if self.kind == "gru":
            _, h = self.body(x)
            return h[-1]
        if self.kind == "mlp":
            r = self.body["inp"](x.flatten(1))
            for b in self.body["blocks"]:
                r = r + b(r)
            return r
        h = x.transpose(1, 2)
        for i, c in enumerate(self.body):
            o = c(h)
            h = o + h if i and o.shape == h.shape else o
        return h[:, :, -1]

    def forward(self, x, xf=None, s=None):
        e = self.emb(s) if self.emb is not None else None
        return self.head(self.represent(x), xf, e)

    def shared_keys(self, scope):
        """Parameter names shared with the server: 'all' (except embeddings), 'body' (FedPer/FedRep)."""
        keys = [k for k in self.state_dict() if not k.startswith("emb.")]
        return keys if scope == "all" else [k for k in keys if k.startswith("body.")]
