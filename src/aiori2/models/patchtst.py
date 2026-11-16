"""Compact PatchTST-style forecaster (Nie et al., 2023) for the 12-step AIORI input.

1. Patching: the history (B, L, C) is end-padded by repeating the last step (PatchTST "padding_patch=end": stride
   steps, plus more if needed so the last patch is complete), then cut into patches of `patch_len` with `stride`
   (L=12, patch 4, stride 2 -> 6 overlapping patches).
2. Each patch (patch_len x C values; C=1 is the univariate, channel-independent case) is linearly projected to
   d_model, plus a learnable positional embedding per patch.
3. A pre-norm Transformer encoder (multi-head self-attention over patches, GELU feed-forward).
4. Flatten head: (n_patches * d_model) -> horizon, a direct 12-step forecast.
Optional RevIN-style instance normalisation (off by default: inputs are already scaled, and 12-step windows can be
nearly constant). No pretrained weights. With only 12 inputs the patch count is small; a longer-context variant
would be a separate ablation with its own data rules.
"""

import torch
from torch import nn


class PatchTST(nn.Module):
    def __init__(self, in_channels=1, input_len=12, horizon=12, patch_len=4, stride=2, d_model=64, heads=4, layers=2,
                 ff=128, dropout=0.1, revin=False):
        super().__init__()
        if patch_len > input_len or stride < 1:
            raise ValueError("need 1 <= stride and patch_len <= input_len")
        pad = stride
        while (input_len + pad - patch_len) % stride:
            pad += 1                                                   # incomplete last patch -> pad until complete
        self.patch_len, self.stride, self.pad, self.revin = patch_len, stride, pad, revin
        self.n_patches = (input_len + pad - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * in_channels, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(self.n_patches * d_model, horizon))

    def patchify(self, x):
        """(B, L, C) -> (B, n_patches, patch_len * C), channel values of one time step kept together."""
        x = torch.cat([x, x[:, -1:, :].expand(-1, self.pad, -1)], 1)
        p = x.unfold(1, self.patch_len, self.stride)                   # (B, n, C, patch_len)
        return p.transpose(2, 3).reshape(x.shape[0], self.n_patches, -1)

    def forward(self, x):
        if self.revin:
            mu, sd = x[..., :1].mean(1, keepdim=True), x[..., :1].std(1, keepdim=True) + 1e-5
            x = torch.cat([(x[..., :1] - mu) / sd, x[..., 1:]], -1)
        z = self.drop(self.proj(self.patchify(x)) + self.pos)
        y = self.head(self.encoder(z))
        return y * sd[:, 0] + mu[:, 0] if self.revin else y
