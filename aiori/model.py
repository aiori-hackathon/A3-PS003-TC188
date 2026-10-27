"""Compact sensor-agnostic GRU forecaster: (B, input_len, F) -> (B, horizon) z-scaled flow.

One parameter set for all sensors (no embeddings, no normalisation layers or buffers), so the
same state_dict works centralised, per client, and under FedAvg.
"""

from torch import nn

from aiori.windows import HORIZON


class ForecastGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, horizon=HORIZON):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, h = self.gru(x)          # h: (1, B, hidden), the last hidden state
        return self.head(h[-1])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
