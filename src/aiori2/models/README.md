# aiori2.models — five forecasters (AIORI 2.0, Phase 4)

All five follow one contract: `model(x)` with `x` of shape `(B, 12, C)` float32 (scaled flow; `C = 2` adds the
0/1 observed flag) returns `(B, 12)` forecasts in the same scaled units. No sensor embeddings, scalers, client data
or buffers live in a model, so one architecture serves every arm and every client, and the whole `state_dict` can be
exchanged in federated training.

```python
from aiori2.models import build_model, count_parameters, save_checkpoint, load_checkpoint, average_states
m = build_model("patchtst", preset="medium", in_channels=1, seed=0)   # mlp | gru | tcn | patchtst | nhits
```

Presets: `aiori2/configs/models/presets.json` (small / medium / large; default medium). Overrides by keyword,
e.g. `build_model("gru", hidden=128)`. `seed` gives deterministic initialisation without touching the global RNG.

| Model | Origin | Medium preset | Params (small / medium / large) |
|---|---|---|---|
| MLP | **reused**: `aopt.Forecaster("mlp")` loaded from its file (N-BEATS-style residual MLP: input linear, blocks of LayerNorm–Linear–GELU–Dropout–Linear with skip, linear head) | hidden 128, 3 blocks, dropout 0.1 | 18,508 / 103,052 / 534,796 |
| GRU | **reused**: `aopt.Forecaster("gru")` (GRU, last hidden state → linear head); with 1 layer it equals the study's `aiori.model.ForecastGRU` (weight-copy test) | hidden 96, 2 layers, dropout 0.1 | 13,644 / 85,548 / 336,972 |
| TCN | **adapted**: `aiori2.models.tcn` generalises the study TCN to any kernel/dilation base; with kernel 3, base 2 it equals `aopt.Forecaster("tcn")` (weight-copy test) | 96 channels, 3 layers, kernel 3, dilation 1/2/4 (receptive field 15) | 6,732 / 57,036 / 233,452 |
| PatchTST | **new** | patch 4, stride 2 (end-padded → 6 patches), d_model 64, 4 heads, 2 pre-norm layers, ff 128, flatten head | 19,820 / 72,396 / 408,332 |
| N-HiTS | **new** | 3 stacks × 1 block, MaxPool (2, 2, 1), forecast points (3, 6, 12) linearly interpolated, MLP 2 × 128 | 17,913 / 60,345 / 437,874 |

The Phase 4 MLP shares the architecture family of the deployed optimised MLP (`candidate_v1`) but is **not** that
model: candidate_v1 uses a 288-step context, future calendar/profile covariates and a sensor embedding. Neither
`candidate_v1` nor any existing checkpoint is used or changed.

## Training loss

`aiori2.losses.masked_loss(pred, y_raw, mask, mean, std, kind="mae"|"mse")`: raw targets are scaled with the
stream's own training scaler after invalid entries are replaced (`torch.where`), and only valid elements are
selected, so invalid targets contribute exactly zero to the loss and gradients; the mean is over valid elements.
An all-invalid batch returns `None` and `train_step` skips the update. Evaluation stays in raw flow units
(`aiori2.data.raw_metrics`, predictions inverted with `loader.inverse`).

## Federated parameter exchange (for Phase 5)

Clients build the same architecture from the same config; `average_states(states, weights)` averages every
floating-point entry in float64 and requires non-floating entries to agree. Optimizer state is not exchanged. No
model has BatchNorm or other running statistics. The N-HiTS last-block backcast weights receive no gradient (its
backcast feeds nothing, as in the original design); they stay at their initial values and average harmlessly.

## Limits

- 12-step inputs give PatchTST only 6 patches; a longer-context PatchTST would be a separate ablation with its own
  data rules, not part of the main comparison.
- Presets are untuned starting points; capacity is comparable in order of magnitude, not identical.

## Smoke test and tests

```
python -m aiori2.models.smoke --device cpu      # or --device cuda (GPU environment)
python -m pytest src/tests/test_aiori2_phase4.py
```
The smoke test uses at most 105 real windows per history per model and writes its report to
`data/processed/aiori2_3yr/phase4/` and throwaway checkpoints to a new `results/aiori2/phase4_smoke/<time>/` folder.
