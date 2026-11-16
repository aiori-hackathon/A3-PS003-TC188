# Running AIORI 2.0 experiments (Phase 6)

All commands run from `src/` (or set `AIORI2_BASE_DIR=<path to src>`). Outputs go to `results/aiori2/runs/<run_id>/`
(git-ignored). Test data is never read by any command here.

## One run

```
python -m aiori2.run --set arm=local --set history=B --set model=tcn --dry-run          # validate only
python -m aiori2.run --set arm=centralised --set model=gru                              # defaults: configs/runs/defaults.json
python -m aiori2.run --set arm=fedavg_geo --set algorithm=scaffold --set model=patchtst --set fl.local_steps=100
python -m aiori2.run --config my_run.json --set budget.epochs=3 --stop-after-epochs 1   # pause after 1 epoch (status interrupted)
python -m aiori2.run --config my_run.json --set budget.epochs=3 --resume                # continue it
```
Smoke-size budget: `--set budget.epochs=2 --set budget.windows_per_epoch=4096 --set validation.max_windows=50000
--set validation.final_full=false --set fl.local_steps=4`.

- `run_id` = arm-history-model-preset-algorithm-seed-<config hash>; device and output folder are not part of it, so a
  run can resume on another machine. An existing run directory is never reused for a different config; a finished
  run is skipped; an unfinished one needs `--resume`.
- Each run dir: `config.json`, `run.json` (status, hashes, environment, per-epoch history, best, counters, resumes,
  failure), `metrics.jsonl`, `latest.pt`, `best.pt` (tensors + JSON, `weights_only` loadable, atomic writes).
- `<out_root>/registry.jsonl` logs start/resume/pause/finish/failure events.

### Budget and fairness
`budget.windows_per_epoch` sampled training windows per epoch **in total**: centralised samples them from all 200
sensors; local gives each of the four models a quarter; FL runs `ceil(W / (4 × local_steps × batch))` rounds per
epoch. A and B use the same budget, so B gains history, not optimizer steps. Actual windows, optimizer steps, rounds,
FL bytes and an approximate forward FLOP count per window (`torch.utils.flop_counter`; RNNs may be undercounted) are
recorded — equal configured epochs are not claimed to mean equal computation.

### Evaluation
Validation only: masked raw-flow MAE, RMSE and WAPE (= Σ|error| / Σ|y| over valid targets, accumulated over batches,
never averaged per batch), with the valid count and WAPE denominator, per geographic and IID client and per sensor.
Every arm and history scores the same fixed subset of validation windows (`validation.max_windows`, seed 0; `null` =
all); with `final_full: true` the best checkpoint is re-scored on all validation windows. Local-arm predictions come
from each sensor's own client model; all predictions are inverted with the arm's scaler. The constant-run sensitivity
mask is optional (`validation.sensitivity_mask`) and never changes the primary mask. Early stopping uses validation
WAPE.

## Tracking and plan

```
python -m aiori2.track summary [--csv results/aiori2/summary.csv]
python -m aiori2.track compare-ab
python -m aiori2.plan status --stage 1
python -m aiori2.plan estimate --stage 1 --bench data/processed/aiori2_3yr/phase6/bench_cuda_b128_<time>.json --set batch_size=128
python -m aiori2.plan run --stage 1 --only history=A,model=gru --max-runs 2              # skips finished runs, resumes interrupted ones
python -m aiori2.plan status --stage 2                                                   # needs all Stage-1 runs (top-2 by mean val WAPE)
python -m aiori2.plan write --stage 1 --dir plan_stage1                                  # one JSON per config, e.g. for the DGX
```
Stage 1: 40 runs. Stage 2: 64 configs, 8 of them (FedAvg) identical to Stage-1 runs, so 56 new. Stage 3: chosen
finalists × extra seeds (`--finalists RUN_ID,... --seeds 1,2`).

## Benchmark

`python -m aiori2.bench --device cuda --batch 128` (bounded; writes `data/processed/aiori2_3yr/phase6/bench_*.json`).

## DGX (preparation only)

1. Copy, keeping the layout under one repository root:
   - code: every file in `data/processed/aiori2_3yr/manifests/dgx_code_manifest.json` (68 files: `src/aiori2/`,
     `src/aiori/`, `src/experiments/accuracy_optimization/aopt/models.py`, tests, `requirements.txt`, `pytest.ini`);
   - data: every file in `data/processed/aiori2_3yr/manifests/dgx_transfer_manifest.json` (1.68 GB, conservative
     policy) plus the two manifests themselves. Raw XTraffic files are not needed.
2. `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r src/aiori2/requirements-dgx.txt`
   (choose the torch CUDA wheel for the driver, see the file).
3. `cd src && python -m aiori2.dgx_check` — GPU model, the memory this process can really allocate, driver/CUDA,
   PyTorch build, a model forward/backward on the GPU.
4. `python -m aiori2.dgx_check --verify` — re-hashes all transferred data and code.
5. `python -m pytest tests/test_aiori2_phase6.py` (synthetic, fast), then one smoke run, then
   `python -m aiori2.bench --device cuda --batch 128` before any estimate is trusted.
6. Train with `python -m aiori2.plan run --stage 1 ...`; resume with the same command after an interruption; export
   with `python -m aiori2.track summary --csv ...` and copy `results/aiori2/runs/` back.
