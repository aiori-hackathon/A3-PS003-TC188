# Phase 7 implementation plan — federated B1 GRU

**Date:** 2026-10-27. **Status: PROPOSED; not started.** Implements `docs/phase7_technical_spec.md`.

**Constraints:**
- No changes to frozen preprocessing, selections, splits, processed artifacts, or Phase 6 results (`results/phase6/`).
- The test split stays closed.
- No full-year data use.
- No commit or push without approval.

---

## 1. Files

| File | Contents | New / changed |
|---|---|---|
| `aiori/federated.py` | see the list below | new |
| `experiments/run_phase7.py` | CLI: `--config`, `--seed`, `--run {primary, missing_validation}`, `--smoke` (2 rounds or epochs × 5 batches per client). Validation only; the test split refused as in Phase 6. Writes `results/phase7/<run>/<arm>_seed<k>[_smoke]/` | new |
| `experiments/configs/phase7_fedavg_geo_b1.json`, `phase7_fedavg_iid_b1.json`, `phase7_local_b1.json` | §3 | new |
| `tests/test_federated.py` | algorithm, budget and reproducibility tests | new |
| `tests/test_federated_isolation.py` | client/server isolation and leakage tests | new |
| Phase 6 modules | **reused unchanged**: `dataset`, `model`, `losses`, `training` (`train_one_epoch`, `predict`, `fit`, `seed_everything`, `configure_torch`), `evaluation`, `metrics` | no change planned. If a change proves necessary, it will be reported and approved first |

**Contents of `aiori/federated.py`:**
- `subset_run(run, positions)`: physically sliced, read-only client copy with re-mapped window indices and a global-position map.
- `geographic_partition(run)` and `iid_partition(run, partition_seed)`.
- `payload_bytes(state_dict)`.
- `Client`:
  - `local_train(global_state, round, epochs)` returns `(state, n_k, stats)`;
  - `validation_sums(state)`;
  - `predict_val(state)`.
- `Server`:
  - `aggregate(states, weights)` in float64;
  - a byte log;
  - early-stopping bookkeeping.
- `run_fedavg(...)` and `run_local_only(...)`.

**Implementation notes:**
- **Local-only:** reuses `training.fit` on each client's subset run. Each client stops on its own validation split.
- **Payload:** `payload_bytes` returns the `torch.save` size of the `state_dict` in an in-memory buffer. Its raw float32 size is reported alongside.
- **Client loaders:** built with the existing `dataset.make_loader`, with a seed derived from `SeedSequence([seed, round, client_idx])`.

## 2. Steps

| Step | Work | Depends on | Size |
|---|---|---|---|
| 0 | Approve this plan and the decisions in §6 | — | — |
| 1 | `subset_run`, partitions, `payload_bytes`; isolation tests on synthetic and real runs | 0 | S |
| 2 | `Client` and `Server` (aggregation, byte log); synthetic algorithm tests (§4.1) | 1 | M |
| 3 | `run_fedavg` and `run_local_only` with early stopping, metric sums, prediction-record merge; equivalence and reproducibility tests | 2 | M |
| 4 | `run_phase7.py` and configs; guard tests | 3 | S |
| 5 | **Tiny real-data smoke tests** (§4.3): each arm × seed 0 × `--smoke` on the primary run and the missing-data run | 4 | S |
| 6 | **Stop: report the smoke results; request approval** for the controlled January experiments | 5 | — |
| 7 | (after approval) **Controlled January experiments**: local-only, FedAvg-IID and FedAvg-geographic × seeds 0, 1, 2 on the primary pilot; validation only | 6 | ≈ 15–30 min CPU |
| 8 | (after approval) Comparison report including centralised B1 (reused) | 7 | S |

Estimated effort for steps 1–5: about 1.5–3 working days. S = under half a day, M = 1 day.

## 3. Proposed experiment configurations (drafts; files are created in step 4)

Shared fields (identical to `experiments/configs/phase6_b1.json`): `calendar: true`, `hidden_size: 64`, `optimizer: adam`, `lr: 0.001`, `batch_size: 512`, `grad_clip: 1.0`, `lr_schedule: null`, `early_stopping_metric: "validation pooled MAE, raw units"`, `patience: 5`, `clip_negative_predictions: false`, `seeds: [0, 1, 2]`, `threads: 8`.

```json
{ "arm": "fedavg_geo_b1",  "partition": "geographic",
  "rounds_max": 50, "local_epochs": 1, "client_fraction": 1.0,
  "aggregation_weights": "train_windows", "optimizer_state": "reset_each_round",
  "aggregation_dtype": "float64" }

{ "arm": "fedavg_iid_b1",  "partition": "iid_sensor_balanced", "partition_seed": 20261027,
  "rounds_max": 50, "local_epochs": 1, "client_fraction": 1.0,
  "aggregation_weights": "train_windows", "optimizer_state": "reset_each_round",
  "aggregation_dtype": "float64" }

{ "arm": "local_b1",       "partition": "geographic",
  "max_epochs": 50, "early_stopping_scope": "per_client_own_validation" }
```

**Centralised B1** is the existing `phase6_b1.json` runs (not rerun).

**Smoke mode** overrides: 2 rounds or epochs, 5 batches per client per local epoch, seed 0.

## 4. Test and validation checklist

### 4.1 Synthetic unit tests (no real data)
- [ ] `aggregate` equals a NumPy weighted mean for every tensor. Equal weights give the plain mean; weights normalise to 1; dtype stays float32; float64 accumulation means client order does not matter.
- [ ] Aggregating K identical models returns that model exactly.
- [ ] **Equivalence:** FedAvg with one client holding all sensors, E = 1, R = 1 and the same shuffle seed gives weights identical to one centralised epoch (`train_one_epoch`).
- [ ] FedAvg with E = 1: each client's step count equals ⌈n_k / 512⌉. Total sample-passes per round equal the centralised per-epoch count.
- [ ] Adam state is fresh each round: the client optimiser is constructed inside `local_train`. A test shows two rounds with identical inputs give identical updates.
- [ ] The same seed gives identical global weights, byte logs and metrics; a different seed gives different weights. Changing one client's shuffle seed changes no other client's batch order.
- [ ] Early stopping restores the best round, using a scripted validation sequence (as in Phase 6).
- [ ] `payload_bytes` equals `len(torch.save(state_dict))`. The per-round log has 2 × K entries, and cumulative bytes are monotone.
- [ ] Batches with no observed target are skipped and counted in the client stats.
- [ ] `iid_partition`: balanced (10/10/10/10 for 40), disjoint, covering, deterministic for a fixed partition seed, and different from the geographic partition.

### 4.2 Isolation and leakage tests
- [ ] Client sensor sets are disjoint and their union is all sensors, for both partitions.
- [ ] Each `Client`'s arrays equal exactly the full-run rows of its own sensors. Its window indices map back to the full-run windows of those sensors for train and val. The client holds no `Run` reference (checked by attribute inspection).
- [ ] Every training batch a client builds contains only its own sensors and only `train` windows. Validation batches are built only under `no_grad`.
- [ ] `Server` methods accept only `state_dict`, integers and metric-sum dictionaries. A test passes a dataset or array and expects a `TypeError`. The `Server` holds no data attributes.
- [ ] Validation pooled MAE from the server's summed errors equals the MAE recomputed from the merged prediction records (exact to float tolerance).
- [ ] The runner refuses `--evaluate test` (Phase 6 guard), and the test windows are not loaded (`load_run(..., splits=("train", "val"))`).
- [ ] Fingerprints of `data/`, the raw files and `results/phase6/` are unchanged before and after every real run.

### 4.3 Tiny real-data smoke tests (step 5; each arm, seed 0, `--smoke`)
- [ ] Primary run: geographic and IID FedAvg, 2 rounds × 5 batches per client; local-only, 2 epochs × 5 batches.
- [ ] Missing-data run (8 sensors; 2 per geographic client): the same. It must exercise masked validation targets.
- [ ] All losses, gradient norms, aggregated weights and predictions are finite. Skipped batches and clipped steps are reported.
- [ ] Bytes per round equal 8 × 59,541 = 476,328 B for 4 clients.
- [ ] Per-round wall-clock is recorded to refine the §5 estimate.

### 4.4 Controlled January experiments (step 7, after approval)
- [ ] 3 arms × seeds 0–2; validation only; the existing centralised B1 reused.
- [ ] Report per arm (mean ± SD):
  - pooled, per-horizon, per-client and per-sensor metrics;
  - normalised MAE and negative rate;
  - best round, rounds run and early stop;
  - bytes to best and total;
  - convergence curves.
- [ ] Paired per-seed differences for the three comparisons (spec §6), using the "consistent" criterion.
- [ ] Limitations section (spec §7) included verbatim in the report.

## 5. Estimated CPU runtime and memory

Measured on 2026-10-27 (24-thread Xeon, 8 torch threads, B1): one client epoch on 63,130 windows (124 batches) takes **0.95 s**. Validation prediction on one client's 8,530 windows takes **0.023 s**. A centralised epoch takes 3.75 s (494 batches).

| Item | Estimate |
|---|---|
| FedAvg round (4 clients × E = 1, sequential) | 4 × 0.95 s + 4 × 0.02 s validation + serialisation and aggregation (milliseconds) ≈ **4 s** |
| FedAvg run | about 23–29 rounds expected (by analogy with the centralised epochs; not guaranteed), so ≈ 1.5–2 min; **≤ 3.5 min** at 50 rounds |
| Local-only run (4 clients × ≤ 50 epochs) | ≈ 1.5–2 min expected; **≤ 3.3 min** |
| Smoke tests (step 5) | under 10 s per arm and run, plus about 3 s process start |
| Controlled experiments (step 7: 3 arms × 3 seeds) | ≈ **15 min expected, ≤ 30 min** |
| New tests (synthetic) | a few seconds; full suite ≈ 10 s |
| Memory | full-run arrays about 7 MB for 40 sensors; client copies about 7 MB more in total; models 57 KB each (global, 4 local, best snapshot); per-round byte logs negligible. Process peak is dominated by PyTorch itself: a few hundred MB (estimate; not measured) |
| Disk per run | `metrics.json` with round history (≈ 60 KB), merged validation predictions (≈ 3 MB), best checkpoint (58 KB). About 30 MB for step 7 in total, under the git-ignored `results/phase7/` |

**All runtimes above are estimates from component timings.** Step 5 will measure them.

## 6. Decisions requiring approval

| # | Decision | Recommendation |
|---|---|---|
| P7-1 | Arms: centralised (reuse the Phase 6 B1 runs), local-only, FedAvg-IID, FedAvg-geographic | approve |
| P7-2 | IID-like = seeded balanced **sensor-level** random partition, one fixed partition seed (20261027) for all training seeds | approve |
| P7-3 | E = 1, R_max = 50, patience 5 rounds, full participation (C = 1.0) | approve; E = 5 or partial participation only as later ablations |
| P7-4 | Aggregation weights n_k = training windows with ≥ 1 observed target; float64 accumulation | approve (confirms D-F) |
| P7-5 | Adam re-initialised every round; optimiser state never shared | approve (confirms D-F) |
| P7-6 | Local-only early stopping per client on its own validation | approve |
| P7-7 | Communication = `torch.save` payload bytes (59,541 B per model), raw float32 also reported | approve |
| P7-8 | "Consistent difference" = same sign on all 3 seeds and > 2 × the larger seed SD; no significance claims | approve |
| P7-9 | Outputs in `results/phase7/` (git-ignored); new code and docs uncommitted until D-G is decided | approve |

**Not proposed now [future, separate approval each]:**
- test-set evaluation;
- the ≈ 200-sensor experiment;
- the behaviour-based stress partition;
- full-year 2023 scaling (the 12 monthly files are already local; only January has an integrity baseline);
- FedAvg ablations (E = 5, persistent Adam, partial participation, FedProx).
