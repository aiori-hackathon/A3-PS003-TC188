# Phase 7 technical specification — federated B1 GRU (FedAvg)

**Date:** 2026-10-27. **Status: PROPOSED; nothing implemented.**

**Labels:**
- **[approved]:** approved in writing by the project lead (source given).
- **[proposed]:** a recommendation awaiting approval.
- **[future]:** recorded only; needs a separate approval.

**Inputs (all frozen):**
- Preprocessing checkpoint `f45bc9a`; artifacts in `data/processed/`.
- Phase 6 code (`aiori/dataset.py`, `model.py`, `losses.py`, `training.py`, `evaluation.py`, `metrics.py`, `baselines.py`).
- Phase 6 results in `results/phase6/`.

**Companion documents:**
- `docs/phase7_implementation_plan.md`: build steps, test checklist, runtime.
- `docs/phase6_technical_spec.md`: model, loss, metrics.
- `docs/research_validity_decisions.md`: R1/R2.

---

## 0. Fixed inputs and decisions already made

| Item | Value | Status |
|---|---|---|
| Model | `ForecastGRU(input_size=4, hidden=64)` → `Linear(64, 12)`; **14,220 parameters** | [approved] B1 adopted 2026-10-27 (validation-only selection) |
| Input features (B1) | z-scaled causally filled flow; observed flag; sin/cos of time-of-day slot (t mod 288) for each of 12 input steps | [approved] |
| Loss | masked MSE in z-space (`torch.where`); batches with no observed target take no step | [approved] Phase 6 |
| Local optimiser | Adam, lr 1e-3, batch 512, gradient clip 1.0, no LR schedule | [approved] D-C (carried over unchanged) |
| Early-stopping metric | pooled validation MAE in raw units, patience 5 | [approved] D-D |
| Negative predictions | not clipped in primary metrics; rate reported | [approved] D-E |
| Data | the frozen January 40-sensor primary pilot (engineering testbed) and the 8-sensor missing-data validation run (pipeline checks only, never pooled) | [approved] A3, A4 |
| Geographic clients | `proposed_client` of `sensor_selection.csv`: 4 clients × 10 sensors | [approved] A7 |
| Test split | **closed**. Not loaded by any Phase 7 code path in development | [approved] |

**Centralised B1 reference results (validation, seeds 0–2):**
- MAE 19.743 ± 0.093; RMSE 31.807 ± 0.142.
- Best epochs 18, 18, 24; epochs run 23, 23, 29.
- About 3.75 s of training per epoch (494 optimiser steps).

---

## 1. Arms

| Arm | Who trains | Data seen by each trainer | Models produced | Status |
|---|---|---|---|---|
| **Centralised B1** | one trainer | all 40 sensors' training windows (252,520) | 1 | [proposed] **reuse the existing Phase 6 runs** `results/phase6/primary/centralised_b1_seed{0,1,2}` (identical code, config and data); no rerun |
| **Local-only B1** | each client independently, no communication | its own 10 sensors (63,130 train windows) | 4 (one per client) | [proposed] |
| **IID-like FedAvg B1** | 4 clients + server | a **seeded random balanced sensor partition** (10/10/10/10 of the same 40 sensors) | 1 global | [proposed] |
| **Geographic FedAvg B1** | 4 clients + server | the geographic partition (`proposed_client`) | 1 global | [proposed] |

**IID-like partition (proposed definition):**
- Sensors are shuffled with a fixed partition seed (`20261027`) and dealt round-robin into 4 groups of 10.
- The partition is generated once, written to a JSON file in `results/phase7/`, and **reused for all training seeds**. Seed-to-seed variation therefore reflects training randomness only.
- Every sensor has 6,313 training windows, so all IID clients also have 63,130 windows.
- **Why sensor-level, not window-level:**
  - A window-level shuffle would give every client every sensor's history. That breaks the rule that one sensor belongs to one owner, and with it the per-sensor scaler ownership.
  - The sensor-level partition keeps each sensor's data and scaler at exactly one client in both FedAvg arms. The only difference between the arms is then *which* sensors are grouped together.
- **Expectation, from R2:** geographic clients are only weakly heterogeneous, and 10 sensors per client gives about 17% power. The pilot is therefore **not expected** to separate IID from geographic FedAvg. A null difference must be reported as "not detected in an underpowered pilot", not as "no effect".

**Local-only predictions:** each sensor is forecast by its own client's model. Pooled local-only metrics come from concatenating the four clients' predictions over the identical validation windows.

---

## 2. FedAvg algorithm

### 2.1 Round structure (synchronous, simulated in one process)

```
w_0 = ForecastGRU(4) initialised after seed_everything(seed)   # identical to the centralised init for the same seed
for r = 1 .. R_max:
    S_r = all clients (full participation)                       # sampling fraction C = 1.0
    for k in S_r (sorted by client name):
        download: client k receives a copy of w_{r-1}              # bytes logged
        local:    k trains E local epochs on its own train windows
                  with a fresh Adam(lr=1e-3), batch 512, clip 1.0, shuffle generator seed g(seed, r, k)
        upload:   k sends w_r^k and n_k                             # bytes logged
    aggregate:  w_r = Σ_k (n_k / Σ_j n_j) · w_r^k   (every float tensor of the state_dict)
    validate:   each client predicts its own val windows with w_r and returns (Σ|e|, Σe², n, per-horizon sums)
                server pools the sums -> validation MAE_r
    early stop: keep the best w_r by MAE_r; stop after 5 rounds without improvement
return best w_r
```

### 2.2 Parameters

| Parameter | Proposed value | Rationale |
|---|---|---|
| Client sampling | **full participation** (C = 1.0; all 4 clients every round). The implementation accepts C < 1 with seeded sampling, but it is not used | only 4 clients; partial participation would add variance and confound the IID vs geographic comparison |
| Local epochs E | **1** | one round then costs exactly one centralised epoch (see §3) |
| Max rounds R_max | **50** | equals the centralised maximum of 50 epochs |
| Early stopping | patience **5 rounds** on pooled validation MAE of the **global** model; restore the best round | same rule as centralised |
| Aggregation weights | `n_k` = number of the client's **training windows with ≥ 1 observed target** (the window-index length) | sample-weighted FedAvg (McMahan et al.); in the pilot all clients have 63,130, so weights are equal (0.25). Recorded per round |
| Aggregated tensors | all six `state_dict` tensors (`gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.bias_ih_l0`, `gru.bias_hh_l0`, `head.weight`, `head.bias`); no buffers exist | the whole model is shared; nothing is client-specific |
| Aggregation precision | accumulate in float64, cast back to float32 | removes order-dependent rounding |
| Optimiser state | **Adam re-initialised at the start of every round on every client**; never sent to the server | D-F recommendation. The communicated state stays exactly the model; clients stay stateless between rounds. Persistent client Adam state is a possible later ablation [future] |
| Local learning rate | 1e-3 (no server learning rate; plain FedAvg, i.e. server lr 1) | |
| Gradient clipping | 1.0 per local step | as centralised |

### 2.3 Reproducibility controls

| Control | Specification |
|---|---|
| Seeds | training seeds 0, 1, 2 (same as Phase 6). `seed_everything(seed)` immediately before building `w_0`. Arms with the same seed therefore start from **identical initial weights** (centralised, both FedAvg arms, and all 4 local-only models) |
| Client shuffle streams | each client's loader uses `torch.Generator().manual_seed(g)`, with `g` derived from `numpy.random.SeedSequence([seed, round, client_idx]).generate_state(1)[0]`. No global RNG is shared between clients, so the client iteration order cannot change another client's data order |
| Client order | clients processed in sorted name order; aggregation in float64 is order-independent anyway |
| Determinism | `torch.use_deterministic_algorithms(True)`; `torch.set_num_threads(8)`; CPU only; versions recorded |
| Partition | the IID-like partition JSON and the geographic membership are saved with every run, with their SHA-256 |
| Config hash | every run records the SHA-256 of its config file; the runner refuses the test split exactly as in Phase 6 |

---

## 3. Fair comparison and budget matching

**Unit of budget:** one *sample-pass*, meaning one training window used in one forward/backward pass. One centralised epoch is 252,520 sample-passes in 494 optimiser steps.

| Arm | Per round / epoch | Sample-passes | Optimiser steps | Max | Matching |
|---|---|---:|---:|---|---|
| Centralised | 1 epoch | 252,520 | 494 | 50 epochs | reference |
| Geographic or IID FedAvg, E = 1 | 1 round (4 clients × 1 local epoch) | 4 × 63,130 = 252,520 | 4 × 124 = 496 | 50 rounds | **identical sample-passes per round/epoch; same maximum; same patience** |
| Local-only (per client) | 1 epoch | 63,130 | 124 | 50 epochs | each client gets exactly the local work it would do in the same number of FedAvg rounds |

**Rules held identical across arms:**
- architecture, initial weights per seed, features, loss, batch size, learning rate, clipping;
- early-stopping rule (patience 5 on pooled validation MAE in raw units);
- validation windows, masks and targets (same 34,120 windows);
- evaluation code and metric definitions;
- seeds 0–2, reported as mean ± SD.

**Local-only early stopping:** each client stops on its **own** validation MAE (it cannot see others' validation data). This mirrors what a real isolated operator could do. Reported metrics pool the four clients' best models.

**What differs, by design:**
- FedAvg sees all sensors only through averaging.
- Local-only sees one client's sensors.
- Centralised sees all data jointly.
- FedAvg runs 4 × 124 = 496 steps per round versus 494 centralised steps per epoch, because each client rounds up its own last partial batch. This 0.4% difference is reported, not corrected.

**Wall-clock is not a budget unit:** the simulation runs clients sequentially. Wall-clock is reported for information only.

---

## 4. Communication accounting

| Item | Specification |
|---|---|
| Payload | the `state_dict` serialised with `torch.save` to an in-memory buffer; **59,541 B** for B1 (measured 2026-10-27). Raw float32 is 56,880 B, so serialisation overhead is 2,661 B (4.7%). Both numbers are reported |
| Counted transfers | per round: one download and one upload per participating client, plus the scalars `n_k` and the validation metric sums (a few dozen bytes, logged separately) |
| Per round (4 clients) | 4 × 59,541 down + 4 × 59,541 up = **476,328 B ≈ 0.48 MB** |
| Per run | 0.48 MB × rounds run (≈ 23–29 expected, from the centralised epochs; ≤ 50 by design): **≈ 11–14 MB expected, ≤ 23.8 MB maximum** |
| Not counted | protocol/TLS overhead, retries, compression (none applied); reported as **model-payload bytes** |
| Local-only and centralised | 0 model bytes. Centralised would require moving raw data; for context only: 40 sensors × 6,336 train steps × 4 B ≈ 1.0 MB of raw float32 flow (not a like-for-like comparison) |
| Convergence | report validation MAE per round, plus cumulative bytes at (a) the best round and (b) the first round within 1% of the arm's best MAE. Also the first round that beats the historical-average baseline (25.779) |

---

## 5. Data isolation and leakage safeguards

### 5.1 Train / validation / test
1. **Window index:** every client dataset is built from `run.windows["train"]` and `run.windows["val"]` only. `load_run(..., splits=("train", "val"))` keeps the test index out of memory.
2. **Optimiser updates:** a client's local trainer accepts only a `train` dataset and raises otherwise. This is the Phase 6 guard, reused.
3. **Model selection:** the server's early stopping consumes only validation metric sums. There is no test path in the Phase 7 runner except the Phase 6 style `--final --approved-config-sha256` guard, which is **not used in Phase 7**.
4. **Statistics:** the per-sensor scaler and time-of-day medians are the frozen training-only statistics (verified in Phase 6). Calendar features carry no target information (verified 2026-10-27: 400/400 windows unchanged when each sensor's data from t0 onward was corrupted).

### 5.2 Between clients (simulation discipline)

The simulation runs in one process, so isolation is enforced by construction and by tests, not by the operating system:

- **Physical data subsetting:** each `Client` holds a **sliced copy** of the run arrays for its own sensors only (`subset_run(run, positions)`). That copy contains:
  - inputs, flags, targets, masks and raw flow;
  - the scaler mean/std and time-of-day medians;
  - window indices re-mapped to local positions.

  A global-position map is kept for reporting. The client object holds no reference to the full `Run`.
- **The server object** holds the global model, `n_k`, metric sums and byte logs. It holds **no** dataset, scaler, window index or predictions. Its interface accepts only `state_dict`s, integers and metric-sum dictionaries.
- **Prediction records:** each client writes its own validation predictions (raw units) into the run folder. The server never receives them. The local experiment harness merges them for reporting, which is analysis outside the federated protocol.
- **Tests (see the plan, §4):**
  - client sensor sets are disjoint and their union is the 40 sensors;
  - client arrays equal the full-run rows of their own sensors and nothing else;
  - the server receives only allowed types;
  - each client's batches contain only its own sensors.

### 5.3 What is shared, explicitly

| Shared with server | Not shared |
|---|---|
| model `state_dict` (each round, both directions); `n_k`; validation Σ\|e\|, Σe², n (overall and per horizon) | raw or scaled series, masks, window indices, scaler parameters, time-of-day medians, predictions, gradients, optimiser state, sensor identities |

**Residual privacy note:** sharing model weights and aggregate validation sums is standard FedAvg. It is not a formal privacy guarantee: no differential privacy or secure aggregation is applied. Out of scope.

---

## 6. Evaluation (validation only)

Metrics follow `docs/phase6_technical_spec.md` §4 and are computed by the existing `aiori.evaluation.score` on raw units:
- pooled (micro) MAE/RMSE; MAE in scaled units;
- per horizon (5 … 60 min); per client (micro and macro); per sensor;
- negative-prediction rate; reference forecasts.

**FedAvg arms:**
- The global model is scored on every client's validation windows. Pooled values come from the clients' summed errors.
- An **exact-equality check** compares pooled metrics computed from the sums with metrics recomputed from the merged prediction records (as in Phase 6).

**Reported per arm, seeds 0–2 (mean ± SD):**
- the tables above;
- best round or epoch;
- rounds or epochs run;
- bytes (per round, cumulative to best, total);
- wall-clock;
- convergence curve (validation MAE vs round, and vs cumulative bytes).

**Comparisons:**

| Comparison | Question |
|---|---|
| FedAvg-geographic vs centralised | the cost of federation under the natural partition |
| FedAvg-geographic vs local-only | the benefit of collaboration |
| FedAvg-IID vs FedAvg-geographic | the effect of partition heterogeneity (underpowered; see §1) |

Paired differences are taken per seed, mean ± SD. With 3 seeds no significance test is claimed. A difference is called "consistent" only if it has the same sign on all 3 seeds **and** exceeds 2 × the larger arm's seed SD.

The **missing-data validation run** (8 sensors) is used only for pipeline smoke tests in Phase 7, never for arm comparisons. Its geographic clients have 2 sensors each.

---

## 7. Known research limitations (to be repeated in every Phase 7 report)

1. **Row-to-ID mapping:** SUPPORTED BY STRONG INDIRECT EVIDENCE; UNCONFIRMED BY PROVENANCE (A6). Every geographic client label is derived through `node_order`, so it is provisional.
2. **Small, underpowered pilot:** 40 sensors (10 per client). Geographic clients are weakly heterogeneous (largest η² ≈ 0.10), and detection power is about 17% at 10 sensors per client. Phase 7 on the pilot is an **engineering validation of the federated pipeline**, not evidence about geographic non-IID effects. That requires the ≈ 200-sensor experiment [future].
3. **Completeness-biased selection:** the pilot was chosen for completeness and is not representative of the client populations.
4. **One month, short validation:** January 2023 only. Validation is 3 weekdays (23–25 January, Mon–Wed), with no weekend in validation. B1 itself was selected on this validation set.
5. **Equal client sizes:** all clients have 63,130 windows, so the pilot has no quantity skew and FedAvg weighting is effectively uniform.
6. **Simulation, not deployment:** sequential clients in one process; isolation by construction and tests; no network, stragglers, dropouts, compression or privacy mechanisms.
7. **Timestamps:** reconstructed from the upstream convention (2023-01-01 00:00, 5-minute steps); timezone unproven. The time-of-day pattern is consistent with local time.
8. **Test split closed:** every Phase 7 number is a validation number. Final test evaluation needs a separate approval of frozen configurations.

---

## 8. Later phases (recorded only; each needs separate approval)

| Phase | Content | Notes |
|---|---|---|
| Final test evaluation [future] | one pass of the four frozen arms × 3 seeds on the test split | requires approved config hashes (runner guard) |
| ≈ 200-sensor geographic experiment [future] | about 50 sensors per client (power ≈ 0.92); new selection policy, stratified by sensor type or flow level | new selection, preprocessing run and checkpoint; same Phase 7 code |
| Behaviour-based stress partition [future, optional] | strong controlled heterogeneity from training-period features | labelled non-geographic |
| **Full-year 2023 scaling [future]** | apply the pipeline to all 12 months | see below |

**Full-year scaling notes (no action now):**
- **Local files:** `year_2023/` already holds the 12 monthly files `2023_p01.npy` … `2023_p12.npy`, about 20 GB in total and git-ignored, so **no download would be needed**. Only `2023_p01.npy` is covered by the local integrity baseline and has been read. The other 11 have not been opened, hashed or audited.
- **What the phase would need:**
  - an integrity baseline for p02–p12;
  - a node-order and shape check per month;
  - a re-run of the flow audit, eligibility and duplicate analysis per month, or for the year;
  - a new chronological split design (for example train Jan–Sep, val Oct, test Nov–Dec, with seasonal and holiday checks);
  - a re-fit of training-only scalers;
  - new window indices.
- **Training cost:** about 12× the January windows for the same 40 sensors, so about 45 s per epoch or round on CPU (estimate). Larger sensor counts scale linearly.
- **Approval:** this would be a new preprocessing checkpoint and requires its own approval.
