# Phase 9 results — nine-month (Jan 1 – Sep 30, 2023) traffic-flow forecasting

Generated automatically by the Phase 9 runner at 2026-10-28T22:17:22Z (no manual editing). Aggregate only: no station IDs.
Row-to-ID mapping and geography are provisional (supported by strong indirect evidence; unconfirmed by provenance).

## Dataset

- Months: 2023_p01.npy, 2023_p02.npy, 2023_p03.npy, 2023_p04.npy, 2023_p05.npy, 2023_p06.npy, 2023_p07.npy, 2023_p08.npy, 2023_p09.npy (January–September only).
- Split (verified programmatically): train 2023-01-01 – 2023-07-10 (191 days), validation 2023-07-11 – 2023-08-07 (28), test 2023-08-08 – 2023-09-30 (54).
- Training-period eligibility: 9,143 of 16,972 sensors; exclusions {'eligible': 9143, 'low_coverage': 4052, 'fully_missing': 2183, 'exact_copy': 1572, 'too_many_zeros': 15, 'constant': 7}.
- Sensor-order consistency across months (rank correlation of monthly means): [0.995, 0.995, 0.995, 0.996, 0.996, 0.995, 0.995, 0.994]; passed: True.
- Daylight saving (12 Mar): no one-hour shift at the 12 Mar transition (best lag within 10 min): daily profiles are in local clock time; step counts per month are complete, so no hour is missing or duplicated.
- Sensors: 200 (50 per client). Retained from the January study: 156; replacements: {'replacement_same_client': 34, 'replacement_same_stratum': 9, 'replacement_same_type': 1} (reasons {'low_coverage': 44}); type groups {'mainline': 153, 'hov': 35, 'ramp': 10, 'other': 2}.
- Windows (inputs and targets inside one split): {'train': 10994346, 'val': 1606706, 'test': 3098409}.
- Checkpoint verification: passed (19/19 checks).

## Training runs

| Run | Status | Validation MAE |
|---|---|---:|
| centralised_seed0 | complete | 16.529 |
| centralised_seed1 | complete | 16.590 |
| centralised_seed2 | complete | 16.610 |
| fedavg_geo_seed0 | complete | 16.809 |
| fedavg_geo_seed1 | complete | 16.832 |
| fedavg_geo_seed2 | complete | 16.840 |
| fedavg_iid_seed0 | complete | 16.614 |
| fedavg_iid_seed1 | complete | 16.758 |
| fedavg_iid_seed2 | complete | 16.788 |
| local_seed0 | complete | 16.510 |
| local_seed1 | complete | 16.460 |
| local_seed2 | complete | 16.461 |

Interrupted runs are never reported as complete; only runs with a verified completion marker were frozen and tested.

## Held-out test results (Aug 8 – Sep 30, 2023; evaluated once)

Test windows: 3,098,409. Seeds per arm are averaged per cell. SESOI = 0.1658 MAE (1% of the mean centralised validation MAE), fixed before the test.

| Model | Seeds | Test MAE | Test RMSE | Normalised MAE | Negative predictions | Validation MAE | Val → test | MAE with clipping at 0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Centralised GRU | 0, 1, 2 | **17.134** | 28.591 | 0.1946 | 0.51% | 16.576 | +0.558 | 17.129 |
| Geographic FedAvg | 0, 1, 2 | **17.362** | 28.927 | 0.1972 | 0.60% | 16.827 | +0.534 | 17.356 |
| Local-only GRU | 0, 1, 2 | **16.990** | 28.301 | 0.1929 | 0.56% | 16.477 | +0.513 | 16.984 |
| IID FedAvg | 0, 1, 2 | **17.274** | 28.827 | 0.1961 | 0.55% | 16.720 | +0.553 | 17.268 |
| persistence (reference) | — | 24.421 | 40.178 | — | — | — | — | — |
| seasonal naive daily (reference) | — | 32.309 | 59.283 | — | — | — | — | — |
| historical average (reference) | — | 35.776 | 61.241 | — | — | — | — | — |

**Pre-declared final model (lowest validation MAE, fixed before the test):** Local-only GRU, clipped at 0 predictions (clipping chosen on validation only).


**Improvement over reference forecasts (MAE):** Centralised GRU: persistence 29.8%, seasonal naive daily 47.0%, historical average 52.1%; Geographic FedAvg: persistence 28.9%, seasonal naive daily 46.3%, historical average 51.5%; Local-only GRU: persistence 30.4%, seasonal naive daily 47.4%, historical average 52.5%; IID FedAvg: persistence 29.3%, seasonal naive daily 46.5%, historical average 51.7%

### Pre-declared comparisons

| | Δ MAE | 95% CI (wider of sensor / freeway bootstrap) | Holm p | Matched seeds | Statistically detectable | Practically meaningful (≥ SESOI) | Verdict |
|---|---:|---|---:|---|---|---|---|
| C1 Geographic FedAvg − Centralised GRU | +0.228 | [0.147, 0.318] | 0.0020 | [0, 1, 2] | yes | yes | supported on the nine-month test split |
| C2 Geographic FedAvg − Local-only GRU | +0.372 | [0.208, 0.548] | 0.0020 | [0, 1, 2] | yes | yes | supported on the nine-month test split |
| C3 IID FedAvg − Geographic FedAvg | -0.088 | [-0.138, -0.043] | 0.0020 | [0, 1, 2] | yes | no | detectable but below the practical threshold on the nine-month test split |
| C4 Local-only GRU − Centralised GRU | -0.144 | [-0.275, 0.000] | 0.0510 | [0, 1, 2] | no | no | not distinguishable on the nine-month test split |

### Per client (test MAE)

| Client | Centralised GRU | Geographic FedAvg | Local-only GRU | IID FedAvg | persistence |
|---|---:|---:|---:|---:|---:|
| client_0_la_ventura | 18.84 | 19.37 | 18.37 | 19.11 | 26.19 |
| client_1_bay_area | 22.77 | 22.85 | 22.34 | 22.84 | 31.90 |
| client_2_coastal_south_central | 13.14 | 13.35 | 13.54 | 13.28 | 20.49 |
| client_3_inland_central | 13.79 | 13.88 | 13.71 | 13.87 | 19.09 |

### Per horizon (test MAE)

| Minutes | Centralised GRU | Geographic FedAvg | Local-only GRU | IID FedAvg | persistence |
|---|---:|---:|---:|---:|---:|
| 5 | 12.96 | 13.05 | 13.03 | 13.00 | 15.00 |
| 15 | 14.96 | 15.10 | 14.95 | 15.04 | 18.78 |
| 30 | 17.02 | 17.23 | 16.87 | 17.13 | 23.54 |
| 45 | 18.78 | 19.09 | 18.54 | 18.98 | 28.60 |
| 60 | 20.68 | 21.01 | 20.33 | 20.87 | 33.57 |

### August vs September, weekday vs weekend (test MAE)

| Model | Aug 8–31 | September | Weekday | Weekend |
|---|---:|---:|---:|---:|
| Centralised GRU | 17.125 | 17.141 | 17.623 | 15.861 |
| Geographic FedAvg | 17.370 | 17.355 | 17.886 | 15.997 |
| Local-only GRU | 16.999 | 16.982 | 17.370 | 16.000 |
| IID FedAvg | 17.281 | 17.268 | 17.783 | 15.947 |

### Runtime and communication

| Run | Training minutes | Model traffic (MB, total / to best) | Peak MB |
|---|---:|---|---:|
| centralised_seed0 | 154.6 | 0 | 1535 |
| centralised_seed1 | 92.1 | 0 | 1572 |
| centralised_seed2 | 86.2 | 0 | 1559 |
| fedavg_geo_seed0 | 196.9 | 16.6 / 14.3 | 1451 |
| fedavg_geo_seed1 | 171.1 | 16.6 / 14.3 | 1455 |
| fedavg_geo_seed2 | 148.5 | 15.2 / 12.8 | 1456 |
| fedavg_iid_seed0 | 192.4 | 19.5 / 17.1 | 1454 |
| fedavg_iid_seed1 | 130.2 | 13.3 / 10.9 | 1456 |
| fedavg_iid_seed2 | 93.0 | 10.9 / 8.6 | 1455 |
| local_seed0 | 137.0 | 0 | 1257 |
| local_seed1 | 130.2 | 0 | 1260 |
| local_seed2 | 103.6 | 0 | 1260 |

Figures: `docs/phase9_figures/` (MAE/RMSE, baselines, per client, per horizon, learning curves).

### Limitations

- One dataset (California XTraffic, flow only), 200 of the eligible sensors, January–September 2023; geography provisional.
- Settings are the Phase 8 B1 settings without retuning; budgets were not reduced. Runs not completed before the training cutoff are excluded, not reported.
- The bootstrap captures sensor sampling; with fewer than three matched seeds, training randomness is not quantified.
- The test split was evaluated once; no model, threshold or setting was changed after it.
- No claim of state-of-the-art performance is made: there is no comparable published benchmark on this exact data and protocol.

## Integrity

- Frozen files re-hashed after the test: unchanged.
- Phase 8 January artifacts (21 checkpoints, 66 run files) unchanged: True.
- January raw file hash equals the 2026-10-24 baseline: True.
- October–December files opened: False.

## Decisions made autonomously

- 2026-10-28T12:06:04Z parallel training processes = 3 (measured throughput {1: {'ms_per_step': 10.42030500054049, 'throughput_steps_per_s': 95.96648082259887}, 2: {'ms_per_step': 12.4579691667653, 'throughput_steps_per_s': 160.54204313325897}, 3: {'ms_per_step': 12.505452222346017, 'throughput_steps_per_s': 240.06844642306555}})
