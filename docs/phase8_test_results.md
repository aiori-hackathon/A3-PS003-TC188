# Phase 8 held-out test results — January 26–31, 2023 (one-time, pre-registered)

**Date:** 2026-10-28. **Stages:** E3, the single test pass, and E4, the pre-declared analysis, run under `docs/phase8_test_protocol.md` (SHA-256 `4cd54367…899c`).

**Scope:**
- Aggregate only: no station IDs.
- District, client and type labels depend on the provisional row-to-ID mapping ("supported by strong indirect evidence; unconfirmed by provenance").

**Runs (local, git-ignored):**
- Sources: `results/phase8/test/analysis_test.json` and `summary_test.md`.
- **E3:** 12 frozen runs (21 checkpoints) and 3 reference forecasts, on 343,246 test windows. It took 46.8 s, with peak memory 634 MB.
- **E4:** 6.2 s, peak memory 1.2 GB.

**Pre-registration chain:**
- **Frozen beforehand:** checkpoints, code, SESOI (0.1638) and statistics, all set before any test value existed.
- **E2:** reproduced every validation prediction byte-for-byte.
- **E4 code:** hashed before E3 (`534475a9…5204`).
- **Single pass:** both E3 and E4 ran once.

---

## 1. Final results table (test, seeds averaged; validation for comparison)

| Model | Test MAE | Test RMSE | Normalised MAE | Sensor-average MAE | Negative predictions | Validation MAE | Validation → test change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Centralised | **15.789** | 27.364 | 0.2488 | 15.781 | 1.02% | 16.383 | −0.594 |
| Local-only | 15.890 | 27.393 | 0.2494 | 15.883 | 1.39% | 16.458 | −0.567 |
| IID FedAvg | 15.882 | 27.535 | 0.2498 | 15.874 | 1.19% | 16.477 | −0.595 |
| Geographic FedAvg | 15.873 | 27.518 | 0.2500 | 15.865 | 1.12% | 16.495 | −0.621 |
| Persistence (reference) | 21.508 | 36.998 | — | — | — | 22.671 | |
| Historical average (reference) | 26.585 | 46.059 | — | — | — | 22.543 | |
| Same time yesterday (reference) | 29.250 | 53.382 | — | — | — | 28.814 | |

Units are vehicles per 5 minutes. There are 4,117,386 observed test cells (1,566 masked). The negative-prediction rates fall to 0.5–0.8% on the weekend.

**Per-seed test MAE (seeds 0 / 1 / 2):**

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Centralised | 15.751 | 15.880 | 15.735 |
| Local-only | 15.996 | 15.797 | 15.878 |
| IID FedAvg | 15.907 | 15.800 | 15.939 |
| Geographic FedAvg | 15.800 | 15.999 | 15.821 |

**Against the reference forecasts:**
- Every learned model beats the best reference forecast, persistence, by about 5.6–5.7 MAE (26%).
- They beat the historical average by about 10.7 MAE (40%).
- The historical average degrades from validation to test (22.5 → 26.6). Persistence and the GRU models improve, partly because test includes a low-flow weekend.

## 2. Pre-registered contrasts (SESOI 0.1638; seeds averaged per cell; paired)

| | Test Δ | Test 95% CI (wider) | Holm p | Test verdict | Validation Δ, verdict | Consistent? |
|---|---:|---|---:|---|---|---|
| **C1** geographic FedAvg − centralised | **+0.084** | [0.043, 0.134] | **0.002** | detectable but below the practical threshold on January test data | +0.112, detectable but below the threshold | **yes** |
| **C2** geographic FedAvg − local-only | −0.017 | [−0.122, 0.088] | 0.91 | not distinguishable on January test data | +0.037, not distinguishable | **yes** |
| **C3** IID FedAvg − geographic FedAvg | +0.008 | [−0.013, 0.030] | 0.91 | not distinguishable on January test data | −0.018, not distinguishable | **yes** |
| **C4** local-only − centralised | +0.102 | [0.007, 0.205] | 0.099 | detectable but below the practical threshold on January test data † | +0.075, not distinguishable | **no** |

Sensor-bootstrap and freeway-cluster intervals were both computed. The freeway interval was the wider for every contrast, and the reported p is the larger of the two.

**† C4:** the label comes from the pre-registered rule, which applies "detectable" whenever the 95% interval excludes 0. Read it with care:
- the interval excludes 0 only narrowly (lower bound 0.007);
- the Holm-adjusted p of 0.099 **does not** survive the multiple-comparison correction;
- the per-seed differences change sign (+0.245, −0.083, +0.143);
- without seed 0 it shrinks to +0.03.

Treat C4 as **weak, not multiplicity-robust evidence** that local-only is slightly worse than centralised. This is the one inconsistency with validation.

**Statistical versus practical significance:**
- **Statistically detectable after Holm:** only C1.
- **Practically meaningful (|Δ| ≥ 0.1638):** none. Every point estimate is below the threshold.
- **C1's upper bound:** at 0.134 it is now also below the SESOI. On test, a practically relevant federation cost is ruled out at 95% confidence; on validation the interval had reached the threshold.
- **C3:** the whole interval sits inside ±0.031, far inside ±SESOI.

## 3. Breakdowns (descriptive, or exploratory intervals; no claims)

### Per client (test MAE)
| Client | Centralised | Local-only | IID FedAvg | Geographic FedAvg | Persistence |
|---|---:|---:|---:|---:|---:|
| LA/Ventura | 17.30 | 17.26 | 17.45 | 17.46 | 23.05 |
| Bay Area | 18.25 | 18.43 | 18.36 | 18.34 | 25.05 |
| Coastal/South Central | 14.38 | 14.43 | 14.46 | 14.43 | 19.95 |
| Inland/Central | 13.22 | 13.45 | 13.25 | 13.26 | 17.98 |

C1 by client: LA/Ventura +0.16 [0.06, 0.28], Bay Area +0.08, Coastal +0.05, Inland +0.05. As on validation, the federation cost is concentrated in LA/Ventura.

### Per horizon (test MAE at 5 / 30 / 60 min)
| | 5 min | 30 min | 60 min |
|---|---:|---:|---:|
| Centralised | 12.35 | 15.69 | 18.61 |
| Local-only | 12.43 | 15.81 | 18.71 |
| IID FedAvg | 12.38 | 15.80 | 18.76 |
| Geographic FedAvg | 12.38 | 15.77 | 18.76 |
| Persistence | 14.31 | 20.73 | 28.77 |

The C1 gap grows with horizon: +0.04 at 5 min, +0.08 at 30 min, +0.14 at 60 min. The same pattern appeared on validation.

### Per test day and weekday versus weekend (test MAE)
| | Thu 26 | Fri 27 | Sat 28 | Sun 29 | Mon 30 | Tue 31 | Weekday | Weekend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralised | 17.49 | 16.75 | 15.24 | 13.72 | 15.39 | 16.19 | 16.45 | 14.48 |
| Local-only | 17.52 | 16.78 | 15.34 | 13.95 | 15.51 | 16.28 | 16.52 | 14.65 |
| IID FedAvg | 17.60 | 16.81 | 15.32 | 13.84 | 15.47 | 16.29 | 16.54 | 14.58 |
| Geographic FedAvg | 17.62 | 16.81 | 15.30 | 13.78 | 15.47 | 16.30 | 16.55 | 14.54 |

**Weekday versus weekend contrasts (exploratory, 95% sensor-bootstrap CI):**

| | Weekday | Weekend |
|---|---|---|
| C1 | +0.095 [0.05, 0.14] | +0.063 [0.03, 0.10] |
| C2 | +0.028 [−0.08, 0.13] | −0.106 [−0.19, −0.02] |
| C3 | −0.006 [−0.03, 0.02] | +0.037 [0.02, 0.06] |
| C4 | +0.067 [−0.03, 0.18] | +0.169 [0.09, 0.26] |

The local-only arm does worse on weekends, which drives C2 and C4 there.

**Leave one test day out:** C1 stays between +0.077 and +0.089, C3 between −0.001 and +0.013, C2 between −0.040 and +0.013, and C4 between +0.075 and +0.117. No single day drives any contrast.

### Other exploratory measures
- **RMSE:** C1 +0.15 [0.08, 0.23], C2 +0.13 [−0.05, 0.29], C3 +0.02, C4 +0.03.
- **Normalised MAE:** all differences are ≤ 0.0012 (about 0.5% of 0.249).
- **By type group and flow tertile (C1):** the cost is in mainline and HOV (+0.10 each) and in high-flow sensors (+0.17). Ramps (+0.02) and low-flow sensors (+0.03) are close to 0.
- **Per sensor:** geographic FedAvg beats centralised on 75 of 200 sensors; IID beats geographic on 106 of 200.

## 4. Validation versus test
| Finding | Validation | Test | Agreement |
|---|---|---|---|
| Arm ranking (MAE) | centralised < local-only < IID < geographic | centralised < geographic ≈ IID < local-only | Centralised is best on both. The other three are within 0.02 on test and switch order. Differences among them are not distinguishable on either |
| C1 federation cost | +0.112, detectable, below threshold, CI reaching the SESOI | +0.084, detectable, below threshold, CI below the SESOI | **consistent, and firmer on test** |
| C2 FedAvg versus local-only | not distinguishable | not distinguishable | consistent |
| C3 geographic versus IID | not distinguishable, within ±SESOI | not distinguishable, within ±0.031 | consistent |
| C4 local-only versus centralised | not distinguishable (+0.075) | interval narrowly excludes 0 (+0.102); Holm p 0.099 | **inconsistent label**: the test evidence is weak and seed-dependent |
| Horizon and client patterns | C1 grows with horizon; concentrated in LA/Ventura and high-flow sensors | same | consistent |
| Level | MAE about 16.4–16.5 | MAE about 15.8–15.9 (weekend days are easier) | expected shift; no arm-specific generalisation gap |

## 5. Conclusions (January 2023, 200 sensors, flow)
1. **Federation cost (RQ1):** geographic FedAvg is about 0.08–0.11 MAE (about 0.5–0.7%) worse than centralised training. It is **statistically detectable on both validation and held-out test, and below the practical threshold**; on test, a practically relevant cost is excluded at 95%. The cost grows with forecast horizon and concentrates on high-flow sensors and LA/Ventura.
2. **Collaboration versus isolation (RQ2):** FedAvg and local-only training are **not distinguishable** on either split. The pilot's large FedAvg penalty did not hold at 50 sensors per client.
3. **Geographic non-IID partitioning (RQ3):** it **does not measurably hurt** FedAvg relative to an IID partition of the same sensors. The test interval lies within ±0.03, far inside ±SESOI.
4. **Local-only versus centralised:** at most a small, weak disadvantage for local-only. It shows on test and on weekends, but is not robust to multiplicity or seeds.
5. **Practical reading:** on this sample, a federated geographic deployment would give accuracy within about 1% of pooled training, with model traffic of about 0.5 MB per round. That is a small, quantified price for not sharing raw data.

## 6. Limitations
- **Provisional geography:** all geography is provisional (the row-to-ID mapping is unconfirmed).
- **Scope:** results cover one month (January), flow only, and 200 of 11,140 eligible sensors. The test window is 6 days with one weekend and no holiday.
- **One budget:** one set of hyperparameters (the B1 pilot settings) was used for every arm without retuning, and one run hit the 50-round cap.
- **Uncertainty:** the bootstrap reflects sensor sampling only. Seeds are averaged first, so training randomness enters only through the 3-seed average. The freeway-cluster bootstrap approximates spatial correlation.
- **The wording rule's interval is not multiplicity-adjusted.** The C4 label shows this; Holm p-values are reported next to every verdict.
- **Simulation:** federation is simulated sequentially, with equal client sizes and byte counts rather than a real network.
- **The January test split is now used.** Any further modelling must be evaluated on new data.

## 7. Integrity
- E3 and E4 each ran once, under the hash gates.
- The 21 checkpoints, all earlier artifacts, the raw files and the frozen code were unchanged afterwards.
- No retraining, tuning or threshold change took place.
- Predictions and per-sensor outputs stay in git-ignored `results/phase8/test/`.
