# Phase 8 results — 200-sensor geographic federated experiment (validation only)

**Date:** 2026-10-28. **Stage:** S5, the pre-declared analysis from `docs/phase8_200_sensor_technical_spec.md` §5.

**Scope:**
- All numbers are on the **January validation split** (23–25 Jan, Mon–Wed). The test split was not read, and no model was retrained.
- The report is aggregate only: no station IDs.
- District, client and type labels come from the provisional row-to-ID mapping. Its status is "supported by strong indirect evidence; unconfirmed by provenance".

**Sources (local, git-ignored):**
- `results/phase8/geo200/*` (12 runs);
- `results/phase8/analysis/sesoi_log.json`;
- `results/phase8/analysis/analysis.json`.

The analysis code is `aiori/phase8_analysis.py` and `scripts/phase8_analysis.py`, tested in `tests/test_phase8_analysis.py`.

---

## 1. Design recap

| Item | Value |
|---|---|
| Sensors | 200 (50 per geographic client); type groups: mainline 128, ramp 41, HOV 26, other 5 |
| Arms × seeds | centralised, local-only, IID FedAvg, geographic FedAvg × seeds 0, 1, 2 (12 runs); B1 GRU and settings unchanged from the pilot |
| Validation cells | 2,047,176 observed target cells in 170,600 overlapping windows (12 horizons); identical cells and masks in all 12 runs |
| Primary endpoint | pooled validation MAE (raw flow units: vehicles per 5 min). Seeds are averaged per cell; arms are paired on identical cells |
| Contrasts | C1 = geographic FedAvg − centralised; C2 = geographic FedAvg − local-only; C3 = IID FedAvg − geographic FedAvg; C4 = local-only − centralised |
| SESOI | **0.1638** = 1% of the centralised mean pooled MAE (16.383). Recorded at 2026-10-28 09:25:26 UTC, before any contrast or bootstrap was computed. The 1% rule was pre-declared on 2026-10-27. The S4 report had already shown descriptive means, so this threshold is pre-analysis but not blind to them |

### Dependence handling
- The overlapping cells are **not** treated as independent.
- **Unit of resampling:** the whole sensor, carrying all of its 10,236 cells across 3 days and 12 horizons. Sensors are resampled within each geographic client (stratified), 2,000 times, with seed 20261028. The same draws are used for every arm, so the contrasts are paired.
- **Spatial sensitivity:** a second bootstrap resamples **freeway-direction clusters** (provisional `Fwy` + `Direction`) within each client. There are 83 clusters of 1–8 sensors (median 2).
- **Reported interval:** the wider of the two.
- **Primary p-value:** the larger of the two. It is two-sided (2 × the share of replicates on the far side of 0) and floored at 1/2000 = 0.0005.
- **Multiplicity:** Holm adjustment across C1–C4.

### Wording rules (pre-declared)
| Situation | Wording |
|---|---|
| Interval excludes 0, Holm p < 0.05 **and** \|Δ\| ≥ SESOI | "supported on January validation data" |
| Interval excludes 0, but \|Δ\| < SESOI or Holm p ≥ 0.05 | "detectable but below the practical threshold" |
| Otherwise | "not distinguishable on this validation set" |

**Three kinds of statement are kept separate below:**
- **Statistical:** whether the interval excludes 0, and the Holm-adjusted p.
- **Practical:** whether \|Δ\| ≥ SESOI.
- **Descriptive:** per-seed and per-sensor counts, which carry no inferential weight.

## 2. Arm results (descriptive)

| Arm | Pooled MAE (seed-averaged) | Per-seed MAE (0 / 1 / 2) | Sensor MAE median (IQR) |
|---|---:|---|---|
| Centralised | 16.383 | 16.339 / 16.495 / 16.314 | 13.06 (6.74–24.73) |
| Local-only | 16.458 | 16.604 / 16.311 / 16.458 | 13.18 (6.82–25.82) |
| IID FedAvg | 16.477 | 16.501 / 16.380 / 16.549 | 13.24 (6.82–25.12) |
| Geographic FedAvg | 16.495 | 16.437 / 16.608 / 16.439 | 13.11 (6.92–24.97) |
| *Historical average (reference)* | *22.543* | | |

**MAE by horizon (5 / 30 / 60 min):**
- centralised 12.50 / 16.25 / 19.53;
- local-only 12.60 / 16.37 / 19.55;
- IID FedAvg 12.54 / 16.37 / 19.65;
- geographic FedAvg 12.56 / 16.38 / 19.68.

**What this shows:** all four arms beat the historical-average reference by about 27%. The spread between arms (0.11 MAE, about 0.7%) is small compared with how much errors vary from sensor to sensor.

## 3. Primary contrasts (pooled MAE)

| | Δ (a − b) | Sensor-bootstrap 95% CI | Freeway-cluster 95% CI | p (conservative) | Holm p | \|Δ\| ≥ SESOI? | Verdict |
|---|---:|---|---|---:|---:|---|---|
| **C1** geographic FedAvg − centralised | **+0.112** | [0.057, 0.167] | [0.055, 0.178] | ≤ 0.0005 | 0.002 | no (0.112 < 0.164) | **detectable but below the practical threshold** |
| **C2** geographic FedAvg − local-only | +0.037 | [−0.066, 0.143] | [−0.078, 0.167] | 0.58 | 0.58 | no | not distinguishable on this validation set |
| **C3** IID FedAvg − geographic FedAvg | −0.018 | [−0.043, 0.007] | [−0.048, 0.012] | 0.24 | 0.49 | no | not distinguishable on this validation set |
| **C4** local-only − centralised | +0.075 | [−0.026, 0.187] | [−0.024, 0.179] | 0.16 | 0.47 | no | not distinguishable on this validation set |

The freeway-cluster interval was the wider one for C1, C2 and C3; the sensor interval was wider for C4. All reported intervals use the wider of the two.

**Reading the table:**
- **C1, the cost of federation versus pooling:** geographic FedAvg is worse than centralised training by about 0.11 vehicles per 5 min (0.7%).
  - The interval excludes 0 under both bootstraps, and the difference survives Holm.
  - The CI's upper end (0.167–0.178) reaches the SESOI, so a practically relevant cost is **not ruled out**. The point estimate is below it.
- **C2, collaboration versus isolation:** no difference is detectable. The interval allows anything from geographic FedAvg being 0.07 better to 0.17 worse than local-only.
  - This differs from the 40-sensor pilot, where FedAvg was worse than local-only by +0.29.
- **C3, geographic versus IID partitioning:** no detectable effect, and the whole interval lies well inside ±SESOI. Geographic partitioning does **not** cost a practically meaningful amount on this sample; this is equivalence-style evidence, not a formal equivalence test.
- **C4, local-only versus centralised:** not distinguishable, with a wide interval. Local-only results vary the most from seed to seed.

## 4. Sensitivity analyses

### 4.1 Leave one validation day out
Point estimates only; three days show stability, not an interval.

| Day dropped | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| none (all 3 days) | +0.112 | +0.037 | −0.018 | +0.075 |
| Mon 23 Jan | +0.130 | +0.050 | −0.018 | +0.080 |
| Tue 24 Jan | +0.100 | +0.046 | −0.019 | +0.054 |
| Wed 25 Jan | +0.105 | +0.014 | −0.017 | +0.090 |

Single days: C1 is +0.075 / +0.134 / +0.126, and C3 is −0.016 to −0.020. **Signs and orders of magnitude are stable across days.** No single day drives any contrast.

### 4.2 Seeds (descriptive only; seeds measure training randomness, not sampling)

| | Seed 0 | Seed 1 | Seed 2 | Seeds 1–2 only |
|---|---:|---:|---:|---:|
| C1 | +0.098 | +0.112 | +0.125 | +0.119 |
| C2 | −0.167 | +0.296 | −0.019 | +0.139 |
| C3 | +0.064 | −0.228 | +0.110 | −0.059 |
| C4 | +0.265 | −0.184 | +0.144 | −0.020 |

- **C1** has the same sign and similar size in every seed.
- **C2–C4** change sign between seeds. That is training-seed variability, mostly from the local-only arm and from IID versus geographic run lengths. This agrees with those contrasts being not distinguishable. **The same sign across 3 seeds is not treated as evidence.**

### 4.3 Exploratory breakdowns (95% sensor-bootstrap CIs; no claims, no multiplicity control)
- **RMSE:**
  - C1 +0.15 [0.03, 0.27];
  - C2 +0.23 [0.04, 0.44];
  - C3 +0.02 [−0.05, 0.08];
  - C4 −0.08 [−0.28, 0.14].
  - RMSE is driven by large errors on high-flow mainline sensors. Local-only has the *lowest* pooled RMSE (29.07) despite a higher MAE.
- **Normalised MAE (z-units):** all differences are ≤ 0.002 (about 0.6% of 0.256). C1, C2 and C3 have intervals excluding 0, but they are negligible in size.
- **Horizon:**
  - The C1 gap grows with horizon: +0.07 at 5 min, +0.13 at 30 min, +0.15 at 60 min, with all intervals above 0.
  - C2 drifts from −0.03 at 5 min to +0.13 at 60 min, with every interval including 0.
  - C3 is small at every horizon (≤ 0.04).
- **Client (C1):** LA/Ventura +0.24 [0.09, 0.41]; Bay Area +0.08; Coastal +0.07; Inland +0.06. The C1 cost is concentrated in LA/Ventura, the client with the largest flows on average. C3 per client is between −0.05 and +0.01, with every interval including 0.
- **Type group (C1):** mainline +0.14, HOV +0.13, ramp +0.02, other +0.03. Federation costs most on high-volume sensors.
- **Flow tertile (C1):** high +0.19, mid +0.09, low +0.06.
- **Per sensor (descriptive):**
  - geographic FedAvg beats centralised on 77 of 200 sensors (median per-sensor Δ +0.02);
  - it beats local-only on 92 of 200;
  - IID beats geographic on 121 of 200 (median −0.01);
  - local-only and centralised split 100 / 100.

## 5. Geographic FedAvg seed 0 reached the 50-round cap

This uses only the saved validation-MAE history per round; nothing was retrained.

| | Geographic s0 | Geographic s1 | Geographic s2 | IID s0 | IID s1 | IID s2 |
|---|---|---|---|---|---|---|
| Rounds run / best | **50 / 46** (cap) | 28 / 23 | 43 / 38 | 36 / 31 | 46 / 41 | 28 / 23 |
| Best validation MAE | 16.437 | 16.608 | 16.439 | 16.501 | 16.380 | 16.549 |
| First round within 1% / 0.5% of best | 26 / 37 | 20 / 23 | 25 / 30 | 23 / 30 | 33 / 39 | 18 / 19 |
| Gain over the 10 rounds before best | 0.086 | 0.323 | 0.150 | 0.199 | 0.173 | 0.300 |

**Findings:**
- At the cap it was 4 rounds past its best, one round short of the patience-5 stop. The last 8 rounds sat between 16.44 and 16.51.
- It improved least over its final 10 rounds (0.086), so it was **on a plateau, not still descending**.
- Its best score matches seed 2, which stopped early.
- The cap therefore probably changed little. The improvement that more rounds could have brought is unknown, but the plateau suggests it is far smaller than the SESOI.
- **Comparisons are unaffected in practice:** without seed 0 (matched seeds 1–2, all arms), C1 = +0.119, which is unchanged in substance.
- **Communication note:** this run used the most model traffic (23.8 MB in total, 21.9 MB up to its best round). Seeds 1 and 2 reached within 1% of their best after 20–25 rounds (9.5–11.9 MB).
- No budget change is proposed. Communication-efficiency questions, such as stopping at a tolerance, would be new analyses needing separate approval.

## 6. Conclusions (January validation data only)

1. **Federation cost (RQ1):** geographic FedAvg's MAE is higher than centralised training by +0.11 (0.7%). The difference is statistically detectable after Holm but below the pre-declared practical threshold. It is concentrated at longer horizons, high-flow sensors and LA/Ventura. Wording: **"detectable but below the practical threshold"**.
2. **Collaboration versus isolation (RQ2):** **not distinguishable.** The pilot's finding that FedAvg was worse than local-only (+0.29 with 10 sensors per client) does **not** carry over to 50 sensors per client.
3. **Effect of geographic partitioning (RQ3):** **not distinguishable**, with an interval inside ±SESOI. The geographic clients' non-IID structure does not measurably hurt FedAvg here.
4. **Stability (RQ4):** these conclusions hold under the freeway-cluster bootstrap, leave-one-day-out, and the matched-seed sensitivity without seed 0.
5. **Nothing here is "confirmed".** Confirmation would need the closed test split, run once on frozen configurations, in a separately approved phase.

## 7. Limitations
- **Provisional geography:** client, District, type and freeway labels rest on the unconfirmed row-to-ID mapping. Heterogeneity is defined by this geography, not by verified locations.
- **Short validation period:** three consecutive weekdays, with no weekends or holidays. Validation was also used for early stopping in every run, so the comparisons are optimistic for all arms equally.
- **SESOI not blind:** it was fixed after the descriptive S4 means had been shown.
- **Spatial dependence:** the freeway-cluster bootstrap handles same-freeway correlation only approximately. Sensors on different freeways can still be correlated, for example through region-wide weather or events.
- **Seeds:** three seeds measure training randomness only roughly. The bootstrap reflects sensor sampling, not training randomness, because the seeds are averaged first.
- **Single budget:** one budget and one hyperparameter set (the B1 pilot settings, not retuned for 200 sensors). One run hit the round cap.
- **Simulation:** sequential, in one process. Communication is counted in bytes, with no network effects. Client sizes are equal, so there is no quantity skew.
- **Scope:** January only, flow only, 200 of 11,140 eligible sensors.

## 8. Recommendation for the next stage (each item needs separate approval)
1. **Commit the reviewed S1–S5 code, tests and aggregate docs** (D-G / D8-12). Reconcile with the rewritten remote by branching from `origin/main` first, with no history rewrite. Keep manifests, checkpoints and results local.
2. **Optional, before any test evaluation:** a pre-registered test-split confirmation of C1–C4 on the frozen Phase 8 configurations and checkpoints, run once. It would score the existing 12 best checkpoints on the January test split (26–31 January, including a weekend). This is the only way to move beyond "validation" wording.
3. **Later, separate phases:**
   - full-year scaling (11 unopened months);
   - communication-efficiency variants (fewer rounds, compression);
   - quantity-skewed client sizes.
