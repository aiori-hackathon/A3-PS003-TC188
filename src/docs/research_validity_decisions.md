# Research validity decisions — R1 sensor mapping, R2 client heterogeneity

**Date:** 2026-10-27.
**Source investigation:** `docs/pre_phase6_research_validity.md`. Scripts are in `scripts/analysis/`; their reproduction is described in §5.

**Labels:**
- **APPROVED:** explicitly approved by the project lead in writing, with the source noted.
- **PROPOSED:** recommended, not approved.
- **FUTURE:** recorded design, not scheduled.

All analyses used training-period data only (steps [0, 6336)).

---

## 1. R1: public tensor row ↔ station ID (`node_order.npy`)

### Official status (APPROVED wording, design brief 2026-10-27)
> **SUPPORTED BY STRONG INDIRECT EVIDENCE; UNCONFIRMED BY PROVENANCE.**

This must not be reported as VERIFIED. Every station ID, District, County and geographic client label derived through `node_order` stays **provisional**.

### Evidence preserved

| Check | Result | Direction |
|---|---|---|
| Residual correlation (own day-type profile removed), metadata neighbours ≤ 1 mile on the same freeway and direction (5,756 Mainline pairs) | median **0.859** | supports |
| Matched distant controls (same District, > 50 km) | median **0.349** | — |
| Within-District label shuffle | median 0.362–0.371 | — |
| Neighbour vs control discrimination | AUC **0.924**; 83.0% of neighbour pairs above the controls' 95th percentile; holds in all 8 Districts | supports |
| Sensor type vs flow | Mainline/ramp flow ratio **10.55** vs shuffle null ≤ 1.03 | supports |
| Road width vs Mainline mean flow | rank correlation **0.77** | supports |
| Whole-month copied-series groups | 97 of 97 lie within one District | supports |
| Upstream 4-sensor labelled sample (`raw_flow_sel.npy`, labels from internal `node_list`) | each series matches a public tensor row exactly, but **not** the row `node_order` assigns to its label | **contradicts** |
| Same upstream sample vs its own metadata | "co-located" series correlate at random-pair level (0.25–0.34); the "on-ramp" carries mainline flow | weakens the contradicting sample |

### Limitations
- The evidence is population-level and indirect. It rules out gross misalignment (random, shuffled within District, shifted) but **not** local swaps between similar co-located sensors.
- No export script, manifest or maintainer statement exists. The one direct linkage found (the upstream sample) disagrees with `node_order`.
- The upstream sample has 4 series; 2 of them are copied profiles.

### Remaining independent verification routes (none started; each needs approval)
1. **Caltrans PeMS match:** download January 2023 5-minute station flow for a few station IDs (needs a PeMS account) and compare with the tensor rows `node_order` assigns.
2. **Official export script or manifest** linking public tensor rows to station IDs.
3. **Written confirmation from the XTraffic maintainers.**
4. **The `node_list.npy` vs `node_order.npy` discrepancy:** obtain the internal `node_list` and compare, and/or report the §3.4 discrepancy of the investigation to the maintainers.

No new data has been downloaded for these routes and no maintainer has been contacted.

---

## 2. R2: heterogeneity of the four geographic clients

### Findings preserved

| Measure | Value |
|---|---|
| Eligible population (training-period eligibility) | **11,188** sensors |
| Primary pilot | **40** sensors (10 per client) |
| Pilot multivariate test (pseudo-F 1.34 vs random balanced partitions) | **p = 0.18** |
| Population multivariate test (pseudo-F 82.9; null 95th percentile 1.73) | **p < 0.005** |
| Largest client-explained feature variance (η²), population | **≈ 10%** (night share 0.10; all others ≤ 0.05) |
| Detection power, random k per client | **17%** at 10, 49% at 25, **92%** at 50, 100% at 100 (a separate run of the same test gave 21% at 10) |
| Effect of per-sensor normalisation | removes level differences; z-space between-client W1 0.068 vs within-client 0.053 |

### Interpretation (as recorded)
- The geographic clients are **not IID**, but they are **weakly** heterogeneous.
- The 40-sensor pilot is **underpowered**, so its null result is absence of evidence, not evidence of IID.
- The pilot is **suitable for validating the forecasting and federated pipelines** and **insufficient for strong geographic non-IID conclusions**.

### Limitations
- January training days only.
- The features are descriptive; no model-based transfer measure was used.
- Permutation counts are modest.
- The pilot was selected for completeness and is not representative of client populations (e.g. coastal pilot median flow 276 vs 73 in the population).

---

## 3. Decisions

| # | Decision | Status | Source |
|---|---|---|---|
| A1 | Training-period-only sensor eligibility | APPROVED | review revision 2, 2026-10-24 |
| A2 | Exclude all members of exact-copy groups (precautionary, not "invalid"), including the 42 training-only copies | APPROVED | 2026-10-24 |
| A3 | District-floor pilot selection (40 sensors, 10 per client) | APPROVED | 2026-10-24 |
| A4 | Separate 8-sensor missing-data validation set; synthetic-only coverage for the empty-window drop and mean fallback | APPROVED | 2026-10-24 |
| A5 | 22 / 3 / 6-day split unchanged; weekday-only validation documented | APPROVED | 2026-10-24 |
| A6 | R1 official status wording (§1) | APPROVED | design brief 2026-10-27 |
| A7 | Keep the four-client geographic partition for the initial implementation | APPROVED | design brief 2026-10-27 |
| A8 | Preserve the R1/R2 scripts in `scripts/analysis/` | APPROVED (executed) | design brief 2026-10-27 |
| P1 | Update older documents that still say "supported but unconfirmed" to the A6 wording | PROPOSED | — |
| P2 | Phase 6 architecture, training and evaluation details (`docs/phase6_technical_spec.md`) | PROPOSED | — |
| P3 | Pursue verification route 1 (PeMS) and/or 3–4 (maintainers) | PROPOSED | — |

---

## 4. Future experimental design (recorded; not scheduled, nothing generated)

| Experiment | Purpose | Status / prerequisite |
|---|---|---|
| **Primary engineering pilot:** the frozen 40 sensors, 4 geographic clients | validate the forecasting and federated pipelines; exploratory comparisons only | current |
| **Larger geographic experiment:** ≈ 200 sensors (≈ 50 per client, power ≈ 0.92) | support geographic non-IID conclusions | FUTURE; needs a separately approved selection policy (e.g. stratified by sensor type or flow level to reduce completeness bias) |
| **Behaviour-based non-IID stress partition** | test FedAvg under strong, controlled heterogeneity | FUTURE, optional; built from training-period features only; clearly labelled as non-geographic; fixed before any test evaluation |
| **Normalisation ablation** (per-sensor vs global or per-client scaling) | measure how scaling changes effective heterogeneity | FUTURE, optional |
| Quantity-skew partition | add the size heterogeneity currently absent (all clients have 63,130 training windows) | FUTURE, optional |

---

## 5. Reproducing the investigation

**Scripts (copied unchanged from the investigation scratch folder; SHA-256 recorded):**

| Script | Reproduces | Inputs | Seed |
|---|---|---|---|
| `scripts/analysis/r1_spatial.py <path to raw_flow_sel.npy>` | R1 tests A, B, C | `year_2023/2023_p01.npy`, `node_order.npy`, `data/processed/sensor_metadata_aligned.csv`, `data/processed/flow_sensor_audit.csv`, and the upstream `causal_analysis/data/raw_flow_sel.npy` from github.com/XAITraffic/XTraffic (not in the project) | `default_rng(20261027)` |
| `scripts/analysis/r2_heterogeneity.py` | R2 features, η², pseudo-F, W1, power at 10 per client, cluster illustration | `year_2023/2023_p01.npy`, `data/processed/flow_sensor_audit.csv`, `sensor_selection.csv`, `pilot_preprocessed.npz` | `default_rng(7)` |
| `scripts/analysis/r2_power.py` | power at 10 / 25 / 50 / 100 per client | as r2 (first part of `r2_heterogeneity.py` plus a power loop) | `default_rng(7)` |

- **How to run:** from the project root with `.venv/Scripts/python.exe -W ignore <script> [args] > out.json`. Results go to stdout as JSON; the scripts write no files.
- **Dependencies:** numpy and pandas only (the versions in `requirements.txt`).
- **Reference outputs:** `scripts/analysis/original_results/r1_results.json` and `r2_results.json` (verbatim). The power output was printed to the console only: `{"10": 0.17, "25": 0.49, "50": 0.92, "100": 1.0}`.
- **Data note:** `r1_results.json` and the investigation document contain sensor-level identifiers, row indices and upstream station IDs. Treat them like the selection manifests before publishing.

### 5.1 Reproduction run (2026-10-27, the project copies in `scripts/analysis/`)

| Script | Runtime (24-thread Xeon, 63.5 GB RAM) | Result |
|---|---:|---|
| `r1_spatial.py` | 23 s | stdout **byte-identical** to `original_results/r1_results.json` |
| `r2_heterogeneity.py` | 30 s | stdout **byte-identical** to `original_results/r2_results.json` |
| `r2_power.py` | 41 s | `{"10": 0.17, "25": 0.49, "50": 0.92, "100": 1.0}`, identical to the original console output |

- **Memory:** each script reads the training slice of the memory-mapped tensor in batches. Peak memory was not measured; the largest in-memory block is about 0.3 GB, the training slice for about 11k sensors.
- **No changes:** before and after, SHA-256 hashes of `data/`, `aiori/` and `tests/` were identical.
- **External dependency:** `r1_spatial.py` Test C needs the upstream `raw_flow_sel.npy`, which is not in the project. Without it, run the Test A/B part only, or fetch the file (needs approval).

---

## 6. Outstanding research questions
1. Which ordering, `node_order` or the internal `node_list`, was used to write the public tensor? Why do they disagree on the upstream sample?
2. Are individual pilot sensors correctly identified? Only an external match can show this.
3. How heterogeneous are clients in other months, and on weekends (validation is weekdays only)?
4. Should per-sensor normalisation remain the primary condition given that it reduces the non-IID contrast?
5. What selection policy should the ≈ 200-sensor experiment use?
