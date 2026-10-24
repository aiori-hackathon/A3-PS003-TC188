# Checkpoint — 2026-10-24 (PS-003, pre-Phase 6)

**Purpose:** freeze the AIORI Hackathon 3 PS-003 data pipeline at a verified state before any modelling. Nothing in this document starts Phase 6 or trains or evaluates a model.

**Workspace:** the project root. At freeze time it was **not a Git repository** (`git rev-parse` → "not a git repository"). Git was initialised afterwards, only to publish this checkpoint (see "Publication" below).

## Complete

| Area | State |
|---|---|
| Sensor eligibility | Training period only: steps [0, 6336), 2023-01-01 to 01-22. Whole-month statistics are informational `month_*` columns only. 11,188 eligible sensors. |
| Exact-copy exclusion | Precautionary: all members of training-period exact-copy groups are excluded (1,944 sensors / 104 groups, including 42 copies only during training). Excluded sensors are **not** judged invalid. |
| Primary pilot | 40 sensors, 10 per provisional client, with District floors. Coastal = District 5 ×1 + District 12 ×9; inland = District 3 ×1, 6 ×4, 8 ×4, 10 ×1. All 100% observed. |
| Missing-data validation set | 8 separate sensors (99–<100% January coverage, val/test gaps). Never mixed into the primary pilot. Exercises fill (121 inputs) and target masks (1,452 cells) on real data. |
| Split | 22 / 3 / 6 days: train [0, 6336), val [6336, 7200), test [7200, 8928). Unchanged. |
| Preprocessing | Per-sensor z-score fitted on training observations; causal input fill (forward fill ≤ 6 steps within a split → train time-of-day median → train mean); targets never imputed (NaN + mask); zero flow is valid. |
| Windows | 12 → 12 index windows; targets fully inside one split. Primary: 252,520 / 34,120 / 68,680. |
| Verification | `scripts/verify_artifacts.py` (85 checks), 55 synthetic pytest tests, local raw-file integrity baseline. |
| Archive | Revision 1 (whole-month eligibility) outputs in `data/processed/archive/v1_month_eligibility/`. |

## Explicitly not started
Phase 6 (compact GRU, masked loss) and everything after it: centralized, local-only, IID-like FedAvg and geographic FedAvg training; communication-byte logging; three-seed experiments; evaluation, plots and the report. No model has been trained or evaluated. PyTorch is not installed.

## Decisions recorded
1. **Training-period-only exact copies:** keep excluding all members of exact-copy groups, including the 42 sensors that are copies only during training. This is a conservative policy; those sensors are not called definitively invalid. Recorded in `docs/preprocessing_review.md` (§1 and "Final pre-Phase-6 decisions"), `docs/implementation_plan.md` (decision 7) and `docs/data_quality_notes.md` §2.
2. **No forced 12-step gap:** the primary missing-data validation set is not changed just to obtain a ≥ 12-step gap. Dropping windows whose targets are all missing, and the training-mean fallback, stay covered by synthetic tests. Any naturally qualifying real-data stress case would be a **separate** set, not a change to this one.

## Final checks run for this checkpoint (2026-10-24)
All checks were read-only. Before and after, SHA-256 hashes of 46 files under `data/`, `aiori/`, `scripts/`, `tests/`, `docs/`, `pytest.ini` and `rebuild_aiori_metadata.py` were identical, so **no file changed**.

**1. Tests** (pytest cache and bytecode writing disabled):
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```
Result: **55 passed**.

**2. Real-data verification, read-only.** `scripts/verify_artifacts.py` normally rewrites `data/processed/verification_results.json` at the end. To avoid modifying that artifact, its check functions were called in the same order as its `main()`, without the final write:
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts'); import verify_artifacts as v; r=v.Results(); v.check_raw(r); a,n=v.check_inputs(r); g=v.check_assignment(r,n); s=v.splits.make_splits(a.shape[1]); au=v.check_audit(r,a,s); v.check_manifests(r,a,n,g,au,s); [v.check_preprocessed(r,a,l,o,m) for l,(o,m) in v.PREPROCESSED_RUNS.items()]; print(len(r.failed),'failed')"
```
Result: **85 PASS, 0 FAIL** (16 INFO).

**Saved results:** the existing `data/processed/verification_results.json` (run 2026-10-24 06:58 UTC: 85 passed, 0 failed) is unchanged. **No saved pytest report file exists**; pytest output is console-only, and the result is recorded here.

**3. Raw-file integrity.** The SHA-256 and size of `year_2023/2023_p01.npy`, `node_order.npy` and `sensor_meta_feature.csv` match `data/raw_integrity_baseline.json`. That baseline is a **local baseline recorded 2026-10-24 06:55 UTC, after download**. It is **not** the original-download hash or a publisher checksum, so it shows only that the raw files haven't changed since it was recorded. The baseline was not rewritten.

## Key artifacts
| Path | Content |
|---|---|
| `data/processed/flow_sensor_audit.csv` | per-sensor training-period stats, eligibility, `month_*` info |
| `data/processed/sensor_selection.csv` | primary 40-sensor manifest |
| `data/processed/missing_validation_selection.csv` | 8-sensor validation manifest |
| `data/processed/split_config.json`, `scaler_parameters.json`, `pilot_preprocessed.npz`, `window_index.npz` | primary preprocessing inputs for Phase 6 |
| `data/processed/missing_validation/` | the same artifacts for the validation set |
| `data/processed/proposed_client_assignment.csv`, `sensor_metadata_aligned.csv` | provisional District→client assignment and aligned metadata |
| `data/processed/flow_quality_report.md`, `preprocessing_report.md` | generated summaries |
| `data/processed/verification_results.json` | saved verification results (85/0) |
| `data/raw_integrity_baseline.json` | local raw-file hash baseline |
| `data/processed/archive/v1_month_eligibility/` | archived revision 1 outputs |
| `docs/implementation_plan.md`, `docs/preprocessing_review.md`, `docs/data_quality_notes.md` | plan, review, data notes |

## Known limitations
- **Row-to-station-ID mapping is unconfirmed.** That public `2023_p01.npy` row `i` is sensor `node_order.npy[i]` is supported but unproven. Every station ID, District, County and geographic client label is **provisional**, and so is any geographic non-IID conclusion.
- Channel 0 = flow, in vehicles per 5 minutes, is a working convention. Timestamps are reconstructed (2023-01-01 00:00, 5-minute steps); the timezone is unproven.
- January only. Validation is Mon–Wed, while test includes a weekend.
- Exact-copy exclusion is precautionary and catches only exact copies; near-copies are not detected.
- Two missing-data paths are verified only synthetically: the training-mean fallback, and dropping windows whose targets are all missing.
- No version control. The checkpoint is a working-directory state only, protected by nothing but this document and the hashes above.

## Publication (2026-10-24)
The checkpoint is published to the team repository `aiori-hackathon/a3-ps003-tc188`, on top of the existing `main` history.

**Published:**
- source (`aiori/`, `rebuild_aiori_metadata.py`), tests (`tests/`, `pytest.ini`) and `scripts/verify_artifacts.py`;
- `requirements.txt`, `.gitignore`, `docs/` (local machine paths removed);
- `data/raw_integrity_baseline.json`: relative paths and hashes. It is a local baseline, not the original-download hash.

**Kept local (excluded by `.gitignore`):**
- the raw XTraffic files (`year_2023/`, `node_order.npy`, `sensor_meta_feature.csv`, `incidents_y2023.csv`);
- everything in `data/processed/`:
  - derived `.npz` arrays and the audit table;
  - aligned sensor metadata and the client assignment, which include coordinates;
  - both selection manifests (`sensor_selection.csv`, `missing_validation_selection.csv`);
  - generated reports and `verification_results.json`, which contain local paths;
  - the v1 archive;
- `.venv/` and caches.

The derived files are held back because redistribution of dataset-derived content (CC BY-NC 4.0) has not been settled by the team.

**To regenerate them**, place the raw files as above and run:
```bash
.venv/Scripts/python.exe rebuild_aiori_metadata.py
.venv/Scripts/python.exe -m aiori.flow_audit
.venv/Scripts/python.exe -m aiori.prepare_pilot
.venv/Scripts/python.exe -m aiori.prepare_pilot --selection data/processed/missing_validation_selection.csv --out-dir data/processed/missing_validation --min-sensors 4 --run-label "missing-data validation set (not the primary pilot)"
.venv/Scripts/python.exe scripts/verify_artifacts.py
```

## Next step (not executed)
Phase 6: compact GRU and masked loss, per `docs/implementation_plan.md`. It must not be started under this checkpoint without a new explicit instruction.
