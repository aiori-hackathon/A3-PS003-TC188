# AIORI 2.0 — Phase 8C (Stage 3) report

Written 2026-11-15 after the batch finished and the one-time test evaluation ran. All numbers come from files in
`src/results/aiori2/phase8c_stage3/`. WAPE values are percentages of raw flow; lower is better.

## 1. Status: **PASS**

- All 24 of 24 planned runs finished, and every run was verified on disk.
- The model selection was frozen from validation results only. Its manifest SHA-256 is `b3efd3be441a…`.
- The July–December 2024 test split was evaluated exactly once, from that frozen manifest.
- The protected AIORI files are unchanged.
- **Caveat:** training was interrupted by unclean shutdowns. These were handled through the audited recovery process (§6), and the resumed runs are equivalent to uninterrupted runs.

## 2. Stage 1 and Stage 2 preservation
- `results/aiori2/phase7_stage1/` holds 524 files. The newest was modified at 2026-11-14 08:42, which is before Stage 3 started at 09:06. Nothing was written there during Stage 3.
- Stage 3 wrote only to its own folders: `phase8c_stage3/` and `phase8c_stage3_watchdog/`.

## 3. Finalist matrix and rationale

The finalist matrix was registered before training began. It is recorded in `stage3_protocol.json` (sha `47436d98…`) and was not changed after results were seen.

- **Configurations:** the four Stage 2 finalists, each run on both the IID and geographic partitions, giving 8 cells:
  - N-HiTS · FedAvgM · history B
  - MLP · FedAvgM · history A
  - N-HiTS · FedAdam · history B
  - N-HiTS · FedAvg · history B, which is the baseline
- **Settings:** every finalist kept its Stage 2 per-algorithm settings. FedAvgM used server momentum 0.9; FedAdam used server learning rate 0.01.

## 4. Seeds and runs

- **Seeds:** 3 seeds (20261027, 20261114, 20261115) for each of the 8 cells, giving 24 runs.
- **Seed 20261027 was retrained from scratch** under the 20-epoch budget, so all three seeds had the same training exposure.
- **Stage 2 kept separate:** the 10-epoch Stage 2 runs are reported on their own and never pooled with these runs.

## 5. Training budget and communication

- **Budget:** at most 20 epochs. Each epoch has 40 rounds and 2,048,000 training windows: 4 clients × 100 local steps × batch 128.
- **Early stopping:** patience 3 and min_delta 1e-4, measured on the fixed 1M-window validation subset. The best checkpoint was then re-scored on the full validation set of 125,101,362 targets.
- **Totals:** 18,920 rounds, which is 23 runs × 800 plus 13 × 40 for run 6, which stopped early. There were 7,568,000 local steps, and none were skipped.
- **Upload per run:**
  - about 0.76 GB for N-HiTS (60,345 parameters);
  - about 1.32 GB for MLP (103,052 parameters).

## 6. Runtime, resources and failures

- **Timeline:** wall clock ran from 2026-11-14 09:10 to 2026-11-15 00:40. Pure training time totalled 10.15 GPU-hours.
- **Time per run:** about 26.8 min for N-HiTS and about 21.2 min for MLP.
- **Hardware:** one RTX PRO 4000, local only. Peak GPU memory was about 77 MB per run.
- **No training failures:** no NaN values, no skipped steps and no failed runs.
- **Infrastructure incidents:**
  - Unclean shutdowns (Kernel-Power 41) at about 10:01 and 13:12–13:20.
  - Three more restarts at 16:28–16:33, which the owner confirmed were caused by their own BIOS power changes. The owner also confirmed the PSU, UPS, socket and cable were physically checked.
  - The watchdog block was cleared through the audited process at 16:49:03, recorded in `phase8c_stage3_watchdog/incidents.jsonl`.
- **Resumes:**
  - Run 2 (N-HiTS FedAvgM IID s20261114) resumed from epoch 16 at 11:34.
  - Run 6 (N-HiTS FedAvgM geo s20261115) resumed from epoch 8 at 16:49.
  - Federated learning randomness is seeded per round, so a resume is equivalent to an uninterrupted run. This was checked earlier: epochs 1–10 matched Stage 2 bit for bit.
- **After the 16:49 resume:**
  - Windows logged no Kernel-Power 41, 6008, WHEA or NVIDIA-driver events.
  - The only error-level event was the MWAgent service, which crashed at 17:36 and was restarted automatically by Windows. MWAgent appears to belong to eScan; that is inferred from the name, not confirmed. It is not a hardware event.
- `auto_resume` stayed `false` throughout.

## 7. Matched-seed validation comparison (full validation, 3 seeds per cell)

| Rank | Configuration | Score (mean of IID and geo) | IID mean ± SD | Geo mean ± SD | SD over 6 runs |
|---|---|---|---|---|---|
| 1 | **MLP FedAvgM A** | **10.866** | 10.840 ± 0.024 | 10.891 ± 0.019 | 0.034 |
| 2 | N-HiTS FedAvgM B | 10.905 | 10.878 ± 0.005 | 10.933 ± 0.027 | 0.035 |
| 3 | N-HiTS FedAdam B | 10.915 | 10.895 ± 0.004 | 10.935 ± 0.004 | 0.022 |
| 4 | N-HiTS FedAvg B (baseline) | 11.065 | 11.057 ± 0.002 | 11.074 ± 0.004 | 0.010 |

Exact per-cell SDs, ranges, MAE, RMSE, per-client values and worst-client values are in `stage3_cells.csv`. Per-run values are in `stage3_runs.csv`.

**Algorithm.** These compare against FedAvg on N-HiTS B, matched by seed:
- FedAvgM improves WAPE by **−0.179** on IID and **−0.141** on geographic data.
- FedAdam improves it by **−0.162** on IID and **−0.139** on geographic data.
- The improvement holds for every seed. FedAvgM and FedAdam are close to each other.

**Model.** The Stage 3 MLP-vs-N-HiTS comparison is confounded, because MLP used history A and N-HiTS used history B. Stage 2 has history-matched FedAvgM pairs (10 epochs, one seed). In those, the MLP-minus-N-HiTS gap was:
- history A: +0.003 (IID) and +0.002 (geo);
- history B: +0.014 (IID) and +0.000 (geo).

The two models were tied at 10 epochs. The MLP's lead at 20 epochs comes from it gaining more from the longer training (−0.10 vs about −0.06 for N-HiTS B FedAvgM). A 20-epoch N-HiTS run on history A was not tested, so the model and history effects cannot be fully separated.

**IID vs geographic.** The geographic split costs +0.04 to +0.06 WAPE for the federated finalists. For the FedAvg baseline the cost is +0.017.

**Per client (geographic).** Client 0, LA/Ventura, is the easiest at about 9.4. Client 3, Inland Central, is the hardest at about 12.8. The order of difficulty is the same for every cell.

## 8. Convergence (`plot_convergence_IID.svg`, `plot_convergence_geo.svg`)
- 23 of 24 runs reached their best epoch at 19 or 20. They were still improving slowly at the 20-epoch cap, so a larger budget would probably help a little.
- **Run 6 is the exception.** N-HiTS FedAvgM geo s20261115 early-stopped at epoch 13, with its best epoch at 10.
  - Its subset WAPE was 10.966 at epoch 10, then 10.972, 10.969 and 10.988. That is three epochs without improvement, so the patience-3 rule stopped it legitimately.
  - Its training loss was smooth across the resume at epoch 8.
  - This run raises the geographic SD for N-HiTS FedAvgM to 0.027. It does not change the ranking, because the runner-up still trails the winner without it.
- **Gain from doubling the budget** (Stage 2 10-epoch vs Stage 3 20-epoch, seed 20261027), measured in full-validation WAPE:

| Configuration | Partition | 10 epochs | 20 epochs |
|---|---|---|---|
| MLP A | IID | 10.939 | 10.813 |
| MLP A | geo | 10.977 | 10.885 |
| N-HiTS FedAvg B | IID | 11.130 | 11.054 |

## 9. Frozen selection manifest
- **Protocol rule:** the winner is the configuration with the lowest score, MLP FedAvgM A at 10.866. The runner-up is tested only if it is within one pooled seed SD of the winner.
- **Runner-up excluded:**
  - The runner-up's gap is 0.039.
  - The pooled SD across all 8 cells is 0.015.
  - The two alternative SD readings are 0.021 (winner and runner-up cells) and 0.022 (winner cells only).
  - The gap is larger under all three readings, so the decision does not depend on how "pooled SD" is read.
- **Frozen set (15 checkpoints plus persistence):**
  - the winner: 6 runs;
  - the baseline N-HiTS FedAvg B: 6 runs;
  - Stage 1 references: local GRU A (best Stage 1 run), centralised MLP A and local MLP A. These are 10-epoch, single-seed runs, flagged as unmatched exposure.
  - persistence for histories A and B.
- **Files and hashes:**

| File | SHA-256 |
|---|---|
| `final_manifest.json` | `b3efd3be441a04420efbe193d7aa89a4894b1f3948341fd8285ffdd376511c2c` (also in `final_manifest.sha256`) |
| `selection.json` | `de41708c…` |
| `validation_analysis.py` | `b69a4f6b…` |
| `stage3_validation_summary.json` | `9fed30bd…` |

  The manifest also records the hash of every `best.pt` and `run.json`, the scalers, the test windows, the valid-target mask, the data manifest and the evaluation code.

## 10. Final test results (Jul–Dec 2024; one evaluation; 124,985,327 valid targets; same target set for all)

| Group | Test WAPE mean ± SD [range] | MAE | RMSE | Worst client | Validation mean |
|---|---|---|---|---|---|
| **MLP FedAvgM A, IID** | **10.506 ± 0.031** [10.471, 10.528] | 20.24 | 34.11 | 12.70 | 10.841 |
| **MLP FedAvgM A, geo** | **10.546 ± 0.026** [10.529, 10.576] | 20.32 | 34.18 | 12.62 | 10.891 |
| N-HiTS FedAvg B, IID | 10.728 ± 0.004 [10.724, 10.732] | 20.67 | 34.88 | 12.81 | 11.057 |
| N-HiTS FedAvg B, geo | 10.742 ± 0.005 [10.737, 10.746] | 20.69 | 34.89 | 12.74 | 11.074 |

- **Winner vs baseline, matched by seed:** −0.222 on IID (range −0.253 to −0.204) and −0.195 on geographic (range −0.217 to −0.161). The winner is better for all 6 pairs. These gaps match the validation gaps.
- **Test vs validation:** test WAPE is about 0.33 lower than validation for every group. That is a shift between the two periods, not a change in ranking.

## 11. Persistence and per-client comparison (test)

- **Persistence:** test WAPE is **12.495** for both histories. It repeats the last observation, so the length of history doesn't matter.
- **Winner vs persistence:** the winner improves on persistence by about 1.97 points, or about 16% relative.

Per-geographic-client test WAPE:

| Model | c0 LA/Ventura | c1 Bay Area | c2 Coastal South Central | c3 Inland Central |
|---|---|---|---|---|
| Winner, IID, mean | ≈ 8.67 | ≈ 10.52 | ≈ 11.40 | ≈ 12.70 |
| Baseline, IID, mean | ≈ 8.95 | ≈ 10.71 | ≈ 11.66 | ≈ 12.81 |
| Persistence | 10.52 | 12.59 | 13.65 | 14.42 |

**By horizon (winner):**
- 5-minute horizon: about 7.0 WAPE.
- 60-minute horizon: about 13.9 WAPE.
- Persistence at 60 minutes: 17.1.

## 12. Chosen federated model and non-federated references
- **Chosen federated model:** MLP · FedAvgM (server momentum 0.9) · history A · 20-epoch budget. The best single checkpoint is `fedavg_iid-A-mlp-medium-fedavgm-s20261027-1387d83baa/best.pt`, with validation 10.813 and test 10.471. **Report the 3-seed mean, not the best seed.**
- **Non-federated references on test** (10 epochs, single seed):

| Reference | Test WAPE |
|---|---|
| local GRU A | **10.299** |
| local MLP A | 10.368 |
| centralised MLP A | 10.424 |

  All three beat the federated winner, by about 0.08–0.25, despite having half the training budget. Federated learning has not yet matched per-client local training on this data.

## 13. Limitations and remaining scientific risks
- **Model vs history is confounded** in Stage 3, because MLP was run on history A and N-HiTS on history B. No 20-epoch N-HiTS A run exists.
- **The non-federated references aren't matched to the federated runs.** They used 10 epochs and one seed, so the federated-vs-local gap may change at matched exposure.
- **Three seeds is a small sample.** The SDs are rough, especially for the low-variance FedAdam and FedAvg cells.
- **The 20-epoch cap is binding.** Most runs were still improving at epoch 20.
- **Two runs were resumed after power loss.** The resume is equivalent by design and was verified on Stage 2, but the two resumed runs themselves were not re-run without interruption.
- **The test split is now spent.** Any further selection must use validation only, and these test numbers can't be used for tuning.
- **Earlier data caveats still apply,** such as provisional sensor-to-station mapping and client labels.

## 14. Paths (all under `src/results/aiori2/phase8c_stage3/`)
- **Runs and checkpoints:** `<run_id>/run.json`, `best.pt` and `latest.pt` for 24 run folders, plus `registry.jsonl` and `stage3_console.log`.
- **Protocol:** `stage3_protocol.json`.
- **`analysis/` folder:**
  - CSVs and summaries:
    - `stage3_runs.csv`, `stage3_cells.csv`, `stage3_configs.csv`;
    - `stage3_validation_summary.json`, which includes the training curves;
    - `final_test_table.csv`, `final_test_summary.json`, `final_test_results.json`, `final_test_console.log`.
  - Selection: `selection.json`, `final_manifest.json` and `final_manifest.sha256`.
  - Figures: `plot_cells_seeds.svg`, `plot_convergence_IID.svg`, `plot_convergence_geo.svg`, `plot_per_client.svg`, `plot_test_wape.svg`.
  - Scripts: `validation_analysis.py`, `test_analysis.py`, `svgplot.py`. matplotlib is not installed, so the plots are plain SVG.
- **Watchdog:** `src/aiori2_watchdog/`. Its state is in `src/results/aiori2/phase8c_stage3_watchdog/`.
- **Backups:** `D:/AIORI_backups/`, holding the pre-Phase 8C backup, the pre-watchdog backup and the Stage 3 snapshot.

## 15. Original AIORI production and localhost files unchanged
- **Protected data/results baseline:** **176 of 176** files match (`protected_baseline.json`).
- **Production app baseline:** **41 of 41** files match (`protected_production_phase6.json`). This covers the backend, frontend, tunnel and deploy files.
- **Git:** HEAD is still `dad975f` and there are no tracked modifications. Nothing was committed or pushed. The new files are untracked.
- **Not touched:** the dashboard, Cloudflare tunnel, production watchdog and deployment settings. No DGX or cloud resources were used.

## 16. Recommended Phase 9 integration plan (not started)
1. Add the MLP FedAvgM A model as a **versioned, experimental candidate**. Keep the current production model as default, with rollback available.
2. Show the 3-seed test mean ± SD, and separate validation from test figures. Never relabel earlier production metrics.
3. Before any switch, confirm:
   - the input contract (history A window, scaler scope, valid-target mask);
   - output units and timestamp alignment.
   Then run inference smoke tests and measure latency.
4. Optionally, use validation only, never the test split, to address the open questions:
   - a matched-exposure local-vs-federated comparison;
   - a 20-epoch N-HiTS A run to separate model from history;
   - a larger epoch budget.
5. Keep `auto_resume` off. Remove the watchdog scheduled task (`remove_task.ps1`) once it is no longer needed. Removing it needs the owner's approval.

**Stopped here.** No Phase 9 work, dashboard change, deployment, commit or push has been done.
