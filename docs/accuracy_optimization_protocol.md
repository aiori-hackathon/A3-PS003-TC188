# Accuracy optimisation: pre-declared protocol

Written on 2026-10-29 and committed at 07:11 UTC (commit 9cd5881), after stage-1 screening and before any stage-2 or federated result existed. (An earlier draft of this line said "about 07:50"; the commit time is authoritative.)

## Data use
- **Training:** Jan 1 – Jul 10, 2023, the frozen Phase 9 training split. It alone fits every parameter, scaler, profile and lag statistic.
- **Validation:** Jul 11 – Aug 7, 2023, the exact Phase 9 validation windows (1,606,706 windows; mask = finite truth). It is used for early stopping, for selection, and for ensemble and residual fitting.
  - Ensemble and residual fitting use 2-fold blocked cross-validation: fit on two weeks, score the other two, report out-of-fold.
  - Longer-context and lag features may read *inputs* from earlier periods; targets are unchanged.
- **Phase 9 held-out test (Aug 8 – Sep 30):** not used for any decision. Its frozen result, 8.672% WAPE for the deployed model, stays the reference figure.
- **October – December 2023:** never opened before this protocol. It is reserved for one independent evaluation, run once, after selection is final.

## Metrics
These are the frozen study definitions, applied to raw units and observed cells only, with negative forecasts clipped at 0:
- WAPE = 100 · Σ|a − p| / Σ|a|
- sMAPE as in `app/analysis/percentage_error.py`
- MAE and RMSE

## Baseline
The reproduced local-only GRU, Phase 9 local_seed1, clipped at 0, has validation WAPE 8.223% and MAE 16.451. The GPU pipeline reproduces its frozen validation predictions to within 0.007 vehicles.

## Selection rule
1. Screen single-seed runs. Take the three configurations with the lowest validation WAPE.
2. Re-run those with seeds 0, 1 and 2. The **selected candidate** is the configuration with the lowest mean validation WAPE over the three seeds.
   - It is served as the seed with the lowest validation WAPE, which mirrors the Phase 9 rule.
3. A federated or personalised method is preferred over local-only training only if its mean validation WAPE is lower.
4. An ensemble or residual correction is adopted only if it lowers out-of-fold validation WAPE by at least 0.05 points, and does not raise any client's WAPE by more than 0.05 points.

## One-time independent evaluation (October – December 2023)
- **Windows:** all windows whose 12 targets lie in Oct 1 – Dec 31, 2023 and include at least one observed target.
  - Inputs may use earlier observations, causally filled.
  - The Oct – Dec segment is filled exactly as the study fills each split, using training-period statistics only.
- **Models scored, once, with no change after opening:**
  - (a) the selected candidate;
  - (b) the frozen Phase 9 deployed model, local_seed1 clipped;
  - (c) persistence.
- **Reported:** WAPE, MAE, RMSE and sMAPE, plus breakdowns by client, horizon and month.
- **Interpretation:**
  - A below-6% claim needs (a) below 6% WAPE on this period.
  - Otherwise the result is reported as measured.
  - Oct – Dec includes holiday periods and lies 3–5 months after the training data, so it is a harder, shifted period than validation.
- **No tuning after opening.** No model, feature, threshold or selection is changed after the October – December targets are opened.
