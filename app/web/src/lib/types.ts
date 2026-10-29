/** Response shapes of the Python inference API (app/backend/server.py). Every number is read from frozen Phase 9 artifacts. */

export type ArmKey = "local" | "centralised" | "fedavg_iid" | "fedavg_geo";
export type BaselineKey = "persistence" | "seasonal_naive_daily" | "historical_average";
export type ClientKey =
  | "client_0_la_ventura"
  | "client_1_bay_area"
  | "client_2_coastal_south_central"
  | "client_3_inland_central";
export type ModelKey = "final" | "centralised" | "candidate";

export interface Health {
  status: string;
  device: string;
  cuda_available: boolean;
  torch: string;
  models_loaded: ModelKey[];
  sensors: number;
  uptime_s: number;
  percentage_error_available: boolean;
  candidate_enabled?: boolean;
}

export interface Sensor {
  id: string;
  client: ClientKey;
  client_name: string;
  type_group: string;
  median_test_flow: number;
}

export interface DemoExample {
  sensor: string;
  client_name: string;
  time: string;
  label: string;
}

export interface Demo {
  note: string;
  examples: DemoExample[];
}

export interface DeployedModel {
  key: ModelKey;
  label: string;
  arm: string;
  seed: number;
  architecture: string;
  selection: string;
  routing: string;
  negative_clipping: boolean;
  validation_mae: number;
  validation_rmse: number;
  test_mae_this_seed: number;
  test_rmse_this_seed: number;
  checkpoint_sha256: Record<string, string>;
  units: string;
}

export interface ErrorPair {
  mae: number;
  rmse: number;
}

export interface Arm {
  label: string;
  seeds: number[];
  mae: number;
  rmse: number;
  normalised_mae: number;
  sensor_macro_mae: number;
  negative_rate: number;
  clipped_mae: number;
  clipped_rmse: number;
  per_seed: Record<string, ErrorPair>;
  client: Record<ClientKey, ErrorPair>;
  horizon: number[];
  month: Record<string, number>;
  weekday_weekend: { weekday: number; weekend: number };
  validation_mae: number;
  validation_rmse: number;
  validation_to_test_change: number;
  improvement_over_baselines_pct: Record<BaselineKey, number>;
}

export interface Baseline {
  mae: number;
  rmse: number;
  horizon: number[];
  client: Record<ClientKey, number>;
}

export interface Contrast {
  a: ArmKey;
  b: ArmKey;
  matched_seeds: number[];
  diff: number;
  reported_ci95: [number, number];
  p_conservative: number;
  p_holm: number;
  abs_diff_ge_sesoi: boolean;
  verdict: string;
  statistical: boolean;
  practical: boolean;
  single_seed: boolean;
}

export interface RunCost {
  minutes: number;
  communication_mb: number;
  communication_to_best_mb: number;
  best_unit: number | null;
  units_run: number | null;
}

export interface PctBlock {
  wape_pct: number;
  smape_pct: number;
  mae: number;
  per_seed_wape_pct?: number[];
  by_horizon_wape_pct: number[];
  by_horizon_smape_pct?: number[];
  by_client_wape_pct: Record<ClientKey, number>;
}

export interface PercentageError {
  definitions: string;
  source: string;
  arms: Record<ArmKey, { label: string; raw: PctBlock; clipped?: PctBlock }>;
  baselines: Record<BaselineKey, { wape_pct: number; smape_pct: number; mae: number }>;
  cells: { observed: number; actual_zero: number; actual_below_5: number; mean_actual_flow: number };
  target: { wape_target_pct: number; achieved: boolean; final_model_wape_pct: number; gap_pct_points: number };
}

export interface Metrics {
  source: string;
  test_windows: number;
  n_sensors: number;
  units: string;
  arms: Record<ArmKey, Arm>;
  baselines: Record<BaselineKey, Baseline>;
  contrasts: Record<"C1" | "C2" | "C3" | "C4", Contrast>;
  sesoi_mae: number;
  runtime: Record<string, RunCost>;
  final_model: { arm: string; variant: string };
  percentage_error: PercentageError | null;
  optimized: Optimized | null;
  deployed: Record<ModelKey, DeployedModel>;
  split: { train: string; validation: string; test: string };
}

export interface OptBlock {
  wape_pct: number;
  mae: number;
  rmse: number;
  smape_pct: number;
  by_client_wape_pct: Record<ClientKey, number>;
  by_horizon_wape_pct: number[];
}

/** Verified accuracy-optimisation results (validation and the one-time independent Oct-Dec 2023 evaluation). */
export interface Optimized {
  label: string;
  architecture: string;
  training: string;
  federated: boolean;
  enabled_for_forecasts: boolean;
  validation: OptBlock;
  validation_period: string;
  independent: {
    period: string;
    windows: number;
    optimized: OptBlock;
    original_gru: OptBlock;
    optimized_by_month_wape_pct: Record<string, number>;
    original_by_month_wape_pct: Record<string, number>;
    relative_reduction_pct: number;
    persistence?: OptBlock;
    relative_reduction_vs_persistence_pct?: number;
  };
  residual_correction: { validation_oof_wape_pct: number; independent_wape_pct: number; deployed: boolean };
  target: { wape_target_pct: number; achieved: boolean };
}

export interface PredictRequest {
  sensor: string;
  time: string;
  models?: ModelKey[];
  observations?: (number | null)[];
}

export interface Prediction {
  sensor: { id: string; client_name: string; type_group: string };
  mode: "historical" | "custom";
  data_label: string;
  period: string | null;
  units: string;
  forecast_start: string;
  history: { timestamps: string[]; values: (number | null)[] };
  forecast_timestamps: string[];
  forecasts: Partial<Record<ModelKey, { label: string; values: number[]; negative_clipping: boolean }>>;
  actual: { label: string; values: (number | null)[] } | null;
  beyond_study_period: boolean;
  validated: boolean;
  accuracy_note: string;
  inference_ms: number;
  device: string;
}
