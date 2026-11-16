/**
 * Verified results of the 2022–2024 federated study (aiori2, Stages 1–3).
 * Static on purpose: the test split was evaluated once and is now closed, so these numbers never change.
 * Sources (all under src/results/aiori2/): phase8c_stage3/analysis/{PHASE8C_REPORT.md, final_test_table.csv,
 * final_test_summary.json, stage3_cells.csv}, phase7_stage1/{stage2_results.csv, <run>/run.json}.
 * Values are WAPE in percent of raw flow; lower is better.
 */

export const MANIFEST_SHA = "b3efd3be441a04420efbe193d7aa89a4894b1f3948341fd8285ffdd376511c2c";
export const TEST_TARGETS = 124_985_327;
export const VAL_TARGETS = 125_101_362;

export type Kind = "final" | "arm" | "ref";

export interface TestGroup {
  key: string;
  label: string;
  short: string;
  kind: Kind;
  note: string;
  wape: number;
  sd?: number;
  mae: number;
  rmse: number;
  val?: number;
  clients: [number, number, number, number];
  h5: number;
  h60: number;
}

/** Final test, Jul–Dec 2024, same 124,985,327 targets for every row. */
export const TEST: TestGroup[] = [
  { key: "win_iid", label: "MLP + FedAvgM, IID clients", short: "MLP FedAvgM · IID", kind: "final", note: "selected · 3 seeds, 20 epochs", wape: 10.506, sd: 0.031, mae: 20.24, rmse: 34.11, val: 10.841, clients: [8.667, 10.519, 11.396, 12.696], h5: 7.008, h60: 13.94 },
  { key: "win_geo", label: "MLP + FedAvgM, geographic clients", short: "MLP FedAvgM · geo", kind: "final", note: "selected · 3 seeds, 20 epochs", wape: 10.546, sd: 0.026, mae: 20.32, rmse: 34.18, val: 10.891, clients: [8.736, 10.605, 11.419, 12.619], h5: 7.02, h60: 14.016 },
  { key: "base_iid", label: "N-HiTS + FedAvg, IID clients", short: "N-HiTS FedAvg · IID", kind: "arm", note: "FedAvg baseline · 3 seeds", wape: 10.728, sd: 0.004, mae: 20.67, rmse: 34.88, val: 11.057, clients: [8.95, 10.715, 11.657, 12.806], h5: 7.113, h60: 14.301 },
  { key: "base_geo", label: "N-HiTS + FedAvg, geographic clients", short: "N-HiTS FedAvg · geo", kind: "arm", note: "FedAvg baseline · 3 seeds", wape: 10.742, sd: 0.005, mae: 20.69, rmse: 34.89, val: 11.074, clients: [8.981, 10.737, 11.696, 12.738], h5: 7.12, h60: 14.319 },
  { key: "local_gru", label: "Local-only GRU (reference)", short: "Local GRU · ref", kind: "ref", note: "not federated · 1 seed, 10 epochs", wape: 10.299, mae: 19.84, rmse: 33.57, val: 10.601, clients: [8.27, 10.39, 11.277, 12.573], h5: 6.961, h60: 13.526 },
  { key: "local_mlp", label: "Local-only MLP (reference)", short: "Local MLP · ref", kind: "ref", note: "not federated · 1 seed, 10 epochs", wape: 10.368, mae: 19.97, rmse: 33.79, val: 10.681, clients: [8.359, 10.495, 11.374, 12.498], h5: 7.029, h60: 13.644 },
  { key: "central_mlp", label: "Centralised MLP (reference)", short: "Centralised MLP · ref", kind: "ref", note: "not federated · 1 seed, 10 epochs", wape: 10.424, mae: 20.08, rmse: 33.85, val: 10.753, clients: [8.572, 10.476, 11.326, 12.544], h5: 7.003, h60: 13.803 },
  { key: "persistence", label: "Persistence (repeat last reading)", short: "Persistence", kind: "ref", note: "naive forecast", wape: 12.495, mae: 24.07, rmse: 40.24, clients: [10.519, 12.59, 13.654, 14.422], h5: 7.859, h60: 17.076 },
];

export const CLIENT_LABELS = ["LA / Ventura", "Bay Area", "Coastal / South Central", "Inland / Central"] as const;

/** Winner minus FedAvg baseline, matched by seed (test). */
export const WIN_VS_BASE = { iid: -0.222, geo: -0.195 };

/** Stage 1: 5 architectures × 2 histories × 4 training approaches = 40 runs; best full-validation WAPE per architecture (10 epochs, 1 seed). */
export const STAGE1: { model: string; params: number; centralised: number; local: number; iid: number; geo: number }[] = [
  { model: "MLP", params: 103_052, centralised: 10.753, local: 10.681, iid: 11.121, geo: 11.128 },
  { model: "N-HiTS", params: 60_345, centralised: 10.728, local: 10.631, iid: 11.13, geo: 11.132 },
  { model: "GRU", params: 85_548, centralised: 10.632, local: 10.601, iid: 11.608, geo: 11.617 },
  { model: "PatchTST", params: 72_396, centralised: 10.972, local: 11.356, iid: 11.399, geo: 11.394 },
  { model: "TCN", params: 57_036, centralised: 11.141, local: 11.038, iid: 11.399, geo: 11.413 },
];

/** Stage 2: 8 algorithms × MLP/N-HiTS × histories A/B × 2 partitions = 64 runs; best full-validation WAPE per algorithm. */
export const STAGE2: { algo: string; iid: number; geo: number }[] = [
  { algo: "FedAvgM", iid: 10.933, geo: 10.972 },
  { algo: "FedAdam", iid: 10.96, geo: 10.983 },
  { algo: "FedYogi", iid: 10.98, geo: 11.006 },
  { algo: "FedAdagrad", iid: 11.046, geo: 11.064 },
  { algo: "SCAFFOLD", iid: 11.12, geo: 11.131 },
  { algo: "FedAvg", iid: 11.121, geo: 11.129 },
  { algo: "FedProx", iid: 11.121, geo: 11.129 },
  { algo: "FedNova", iid: 11.121, geo: 11.128 },
];

/** Stage 3: 4 finalists × 2 partitions × 3 seeds = 24 runs, 20 epochs; mean full-validation WAPE ± SD. */
export const STAGE3: { config: string; score: number; iid: [number, number]; geo: [number, number] }[] = [
  { config: "MLP · FedAvgM · history A", score: 10.866, iid: [10.84, 0.024], geo: [10.891, 0.019] },
  { config: "N-HiTS · FedAvgM · history B", score: 10.905, iid: [10.878, 0.005], geo: [10.933, 0.027] },
  { config: "N-HiTS · FedAdam · history B", score: 10.915, iid: [10.895, 0.004], geo: [10.935, 0.004] },
  { config: "N-HiTS · FedAvg · history B", score: 11.065, iid: [11.057, 0.002], geo: [11.074, 0.004] },
];

/** Communication in Stage 3 (20-epoch budget, 40 rounds per epoch, four clients). Upload only; download is equal. */
export const COMM = {
  roundsPerRun: 800,
  totalRounds: 18_920,
  mlpGbPerRun: 1.32,
  nhitsGbPerRun: 0.76,
  mlpMbPerRound: 1.65,
  gpuHours: 10.15,
};
