import type { ModelKey, Prediction } from "./types";

/**
 * Which response series to draw as the primary (green) and comparison lines. Optimized-MLP replays compare against the
 * original GRU; everything else shows the original GRU with the centralised GRU. Labels always name the model that
 * produced the values, so no metric of one model is shown beside another model's forecast.
 */
export function displayModels(p: Prediction): { pred: Prediction; primary: string; compare: string } {
  if (p.forecasts.candidate)
    return { pred: { ...p, forecasts: { final: p.forecasts.candidate, centralised: p.forecasts.final } }, primary: "Optimized MLP", compare: "Original GRU" };
  return { pred: p, primary: "Original GRU (local-only)", compare: "Centralised GRU" };
}

export interface ForecastRow {
  t: string;
  observed: number | null;
  final: number | null;
  centralised: number | null;
  actual: number | null;
}

/**
 * 24 rows: the 12 observed readings, then the 12 forecast steps. Each forecast line starts at the last observed reading
 * so it visibly continues from the data it was computed from; "actual" (historical truth) only fills forecast rows.
 */
export function forecastRows(p: Prediction, opts: { models: ModelKey[]; showActual: boolean }): ForecastRow[] {
  const h = p.history.values;
  const lastObserved = h[h.length - 1];
  const rows: ForecastRow[] = p.history.timestamps.map((t, i) => ({ t, observed: h[i], final: null, centralised: null, actual: null }));
  const bridge = rows[rows.length - 1];
  for (const m of opts.models) if ((m === "final" || m === "centralised") && p.forecasts[m] && lastObserved !== null) bridge[m] = lastObserved;
  p.forecast_timestamps.forEach((t, i) => {
    rows.push({
      t,
      observed: null,
      final: opts.models.includes("final") ? (p.forecasts.final?.values[i] ?? null) : null,
      centralised: opts.models.includes("centralised") ? (p.forecasts.centralised?.values[i] ?? null) : null,
      actual: opts.showActual && p.actual ? p.actual.values[i] : null,
    });
  });
  return rows;
}
