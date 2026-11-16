/** Historical Disruption Explorer (Phase 9B): API client, types and pure helpers. Talks to app/backend/replay_server.py. */
import { type ModelInfo, type ObsStatus, type Partition, ReplayApiError } from "./replay";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

async function get<T>(path: string, params: Record<string, string | number> = {}, signal?: AbortSignal): Promise<T> {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/replay/disruptions${path}${q ? `?${q}` : ""}`, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ReplayApiError("The replay service is not reachable. Start it with .venv\\Scripts\\python.exe src\\app\\backend\\replay_server.py, then retry.", 0);
  }
  let data: { error?: { message?: string } } | null = null;
  try {
    data = await res.json();
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
  }
  if (!res.ok) throw new ReplayApiError(data?.error?.message ?? `The replay service returned HTTP ${res.status}.`, res.status);
  if (data === null || typeof data !== "object") throw new ReplayApiError("The replay service returned an unreadable response.", res.status);
  return data as T;
}

export type Category = "sudden_decrease" | "sudden_increase" | "sustained_deviation" | "coordinated_deviation" | "sensor_quality";
export type Confidence = "high" | "medium" | "low";
export type Region = "LA" | "BA" | "CS" | "IC";

export interface EventBrief {
  id: string;
  start: string;
  end: string;
  duration_min: number;
  direction: "decrease" | "increase";
  category: Category;
  region_code: Region;
  n_sensors: number;
  sensors: string[];
  peak_z: number;
  peak_relative: number;
  peak_sensor: string;
  peak_time: string;
  score: number;
  quality_share: number;
  missing_share: number;
  quality_flags: string[];
  confidence: Confidence;
  day_type: string;
  holiday: string | null;
}

export interface SensorDetail {
  sensor: string;
  start: string;
  end: string;
  n_segments: number;
  n_anomalous: number;
  n_missing: number;
  n_quality: number;
  quality_flags: string[];
  peak_z: number;
  peak_relative: number;
  peak_time: string;
  peak_observed: number;
  peak_expected: number;
  score: number;
}

export interface EventDetail extends EventBrief {
  client_geo: string;
  n_anomalous: number;
  sensor_details: SensorDetail[];
  region: string;
  category_text: string;
  quality_text: string[];
  why_flagged: string;
  period: string;
  period_label: string;
  replay: { date: string; time: string; sensor: string };
  forecast: { available: boolean; issue_at_onset: string | null; issue_at_peak: string | null; first_forecast_time: string; reason: string | null };
  detection_note: string;
  units: string;
  timezone: string;
}

export interface EventList {
  total: number;
  page: number;
  page_size: number;
  pages: number;
  sort: string;
  items: EventBrief[];
}

export interface SummaryRow {
  group: string;
  period: string;
  events: number;
  event_pairs: number;
  event_mae: number | null;
  event_wape_pct: number | null;
  reference_pairs: number;
  reference_mae: number | null;
  reference_wape_pct: number | null;
}

export interface DisruptionMeta {
  title: string;
  version: string;
  created_utc: string;
  config: Record<string, unknown>;
  config_sha256: string;
  thresholds: { z_low: number; z_high: number; min_relative: number; calibration: string; n_calibration_readings: number; n_calibration_sensors: number };
  calendar: { weekday_level: number; weekend_level: number; rule_2023: Record<string, { level_2023: number; weekend_like: boolean }> };
  baseline: { method: string; label: string; sha256: string };
  provenance: { source: string; timezone: string; first: string; last: string; n_sensors: number; inputs: string[] };
  summary: {
    n_events: number;
    by_year: Record<string, number>;
    by_category: Record<string, number>;
    by_direction: Record<string, number>;
    by_region: Record<string, number>;
    by_confidence: Record<string, number>;
    duration_min: { p50: number; p90: number; max: number };
    sensors_with_events: number;
  };
  audit: { level_shift_sensor_months: number; chance_coordination: { observed: number; shifted_mean: number; method: string }; scored_readings: number };
  events_sha256: string;
  units: string;
  timezone: string;
  categories: Record<Category, string>;
  quality_flags: Record<string, string>;
  limitations: string[];
  regions: Record<Region, string>;
  first_forecast_time: string;
  robustness_summary: { events_evaluated: number; model: string; rows: SummaryRow[]; method: string; caution: string; partition: Partition } | null;
  score_note: string;
}

export interface EventSeries {
  event: string;
  sensor: string;
  units: string;
  window: { start: string; end: string; truncated: boolean };
  sensor_interval: { start: string; end: string };
  event_interval: { start: string; end: string };
  timestamps: string[];
  observed: (number | null)[];
  status: ObsStatus[];
  expected: (number | null)[];
  threshold_low: (number | null)[];
  threshold_high: (number | null)[];
  z: (number | null)[];
  anomalous: (-1 | 0 | 1)[];
  scored: boolean[];
  quality: string[][];
  dst_nonexistent: number[];
  expected_label: string;
  threshold_label: string;
  unscored_note: string;
}

export interface EventCompare {
  event: string;
  timestamps: string[];
  sensors: { sensor: string; ratio: (number | null)[]; available: number }[];
  shown: number;
  total_sensors: number;
  normalization: string;
}

export interface Metrics {
  n: number;
  mae: number | null;
  wape_pct: number | null;
  mae_by_horizon: (number | null)[];
  missing_targets: number;
}

export interface Robustness {
  event: string;
  sensor: string;
  partition: Partition;
  available: boolean;
  reason?: string;
  model?: ModelInfo;
  issues?: { first: string; last: string; count: number; stride_minutes: number; usable: number };
  anomaly_interval?: { start: string; end: string };
  event_window?: Metrics;
  reference?: Metrics & { days: string[] };
  mae_difference?: number | null;
  periods?: string[];
  method?: string;
  caution?: string;
  warnings?: string[];
}

export interface Filters {
  from: string;
  to: string;
  region: string;
  category: string;
  direction: string;
  sensor: string;
  min_sensors: string;
  min_duration: string;
  min_peak_z: string;
  confidence: string;
  quality: "include" | "exclude" | "only";
  holiday: "include" | "exclude" | "only";
  sort: string;
  page: number;
}

export const DEFAULT_FILTERS: Filters = {
  from: "",
  to: "",
  region: "",
  category: "",
  direction: "",
  sensor: "",
  min_sensors: "",
  min_duration: "",
  min_peak_z: "",
  confidence: "",
  quality: "exclude",
  holiday: "include",
  sort: "-score",
  page: 1,
};

export const PAGE_SIZE = 25;

/** Query parameters for /api/replay/disruptions: empty filters are left out, defaults kept explicit. */
export function listParams(f: Filters): Record<string, string | number> {
  const p: Record<string, string | number> = { sort: f.sort, page: f.page, page_size: PAGE_SIZE, quality: f.quality, holiday: f.holiday };
  for (const k of ["from", "to", "region", "category", "direction", "sensor", "min_sensors", "min_duration", "min_peak_z", "confidence"] as const) {
    const v = f[k].trim();
    if (v !== "") p[k] = v;
  }
  return p;
}

export const disruptionsApi = {
  meta: (s?: AbortSignal) => get<DisruptionMeta>("/meta", {}, s),
  list: (f: Filters, s?: AbortSignal) => get<EventList>("", listParams(f), s),
  event: (id: string, s?: AbortSignal) => get<EventDetail>(`/${id}`, {}, s),
  series: (id: string, sensor: string, s?: AbortSignal) => get<EventSeries>(`/${id}/series`, { sensor }, s),
  compare: (id: string, s?: AbortSignal) => get<EventCompare>(`/${id}/compare`, {}, s),
  robustness: (id: string, sensor: string, partition: Partition, s?: AbortSignal) => get<Robustness>(`/${id}/robustness`, { sensor, partition }, s),
};

export const CATEGORY_LABEL: Record<Category, string> = {
  sudden_decrease: "Sudden decrease",
  sudden_increase: "Sudden increase",
  sustained_deviation: "Sustained deviation",
  coordinated_deviation: "Coordinated deviation",
  sensor_quality: "Sensor-quality anomaly",
};

export function duration(min: number): string {
  if (min < 60) return `${min} min`;
  if (min < 1440) return `${Math.floor(min / 60)} h${min % 60 ? ` ${min % 60} min` : ""}`;
  const d = Math.floor(min / 1440);
  const h = Math.round((min % 1440) / 60);
  return `${d} d${h ? ` ${h} h` : ""}`;
}

/** Signed relative deviation as a percentage of typical flow, e.g. -0.62 -> "-62%". */
export const pct = (rel: number) => `${rel > 0 ? "+" : ""}${Math.round(rel * 100)}%`;

export interface SeriesRow {
  t: string;
  observed: number | null;
  expected: number | null;
  band: [number, number] | null;
  anomaly: number | null;
  missing: number | null;
}

/** Chart rows: observed (null where missing, never 0), typical flow, detection band, anomalous readings as points, and a
 * marker row at the axis floor where a reading is missing or invalid. */
export function seriesRows(s: EventSeries): SeriesRow[] {
  return s.timestamps.map((t, i) => ({
    t,
    observed: s.observed[i],
    expected: s.expected[i],
    band: s.threshold_low[i] !== null && s.threshold_high[i] !== null ? [Math.max(0, s.threshold_low[i]!), s.threshold_high[i]!] : null,
    anomaly: s.anomalous[i] !== 0 ? s.observed[i] : null,
    missing: s.status[i] === "ok" ? null : 0,
  }));
}

/** Rows for the normalised multi-sensor chart: one key per chosen sensor. */
export function compareRows(c: EventCompare, chosen: readonly string[]): Record<string, string | number | null>[] {
  const pick = c.sensors.filter((s) => chosen.includes(s.sensor));
  return c.timestamps.map((t, i) => Object.fromEntries([["t", t], ...pick.map((s) => [s.sensor, s.ratio[i]])]));
}

export function replayHref(e: Pick<EventDetail, "id" | "replay">, sensor?: string, time?: string): string {
  const t = time ?? e.replay.time;
  return `/replay?${new URLSearchParams({ date: t.slice(0, 10), time: t, sensor: sensor ?? e.replay.sensor, event: e.id }).toString()}`;
}
