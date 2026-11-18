/** Historical Traffic Replay (Phase 9A): API client, types and pure helpers. Talks to app/backend/replay_server.py. */

// Own small fetch helper (same behaviour as lib/api.ts, which is a protected production file and is left unchanged).
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ReplayApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function get<T>(path: string, params: Record<string, string | number> = {}, signal?: AbortSignal): Promise<T> {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/replay/${path}${q ? `?${q}` : ""}`, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ReplayApiError("The replay service is not reachable. Start it with .venv\\Scripts\\python.exe src\\app\\backend\\replay_server.py, then retry.", 0);
  }
  let data: { error?: { message?: string } } | null = null;
  try {
    data = await res.json();
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;                 // an aborted body must never look like an empty success
  }
  if (!res.ok) throw new ReplayApiError(data?.error?.message ?? `The replay service returned HTTP ${res.status}.`, res.status);
  if (data === null || typeof data !== "object") throw new ReplayApiError("The replay service returned an unreadable response.", res.status);
  return data as T;
}

export type Partition = "geo" | "iid";
export type Layer = "observed" | "forecast";
export type ObsStatus = "ok" | "missing" | "invalid" | "no_input" | "beyond_dataset";

export interface ModelInfo {
  model_id: string;
  partition: Partition;
  partition_label: string;
  architecture: string;
  algorithm: string;
  training_history: string;
  seed: number;
  run_id: string;
  checkpoint_epoch: number;
  checkpoint_sha256: string;
  manifest_sha256: string;
  scaler_scope: string;
  validation_wape_pct_this_seed: number;
  test_wape_pct_three_seed_mean: number;
  seed_policy: string;
  performance_note: string;
  units: string;
}

export interface ReplayMeta {
  source: string;
  units: string;
  timezone: string;
  n_sensors: number;
  mapped_sensors: number;
  first_timestamp: string;
  last_timestamp: string;
  first_forecast_time: string;
  missing_value_policy: string;
  dates: { date: string; coverage: number }[];
  periods: { key: string; start: string; end: string; label: string }[];
  location_note: string;
  congestion_note: string;
  models: ModelInfo[];
  default_partition: Partition;
}

export interface ReplaySensor {
  id: string;
  client_geo: string;
  region: string;
  client_iid: string;
  type_group: string;
  district: number;
  county: string | null;
  freeway: string | null;
  lat: number | null;
  lng: number | null;
  location_status: "provisional" | "excluded";
  shares_location: boolean;
}

export interface DayCoverage {
  date: string;
  period: string;
  period_label: string;
  valid_sensors: number[];
  dst_nonexistent: number[];
  available_slots: number;
}

export interface Frame {
  time: string;
  layer: Layer;
  units: string;
  period: string;
  dst_nonexistent: boolean;
  values: (number | null)[];
  status: ObsStatus[];
  relative: (number | null)[];
  valid_sensors: number;
  target_time?: string | null;
  horizon?: number;
  partition?: Partition;
  model_id?: string;
  actual?: (number | null)[];
}

export interface DaySeries {
  sensor: string;
  date: string;
  units: string;
  timestamps: string[];
  values: (number | null)[];
  status: ObsStatus[];
  typical: (number | null)[];
  typical_label: string;
}

export interface Forecast {
  sensor: string;
  issued_at: string;
  units: string;
  timezone: string;
  model: ModelInfo;
  period: string;
  period_label: string;
  input: { timestamps: string[]; model_input: number[]; observed: (number | null)[]; imputed: boolean[] };
  forecast: { timestamps: (string | null)[]; values: number[] };
  actual: { label: string; values: (number | null)[]; status: ObsStatus[] };
  comparison: { abs_error: (number | null)[]; n_compared: number; mae: number | null; wape_pct: number | null; note: string };
  warnings: string[];
  inference_ms: number;
  device: string;
}

export const replayApi = {
  meta: (s?: AbortSignal) => get<ReplayMeta>("meta", {}, s),
  sensors: (s?: AbortSignal) => get<ReplaySensor[]>("sensors", {}, s),
  day: (date: string, s?: AbortSignal) => get<DayCoverage>("day", { date }, s),
  frame: (time: string, layer: Layer, partition: Partition, horizon: number, s?: AbortSignal) =>
    get<Frame>("frame", layer === "observed" ? { time, layer } : { time, layer, partition, horizon }, s),
  series: (sensor: string, date: string, s?: AbortSignal) => get<DaySeries>("series", { sensor, date }, s),
  forecast: (sensor: string, time: string, partition: Partition, s?: AbortSignal) => get<Forecast>("forecast", { sensor, time, partition }, s),
};

/* ---------- time on the dataset's naive Pacific wall clock (pure string arithmetic, never the viewer's zone) ---------- */

export const SLOTS = 288;

export const slotTime = (date: string, slot: number) =>
  `${date}T${String(Math.floor(slot / 12)).padStart(2, "0")}:${String((slot % 12) * 5).padStart(2, "0")}`;

export const timeSlot = (t: string) => Number(t.slice(11, 13)) * 12 + Number(t.slice(14, 16)) / 5;

/** Next (dir 1) or previous (dir -1) slot after `slot` that has at least one valid reading, or null. */
export function nextAvailable(counts: readonly number[], slot: number, dir: 1 | -1): number | null {
  for (let s = slot + dir; s >= 0 && s < counts.length; s += dir) if (counts[s] > 0) return s;
  return null;
}

/** Adjacent date in the dataset's date list, or null at either end. */
export function adjacentDate(dates: readonly string[], date: string, dir: 1 | -1): string | null {
  const i = dates.indexOf(date);
  return i < 0 ? null : (dates[i + dir] ?? null);
}

/** Playback steps per second -> milliseconds between frames. */
export const SPEEDS = [1, 2, 4, 8] as const;
export const speedDelay = (stepsPerSecond: number) => Math.round(1000 / stepsPerSecond);

/* ---------- relative-flow colour (not a congestion level) ---------- */

export const REL_LOW = "#1f5fa8";
export const REL_MID = "#adb5bb"; // neutral slate grey: "typical" (was a green-tinted grey)
export const REL_HIGH = "#a85a00";
export const NO_DATA = "#8a949b";

const hex = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const mix = (a: string, b: string, t: number) =>
  `#${hex(a).map((x, i) => Math.round(x + (hex(b)[i] - x) * t).toString(16).padStart(2, "0")).join("")}`;

/** Ratio of flow to the sensor's typical flow -> colour; log2 scale clamped to 0.5x .. 2x. A real zero (or a negative
 * forecast) is a legitimate low value and takes the low end; only a missing ratio is grey. */
export function relColor(ratio: number | null): string {
  if (ratio === null || !Number.isFinite(ratio)) return NO_DATA;
  const t = ratio <= 0 ? -1 : Math.max(-1, Math.min(1, Math.log2(ratio)));
  return t < 0 ? mix(REL_MID, REL_LOW, -t) : mix(REL_MID, REL_HIGH, t);
}

/* ---------- bounded cache for frames, series and forecasts ---------- */

export class Lru<V> {
  private m = new Map<string, V>();
  constructor(private readonly max: number) {}
  get(k: string): V | undefined {
    const v = this.m.get(k);
    if (v !== undefined) {
      this.m.delete(k);
      this.m.set(k, v);
    }
    return v;
  }
  set(k: string, v: V) {
    this.m.delete(k);
    this.m.set(k, v);
    while (this.m.size > this.max) this.m.delete(this.m.keys().next().value as string);
  }
  get size() {
    return this.m.size;
  }
}

/* ---------- forecast chart rows ---------- */

export interface ForecastRow {
  t: string;
  observed?: number | null;
  filled?: number | null;
  forecast?: number | null;
  actual?: number | null;
}

/** Input hour (observed readings, and the filled values the model saw where readings were missing), then forecast and
 * later observations. Steps after the end of the dataset have no timestamp and are dropped. */
export function forecastRows(f: Forecast): ForecastRow[] {
  const rows: ForecastRow[] = f.input.timestamps.map((t, i) => ({
    t,
    observed: f.input.observed[i],
    filled: f.input.imputed[i] ? f.input.model_input[i] : null,
  }));
  f.forecast.timestamps.forEach((t, i) => {
    if (t === null) return;
    rows.push({ t, forecast: f.forecast.values[i], actual: f.actual.values[i] });
  });
  return rows;
}

export const STATUS_TEXT: Record<ObsStatus, string> = {
  ok: "observed",
  missing: "no reading",
  invalid: "reading flagged invalid",
  no_input: "no valid input readings",
  beyond_dataset: "after the end of the dataset",
};

/* ---------- deep links from the Disruption Explorer: /replay?date=..&time=..&sensor=..&event=.. ---------- */

export const EVENT_ID_RE = /^\d{8}T\d{4}-(LA|BA|CS|IC)-[DU]-[0-9a-f]{6}$/;

export interface ReplayLink {
  date?: string;
  slot?: number;
  sensor?: string;
  event?: string;
  problems: string[];
}

/** Validates replay link parameters against the dataset's dates and sensors; anything invalid is ignored and reported. */
export function parseReplayLink(q: URLSearchParams, dates: readonly string[], sensorIds: readonly string[]): ReplayLink {
  const out: ReplayLink = { problems: [] };
  const time = q.get("time");
  let date = q.get("date") ?? undefined;
  if (time !== null) {
    const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})$/.exec(time);
    if (!m || Number(m[2]) > 23 || Number(m[3]) % 5 !== 0 || (date !== undefined && date !== m[1])) out.problems.push("time");
    else {
      date = m[1];
      out.slot = timeSlot(time);
    }
  }
  if (date !== undefined) {
    if (dates.includes(date)) out.date = date;
    else {
      out.problems.push("date");
      out.slot = undefined;
    }
  }
  const sensor = q.get("sensor");
  if (sensor !== null) {
    if (sensorIds.includes(sensor)) out.sensor = sensor;
    else out.problems.push("sensor");
  }
  const event = q.get("event");
  if (event !== null) {
    if (EVENT_ID_RE.test(event)) out.event = event;
    else out.problems.push("event");
  }
  return out;
}
