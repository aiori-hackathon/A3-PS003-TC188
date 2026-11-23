import { CalendarClock, Download, FileUp, History, Info, LineChart as LineIcon, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ForecastChart } from "../components/charts";
import { StudyGate } from "../components/study";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Field,
  Input,
  Legend,
  Note,
  Num,
  NumTh,
  PageHeader,
  Segmented,
  Select,
  Skeleton,
  Switch,
  TableWrap,
  Tag,
} from "../components/ui";
import { api } from "../lib/api";
import { addSteps, beyondStudy, HORIZON, hhmm, nextInterval, niceDate, STEP_MIN, toMinutes } from "../lib/clock";
import { cn } from "../lib/cn";
import { useRequest, type StudyData } from "../lib/data";
import { num, pct } from "../lib/format";
import { MAX_FLOW, parseReading, parseReadingsCsv, readingsTemplate } from "../lib/readings";
import { forecastRows, replayApi, ReplayApiError, STATUS_TEXT, type Forecast as ReplayForecast, type Partition, type ReplayMeta, type ReplaySensor } from "../lib/replay";
import { CLIENT_NAME, CLIENTS, SERIES } from "../lib/study";
import type { ClientKey, ModelKey, Prediction } from "../lib/types";

type Mode = "historical" | "future";

/* ---------- inputs: shared sensor picker ---------- */
function SensorPicker({ data, client, sensor, onClient, onSensor }: { data: StudyData; client: ClientKey; sensor: string; onClient: (c: ClientKey) => void; onSensor: (s: string) => void }) {
  const list = data.sensors.filter((s) => s.client === client);
  return (
    <>
      <Field label="Client group">
        {(id) => (
          <Select id={id} value={client} onChange={(e) => onClient(e.target.value as ClientKey)}>
            {CLIENTS.map((c) => (
              <option key={c} value={c}>
                {CLIENT_NAME[c]}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Sensor">
        {(id) => (
          <Select id={id} value={sensor} onChange={(e) => onSensor(e.target.value)}>
            {list.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} · {s.type_group} · median {s.median_test_flow === null ? "n/a" : Math.round(s.median_test_flow)}
              </option>
            ))}
          </Select>
        )}
      </Field>
    </>
  );
}

/* ---------- inputs: historical ---------- */
const HISTORICAL_MODEL = "Residual MLP + FedAvgM — Frozen federated model";
const PARTITION_NAME: Record<Partition, string> = { geo: "Geographic FedAvgM", iid: "IID FedAvgM" };
const EXAMPLE_SENSORS = ["LA-01", "BA-01", "CS-01", "IC-01"];
// Both timestamps were checked against the frozen service for every example sensor and partition.
const EXAMPLE_TIMES = [
  { time: "2024-02-06T07:30", label: "Morning · validation period" },
  { time: "2024-09-12T17:00", label: "Evening · held-out test period" },
];

type HistoricalExample = (typeof EXAMPLE_TIMES)[number] & { sensor: string };

function HistoricalInputs({ meta, sensors, sensor, issuedAt, examples, setSensor, setIssuedAt, onRun, loading }: {
  meta: ReplayMeta;
  sensors: ReplaySensor[];
  sensor: ReplaySensor;
  issuedAt: string;
  examples: HistoricalExample[];
  setSensor: (s: ReplaySensor) => void;
  setIssuedAt: (time: string) => void;
  onRun: (sensor: ReplaySensor, time: string) => void;
  loading: boolean;
}) {
  const [error, setError] = useState<string | null>(null);
  const [date, time] = issuedAt.split("T");
  const regions = new Map(sensors.map((s) => [s.client_geo, s.region]));
  const first = meta.first_forecast_time;
  const last = meta.last_timestamp;

  const submit = () => {
    const minutes = toMinutes(issuedAt);
    if (minutes === null || minutes % STEP_MIN) return setError("Choose an issue time on a 5-minute boundary.");
    if (issuedAt < first || issuedAt > last) return setError("Choose an issue time within the supported range below.");
    setError(null);
    onRun(sensor, issuedAt);
  };

  return (
    <form className="flex flex-col gap-4" noValidate onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <Field label="Client group">
        {(id) => (
          <Select id={id} value={sensor.client_geo} onChange={(e) => {
            const next = sensors.find((s) => s.client_geo === e.target.value);
            if (next) setSensor(next);
          }}>
            {[...regions].map(([client, region]) => <option key={client} value={client}>{region}</option>)}
          </Select>
        )}
      </Field>
      <Field label="Sensor">
        {(id) => (
          <Select id={id} value={sensor.id} onChange={(e) => {
            const next = sensors.find((s) => s.id === e.target.value);
            if (next) setSensor(next);
          }}>
            {sensors.filter((s) => s.client_geo === sensor.client_geo).map((s) => (
              <option key={s.id} value={s.id}>{s.id} · {s.type_group}{s.freeway ? ` · ${s.freeway}` : ""}</option>
            ))}
          </Select>
        )}
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Date">
          {(id) => <Input id={id} type="date" min={first.slice(0, 10)} max={last.slice(0, 10)} required value={date} onChange={(e) => { setIssuedAt(`${e.target.value}T${time}`); setError(null); }} />}
        </Field>
        <Field label="Issue time (Pacific)">
          {(id) => <Input id={id} type="time" step={300} required value={time} onChange={(e) => { setIssuedAt(`${date}T${e.target.value}`); setError(null); }} />}
        </Field>
      </div>
      <Note>
        Issue times run from {niceDate(first)}, {hhmm(first)} to {niceDate(last)}, {hhmm(last)} on the dataset's Pacific wall clock.
        Availability depends on the selected sensor's preceding hour.
      </Note>
      {error && <p role="alert" className="text-sm text-danger-ink">{error}</p>}
      <Button type="submit" variant="primary" loading={loading} className="w-full">
        {loading ? "Running" : "Run forecast"}
      </Button>
      {examples.length > 0 && (
        <div className="border-t border-line pt-4">
          <p className="mb-2 text-sm font-medium text-fg-2">Historical examples</p>
          <div className="flex flex-col gap-1.5">
            {examples.map((e) => {
              const selected = issuedAt === e.time && sensor.id === e.sensor;
              return (
                <button
                  key={e.time}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => {
                    const next = sensors.find((s) => s.id === e.sensor);
                    if (!next) return;
                    setError(null);
                    onRun(next, e.time);
                  }}
                  className={cn(
                    "flex cursor-pointer flex-col items-start rounded-control border px-3 py-2 text-left transition-colors",
                    selected ? "border-accent/40 bg-accent/[0.07]" : "border-line bg-surface-2/40 hover:border-line-strong hover:bg-surface-2",
                  )}
                >
                  <span className="text-sm font-medium">{e.label}</span>
                  <span className="tnum text-xs text-fg-3">{e.sensor} · {niceDate(e.time)}, {hhmm(e.time)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </form>
  );
}

/* ---------- inputs: future ---------- */
function FutureInputs({ data, client, sensor, setClient, setSensor, onRun, loading }: {
  data: StudyData;
  client: ClientKey;
  sensor: string;
  setClient: (c: ClientKey) => void;
  setSensor: (s: string) => void;
  onRun: (time: string, observations: (number | null)[]) => void;
  loading: boolean;
}) {
  const [start, setStart] = useState(nextInterval);
  const [values, setValues] = useState<string[]>(() => Array(HORIZON).fill(""));
  const [file, setFile] = useState<{ name: string; notes: string[] } | null>(null);
  const [csvErrors, setCsvErrors] = useState<string[]>([]);
  const [formError, setFormError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const startMin = toMinutes(start);
  const times = useMemo(() => (startMin === null ? null : Array.from({ length: HORIZON }, (_, i) => addSteps(start, i - HORIZON))), [start, startMin]);
  const parsed = values.map(parseReading);
  const filled = parsed.filter((p) => "value" in p && p.value !== null).length;

  const load = async (f: File) => {
    setFormError(null);
    if (f.size > 1_000_000) return setCsvErrors(["The file is larger than 1 MB. Keep only the most recent readings."]);
    const r = parseReadingsCsv(await f.text());
    if (!r.ok) {
      setFile(null);
      return setCsvErrors(r.errors);
    }
    setCsvErrors([]);
    setFile({ name: f.name, notes: r.notes });
    setStart(r.start);
    setValues(r.observations.map((v) => (v === null ? "" : String(v))));
  };
  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) void load(f);
    e.target.value = "";
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void load(f);
  };
  const template = () => {
    const url = URL.createObjectURL(new Blob([readingsTemplate(startMin === null ? nextInterval() : start)], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "aiori-readings-template.csv" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const submit = () => {
    if (startMin === null || startMin % STEP_MIN) return setFormError("Choose a forecast start on a 5-minute boundary.");
    if (addSteps(start, -HORIZON) < "2023-01-01T00:00") return setFormError("The hour of readings must start on or after 1 January 2023.");
    if (start >= "2100-01-01T00:00") return setFormError("Choose a start before the year 2100.");
    if (parsed.some((p) => "error" in p)) return setFormError(`Fix the highlighted readings: each is a number from 0 to ${MAX_FLOW}, or blank if missing.`);
    if (!filled) return setFormError("Enter at least one reading, or upload a CSV.");
    setFormError(null);
    onRun(start, parsed.map((p) => ("value" in p ? p.value : null)));
  };

  return (
    <form
      className="flex flex-col gap-4"
      noValidate
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <SensorPicker data={data} client={client} sensor={sensor} onClient={setClient} onSensor={setSensor} />
      <Note>Manual and CSV forecasts use the original 2023 GRU. Enter readings from this sensor so the model can apply its 2023 scaling.</Note>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn("rounded-card border border-dashed p-4 transition-colors", dragging ? "border-accent bg-accent/[0.06]" : "border-line-strong bg-bg/50")}
      >
        <input ref={fileRef} type="file" accept=".csv,text/csv,.txt" className="sr-only" onChange={onFile} aria-label="Upload a CSV of readings" tabIndex={-1} />
        {file ? (
          <div className="flex items-start gap-3">
            <FileUp className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="text-xs text-fg-3">
                Loaded 12 readings ending {hhmm(addSteps(start, -1))}, {niceDate(start)}. {file.notes.join(" ")}
              </p>
            </div>
            <button type="button" className="grid size-7 cursor-pointer place-items-center rounded-control text-fg-3 hover:bg-surface-2 hover:text-fg" aria-label="Remove file" onClick={() => setFile(null)}>
              <X className="size-4" aria-hidden />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-3">
            <p className="text-sm text-fg-2">
              Upload a CSV with <code>timestamp</code> and <code>flow</code> columns; the latest 12 consecutive 5-minute rows are used.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => fileRef.current?.click()}>
                <FileUp className="size-4" aria-hidden />
                Upload CSV
              </Button>
              <Button type="button" variant="ghost" onClick={template}>
                <Download className="size-4" aria-hidden />
                Blank template
              </Button>
            </div>
          </div>
        )}
      </div>
      {csvErrors.length > 0 && (
        <div role="alert" className="rounded-control border border-danger/35 bg-danger/[0.06] px-3 py-2.5 text-xs text-fg-2">
          <p className="mb-1 font-semibold text-fg">This file could not be used.</p>
          <ul className="list-disc space-y-0.5 pl-4">
            {csvErrors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Field label="Forecast from (date)">
          {(id) => (
            <Input id={id} type="date" min="2023-01-01" max="2099-12-31" value={start.slice(0, 10)} onChange={(e) => (setStart(`${e.target.value}T${start.slice(11, 16)}`), setFile(null), setCsvErrors([]))} />
          )}
        </Field>
        <Field label="Time">
          {(id) => <Input id={id} type="time" step={300} value={start.slice(11, 16)} onChange={(e) => (setStart(`${start.slice(0, 10)}T${e.target.value}`), setFile(null), setCsvErrors([]))} />}
        </Field>
      </div>

      <fieldset>
        <legend className="mb-1.5 text-sm font-medium text-fg-2">
          Latest hour of readings <span className="font-normal text-fg-3">· vehicles per 5 min, oldest first</span>
        </legend>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-3">
          {values.map((v, i) => {
            const bad = "error" in parsed[i];
            const label = times ? hhmm(times[i]) : `Reading ${i + 1}`;
            return (
              <label key={i} className="flex flex-col gap-1">
                <span className="tnum text-xs text-fg-3">{label}</span>
                <Input
                  inputMode="decimal"
                  autoComplete="off"
                  value={v}
                  aria-invalid={bad || undefined}
                  aria-label={`Reading at ${label}`}
                  title={bad ? (parsed[i] as { error: string }).error : undefined}
                  placeholder="blank"
                  className="h-8 px-2 text-right placeholder:text-fg-3/60"
                  onChange={(e) => {
                    const next = [...values];
                    next[i] = e.target.value;
                    setValues(next);
                    setFile(null);
                    setCsvErrors([]);
                  }}
                />
              </label>
            );
          })}
        </div>
        <Note className="mt-2">
          {filled} of 12 entered. Blank readings are filled the way the model was trained: the last value for up to 30 minutes, then the sensor's typical
          level for that time of day.
        </Note>
      </fieldset>

      {formError && (
        <p role="alert" className="text-sm text-danger-ink">
          {formError}
        </p>
      )}
      <Button type="submit" variant="primary" loading={loading} className="w-full">
        {loading ? "Running" : "Forecast the next hour"}
      </Button>
    </form>
  );
}

/* ---------- historical result ---------- */
const HISTORICAL_AXIS = { stroke: "var(--color-line-strong)", tickLine: false, tick: { fill: "var(--color-fg-3)", fontSize: 12 } } as const;

function HistoricalResult({ f, compared, region, showActual }: { f: ReplayForecast; compared: ReplayForecast | null; region: string; showActual: boolean }) {
  const primary = PARTITION_NAME[f.model.partition];
  const compareLabel = compared && PARTITION_NAME[compared.model.partition];
  // Beyond the archive the API still returns predictions, but no timestamps for recorded observations.
  const times = f.forecast.timestamps.map((t, i) => t ?? addSteps(f.issued_at, i + 1));
  const rows = forecastRows({ ...f, forecast: { ...f.forecast, timestamps: times } }).map((r, i) => ({
    ...r,
    comparison: i >= f.input.timestamps.length ? compared?.forecast.values[i - f.input.timestamps.length] : undefined,
  }));
  const end = times[times.length - 1];
  const filled = f.input.imputed.some(Boolean);

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <Card aria-live="polite">
        <CardHeader
          title={`${f.sensor} · ${region}`}
          meta={`Issued ${niceDate(f.issued_at)}, ${hhmm(f.issued_at)} Pacific · computed in ${num(f.inference_ms, 2)} ms on ${f.device.toUpperCase()}`}
          action={<Tag><History className="size-3" aria-hidden />Historical forecast</Tag>}
        />
        <CardBody>
          <div className="mb-5 flex gap-2.5 rounded-control border border-line bg-surface-2/50 px-3 py-2.5 text-sm">
            <Info className="mt-0.5 size-4 shrink-0 text-fg-3" aria-hidden />
            <div className="space-y-1 text-fg-2">
              <p className="font-medium text-fg">{HISTORICAL_MODEL}</p>
              <p>{f.forecast.timestamps[0] ? f.period_label : "The forecast hour is beyond the recorded dataset."}</p>
              {f.warnings.map((w) => <p key={w}>{w}</p>)}
            </div>
          </div>

          <div role="figure" aria-label={`Traffic flow at sensor ${f.sensor}, forecast issued at ${hhmm(f.issued_at)} through ${hhmm(end)} Pacific.`} className="-ml-2 select-none">
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={rows} margin={{ top: 22, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="var(--color-grid)" vertical={false} />
                <ReferenceArea x1={f.issued_at} x2={end} fill="var(--color-accent)" fillOpacity={0.045} ifOverflow="visible" />
                <ReferenceLine x={f.issued_at} stroke="var(--color-cursor)" strokeDasharray="3 3"
                  label={{ value: "Forecast", position: "insideTopLeft", fill: "var(--color-accent)", fontSize: 12, dx: 6, dy: -20 }} />
                <XAxis dataKey="t" {...HISTORICAL_AXIS} tickFormatter={hhmm} minTickGap={24} tickMargin={8} />
                <YAxis {...HISTORICAL_AXIS} width={46} tickFormatter={(v: number) => num(v, 0)} domain={[(v: number) => Math.min(0, v), "auto"]} allowDecimals={false} />
                <Tooltip cursor={{ stroke: "var(--color-cursor)" }} labelFormatter={(t) => `${hhmm(String(t))} · ${niceDate(String(t))}`}
                  formatter={(v) => num(Number(v), 1)} isAnimationActive={false}
                  contentStyle={{ background: "var(--color-surface-2)", border: "1px solid var(--color-line-strong)", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "var(--color-fg)", fontWeight: 500 }} />
                <Line dataKey="observed" name="Observed input" stroke={SERIES.observed.color} strokeWidth={SERIES.observed.width}
                  dot={{ r: 2.4, fill: "var(--color-surface)", strokeWidth: 1.4 }} activeDot={{ r: 3.5 }} connectNulls={false} isAnimationActive={false} />
                {filled && <Line dataKey="filled" name="Filled input" stroke="var(--color-fg-3)" strokeWidth={0}
                  dot={{ r: 3, fill: "var(--color-surface)", strokeDasharray: "1 1" }} isAnimationActive={false} />}
                {showActual && <Line dataKey="actual" name="Observed afterwards" stroke={SERIES.actual.color} strokeWidth={SERIES.actual.width}
                  strokeDasharray={SERIES.actual.dash} dot={false} connectNulls={false} isAnimationActive={false} />}
                {compareLabel && <Line dataKey="comparison" name={compareLabel} stroke={SERIES.centralised.color}
                  strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />}
                <Line dataKey="forecast" name={primary} stroke={SERIES.local.color} strokeWidth={SERIES.local.width}
                  dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <Legend items={[
            { label: "Observed, previous hour", color: SERIES.observed.color },
            ...(filled ? [{ label: "Filled input", color: "var(--color-fg-3)", dash: true }] : []),
            { label: primary, color: SERIES.local.color },
            ...(compareLabel ? [{ label: compareLabel, color: SERIES.centralised.color, dash: true }] : []),
            ...(showActual ? [{ label: "Observed afterwards", color: SERIES.actual.color, dash: true }] : []),
          ]} />
          {showActual && (
            <Note className="mt-3">
              {[f, ...(compared ? [compared] : [])].map((result) => (
                <span key={result.model.partition} className="block">
                  {PARTITION_NAME[result.model.partition]} for this sensor and hour
                  {" · "}MAE {num(result.comparison.mae, 2)}
                  {" · "}WAPE {pct(result.comparison.wape_pct, 2)}
                  {" · "}{result.comparison.n_compared} of {HORIZON} steps compared
                </span>
              ))}
              <span>{f.comparison.note}</span>
            </Note>
          )}
        </CardBody>
        <ol className="grid grid-cols-3 border-t border-line sm:grid-cols-6 xl:grid-cols-12" aria-label={`${primary} forecast, vehicles per 5 minutes`}>
          {times.map((t, i) => (
            <li key={t} className="flex flex-col gap-0.5 border-b border-r border-line px-3 py-2.5 last:border-r-0 xl:border-b-0">
              <span className="tnum text-xs text-fg-3" title={t}>{hhmm(t)}</span>
              <span className="tnum text-md font-semibold text-accent">{num(f.forecast.values[i], 0)}</span>
              {showActual && <span className="tnum text-xs text-fg-3">obs <b className="font-medium text-observed">{num(f.actual.values[i], 0)}</b></span>}
            </li>
          ))}
        </ol>
      </Card>

      <Card>
        <CardHeader title="Forecast detail" meta={f.units} />
        <TableWrap label="Forecast detail">
          <thead>
            <tr>
              <th scope="col">Time</th>
              <NumTh scope="col">{primary}</NumTh>
              {compareLabel && <NumTh scope="col">{compareLabel}</NumTh>}
              {showActual && <NumTh scope="col">Observed afterwards</NumTh>}
              {showActual && <NumTh scope="col">Absolute error</NumTh>}
            </tr>
          </thead>
          <tbody>
            {times.map((t, i) => (
              <tr key={t}>
                <td className="tnum" title={t}>{hhmm(t)}</td>
                <Num>{num(f.forecast.values[i], 1)}</Num>
                {compared && <Num className="text-fg-2">{num(compared.forecast.values[i], 1)}</Num>}
                {showActual && <Num className="text-observed">{f.actual.values[i] === null ? STATUS_TEXT[f.actual.status[i]] : num(f.actual.values[i], 0)}</Num>}
                {showActual && <Num>{num(f.comparison.abs_error[i], 1)}</Num>}
              </tr>
            ))}
          </tbody>
        </TableWrap>
        <details className="border-t border-line">
          <summary className="cursor-pointer px-4 py-3 text-sm text-fg-2">Previous 60 minutes · 12 input readings</summary>
          <TableWrap label="Previous hour readings and model inputs">
            <thead>
              <tr><th scope="col">Time</th><NumTh scope="col">Observed input</NumTh><NumTh scope="col">Model input</NumTh></tr>
            </thead>
            <tbody>
              {f.input.timestamps.map((t, i) => (
                <tr key={t}>
                  <td className="tnum" title={t}>{hhmm(t)}</td>
                  <Num>{num(f.input.observed[i], 1)}</Num>
                  <Num>{num(f.input.model_input[i], 1)}{f.input.imputed[i] && " (filled)"}</Num>
                </tr>
              ))}
            </tbody>
          </TableWrap>
        </details>
      </Card>
    </div>
  );
}

/* ---------- future result ---------- */
function FutureResult({ p, models }: { p: Prediction; models: ModelKey[] }) {
  const primary = "Original GRU (local-only)";
  const compare = "Centralised GRU";
  const v = p.forecasts.final?.values ?? [];
  const c = models.includes("centralised") ? p.forecasts.centralised?.values : undefined;
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <Card aria-live="polite">
        <CardHeader
          title={`${p.sensor.id} · ${p.sensor.client_name}`}
          meta={`${niceDate(p.forecast_start)}, from ${hhmm(p.forecast_start)} · computed in ${p.inference_ms} ms on ${p.device.toUpperCase()}`}
          action={(
            <Tag tone={p.beyond_study_period ? "observed" : "neutral"}>
              <CalendarClock className="size-3" aria-hidden />
              {p.beyond_study_period ? "Future forecast · not validated" : "Your readings"}
            </Tag>
          )}
        />
        <CardBody>
          <div className={cn("mb-5 flex gap-2.5 rounded-control border px-3 py-2.5 text-sm", p.beyond_study_period ? "border-observed/30 bg-observed/[0.06]" : "border-line bg-surface-2/50")}>
            <Info className={cn("mt-0.5 size-4 shrink-0", p.beyond_study_period ? "text-observed" : "text-fg-3")} aria-hidden />
            <p className="text-fg-2">
              {p.accuracy_note} <span className="text-fg-3">Data: {p.data_label}.</span>
            </p>
          </div>
          <ForecastChart prediction={p} models={models} showActual={false} primaryLabel={primary} compareLabel={compare} />
          <Legend items={[
            { label: "Your readings", color: SERIES.observed.color },
            { label: primary, color: SERIES.local.color },
            ...(c ? [{ label: compare, color: SERIES.centralised.color, dash: true }] : []),
          ]} />
        </CardBody>
        <ol className="grid grid-cols-3 border-t border-line sm:grid-cols-6 xl:grid-cols-12" aria-label={`${primary} forecast, vehicles per 5 minutes`}>
          {p.forecast_timestamps.map((t, i) => (
            <li key={t} className="flex flex-col gap-0.5 border-b border-r border-line px-3 py-2.5 last:border-r-0 xl:border-b-0">
              <span className="tnum text-xs text-fg-3">{hhmm(t)}</span>
              <span className="tnum text-md font-semibold text-accent">{Math.round(v[i])}</span>
            </li>
          ))}
        </ol>
      </Card>
      <Card>
        <CardHeader title="Forecast detail" meta={p.units} />
        <TableWrap label="Forecast detail">
          <thead>
            <tr><th scope="col">Time</th><NumTh scope="col">{primary}</NumTh>{c && <NumTh scope="col">{compare}</NumTh>}</tr>
          </thead>
          <tbody>
            {p.forecast_timestamps.map((t, i) => (
              <tr key={t}>
                <td className="tnum">{hhmm(t)}</td>
                <Num>{num(v[i], 1)}</Num>
                {c && <Num className="text-fg-2">{num(c[i], 1)}</Num>}
              </tr>
            ))}
          </tbody>
        </TableWrap>
      </Card>
    </div>
  );
}

function ForecastLayout({ tabs, controls, children }: { tabs: ReactNode; controls: ReactNode; children: ReactNode }) {
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <Card className="lg:sticky lg:top-20">
        <div className="border-b border-line p-3">{tabs}</div>
        <CardBody>{controls}</CardBody>
      </Card>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

async function loadHistoricalData(signal: AbortSignal) {
  const [meta, sensors] = await Promise.all([replayApi.meta(signal), replayApi.sensors(signal)]);
  if (!sensors.length) throw new Error("The historical service returned no sensors.");
  return { meta, sensors };
}

function HistoricalWorkspace({ tabs }: { tabs: ReactNode }) {
  const { data, error, run } = useRequest(loadHistoricalData);
  useEffect(() => { void run(); }, [run]);

  if (data) return <HistoricalWorkspaceReady tabs={tabs} {...data} />;
  return (
    <ForecastLayout tabs={tabs} controls={<Note>Historical forecasts use the frozen federated model.</Note>}>
      {error ? <ErrorState title="Historical forecasting is unavailable." message={error} onRetry={() => void run()} /> : <Skeleton className="h-[520px]" label="Loading historical sensors" />}
    </ForecastLayout>
  );
}

async function runHistoricalForecast(
  sensor: ReplaySensor, time: string, partition: Partition, meta: ReplayMeta, sensors: ReplaySensor[], signal: AbortSignal,
) {
  try {
    return { forecast: await replayApi.forecast(sensor.id, time, partition, signal) };
  } catch (error) {
    if (!(error instanceof ReplayApiError) || error.status !== 422 || time < meta.first_forecast_time || time > meta.last_timestamp) throw error;
  }

  const date = time.slice(0, 10);
  const dayStart = `${date}T00:00`;
  const previousDate = addSteps(dayStart, -1).slice(0, 10);
  const [current, previous, observed] = await Promise.allSettled([
    replayApi.series(sensor.id, date, signal),
    previousDate >= meta.first_forecast_time.slice(0, 10) ? replayApi.series(sensor.id, previousDate, signal) : null,
    replayApi.frame(time, "observed", partition, HORIZON, signal),
  ]);

  const start = dayStart > meta.first_forecast_time ? dayStart : meta.first_forecast_time;
  const dayEnd = `${date}T23:55`;
  const end = dayEnd < meta.last_timestamp ? dayEnd : meta.last_timestamp;
  const requested = toMinutes(time) ?? 0;
  let nearest: string | undefined;
  let bestDistance = Infinity;

  // A valid reading can support issuance from its own slot through the next 11 slots.
  for (const result of [previous, current]) {
    if (result.status !== "fulfilled" || !result.value) continue;
    const { timestamps, status } = result.value;
    for (const [index, reading] of timestamps.entries()) {
      if (status[index] !== "ok") continue;
      const first = reading > start ? reading : start;
      const lastWithReading = addSteps(reading, HORIZON - 1);
      const last = lastWithReading < end ? lastWithReading : end;
      if (first > last) continue;
      const candidate = time < first ? first : time > last ? last : time;
      if (candidate === time) continue;
      const distance = Math.abs((toMinutes(candidate) ?? requested) - requested);
      if (distance < bestDistance) {
        nearest = candidate;
        bestDistance = distance;
      }
    }
  }

  const sameRegion = observed.status === "fulfilled"
    ? sensors.filter((s, i) => s.id !== sensor.id && s.client_geo === sensor.client_geo && observed.value.status[i] === "ok")
    : [];
  const sameType = sameRegion.filter((s) => s.type_group === sensor.type_group);
  const alternate = sameType.find((s) => s.freeway === sensor.freeway)
    ?? sameType.find((s) => sensor.freeway && s.freeway?.split("-")[0] === sensor.freeway.split("-")[0])
    ?? sameType[0] ?? sameRegion[0];
  return { gap: { sensor, time, nearest, alternate } };
}

function HistoricalWorkspaceReady({ tabs, meta, sensors }: { tabs: ReactNode; meta: ReplayMeta; sensors: ReplaySensor[] }) {
  const exampleTimes = EXAMPLE_TIMES.filter(({ time }) =>
    time >= meta.first_forecast_time && time <= meta.last_timestamp && meta.dates.some((day) => day.date === time.slice(0, 10) && day.coverage > 0),
  );
  const firstSensor = sensors.find((s) => EXAMPLE_SENSORS.includes(s.id)) ?? sensors[0];
  const firstExampleTime = EXAMPLE_SENSORS.includes(firstSensor.id) ? exampleTimes[0]?.time : undefined;
  const [sensor, setSensor] = useState(firstSensor);
  const [issuedAt, setIssuedAt] = useState(firstExampleTime ?? meta.first_forecast_time);
  const [compare, setCompare] = useState(true);
  const [showActual, setShowActual] = useState(true);
  const partition = meta.default_partition;
  const otherPartition = partition === "geo" ? "iid" : "geo";
  const otherModel = meta.models.find((m) => m.partition === otherPartition);
  const primary = useRequest(runHistoricalForecast);
  const comparison = useRequest(replayApi.forecast);
  const { run } = primary;
  const { run: runComparison, reset: resetComparison } = comparison;

  const runHistorical = (selected: ReplaySensor, time: string) => {
    setSensor(selected);
    setIssuedAt(time);
    void run(selected, time, partition, meta, sensors);
  };
  useEffect(() => {
    if (firstExampleTime) void run(firstSensor, firstExampleTime, partition, meta, sensors);
  }, [firstExampleTime, firstSensor, partition, meta, sensors, run]);

  const f = primary.data?.forecast;
  const gap = primary.data?.gap;
  useEffect(() => {
    if (compare && f && otherModel) void runComparison(f.sensor, f.issued_at, otherPartition);
    else resetComparison();
  }, [compare, f, otherModel, otherPartition, resetComparison, runComparison]);

  const c = comparison.data;
  const compared = compare && f && c && c.sensor === f.sensor && c.issued_at === f.issued_at && c.model.partition === otherPartition ? c : null;
  const exampleSensor = sensors.find((s) => s.client_geo === sensor.client_geo && EXAMPLE_SENSORS.includes(s.id));
  const examples = exampleSensor ? exampleTimes.map((e) => ({ ...e, sensor: exampleSensor.id })) : [];
  const archiveYears = `${meta.first_timestamp.slice(0, 4)}–${meta.last_timestamp.slice(0, 4)}`;
  const regionCount = new Set(sensors.map((s) => s.client_geo)).size;
  const periods = meta.periods.map((p) => {
    const purpose = p.key === "train" ? "training" : p.key === "val" ? "validation" : "held-out test";
    return `${p.label.split(":")[0]} ${purpose}`;
  });

  return (
    <ForecastLayout tabs={tabs} controls={
      <>
        <HistoricalInputs meta={meta} sensors={sensors} sensor={sensor} setSensor={setSensor}
          issuedAt={issuedAt} setIssuedAt={setIssuedAt} examples={examples} onRun={runHistorical} loading={primary.loading} />
        <div className="mt-5 flex flex-col gap-2 border-t border-line pt-4">
          {otherModel && <Switch checked={compare} onChange={setCompare} label={`Compare ${PARTITION_NAME[otherPartition]}`}
            hint="Same architecture, separate frozen checkpoint" />}
          <Switch checked={showActual} onChange={setShowActual} label="Show observed afterwards" hint="Recorded readings for comparison" />
          {compare && comparison.error && <Note>Comparison unavailable. {comparison.error}</Note>}
        </div>
        <Note className="mt-4">
          <span className="block">{archiveYears} XTraffic observations · {meta.n_sensors} sensors · {regionCount} federated clients.</span>
          <span className="block">{periods.join(" · ")}.</span>
          <span className="block">Held-out test WAPE, mean of three seeds:</span>
          {meta.models.map((m) => (
            <span key={m.partition} className="block">{PARTITION_NAME[m.partition]} {pct(m.test_wape_pct_three_seed_mean, 3)}</span>
          ))}
        </Note>
      </>
    }>
      {primary.error ? (
        <ErrorState title="No forecast for this choice." message={primary.error} />
      ) : f ? (
        <div className={cn("transition-opacity", primary.loading && "opacity-60")}>
          <HistoricalResult f={f} compared={compared} region={sensors.find((s) => s.id === f.sensor)?.region ?? sensor.region} showActual={showActual} />
        </div>
      ) : primary.loading ? (
        <Skeleton className="h-[520px]" label="Running the forecast" />
      ) : gap ? (
        <Card>
          <CardHeader title="No usable readings for this hour" meta={`${gap.sensor.id} · ${niceDate(gap.time)}, ${hhmm(gap.time)} Pacific`} />
          <CardBody>
            <p className="text-sm text-fg-2">
              This sensor has no usable readings in the hour ending at {hhmm(gap.time)}. The model needs at least one recorded reading to make a forecast.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {gap.nearest && (
                <Button type="button" onClick={() => runHistorical(gap.sensor, gap.nearest!)}>
                  Try {hhmm(gap.nearest)} with {gap.sensor.id}
                </Button>
              )}
              {gap.alternate && (
                <Button type="button" onClick={() => runHistorical(gap.alternate!, gap.time)}>
                  Try {gap.alternate.id}{gap.alternate.freeway && ` (${gap.alternate.freeway})`} at {hhmm(gap.time)}
                </Button>
              )}
            </div>
            {!gap.nearest && !gap.alternate && <Note>Choose a different date or sensor, or select a historical example.</Note>}
          </CardBody>
        </Card>
      ) : (
        <Card><EmptyState icon={<LineIcon className="size-5" aria-hidden />} title="Choose a historical hour">
          Select a sensor and issue time to forecast the following 60 minutes with {HISTORICAL_MODEL}.
        </EmptyState></Card>
      )}
    </ForecastLayout>
  );
}

function FutureWorkspace({ tabs, data }: { tabs: ReactNode; data: StudyData }) {
  const [sensor, setSensor] = useState(data.demo.examples[0].sensor);
  const client = data.sensors.find((s) => s.id === sensor)?.client ?? CLIENTS[0];
  const setClient = (c: ClientKey) => setSensor(data.sensors.find((s) => s.client === c)!.id);
  const [compare, setCompare] = useState(true);
  const req = useRequest(api.predict);
  const models: ModelKey[] = compare ? ["final", "centralised"] : ["final"];
  const p = req.data;
  const runFuture = (time: string, observations: (number | null)[]) =>
    void req.run({ sensor, time, observations, models: ["final", "centralised"] });

  return (
    <ForecastLayout tabs={tabs} controls={
      <>
        <FutureInputs data={data} client={client} sensor={sensor} setClient={setClient} setSensor={setSensor} onRun={runFuture} loading={req.loading} />
        <div className="mt-5 flex flex-col gap-2 border-t border-line pt-4">
          <Switch checked={compare} onChange={setCompare} label="Compare centralised GRU" hint="Same input, pooled-data model" />
        </div>
      </>
    }>
      {req.error ? (
        <ErrorState title="No forecast for this choice." message={req.error} />
      ) : p ? (
        <div className={cn("transition-opacity", req.loading && "opacity-60")}>
          <FutureResult p={p} models={models} />
        </div>
      ) : req.loading ? (
        <Skeleton className="h-[520px]" label="Running the forecast" />
      ) : (
        <Card>
          <EmptyState icon={<LineIcon className="size-5" aria-hidden />} title="Add the latest hour of readings">
            Upload a CSV or type the 12 most recent 5-minute readings for the chosen sensor. The original 2023 GRU forecasts the following hour.
            Its measured accuracy applies to the 2023 study period.
          </EmptyState>
        </Card>
      )}
      {p && beyondStudy(p.forecast_start) && (
        <Note className="mt-4">
          The original GRU recorded 8.672% test WAPE on 8 August–30 September 2023. Treat forecasts beyond that period as unvalidated estimates.
        </Note>
      )}
    </ForecastLayout>
  );
}

export default function Forecast() {
  const [params, setParams] = useSearchParams();
  const mode: Mode = params.get("mode") === "future" ? "future" : "historical";
  const tabs = (
    <Segmented
      label="Forecast type"
      value={mode}
      onChange={(m) => setParams(m === "future" ? { mode: "future" } : {}, { replace: true })}
      options={[
        { value: "historical", label: "Historical hour" },
        { value: "future", label: "Future forecast" },
      ]}
    />
  );

  return (
    <>
      <PageHeader title="Forecasting">
        The frozen Residual MLP + FedAvgM uses 60 minutes of recorded readings to forecast the next 60 in five-minute steps.
        Future forecasts from your own readings use the original 2023 GRU.
      </PageHeader>
      {mode === "historical" ? <HistoricalWorkspace tabs={tabs} /> : <StudyGate>{(d) => <FutureWorkspace tabs={tabs} data={d} />}</StudyGate>}
    </>
  );
}
