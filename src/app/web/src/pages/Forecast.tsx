import { CalendarClock, Download, FileUp, History, Info, LineChart as LineIcon, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useSearchParams } from "react-router";
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
import { displayModels } from "../lib/series";
import { num, signed } from "../lib/format";
import { MAX_FLOW, parseReading, parseReadingsCsv, readingsTemplate } from "../lib/readings";
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
function HistoricalInputs({ data, client, sensor, setClient, setSensor, onRun, loading }: {
  data: StudyData;
  client: ClientKey;
  sensor: string;
  setClient: (c: ClientKey) => void;
  setSensor: (s: string) => void;
  onRun: (time: string, sensorOverride?: string) => void;
  loading: boolean;
}) {
  const first = data.demo.examples[0];
  const [date, setDate] = useState(first.time.slice(0, 10));
  const [time, setTime] = useState(first.time.slice(11, 16));
  const [picked, setPicked] = useState<number | null>(0);
  const examples = data.demo.examples.map((e, i) => ({ ...e, i })).filter((e) => data.sensors.find((s) => s.id === e.sensor)?.client === client);
  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        setPicked(null);
        onRun(`${date}T${time}`);
      }}
    >
      <SensorPicker data={data} client={client} sensor={sensor} onClient={setClient} onSensor={(s) => (setSensor(s), setPicked(null))} />
      <div className="grid grid-cols-2 gap-3">
        <Field label="Date">{(id) => <Input id={id} type="date" min="2023-07-11" max="2023-09-30" required value={date} onChange={(e) => setDate(e.target.value)} />}</Field>
        <Field label="Forecast from">{(id) => <Input id={id} type="time" step={300} required value={time} onChange={(e) => setTime(e.target.value)} />}</Field>
      </div>
      <Note>Historical forecasts can start from 01:00 on 11 July to 23:00 on 30 September 2023, the validation and held-out test periods.</Note>
      <Button type="submit" variant="primary" loading={loading} className="w-full">
        {loading ? "Running" : "Run forecast"}
      </Button>
      <div className="border-t border-line pt-4">
        <p className="mb-2 text-sm font-medium text-fg-2">Historical examples</p>
        <div className="flex flex-col gap-1.5">
          {examples.map((e) => (
            <button
              key={e.i}
              type="button"
              aria-pressed={picked === e.i}
              onClick={() => {
                setPicked(e.i);
                setSensor(e.sensor);
                setDate(e.time.slice(0, 10));
                setTime(e.time.slice(11, 16));
                onRun(e.time, e.sensor);
              }}
              className={cn(
                "flex cursor-pointer flex-col items-start rounded-control border px-3 py-2 text-left transition-colors",
                picked === e.i ? "border-accent/40 bg-accent/[0.07]" : "border-line bg-surface-2/40 hover:border-line-strong hover:bg-surface-2",
              )}
            >
              <span className="text-sm font-medium">{e.label}</span>
              <span className="tnum text-xs text-fg-3">
                {e.sensor} · {niceDate(e.time)}, {hhmm(e.time)}
              </span>
            </button>
          ))}
        </div>
      </div>
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
      <Note>Use readings from this sensor: the model scales them with the sensor's own 2023 training statistics.</Note>

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
        <p role="alert" className="text-sm text-[#f08680]">
          {formError}
        </p>
      )}
      <Button type="submit" variant="primary" loading={loading} className="w-full">
        {loading ? "Running" : "Forecast the next hour"}
      </Button>
    </form>
  );
}

/* ---------- result ---------- */
function ResultTag({ p }: { p: Prediction }) {
  if (p.mode === "historical")
    return (
      <Tag>
        <History className="size-3" aria-hidden />
        {p.validated ? "Historical · held-out test" : "Historical · validation period"}
      </Tag>
    );
  return (
    <Tag tone={p.beyond_study_period ? "observed" : "neutral"}>
      <CalendarClock className="size-3" aria-hidden />
      {p.beyond_study_period ? "Future forecast · not validated" : "Your readings"}
    </Tag>
  );
}

function Result({ p: raw, models, showActual }: { p: Prediction; models: ModelKey[]; showActual: boolean }) {
  const { pred: p, primary, compare } = displayModels(raw);
  const v = p.forecasts.final?.values ?? [];
  const c = models.includes("centralised") ? p.forecasts.centralised?.values : undefined;
  const a = showActual && p.actual ? p.actual.values : null;
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <Card aria-live="polite">
        <CardHeader
          title={`${p.sensor.id} · ${p.sensor.client_name}`}
          meta={`${niceDate(p.forecast_start)}, from ${hhmm(p.forecast_start)} · computed in ${p.inference_ms} ms on ${p.device.toUpperCase()}`}
          action={<ResultTag p={p} />}
        />
        <CardBody>
          <div className={cn("mb-5 flex gap-2.5 rounded-control border px-3 py-2.5 text-sm", p.beyond_study_period ? "border-observed/30 bg-observed/[0.06]" : "border-line bg-surface-2/50")}>
            <Info className={cn("mt-0.5 size-4 shrink-0", p.beyond_study_period ? "text-observed" : "text-fg-3")} aria-hidden />
            <p className="text-fg-2">
              {p.accuracy_note} <span className="text-fg-3">Data: {p.data_label}.</span>
            </p>
          </div>
          <ForecastChart prediction={p} models={models} showActual={showActual} primaryLabel={primary} compareLabel={compare} />
          <Legend
            items={[
              { label: p.mode === "historical" ? "Observed, previous hour" : "Your readings", color: SERIES.observed.color },
              { label: primary, color: SERIES.local.color },
              ...(c ? [{ label: compare, color: SERIES.centralised.color, dash: true }] : []),
              ...(a ? [{ label: "Observed afterwards", color: SERIES.actual.color, dash: true }] : []),
            ]}
          />
        </CardBody>
        <ol className="grid grid-cols-3 border-t border-line sm:grid-cols-6 xl:grid-cols-12" aria-label={`${primary} forecast, vehicles per 5 minutes`}>
          {p.forecast_timestamps.map((t, i) => (
            <li key={t} className="flex flex-col gap-0.5 border-b border-r border-line px-3 py-2.5 last:border-r-0 xl:border-b-0">
              <span className="tnum text-xs text-fg-3">{hhmm(t)}</span>
              <span className="tnum text-md font-semibold text-accent">{Math.round(v[i])}</span>
              {a && (
                <span className="tnum text-xs text-fg-3">
                  obs <b className="font-medium text-observed">{a[i] === null ? "—" : Math.round(a[i])}</b>
                </span>
              )}
            </li>
          ))}
        </ol>
      </Card>
      <Card>
        <CardHeader title="Forecast detail" meta={p.units} />
        <TableWrap label="Forecast detail">
          <thead>
            <tr>
              <th scope="col">Time</th>
              <NumTh scope="col">{primary}</NumTh>
              {c && <NumTh scope="col">{compare}</NumTh>}
              {a && <NumTh scope="col">Observed afterwards</NumTh>}
              {a && <NumTh scope="col">{primary} error</NumTh>}
            </tr>
          </thead>
          <tbody>
            {p.forecast_timestamps.map((t, i) => (
              <tr key={t}>
                <td className="tnum">{hhmm(t)}</td>
                <Num>{num(v[i], 1)}</Num>
                {c && <Num className="text-fg-2">{num(c[i], 1)}</Num>}
                {a && <Num className="text-observed">{num(a[i], 0)}</Num>}
                {a && <Num>{a[i] === null ? "—" : signed(v[i] - a[i], 1)}</Num>}
              </tr>
            ))}
          </tbody>
        </TableWrap>
      </Card>
    </div>
  );
}

function ForecastWorkspace({ data }: { data: StudyData }) {
  const [params, setParams] = useSearchParams();
  const mode: Mode = params.get("mode") === "future" ? "future" : "historical";
  const first = data.demo.examples[0];
  const [sensor, setSensor] = useState(first.sensor);
  const client = (data.sensors.find((s) => s.id === sensor)?.client ?? CLIENTS[0]) as ClientKey;
  const setClient = (c: ClientKey) => setSensor(data.sensors.find((s) => s.client === c)!.id);
  const [compare, setCompare] = useState(true);
  const [showActual, setShowActual] = useState(true);
  const req = useRequest(api.predict);
  const { run, reset } = req;

  // Historical replay uses the optimized MLP when the server enables it (it needs a week of history); manual 12-reading
  // forecasts always use the original GRU.
  const histModels = useMemo<ModelKey[]>(() => (data.health.candidate_enabled ? ["candidate", "final"] : ["final", "centralised"]), [data.health.candidate_enabled]);
  const runHistorical = useCallback((time: string, s?: string) => void run({ sensor: s ?? sensor, time, models: histModels }), [run, sensor, histModels]);
  const runFuture = (time: string, observations: (number | null)[]) => void run({ sensor, time, observations, models: ["final", "centralised"] });

  // Each mode keeps its own result: switching clears the panel, and historical mode opens on its first example.
  useEffect(() => {
    reset();
    if (mode === "historical") {
      setSensor(first.sensor); // the picker shows the sensor whose result is displayed
      void run({ sensor: first.sensor, time: first.time, models: histModels });
    }
  }, [mode]);

  const models: ModelKey[] = compare ? ["final", "centralised"] : ["final"];
  const p = req.data && (req.data.mode === "historical") === (mode === "historical") ? req.data : null;

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <Card className="lg:sticky lg:top-20">
        <div className="border-b border-line p-3">
          <Segmented
            label="Forecast type"
            value={mode}
            onChange={(m) => setParams(m === "future" ? { mode: "future" } : {}, { replace: true })}
            options={[
              { value: "historical", label: "Historical hour" },
              { value: "future", label: "Future forecast" },
            ]}
          />
        </div>
        <CardBody>
          {mode === "historical" ? (
            <HistoricalInputs data={data} client={client} sensor={sensor} setClient={setClient} setSensor={setSensor} onRun={runHistorical} loading={req.loading} />
          ) : (
            <FutureInputs data={data} client={client} sensor={sensor} setClient={setClient} setSensor={setSensor} onRun={runFuture} loading={req.loading} />
          )}
          <div className="mt-5 flex flex-col gap-2 border-t border-line pt-4">
            <Switch
              checked={compare}
              onChange={setCompare}
              label={mode === "historical" && data.health.candidate_enabled ? "Compare original GRU" : "Compare centralised GRU"}
              hint={mode === "historical" && data.health.candidate_enabled ? "Original local-only GRU, same window" : "Same input, pooled-data model"}
            />
            {mode === "historical" && <Switch checked={showActual} onChange={setShowActual} label="Show observed afterwards" hint="Historical truth, for comparison" />}
          </div>
        </CardBody>
      </Card>

      <div className="min-w-0">
        {req.error ? (
          <ErrorState title="No forecast for this choice." message={req.error} />
        ) : p ? (
          <div className={cn("transition-opacity", req.loading && "opacity-60")}>
            <Result p={p} models={models} showActual={mode === "historical" && showActual} />
          </div>
        ) : req.loading ? (
          <Skeleton className="h-[520px]" label="Running the forecast" />
        ) : (
          <Card>
            <EmptyState icon={<LineIcon className="size-5" aria-hidden />} title="Add the latest hour of readings">
              Upload a CSV or type the 12 most recent 5-minute readings for the chosen sensor. The model forecasts the following hour, for any date from 2023
              on. Forecasts after September 2023 have not been checked against real traffic.
            </EmptyState>
          </Card>
        )}
        {mode === "future" && p && beyondStudy(p.forecast_start) && (
          <Note className="mt-4">
            The study's accuracy figures (test WAPE 8.672%) describe 8 August – 30 September 2023 only. Traffic patterns, sensors and road works change; treat
            this forecast as an unvalidated estimate.
          </Note>
        )}
      </div>
    </div>
  );
}

export default function Forecast() {
  return (
    <>
      <PageHeader title="Forecasting">
        Replay a historical hour from the study, or forecast ahead from your own latest readings. The frozen GRU reads the previous 60 minutes and forecasts the
        next 60 in 5-minute steps, computed on request each time.
      </PageHeader>
      <StudyGate>{(d) => <ForecastWorkspace data={d} />}</StudyGate>
    </>
  );
}
