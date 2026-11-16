import { ChevronLeft, ChevronRight, History, MapPin, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ForecastView, hhmm } from "../components/replay/ForecastView";
import { ReplayMap } from "../components/replay/ReplayMap";
import { Button, ButtonLink, Card, CardBody, CardHeader, EmptyState, ErrorState, Field, Input, Note, PageHeader, Segmented, Select, Skeleton, Tag } from "../components/ui";
import { niceDate } from "../lib/clock";
import { modelName, num } from "../lib/format";
import {
  adjacentDate,
  Lru,
  NO_DATA,
  nextAvailable,
  parseReplayLink,
  REL_HIGH,
  REL_LOW,
  REL_MID,
  replayApi,
  SLOTS,
  slotTime,
  SPEEDS,
  speedDelay,
  STATUS_TEXT,
  type DayCoverage,
  type DaySeries,
  type Forecast,
  type Frame,
  type Layer,
  type Partition,
  type ReplayLink,
  type ReplayMeta,
  type ReplaySensor,
} from "../lib/replay";
import { useRemote } from "../lib/useRemote";

/* Module-level bounded caches: survive page revisits within a session, never grow without limit. */
const dayCache = new Lru<DayCoverage>(64);
const frameCache = new Lru<Frame>(400);
const seriesCache = new Lru<DaySeries>(48);
const forecastCache = new Lru<Forecast>(200);



export default function Replay() {
  const [boot, setBoot] = useState<{ meta: ReplayMeta; sensors: ReplaySensor[] } | { error: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [params] = useSearchParams();
  useEffect(() => {
    const ac = new AbortController();
    setBoot(null);
    Promise.all([replayApi.meta(ac.signal), replayApi.sensors(ac.signal)]).then(
      ([meta, sensors]) => setBoot({ meta, sensors }),
      (e: Error) => e.name !== "AbortError" && setBoot({ error: e.message }),
    );
    return () => ac.abort();
  }, [attempt]);

  return (
    <>
      <PageHeader title="Historical Traffic Replay">
        Step through recorded XTraffic flow readings for the 200 study sensors, five minutes at a time, and ask the frozen
        federated model (MLP + FedAvgM, history A) to forecast the next hour from any moment. This is a replay of past data, not a live feed.
      </PageHeader>
      {boot === null && <Skeleton className="h-[640px]" label="Loading replay data" />}
      {boot && "error" in boot && <ErrorState title="Replay data could not be loaded" message={boot.error} onRetry={() => setAttempt((a) => a + 1)} />}
      {boot && "meta" in boot && (
        <ReplayView
          meta={boot.meta}
          sensors={boot.sensors}
          link={parseReplayLink(
            params,
            boot.meta.dates.filter((d) => d.coverage > 0).map((d) => d.date),
            boot.sensors.map((s) => s.id),
          )}
        />
      )}
    </>
  );
}

function ReplayView({ meta, sensors, link }: { meta: ReplayMeta; sensors: ReplaySensor[]; link: ReplayLink }) {
  const dates = useMemo(() => meta.dates.filter((d) => d.coverage > 0).map((d) => d.date), [meta]);
  const coverage = useMemo(() => new Map(meta.dates.map((d) => [d.date, d.coverage])), [meta]);
  const [date, setDate] = useState(() => link.date ?? [...meta.dates].reverse().find((d) => d.coverage >= 0.9)?.date ?? dates[dates.length - 1]);
  const [slot, setSlot] = useState(link.slot ?? 96);
  const [layer, setLayer] = useState<Layer>("observed");
  const [partition, setPartition] = useState<Partition>(meta.default_partition);
  const [horizon, setHorizon] = useState(12);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(2);
  const [selected, setSelected] = useState<string | null>(link.sensor ?? null);
  const [notice, setNotice] = useState<string | null>(link.problems.length ? `The link's ${link.problems.join(", ")} could not be used and ${link.problems.length > 1 ? "were" : "was"} ignored.` : null);
  const [dateInput, setDateInput] = useState(date);
  const align = useRef<1 | -1 | 0>(0);

  const time = slotTime(date, slot);
  const canForecast = time >= meta.first_forecast_time;
  const model = meta.models.find((m) => m.partition === partition)!;

  const day = useRemote(date, dayCache, (s) => replayApi.day(date, s));
  const counts = day.data?.valid_sensors;
  const frameKey = layer === "forecast" && !canForecast ? null : `${time}|${layer}|${layer === "forecast" ? `${partition}|${horizon}` : ""}`;
  const frame = useRemote(frameKey, frameCache, (s) => replayApi.frame(time, layer, partition, horizon, s), playing ? 0 : 60);
  const shown = frame.data ?? (frame.loading ? frame.last : undefined) ?? null;

  useEffect(() => setDateInput(date), [date]);

  // After moving to another day during playback or stepping, land on that day's first/last slot with readings.
  useEffect(() => {
    if (!counts || align.current === 0) return;
    const dir = align.current;
    align.current = 0;
    if (counts[slot] > 0) return;
    const n = nextAvailable(counts, dir > 0 ? -1 : SLOTS, dir);
    if (n !== null) setSlot(n);
  }, [counts, slot]);

  const step = useCallback(
    (dir: 1 | -1) => {
      if (!counts) return;
      const n = nextAvailable(counts, slot, dir);
      if (n !== null) {
        const gap = Math.abs(n - slot) - 1;
        setNotice(gap > 0 ? `Skipped ${gap} five-minute step${gap > 1 ? "s" : ""} with no valid reading at any sensor.` : null);
        setSlot(n);
        return;
      }
      const nd = adjacentDate(dates, date, dir);
      if (!nd) {
        setPlaying(false);
        setNotice(dir > 0 ? "Reached the end of the dataset (31 December 2024)." : "Reached the start of the dataset (1 January 2022).");
        return;
      }
      align.current = dir;
      setNotice(null);
      setDate(nd);
      setSlot(dir > 0 ? 0 : SLOTS - 1);
    },
    [counts, slot, dates, date],
  );

  // Playback: the next step is scheduled only after the current frame has arrived, so requests never overlap.
  const frameReady = frameKey === null || frame.data !== undefined || frame.error !== undefined;
  useEffect(() => {
    if (!playing || !counts || !frameReady) return;
    const id = window.setTimeout(() => step(1), speedDelay(speed));
    return () => clearTimeout(id);
  }, [playing, counts, frameReady, step, speed]);

  const chooseDate = (d: string) => {
    setDateInput(d);
    if (!dates.includes(d)) return;
    setNotice(null);
    setDate(d);
  };
  const dateError = dateInput && !dates.includes(dateInput) ? `Choose a date between ${dates[0]} and ${dates[dates.length - 1]} that has readings.` : undefined;
  const sensor = selected ? sensors.find((s) => s.id === selected)! : null;
  const idx = sensor ? sensors.indexOf(sensor) : -1;
  const unmapped = sensors.filter((s) => s.lat === null).length;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2" role="note">
        <Tag tone="observed">
          <History className="size-3.5" aria-hidden />
          Historical replay · not live
        </Tag>
        <Tag>2022–2024 · {meta.n_sensors} sensors</Tag>
        <Tag>Pacific local time (inferred)</Tag>
        <Tag>Locations provisional</Tag>
        {link.event && (
          <ButtonLink variant="ghost" className="ml-auto h-8" to={`/disruptions?event=${link.event}`}>
            Back to candidate event {link.event}
          </ButtonLink>
        )}
      </div>

      <Card>
        <CardHeader
          title={
            <span className="tnum">
              {niceDate(time)} · {hhmm(time)}
            </span>
          }
          meta={day.data ? `${day.data.period_label} · ${counts?.[slot] ?? 0} of ${meta.n_sensors} sensors have a valid reading at this time` : "Loading day coverage…"}
          action={day.data?.dst_nonexistent.includes(slot) ? <Tag tone="danger">Clock time skipped by daylight saving</Tag> : undefined}
        />
        <CardBody className="flex flex-col gap-4">
          <div className="grid gap-4 md:grid-cols-[minmax(0,220px)_minmax(0,1fr)]">
            <Field label="Date" error={dateError} hint={!dateError ? `Coverage this day: ${num(100 * (coverage.get(date) ?? 0), 1)}% of readings` : undefined}>
              {(id, d) => (
                <div className="flex gap-1.5">
                  <Button variant="ghost" className="w-9 px-0" aria-label="Previous day" disabled={!adjacentDate(dates, date, -1)} onClick={() => chooseDate(adjacentDate(dates, date, -1)!)}>
                    <ChevronLeft className="size-4" aria-hidden />
                  </Button>
                  <Input id={id} aria-describedby={d} type="date" min={dates[0]} max={dates[dates.length - 1]} value={dateInput} onChange={(e) => chooseDate(e.target.value)} />
                  <Button variant="ghost" className="w-9 px-0" aria-label="Next day" disabled={!adjacentDate(dates, date, 1)} onClick={() => chooseDate(adjacentDate(dates, date, 1)!)}>
                    <ChevronRight className="size-4" aria-hidden />
                  </Button>
                </div>
              )}
            </Field>
            <Field label="Time of day" hint="Bars show how many sensors have a valid reading at each five-minute step; red ticks mark steps with none.">
              {(id, d) => (
                <div className="flex flex-col gap-1.5">
                  <CoverageStrip counts={counts} slot={slot} onPick={setSlot} />
                  <input
                    id={id}
                    aria-describedby={d}
                    type="range"
                    min={0}
                    max={SLOTS - 1}
                    step={1}
                    value={slot}
                    aria-valuetext={`${hhmm(time)}`}
                    onChange={(e) => {
                      setNotice(null);
                      setSlot(Number(e.target.value));
                    }}
                    className="w-full accent-[var(--color-accent)]"
                  />
                </div>
              )}
            </Field>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="ghost" aria-label="Previous reading" onClick={() => step(-1)} disabled={!counts}>
              <SkipBack className="size-4" aria-hidden />
            </Button>
            <Button variant="primary" onClick={() => setPlaying((p) => !p)} disabled={!counts} aria-pressed={playing} className="min-w-[92px]">
              {playing ? <Pause className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
              {playing ? "Pause" : "Play"}
            </Button>
            <Button variant="ghost" aria-label="Next reading" onClick={() => step(1)} disabled={!counts}>
              <SkipForward className="size-4" aria-hidden />
            </Button>
            <label className="ml-1 flex items-center gap-2 text-sm text-fg-2">
              Speed
              <Select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="h-9 w-[150px]" aria-label="Replay speed">
                {SPEEDS.map((s) => (
                  <option key={s} value={s}>
                    {s} step{s > 1 ? "s" : ""} / second
                  </option>
                ))}
              </Select>
            </label>
            <span className="text-xs text-fg-3" aria-live="polite">
              {frame.loading ? `Loading ${hhmm(time)}…` : notice ?? (counts && counts[slot] === 0 ? "No sensor has a valid reading at this time." : "")}
            </span>
          </div>
          {day.error && <ErrorState title="Day coverage could not be loaded" message={day.error} onRetry={day.retry} />}
        </CardBody>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader
            title="Sensor map"
            meta={
              layer === "observed"
                ? `Observed flow at ${hhmm(time)}`
                : canForecast
                  ? `Forecast issued at ${hhmm(time)} for ${shown?.target_time ? hhmm(shown.target_time) : "—"} (${horizon * 5} min ahead) · ${modelName(model.model_id)}`
                  : "Forecast layer unavailable for this time"
            }
            action={
              <div className="flex flex-wrap items-center gap-2">
                <Segmented
                  label="Map layer"
                  value={layer}
                  onChange={setLayer}
                  options={[
                    { value: "observed", label: "Observed" },
                    { value: "forecast", label: "Predicted" },
                  ]}
                />
                {layer === "forecast" && (
                  <Select aria-label="Forecast lead time" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} className="h-10 w-[120px]">
                    {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
                      <option key={h} value={h}>
                        +{h * 5} min
                      </option>
                    ))}
                  </Select>
                )}
              </div>
            }
          />
          <CardBody className="flex flex-col gap-3">
            {layer === "forecast" && !canForecast && (
              <Note className="text-sm text-fg-2">
                Forecasts start at {meta.first_forecast_time.replace("T", " ")}: the model's preprocessing (training history A) begins on 1 January 2023, so 2022 is
                available as observations only.
              </Note>
            )}
            {frame.error && <ErrorState title="Map data unavailable" message={frame.error} onRetry={frame.retry} />}
            <ReplayMap sensors={sensors} frame={shown} selected={selected} onSelect={setSelected} />
            <MapLegend />
            <Note>
              {shown ? `${shown.valid_sensors} of ${meta.n_sensors} sensors have a ${shown.layer === "forecast" ? "forecast" : "valid reading"} at this time. ` : ""}
              {meta.mapped_sensors} of {meta.n_sensors} sensors are mapped{unmapped ? `; ${unmapped} without usable coordinates are listed in the sensor menu only` : ""}. Hollow
              grey markers have no {layer === "forecast" ? "forecast (no valid input reading in the hour before)" : "valid reading"}. {meta.location_note}
            </Note>
          </CardBody>
        </Card>

        <SensorPanel
          sensors={sensors}
          sensor={sensor}
          onSelect={setSelected}
          date={date}
          time={time}
          frameValue={shown && idx >= 0 && shown.layer === "observed" && shown.time === time ? shown.values[idx] : undefined}
          frameStatus={shown && idx >= 0 && shown.layer === "observed" && shown.time === time ? shown.status[idx] : undefined}
          partition={partition}
          setPartition={setPartition}
          canForecast={canForecast}
          meta={meta}
        />
      </div>

      <Card>
        <CardHeader title="About this data" />
        <CardBody className="grid gap-x-8 gap-y-3 text-sm leading-relaxed text-fg-2 md:grid-cols-2">
          <p>
            <b className="font-medium text-fg">Source.</b> {meta.source}. Units: {meta.units}.
          </p>
          <p>
            <b className="font-medium text-fg">Time.</b> {meta.timezone}. Timestamps are shown exactly as stored; on the spring-forward day the skipped hour has no readings.
          </p>
          <p>
            <b className="font-medium text-fg">Missing readings.</b> Policy {meta.missing_value_policy}. Missing readings are shown as missing, never as zero.
          </p>
          <p>
            <b className="font-medium text-fg">Traffic conditions.</b> {meta.congestion_note}
          </p>
          <p className="md:col-span-2">
            <b className="font-medium text-fg">Periods.</b> {meta.periods.map((p) => p.label).join(" · ")}. Observations from 2022 come before the model's training data.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

function CoverageStrip({ counts, slot, onPick }: { counts?: number[]; slot: number; onPick: (s: number) => void }) {
  return (
    <svg
      viewBox={`0 0 ${SLOTS} 20`}
      preserveAspectRatio="none"
      className="h-5 w-full cursor-pointer rounded-[3px] bg-bg"
      aria-hidden
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onPick(Math.max(0, Math.min(SLOTS - 1, Math.floor(((e.clientX - r.left) / r.width) * SLOTS))));
      }}
    >
      {counts?.map((c, i) =>
        c > 0 ? <rect key={i} x={i} y={20 - (18 * c) / 200} width={1} height={(18 * c) / 200} fill="var(--color-ref)" /> : <rect key={i} x={i} y={14} width={1} height={6} fill="var(--color-danger)" />,
      )}
      <rect x={slot - 0.5} y={0} width={2} height={20} fill="var(--color-accent)" />
    </svg>
  );
}

function MapLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-fg-2" aria-label="Map legend">
      <span className="flex items-center gap-2">
        Flow vs. this sensor's typical 2023 flow at this time of day
        <span className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${REL_LOW}, ${REL_MID}, ${REL_HIGH})` }} aria-hidden />
        <span className="tnum text-fg-3">0.5× · typical · 2×</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="inline-block size-2.5 rounded-full border border-dashed border-fg-3" style={{ background: `${NO_DATA}26` }} aria-hidden />
        no valid value
      </span>
      <span className="text-fg-3">Colour is relative flow, not a congestion level.</span>
    </div>
  );
}

type PanelProps = {
  sensors: ReplaySensor[];
  sensor: ReplaySensor | null;
  onSelect: (id: string) => void;
  date: string;
  time: string;
  frameValue?: number | null;
  frameStatus?: string;
  partition: Partition;
  setPartition: (p: Partition) => void;
  canForecast: boolean;
  meta: ReplayMeta;
};

function SensorPanel({ sensors, sensor, onSelect, date, time, frameValue, frameStatus, partition, setPartition, canForecast, meta }: PanelProps) {
  const series = useRemote(sensor ? `${sensor.id}|${date}` : null, seriesCache, (s) => replayApi.series(sensor!.id, date, s));
  const fkey = sensor ? `${sensor.id}|${time}|${partition}` : null;
  const [asked, setAsked] = useState<string | null>(null);
  const want = fkey !== null && (asked === fkey || forecastCache.get(fkey) !== undefined) ? fkey : null;
  const fc = useRemote(want, forecastCache, (s) => replayApi.forecast(sensor!.id, time, partition, s));
  const slotIdx = Number(time.slice(11, 13)) * 12 + Number(time.slice(14, 16)) / 5;
  const value = frameValue !== undefined ? frameValue : series.data?.values[slotIdx];
  const status = frameStatus ?? series.data?.status[slotIdx];

  return (
    <Card>
      <CardHeader
        title={sensor ? `Sensor ${sensor.id}` : "Sensor details"}
        meta={sensor ? [sensor.region, sensor.type_group, sensor.freeway, sensor.county ? `${sensor.county} County` : null, `District ${sensor.district}`].filter(Boolean).join(" · ") : "Choose a sensor on the map or from the list"}
        action={
          <Select aria-label="Choose a sensor" value={sensor?.id ?? ""} onChange={(e) => e.target.value && onSelect(e.target.value)} className="h-9 w-[150px]">
            <option value="">Choose…</option>
            {sensors.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
                {s.lat === null ? " (not mapped)" : ""}
              </option>
            ))}
          </Select>
        }
      />
      {!sensor ? (
        <EmptyState icon={<MapPin className="size-4" aria-hidden />} title="No sensor selected">
          Click a marker to see that sensor's readings for the day and to run a forecast from the selected moment.
        </EmptyState>
      ) : (
        <CardBody className="flex flex-col gap-5">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="tnum text-2xl font-semibold">{value === null || value === undefined ? "—" : num(value, 0)}</span>
            <span className="text-sm text-fg-2">
              {value === null || value === undefined
                ? status
                  ? STATUS_TEXT[status as keyof typeof STATUS_TEXT]
                  : "loading"
                : "vehicles per 5 min"}{" "}
              observed at {hhmm(time)}
            </span>
            {sensor.location_status === "provisional" && <Tag className="ml-auto">Location provisional</Tag>}
          </div>

          <section aria-label={`Readings for ${sensor.id} on ${date}`}>
            <p className="mb-1 text-xs font-medium text-fg-3">Readings on {niceDate(time)}</p>
            {series.error && <ErrorState title="Readings unavailable" message={series.error} onRetry={series.retry} />}
            {series.loading && <Skeleton className="h-[170px]" label="Loading readings" />}
            {series.data && <DayChart s={series.data} time={time} />}
          </section>

          <section aria-label="Forecast" className="flex flex-col gap-3 border-t border-line pt-4">
            <div className="flex flex-wrap items-center gap-2">
              <Segmented
                label="Federated model"
                value={partition}
                onChange={setPartition}
                options={[
                  { value: "geo", label: "Geographic (non-IID)" },
                  { value: "iid", label: "IID" },
                ]}
              />
              <Button variant="primary" disabled={!canForecast || fc.loading} loading={fc.loading} onClick={() => setAsked(fkey)}>
                Forecast next 60 min
              </Button>
            </div>
            {!canForecast && <Note className="text-sm text-fg-2">Forecasts start at {meta.first_forecast_time.replace("T", " ")}; 2022 has observations only.</Note>}
            {fc.error && <ErrorState title="No forecast for this moment" message={fc.error} />}
            {fc.data && <ForecastView f={fc.data} />}
            {!fc.data && !fc.error && !fc.loading && canForecast && (
              <Note>The forecast uses only the 12 readings up to {hhmm(time)}. Readings after {hhmm(time)} are shown afterwards for comparison only.</Note>
            )}
          </section>
        </CardBody>
      )}
    </Card>
  );
}

function DayChart({ s, time }: { s: DaySeries; time: string }) {
  const rows = s.timestamps.map((t, i) => ({ t, observed: s.values[i], typical: s.typical[i] }));
  return (
    <div className="-ml-2 select-none" role="figure" aria-label={`Flow at ${s.sensor} through the day, with the 2023 typical flow for comparison`}>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--color-grid)" vertical={false} />
          <XAxis dataKey="t" {...AXIS} tickFormatter={hhmm} minTickGap={36} />
          <YAxis {...AXIS} width={40} tickFormatter={(v: number) => num(v, 0)} allowDecimals={false} />
          <Tooltip contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-line-strong)", borderRadius: 8, fontSize: 12 }} labelFormatter={(l) => hhmm(String(l))} formatter={(v) => num(Number(v), 0)} isAnimationActive={false} />
          <Line dataKey="typical" name="Typical (2023 median)" stroke="var(--color-ref)" strokeDasharray="3 3" dot={false} strokeWidth={1.2} isAnimationActive={false} />
          <Line dataKey="observed" name="Observed" stroke="var(--color-fg)" dot={false} strokeWidth={1.4} connectNulls={false} isAnimationActive={false} />
          <ReferenceLine x={time} stroke="var(--color-accent)" />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap gap-x-4 text-xs text-fg-3">
        <span>— observed (gaps = missing)</span>
        <span>- - {s.typical_label}</span>
        <span className="text-accent">│ selected time</span>
      </div>
    </div>
  );
}
