import { ArrowDownRight, ArrowUpRight, ChevronLeft, ChevronRight, History, RotateCcw, SearchX, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router";
import { Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ForecastView, hhmm } from "../components/replay/ForecastView";
import { Button, ButtonLink, Card, CardBody, CardHeader, EmptyState, ErrorState, Field, Input, Note, Num, NumTh, PageHeader, Segmented, Select, Skeleton, Stat, StatStrip, TableWrap, Tag } from "../components/ui";
import { niceDate } from "../lib/clock";
import {
  CATEGORY_LABEL,
  compareRows,
  DEFAULT_FILTERS,
  disruptionsApi,
  duration,
  type Category,
  type DisruptionMeta,
  type EventBrief,
  type EventCompare,
  type EventDetail,
  type EventList,
  type EventSeries,
  type Filters,
  listParams,
  pct,
  replayHref,
  type Robustness,
  seriesRows,
} from "../lib/disruptions";
import { modelName, num, publicSource } from "../lib/format";
import { EVENT_ID_RE, Lru, replayApi, type Forecast, type Partition, type ReplaySensor } from "../lib/replay";
import { useRemote } from "../lib/useRemote";

const listCache = new Lru<EventList>(60);
const eventCache = new Lru<EventDetail>(60);
const seriesCache = new Lru<EventSeries>(60);
const compareCache = new Lru<EventCompare>(30);
const robustCache = new Lru<Robustness>(60);
const forecastCache = new Lru<Forecast>(100);
const TOOLTIP = { contentStyle: { background: "var(--color-surface)", border: "1px solid var(--color-line-strong)", borderRadius: 8, fontSize: 12 }, isAnimationActive: false } as const;
const LINE_COLORS = ["var(--color-fg)", "var(--color-accent)", "var(--color-observed)", "var(--color-below)", "var(--color-danger)"];
const stamp = (t: string) => `${t.slice(5, 10)} ${hhmm(t)}`;

export default function Disruptions() {
  const [boot, setBoot] = useState<{ meta: DisruptionMeta; sensors: ReplaySensor[] } | { error: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const ac = new AbortController();
    setBoot(null);
    Promise.all([disruptionsApi.meta(ac.signal), replayApi.sensors(ac.signal)]).then(
      ([meta, sensors]) => setBoot({ meta, sensors }),
      (e: Error) => e.name !== "AbortError" && setBoot({ error: e.message }),
    );
    return () => ac.abort();
  }, [attempt]);

  return (
    <>
      <PageHeader title="Historical Disruption Explorer">
        Browse unusual traffic-flow periods found in the recorded 2022–2024 XTraffic readings, compare them with each sensor's typical 2023 flow, replay them on the
        map and see how the frozen federated model (MLP + FedAvgM) forecast through them. Every entry is a candidate anomaly found after the fact; its cause is not known.
      </PageHeader>
      {boot === null && <Skeleton className="h-[640px]" label="Loading the event catalog" />}
      {boot && "error" in boot && <ErrorState title="The disruption catalog could not be loaded" message={boot.error} onRetry={() => setAttempt((a) => a + 1)} />}
      {boot && "meta" in boot && <Explorer meta={boot.meta} sensors={boot.sensors} />}
    </>
  );
}

function Explorer({ meta, sensors }: { meta: DisruptionMeta; sensors: ReplaySensor[] }) {
  const [params, setParams] = useSearchParams();
  const linked = params.get("event");
  const eventId = linked && EVENT_ID_RE.test(linked) ? linked : null;
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const key = JSON.stringify(listParams(filters));
  const list = useRemote(key, listCache, (s) => disruptionsApi.list(filters, s), 150);
  const shown = list.data ?? (list.loading ? list.last : undefined);
  const set = (patch: Partial<Filters>) => setFilters((f) => ({ ...f, ...patch, page: "page" in patch ? patch.page! : 1 }));
  const open = (id: string | null) =>
    setParams(
      (p) => {
        const q = new URLSearchParams(p);
        if (id) q.set("event", id);
        else q.delete("event");
        return q;
      },
      { replace: false },
    );

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2" role="note">
        <Tag tone="observed">
          <History className="size-3.5" aria-hidden />
          Retrospective · not live
        </Tag>
        <Tag>Candidate anomalies · causes not verified</Tag>
        <Tag>Pacific local time (inferred)</Tag>
        <Tag>Detector {meta.version}</Tag>
      </div>

      <Overview meta={meta} />

      <Card>
        <CardHeader
          title="Find candidate events"
          meta={shown ? `${num(shown.total, 0)} matching event${shown.total === 1 ? "" : "s"}` : "Loading…"}
          action={
            <Button variant="ghost" onClick={() => setFilters(DEFAULT_FILTERS)}>
              <RotateCcw className="size-3.5" aria-hidden />
              Reset filters
            </Button>
          }
        />
        <CardBody className="flex flex-col gap-4">
          <FilterBar meta={meta} sensors={sensors} f={filters} set={set} />
          {list.error && <ErrorState title="Events could not be loaded" message={list.error} onRetry={list.retry} />}
          {!shown && !list.error && <Skeleton className="h-[320px]" label="Loading events" />}
          {shown && shown.total === 0 && (
            <EmptyState icon={<SearchX className="size-4" aria-hidden />} title="No candidate events match these filters">
              Widen the date range, lower the minimum sensors or magnitude, or include sensor-quality anomalies.
            </EmptyState>
          )}
          {shown && shown.total > 0 && <EventTable list={shown} selected={eventId} onOpen={open} busy={list.loading} />}
          {shown && shown.pages > 1 && (
            <nav className="flex items-center justify-end gap-2 text-sm text-fg-2" aria-label="Event pages">
              <Button variant="ghost" disabled={filters.page <= 1} onClick={() => set({ page: filters.page - 1 })} aria-label="Previous page">
                <ChevronLeft className="size-4" aria-hidden />
              </Button>
              <span className="tnum" aria-live="polite">
                Page {shown.page} of {num(shown.pages, 0)}
              </span>
              <Button variant="ghost" disabled={filters.page >= shown.pages} onClick={() => set({ page: filters.page + 1 })} aria-label="Next page">
                <ChevronRight className="size-4" aria-hidden />
              </Button>
            </nav>
          )}
        </CardBody>
      </Card>

      {eventId ? <EventPanel key={eventId} id={eventId} meta={meta} onClose={() => open(null)} /> : <Note>Choose an event in the list to see its readings, its affected sensors and the model's forecasts around it.</Note>}

      <AggregateCard meta={meta} />
      <AboutCard meta={meta} />
    </div>
  );
}

/* ------------------------------------------------------------------ overview */

function Overview({ meta }: { meta: DisruptionMeta }) {
  const s = meta.summary;
  const q = s.by_category.sensor_quality ?? 0;
  return (
    <Card>
      <CardHeader title="Catalog overview" meta={`${meta.provenance.first.slice(0, 10)} to ${meta.provenance.last.slice(0, 10)} · ${meta.provenance.n_sensors} sensors · built ${meta.created_utc.slice(0, 10)}`} />
      <CardBody className="flex flex-col gap-4">
        <StatStrip>
          <Stat label="Candidate events" value={num(s.n_events, 0)} note={`${num(s.n_events - q, 0)} without dominant quality issues`} accent />
          <Stat label="Sensor-quality anomalies" value={num(q, 0)} note="traffic change and sensor fault indistinguishable" />
          <Stat label="Sensors with events" value={`${s.sensors_with_events} / ${meta.provenance.n_sensors}`} />
          <Stat label="Median duration" value={duration(s.duration_min.p50)} note={`90% under ${duration(s.duration_min.p90)}`} />
        </StatStrip>
        <div className="grid gap-4 text-sm md:grid-cols-3">
          <Breakdown title="By type" rows={Object.entries(s.by_category).map(([k, v]) => [CATEGORY_LABEL[k as Category] ?? k, v])} />
          <Breakdown title="By year" rows={Object.entries(s.by_year)} />
          <Breakdown title="By region (dataset groups)" rows={Object.entries(s.by_region).map(([k, v]) => [meta.regions[k as keyof typeof meta.regions] ?? k, v])} />
        </div>
        <Note>
          A reading is anomalous when it is at least {Math.round(meta.thresholds.min_relative * 100)}% away from the sensor's typical 2023 flow for that time and
          day type and beyond z = {meta.thresholds.z_low.toFixed(2)} / +{meta.thresholds.z_high.toFixed(2)} (thresholds calibrated on 2023 only). Events need 15
          minutes of such readings. Multi-sensor groups: {meta.audit.chance_coordination.observed} found, about {Math.round(meta.audit.chance_coordination.shifted_mean)} expected
          by chance alone.
        </Note>
      </CardBody>
    </Card>
  );
}

function Breakdown({ title, rows }: { title: string; rows: [string, number][] }) {
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-fg-3">{title}</p>
      <dl className="flex flex-col gap-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 border-b border-line/60 pb-1">
            <dt className="text-fg-2">{k}</dt>
            <dd className="tnum">{num(v, 0)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/* ------------------------------------------------------------------ filters and list */

function FilterBar({ meta, sensors, f, set }: { meta: DisruptionMeta; sensors: ReplaySensor[]; f: Filters; set: (p: Partial<Filters>) => void }) {
  const min = meta.provenance.first.slice(0, 10);
  const max = meta.provenance.last.slice(0, 10);
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
      <Field label="From">{(id) => <Input id={id} type="date" min={min} max={max} value={f.from} onChange={(e) => set({ from: e.target.value })} />}</Field>
      <Field label="To">{(id) => <Input id={id} type="date" min={min} max={max} value={f.to} onChange={(e) => set({ to: e.target.value })} />}</Field>
      <Field label="Type">
        {(id) => (
          <Select id={id} value={f.category} onChange={(e) => set({ category: e.target.value })}>
            <option value="">All types</option>
            {Object.entries(CATEGORY_LABEL).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Region (dataset group)">
        {(id) => (
          <Select id={id} value={f.region} onChange={(e) => set({ region: e.target.value })}>
            <option value="">All regions</option>
            {Object.entries(meta.regions).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Sensor">
        {(id) => (
          <Select id={id} value={f.sensor} onChange={(e) => set({ sensor: e.target.value })}>
            <option value="">All sensors</option>
            {sensors.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Direction">
        {(id) => (
          <Select id={id} value={f.direction} onChange={(e) => set({ direction: e.target.value })}>
            <option value="">Both</option>
            <option value="decrease">Below typical</option>
            <option value="increase">Above typical</option>
          </Select>
        )}
      </Field>
      <Field label="Minimum sensors">
        {(id) => (
          <Select id={id} value={f.min_sensors} onChange={(e) => set({ min_sensors: e.target.value })}>
            <option value="">Any</option>
            {[2, 3, 5, 10, 20].map((n) => (
              <option key={n} value={n}>
                {n} or more
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Minimum duration">
        {(id) => (
          <Select id={id} value={f.min_duration} onChange={(e) => set({ min_duration: e.target.value })}>
            <option value="">Any</option>
            {[30, 60, 120, 360, 1440].map((n) => (
              <option key={n} value={n}>
                {duration(n)}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Minimum |z| at peak">
        {(id) => (
          <Select id={id} value={f.min_peak_z} onChange={(e) => set({ min_peak_z: e.target.value })}>
            <option value="">Any</option>
            {[8, 10, 15, 20, 30].map((n) => (
              <option key={n} value={n}>
                {n} or more
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label="Confidence in readings">
        {(id) => (
          <Select id={id} value={f.confidence} onChange={(e) => set({ confidence: e.target.value })}>
            <option value="">Any</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </Select>
        )}
      </Field>
      <Field label="Holidays">
        {(id) => (
          <Select id={id} value={f.holiday} onChange={(e) => set({ holiday: e.target.value as Filters["holiday"] })}>
            <option value="include">Include</option>
            <option value="exclude">Exclude</option>
            <option value="only">Only holidays</option>
          </Select>
        )}
      </Field>
      <Field label="Sort by">
        {(id) => (
          <Select id={id} value={f.sort} onChange={(e) => set({ sort: e.target.value })}>
            <option value="-score">Largest score</option>
            <option value="-peak">Largest peak |z|</option>
            <option value="-duration">Longest</option>
            <option value="-sensors">Most sensors</option>
            <option value="start">Oldest first</option>
            <option value="-start">Newest first</option>
          </Select>
        )}
      </Field>
      <div className="sm:col-span-2 lg:col-span-4 xl:col-span-6">
        <Segmented
          label="Sensor-quality anomalies"
          value={f.quality}
          onChange={(v) => set({ quality: v })}
          className="max-w-[520px]"
          options={[
            { value: "exclude", label: "Hide quality anomalies" },
            { value: "include", label: "Show all" },
            { value: "only", label: "Only quality anomalies" },
          ]}
        />
      </div>
    </div>
  );
}

function CategoryTag({ c }: { c: Category }) {
  return <Tag tone={c === "sensor_quality" ? "danger" : c === "coordinated_deviation" ? "accent" : "neutral"}>{CATEGORY_LABEL[c]}</Tag>;
}

function Direction({ e }: { e: Pick<EventBrief, "direction" | "peak_relative"> }) {
  const Icon = e.direction === "decrease" ? ArrowDownRight : ArrowUpRight;
  return (
    <span className="inline-flex items-center gap-1">
      <Icon className={e.direction === "decrease" ? "size-3.5 text-below" : "size-3.5 text-observed"} aria-hidden />
      {e.direction === "decrease" ? "Below" : "Above"} typical ({pct(e.peak_relative)})
    </span>
  );
}

function EventTable({ list, selected, onOpen, busy }: { list: EventList; selected: string | null; onOpen: (id: string) => void; busy: boolean }) {
  return (
    <div className={busy ? "opacity-60 transition-opacity" : undefined} aria-busy={busy}>
      <TableWrap label="Candidate events">
        <thead>
          <tr>
            <th>Candidate event</th>
            <th>Start</th>
            <th>Type</th>
            <th>Deviation at peak</th>
            <NumTh>Duration</NumTh>
            <NumTh>Sensors</NumTh>
            <NumTh>Peak |z|</NumTh>
            <NumTh>Score</NumTh>
            <th>Region</th>
            <th>Readings</th>
          </tr>
        </thead>
        <tbody>
          {list.items.map((e) => (
            <tr key={e.id} className={e.id === selected ? "bg-surface-3/60" : undefined}>
              <td>
                <button type="button" className="tnum whitespace-nowrap text-left text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-accent" onClick={() => onOpen(e.id)} aria-current={e.id === selected ? "true" : undefined}>
                  {e.id}
                </button>
              </td>
              <td className="tnum whitespace-nowrap">{e.start.replace("T", " ")}</td>
              <td>
                <CategoryTag c={e.category} />
              </td>
              <td className="whitespace-nowrap">
                <Direction e={e} />
              </td>
              <Num>{duration(e.duration_min)}</Num>
              <Num>{e.n_sensors}</Num>
              <Num>{num(Math.abs(e.peak_z), 1)}</Num>
              <Num>{num(e.score, 1)}</Num>
              <td>{e.region_code}</td>
              <td className="whitespace-nowrap">
                {e.confidence === "high" ? "Clean" : e.confidence === "medium" ? "Some flags" : "Quality issues"}
                {e.holiday ? <span className="text-fg-3"> · holiday</span> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </TableWrap>
    </div>
  );
}

/* ------------------------------------------------------------------ event detail */

function EventPanel({ id, meta, onClose }: { id: string; meta: DisruptionMeta; onClose: () => void }) {
  const ev = useRemote(id, eventCache, (s) => disruptionsApi.event(id, s));
  const [sensor, setSensor] = useState<string | null>(null);
  if (ev.error)
    return (
      <Card>
        <CardHeader title={`Candidate event ${id}`} action={<Button variant="ghost" onClick={onClose}>Close</Button>} />
        <CardBody>
          <ErrorState title="This event could not be loaded" message={ev.error} onRetry={ev.retry} />
        </CardBody>
      </Card>
    );
  if (!ev.data) return <Skeleton className="h-[520px]" label="Loading event" />;
  const e = ev.data;
  const sid = sensor ?? e.peak_sensor;
  return (
    <div className="flex flex-col gap-5" aria-label={`Candidate event ${e.id}`}>
      <Card>
        <CardHeader
          title={
            <span className="flex flex-wrap items-center gap-2">
              <span className="tnum">Candidate event {e.id}</span>
              <CategoryTag c={e.category} />
            </span>
          }
          meta={`${niceDate(e.start)} · ${e.start.slice(11)} to ${e.end.slice(0, 10) === e.start.slice(0, 10) ? e.end.slice(11) : e.end.replace("T", " ")} · ${e.region} · ${e.period_label}`}
          action={
            <div className="flex flex-wrap gap-2">
              <ButtonLink variant="primary" to={replayHref(e, sid)}>
                <History className="size-4" aria-hidden />
                Open in Historical Traffic Replay
              </ButtonLink>
              <Button variant="ghost" onClick={onClose}>
                Close
              </Button>
            </div>
          }
        />
        <CardBody className="flex flex-col gap-4">
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3 lg:grid-cols-6">
            <Fact k="Duration" v={duration(e.duration_min)} />
            <Fact k="Affected sensors" v={String(e.n_sensors)} />
            <Fact k="Anomalous readings" v={num(e.n_anomalous, 0)} />
            <Fact k="Peak deviation" v={`${pct(e.peak_relative)} (z ${num(e.peak_z, 1)})`} />
            <Fact k="Peak" v={`${e.peak_sensor} at ${hhmm(e.peak_time)}`} />
            <Fact k="Score" v={num(e.score, 1)} />
          </dl>
          <p className="text-sm leading-relaxed text-fg-2">
            <b className="font-medium text-fg">Why it was flagged. </b>
            {e.why_flagged} {e.category_text}.
          </p>
          <div className="flex flex-wrap gap-2 text-xs">
            <Tag>{e.day_type === "weekend_like" ? "Weekend-like day" : "Weekday-like day"}</Tag>
            {e.holiday && <Tag tone="observed">Holiday: {e.holiday}</Tag>}
            <Tag tone={e.confidence === "high" ? "accent" : e.confidence === "low" ? "danger" : "neutral"}>Reading confidence: {e.confidence}</Tag>
            <Tag>{Math.round(e.missing_share * 100)}% of readings in the interval missing or invalid</Tag>
          </div>
          {e.quality_text.length > 0 && (
            <Note className="text-fg-2">
              <TriangleAlert className="mr-1 inline size-3.5 text-observed" aria-hidden />
              Quality flags on {e.quality_share > 0 && e.quality_share < 0.01 ? "under 1" : Math.round(e.quality_share * 100)}% of the anomalous readings: {e.quality_text.join("; ")}.
            </Note>
          )}
          <Note>
            {e.detection_note} {meta.score_note}
          </Note>
        </CardBody>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <SeriesCard e={e} sensor={sid} />
        <SensorsCard e={e} sensor={sid} onSelect={setSensor} />
      </div>
      {e.n_sensors > 1 && <CompareCard e={e} />}
      <ForecastCard e={e} sensor={sid} />
    </div>
  );
}

const Fact = ({ k, v }: { k: string; v: string }) => (
  <div>
    <dt className="text-xs text-fg-3">{k}</dt>
    <dd className="tnum font-medium">{v}</dd>
  </div>
);

function SeriesCard({ e, sensor }: { e: EventDetail; sensor: string }) {
  const s = useRemote(`${e.id}|${sensor}`, seriesCache, (sig) => disruptionsApi.series(e.id, sensor, sig));
  const multiDay = s.data ? s.data.timestamps[0].slice(0, 10) !== s.data.timestamps[s.data.timestamps.length - 1].slice(0, 10) : false;
  const fmt = multiDay ? stamp : hhmm;
  const missing = s.data ? s.data.status.filter((x) => x !== "ok").length : 0;
  return (
    <Card>
      <CardHeader title={`Observed and typical flow · ${sensor}`} meta={s.data ? `${s.data.window.start.replace("T", " ")} to ${s.data.window.end.replace("T", " ")} · ${s.data.units}` : "Loading…"} />
      <CardBody className="flex flex-col gap-3">
        {s.error && <ErrorState title="Readings unavailable" message={s.error} onRetry={s.retry} />}
        {!s.data && !s.error && <Skeleton className="h-[260px]" label="Loading readings" />}
        {s.data && (
          <>
            <div className="-ml-2 select-none" role="figure" aria-label={`Flow at ${sensor} around the event, with the typical 2023 flow and the detection thresholds`}>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={seriesRows(s.data)} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--color-grid)" vertical={false} />
                  <XAxis dataKey="t" {...AXIS} tickFormatter={fmt} minTickGap={40} />
                  <YAxis {...AXIS} width={44} tickFormatter={(v: number) => num(v, 0)} allowDecimals={false} />
                  <Tooltip {...TOOLTIP} labelFormatter={(l) => String(l).replace("T", " ")} formatter={(v, n) => (Array.isArray(v) ? `${num(Number(v[0]), 0)}–${num(Number(v[1]), 0)}` : n === "Missing or invalid" ? "no reading" : num(Number(v), 0))} />
                  <ReferenceArea x1={s.data.sensor_interval.start} x2={s.data.sensor_interval.end} fill="var(--color-observed)" fillOpacity={0.08} stroke="none" ifOverflow="hidden" />
                  <Area dataKey="band" name="Detection range" stroke="none" fill="var(--color-ref)" fillOpacity={0.18} isAnimationActive={false} connectNulls={false} />
                  <Line dataKey="expected" name="Typical 2023 flow" stroke="var(--color-fg-3)" strokeDasharray="4 3" dot={false} strokeWidth={1.3} isAnimationActive={false} />
                  <Line dataKey="observed" name="Observed" stroke="var(--color-fg)" dot={false} strokeWidth={1.5} connectNulls={false} isAnimationActive={false} />
                  <Line dataKey="anomaly" name="Anomalous reading" stroke="none" dot={{ r: 2.4, fill: "var(--color-observed)", stroke: "none" }} activeDot={false} isAnimationActive={false} />
                  <Line dataKey="missing" name="Missing or invalid" stroke="none" dot={{ r: 2, fill: "var(--color-danger)", stroke: "none" }} activeDot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-2">
                <span>— observed (gaps = missing)</span>
                <span className="text-fg-3">- - typical 2023 flow (reference, not a measurement)</span>
                <span className="text-fg-3">▒ detection range</span>
                <span className="text-observed">● anomalous reading · shaded: this sensor's anomaly</span>
                {missing > 0 && <span className="text-danger-ink">● {missing} missing or invalid</span>}
              </div>
            </div>
            <Note>
              {s.data.threshold_label}. {s.data.unscored_note}
              {s.data.window.truncated ? " This episode is longer than three days; the chart shows three days around this sensor's peak." : ""}
              {s.data.dst_nonexistent.length ? " The window includes the clock hour skipped by daylight saving, which has no readings." : ""}
            </Note>
          </>
        )}
      </CardBody>
    </Card>
  );
}

function SensorsCard({ e, sensor, onSelect }: { e: EventDetail; sensor: string; onSelect: (s: string) => void }) {
  const rows = [...e.sensor_details].sort((a, b) => Math.abs(b.peak_z) - Math.abs(a.peak_z));
  return (
    <Card>
      <CardHeader title="Affected sensors" meta={`${e.n_sensors} sensor${e.n_sensors > 1 ? "s" : ""} · choose one to chart, replay or forecast`} />
      <div className="max-h-[360px] overflow-auto">
        <TableWrap label="Affected sensors">
          <thead>
            <tr>
              <th>Sensor</th>
              <th>Anomaly</th>
              <NumTh>Peak obs. / typical</NumTh>
              <NumTh>|z|</NumTh>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.sensor} className={s.sensor === sensor ? "bg-surface-3/60" : undefined}>
                <td>
                  <button type="button" className="text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-accent" aria-pressed={s.sensor === sensor} onClick={() => onSelect(s.sensor)}>
                    {s.sensor}
                  </button>
                </td>
                <td className="tnum whitespace-nowrap">
                  {hhmm(s.start)}–{s.end.slice(0, 10) === s.start.slice(0, 10) ? hhmm(s.end) : stamp(s.end)}
                </td>
                <Num>
                  {num(s.peak_observed, 0)} / {num(s.peak_expected, 0)}
                </Num>
                <Num>{num(Math.abs(s.peak_z), 1)}</Num>
                <td className="text-xs text-fg-3">{s.quality_flags.length ? s.quality_flags.join(", ").replaceAll("_", " ") : "—"}</td>
              </tr>
            ))}
          </tbody>
        </TableWrap>
      </div>
      <CardBody>
        <Note>Grouping is by the study's dataset region and timing only. It does not show that these sensors share a road or a cause; locations are provisional.</Note>
      </CardBody>
    </Card>
  );
}

function CompareCard({ e }: { e: EventDetail }) {
  const c = useRemote(e.id, compareCache, (s) => disruptionsApi.compare(e.id, s));
  const [chosen, setChosen] = useState<string[] | null>(null);
  const pick = chosen ?? c.data?.sensors.slice(0, 4).map((s) => s.sensor) ?? [];
  const toggle = (s: string) => setChosen(pick.includes(s) ? pick.filter((x) => x !== s) : pick.length >= LINE_COLORS.length ? pick : [...pick, s]);
  const multiDay = c.data ? c.data.timestamps[0].slice(0, 10) !== c.data.timestamps[c.data.timestamps.length - 1].slice(0, 10) : false;
  return (
    <Card>
      <CardHeader title="Compare affected sensors" meta={c.data ? `Showing up to ${LINE_COLORS.length} of ${c.data.shown} strongest sensors${c.data.total_sensors > c.data.shown ? ` (event has ${c.data.total_sensors})` : ""}` : "Loading…"} />
      <CardBody className="flex flex-col gap-3">
        {c.error && <ErrorState title="Comparison unavailable" message={c.error} onRetry={c.retry} />}
        {!c.data && !c.error && <Skeleton className="h-[220px]" label="Loading comparison" />}
        {c.data && (
          <>
            <fieldset className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
              <legend className="sr-only">Sensors to compare</legend>
              {c.data.sensors.map((s) => (
                <label key={s.sensor} className="flex items-center gap-1.5">
                  <input type="checkbox" checked={pick.includes(s.sensor)} onChange={() => toggle(s.sensor)} disabled={!pick.includes(s.sensor) && pick.length >= LINE_COLORS.length} className="accent-[var(--color-accent)]" />
                  <span style={{ color: pick.includes(s.sensor) ? LINE_COLORS[pick.indexOf(s.sensor)] : undefined }}>{s.sensor}</span>
                  <span className="text-xs text-fg-3">{Math.round(s.available * 100)}% available</span>
                </label>
              ))}
            </fieldset>
            <div className="-ml-2 select-none" role="figure" aria-label="Flow divided by typical flow for the chosen sensors">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={compareRows(c.data, pick)} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--color-grid)" vertical={false} />
                  <XAxis dataKey="t" {...AXIS} tickFormatter={multiDay ? stamp : hhmm} minTickGap={40} />
                  <YAxis {...AXIS} width={40} tickFormatter={(v: number) => `${num(v, 1)}×`} />
                  <Tooltip {...TOOLTIP} labelFormatter={(l) => String(l).replace("T", " ")} formatter={(v) => `${num(Number(v), 2)}× typical`} />
                  <ReferenceLine y={1} stroke="var(--color-ref)" strokeDasharray="4 3" />
                  {pick.map((s, i) => (
                    <Line key={s} dataKey={s} stroke={LINE_COLORS[i]} dot={false} strokeWidth={1.4} connectNulls={false} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <Note>{c.data.normalization}</Note>
          </>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ forecasts around the event */

function ForecastCard({ e, sensor }: { e: EventDetail; sensor: string }) {
  const [partition, setPartition] = useState<Partition>("geo");
  const [issue, setIssue] = useState<"onset" | "peak">("onset");
  const time = issue === "onset" ? e.forecast.issue_at_onset : e.forecast.issue_at_peak;
  const fkey = time ? `${sensor}|${time}|${partition}` : null;
  const [asked, setAsked] = useState<string | null>(null);
  const fc = useRemote(fkey && (asked === fkey || forecastCache.get(fkey)) ? fkey : null, forecastCache, (s) => replayApi.forecast(sensor, time!, partition, s));
  const rkey = `${e.id}|${sensor}|${partition}`;
  const [askedR, setAskedR] = useState<string | null>(null);
  const rb = useRemote(askedR === rkey || robustCache.get(rkey) ? rkey : null, robustCache, (s) => disruptionsApi.robustness(e.id, sensor, partition, s));

  return (
    <Card>
      <CardHeader title={`Model forecasts around this event · ${sensor}`} meta="Frozen MLP + FedAvgM (history A), the model served by Historical Traffic Replay; nothing is retrained" />
      <CardBody className="flex flex-col gap-5">
        {!e.forecast.available ? (
          <Note className="text-sm text-fg-2">{e.forecast.reason}</Note>
        ) : (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <Segmented
                label="Federated model"
                value={partition}
                onChange={setPartition}
                options={[
                  { value: "geo", label: "Geographic (non-IID)" },
                  { value: "iid", label: "IID" },
                ]}
              />
              <Segmented
                label="Forecast issued"
                value={issue}
                onChange={setIssue}
                options={[
                  { value: "onset", label: `Just before onset (${hhmm(e.forecast.issue_at_onset!)})` },
                  { value: "peak", label: `Before the peak (${hhmm(e.forecast.issue_at_peak!)})` },
                ]}
              />
              <Button variant="primary" loading={fc.loading} disabled={fc.loading} onClick={() => setAsked(fkey)}>
                Forecast next 60 min
              </Button>
              <ButtonLink variant="ghost" to={replayHref(e, sensor, time!)}>
                Open this moment in Replay
              </ButtonLink>
            </div>
            {fc.error && <ErrorState title="No forecast for this moment" message={fc.error} />}
            {fc.data && <ForecastView f={fc.data} />}
            {!fc.data && !fc.error && !fc.loading && (
              <Note>
                The forecast for {sensor} is issued at {time!.replace("T", " ")} from its 12 readings up to then. The event's onset is relative to the event as a whole;
                this sensor's own anomaly may start later.
              </Note>
            )}
          </>
        )}

        <section aria-label="Forecast errors around the event" className="flex flex-col gap-3 border-t border-line pt-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium">Forecast errors inside this anomaly vs. ordinary days</p>
              <p className="text-xs text-fg-3">Exploratory and retrospective. Not a test score.</p>
            </div>
            <Button variant="secondary" loading={rb.loading} disabled={rb.loading} onClick={() => setAskedR(rkey)}>
              Compare forecast errors
            </Button>
          </div>
          {rb.error && <ErrorState title="Comparison unavailable" message={rb.error} />}
          {rb.data && <RobustnessView r={rb.data} />}
        </section>
      </CardBody>
    </Card>
  );
}

function RobustnessView({ r }: { r: Robustness }) {
  if (!r.available) return <Note className="text-sm text-fg-2">{r.reason}</Note>;
  const ev = r.event_window!;
  const ref = r.reference!;
  const rows = ev.mae_by_horizon.map((v, i) => ({ h: `+${(i + 1) * 5}`, event: v, reference: ref.mae_by_horizon[i] }));
  const cell = (v: number | null, unit = "") => (v === null ? "—" : `${num(v, 1)}${unit}`);
  return (
    <div className="flex flex-col gap-3">
      {r.warnings!.map((w) => (
        <Note key={w} className="text-fg-2">
          {w}
        </Note>
      ))}
      <TableWrap label="Forecast errors">
        <thead>
          <tr>
            <th>Window</th>
            <NumTh>Forecast–target pairs</NumTh>
            <NumTh>Missing targets</NumTh>
            <NumTh>MAE</NumTh>
            <NumTh>WAPE</NumTh>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Inside this anomaly ({hhmm(r.anomaly_interval!.start)}–{hhmm(r.anomaly_interval!.end)})</td>
            <Num>{ev.n}</Num>
            <Num>{ev.missing_targets}</Num>
            <Num>{ev.n ? cell(ev.mae) : "Comparison unavailable"}</Num>
            <Num>{cell(ev.wape_pct, "%")}</Num>
          </tr>
          <tr>
            <td>Same times on {ref.days.length ? ref.days.join(", ") : "no reference day"}</td>
            <Num>{ref.n}</Num>
            <Num>{ref.missing_targets}</Num>
            <Num>{ref.n ? cell(ref.mae) : "Comparison unavailable"}</Num>
            <Num>{cell(ref.wape_pct, "%")}</Num>
          </tr>
        </tbody>
      </TableWrap>
      {ev.n > 0 && ref.n > 0 && (
        <div className="-ml-2 select-none" role="figure" aria-label="Mean absolute error by forecast lead time, inside the anomaly and on reference days">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-grid)" vertical={false} />
              <XAxis dataKey="h" {...AXIS} />
              <YAxis {...AXIS} width={40} tickFormatter={(v: number) => num(v, 0)} />
              <Tooltip {...TOOLTIP} labelFormatter={(l) => `${l} min ahead`} formatter={(v) => `${num(Number(v), 1)} vehicles`} />
              <Line dataKey="event" name="Inside anomaly" stroke="var(--color-observed)" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
              <Line dataKey="reference" name="Reference days" stroke="var(--color-fg-3)" strokeDasharray="4 3" strokeWidth={1.5} dot={{ r: 2 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-1 flex gap-4 text-xs text-fg-2">
            <span className="text-observed">— MAE inside the anomaly, by lead time (minutes)</span>
            <span className="text-fg-3">- - reference days</span>
          </div>
        </div>
      )}
      <Note>
        {r.issues!.count} forecasts issued every {r.issues!.stride_minutes} min from {r.issues!.first.replace("T", " ")} to {r.issues!.last.replace("T", " ")} ({modelName(r.model!.model_id)}). {r.method} {r.caution}
      </Note>
    </div>
  );
}

/* ------------------------------------------------------------------ catalog-wide comparison and limitations */

const PERIOD_LABEL: Record<string, string> = { train: "2023 (training year)", val: "Jan–Jun 2024 (validation)", test: "Jul–Dec 2024 (final test period)" };

function AggregateCard({ meta }: { meta: DisruptionMeta }) {
  const s = meta.robustness_summary;
  if (!s) return null;
  const rows = s.rows.filter((r) => r.group === "all" || r.group === "decrease" || r.group === "increase");
  return (
    <Card>
      <CardHeader title="How the model behaved across all candidate events" meta={`${num(s.events_evaluated, 0)} events from 2023 onward · ${modelName(s.model)}`} />
      <CardBody className="flex flex-col gap-3">
        <TableWrap label="Pooled forecast errors inside candidate events and on matched ordinary days">
          <thead>
            <tr>
              <th>Events</th>
              <th>Period</th>
              <NumTh>Events</NumTh>
              <NumTh>MAE inside</NumTh>
              <NumTh>WAPE inside</NumTh>
              <NumTh>MAE reference</NumTh>
              <NumTh>WAPE reference</NumTh>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.group}|${r.period}`}>
                <td>{r.group === "all" ? "All (non-quality)" : r.group === "decrease" ? "Below typical" : "Above typical"}</td>
                <td>{PERIOD_LABEL[r.period] ?? r.period}</td>
                <Num>{num(r.events, 0)}</Num>
                <Num>{r.event_mae === null ? "—" : num(r.event_mae, 1)}</Num>
                <Num>{r.event_wape_pct === null ? "—" : `${num(r.event_wape_pct, 1)}%`}</Num>
                <Num>{r.reference_mae === null ? "—" : num(r.reference_mae, 1)}</Num>
                <Num>{r.reference_wape_pct === null ? "—" : `${num(r.reference_wape_pct, 1)}%`}</Num>
              </tr>
            ))}
          </tbody>
        </TableWrap>
        <Note>
          {s.method} {s.caution} The study's frozen result for this configuration is a three-seed aggregate test WAPE of 10.5%; the numbers here are a different, selected
          sample.
        </Note>
      </CardBody>
    </Card>
  );
}

function AboutCard({ meta }: { meta: DisruptionMeta }) {
  return (
    <Card>
      <CardHeader title="What this explorer can and cannot tell you" />
      <CardBody className="grid gap-x-8 gap-y-4 text-sm leading-relaxed text-fg-2 md:grid-cols-2">
        <div>
          <p className="mb-1 font-medium text-fg">What it shows</p>
          <ul className="list-disc space-y-1 pl-5">
            <li>Genuine historical flow readings that departed from each sensor's typical 2023 pattern.</li>
            <li>When each unusual period started and ended, how large it was and which sensors were involved.</li>
            <li>Several sensors in the same dataset region deviating at overlapping times.</li>
            <li>How the frozen model's forecasts compared with what was later observed.</li>
            <li>Missing readings, invalid readings and sensor-quality warnings.</li>
          </ul>
        </div>
        <div>
          <p className="mb-1 font-medium text-fg">What it does not claim</p>
          <ul className="list-disc space-y-1 pl-5">
            <li>Earthquake, flood, accident or road-damage detection.</li>
            <li>Evacuation routing, traffic simulation or live incident monitoring.</li>
            <li>Validated forecasting accuracy during disasters.</li>
            <li>Congestion levels: the data used here is flow only.</li>
          </ul>
        </div>
        <p className="font-medium text-fg md:col-span-2">{meta.limitations[0]}</p>
        <ul className="list-disc space-y-1 pl-5 md:col-span-2">
          {meta.limitations.slice(1).map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className="md:col-span-2 text-xs text-fg-3">
          Baseline: {meta.baseline.method}. Thresholds: {meta.thresholds.calibration}. {publicSource(meta.provenance.source)}. {meta.timezone}. Catalog events SHA-256 {meta.events_sha256.slice(0, 12)}…
        </p>
      </CardBody>
    </Card>
  );
}
