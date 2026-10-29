import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { hhmm, niceDate } from "../../lib/clock";
import { num } from "../../lib/format";
import { forecastRows } from "../../lib/series";
import { SERIES } from "../../lib/study";
import type { ModelKey, Prediction } from "../../lib/types";

const AXIS = { stroke: "#353535", tickLine: false, tick: { fill: "#8a8a8a", fontSize: 12 } } as const;
const GRID = <CartesianGrid stroke="#262626" vertical={false} />;

type Item = { name?: string | number; value?: unknown; color?: string; dataKey?: unknown };

/** Tooltip in the app's palette: a title row, then one line per visible series. */
function Tip({ active, payload, label, title, digits = 1 }: Partial<TooltipContentProps<number, string>> & { title: (l: string) => string; digits?: number }) {
  const items = ((payload ?? []) as readonly Item[]).filter((p) => typeof p.value === "number");
  if (!active || !items.length) return null;
  return (
    <div className="min-w-[180px] rounded-[8px] border border-line-strong bg-surface-2/95 px-3 py-2 text-xs shadow-[0_8px_24px_rgba(0,0,0,0.45)] backdrop-blur">
      <p className="mb-1.5 font-medium text-fg">{title(String(label))}</p>
      {items.map((p) => (
        <p key={String(p.dataKey)} className="flex items-center justify-between gap-4 py-0.5 text-fg-2">
          <span className="flex items-center gap-2">
            <span className="inline-block size-2 rounded-full" style={{ background: p.color }} aria-hidden />
            {p.name}
          </span>
          <b className="tnum font-medium text-fg">{num(p.value as number, digits)}</b>
        </p>
      ))}
    </div>
  );
}

/** Observed hour, then the 12-step forecast; the shaded band marks what the model produced. */
export function ForecastChart({ prediction, models, showActual, height = 320, primaryLabel = "Original GRU (local-only)", compareLabel = "Centralised GRU" }: { prediction: Prediction; models: ModelKey[]; showActual: boolean; height?: number; primaryLabel?: string; compareLabel?: string }) {
  const rows = useMemo(() => forecastRows(prediction, { models, showActual }), [prediction, models, showActual]);
  const boundary = prediction.history.timestamps[11];
  const end = prediction.forecast_timestamps[11];
  const label = `Traffic flow at sensor ${prediction.sensor.id}: observed hour from ${hhmm(prediction.history.timestamps[0])} and forecast to ${hhmm(end)}.`;
  return (
    <div role="figure" aria-label={label} className="-ml-2 select-none">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={rows} margin={{ top: 22, right: 8, bottom: 0, left: 0 }}>
          {GRID}
          <ReferenceArea x1={boundary} x2={end} fill="#3bb27e" fillOpacity={0.045} ifOverflow="visible" />
          <ReferenceLine
            x={boundary}
            stroke="#454545"
            strokeDasharray="3 3"
            label={{ value: "Forecast", position: "insideTopLeft", fill: "#3bb27e", fontSize: 12, dx: 6, dy: -20 }}
          />
          <XAxis dataKey="t" {...AXIS} tickFormatter={hhmm} minTickGap={24} tickMargin={8} />
          <YAxis {...AXIS} width={46} tickFormatter={(v: number) => num(v, 0)} domain={[0, "auto"]} allowDecimals={false} />
          <Tooltip cursor={{ stroke: "#454545" }} content={<Tip title={(l) => `${hhmm(l)} · ${niceDate(l)}`} />} isAnimationActive={false} />
          <Line dataKey="observed" name="Observed" stroke={SERIES.observed.color} strokeWidth={SERIES.observed.width} dot={{ r: 2.4, fill: "#0e0e0e", strokeWidth: 1.4 }} activeDot={{ r: 3.5 }} connectNulls={false} isAnimationActive={false} />
          {showActual && prediction.actual && (
            <Line dataKey="actual" name="Observed afterwards" stroke={SERIES.actual.color} strokeWidth={SERIES.actual.width} strokeDasharray={SERIES.actual.dash} dot={false} isAnimationActive={false} />
          )}
          {models.includes("centralised") && prediction.forecasts.centralised && (
            <Line dataKey="centralised" name={compareLabel} stroke={SERIES.centralised.color} strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          )}
          <Line dataKey="final" name={primaryLabel} stroke={SERIES.local.color} strokeWidth={SERIES.local.width} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface LineSeries {
  key: string;
  name: string;
  values: number[];
  color: string;
  dash?: string;
  width: number;
}

/** Error at each 5-minute step ahead, for every approach. */
export function HorizonChart({ series, height = 260 }: { series: LineSeries[]; height?: number }) {
  const rows = useMemo(
    () => Array.from({ length: 12 }, (_, i) => Object.fromEntries([["h", `${(i + 1) * 5} min`], ...series.map((s) => [s.key, s.values[i]])])),
    [series],
  );
  return (
    <div role="figure" aria-label="Mean absolute error by forecast horizon, 5 to 60 minutes ahead" className="-ml-2">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          {GRID}
          <XAxis dataKey="h" {...AXIS} minTickGap={16} tickMargin={8} />
          <YAxis {...AXIS} width={40} tickFormatter={(v: number) => num(v, 0)} domain={[0, "auto"]} />
          <Tooltip cursor={{ stroke: "#454545" }} content={<Tip title={(l) => `${l} ahead · MAE`} digits={2} />} isAnimationActive={false} />
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width} strokeDasharray={s.dash} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface BarDatum {
  label: string;
  value: number;
  kind: "final" | "arm" | "ref";
}

const BAR_FILL = { final: "#3bb27e", arm: "#5c5c5c", ref: "#333333" } as const;

/** Horizontal bars, lower is better. The highlighted model is the only coloured bar. */
export function BarCompare({ data, digits = 3, suffix = "", label }: { data: BarDatum[]; digits?: number; suffix?: string; label: string }) {
  const max = Math.max(...data.map((d) => d.value));
  return (
    <div role="figure" aria-label={`${label}: ${data.map((d) => `${d.label} ${num(d.value, digits)}${suffix}`).join(", ")}`}>
      <ResponsiveContainer width="100%" height={data.length * 34 + 8}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 0 }} barCategoryGap={9}>
          <XAxis type="number" hide domain={[0, max * 1.02]} />
          <YAxis type="category" dataKey="label" width={136} axisLine={false} tickLine={false} tick={{ fill: "#a3a3a3", fontSize: 12 }} interval={0} />
          <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.label} fill={BAR_FILL[d.kind]} />
            ))}
            <LabelList dataKey="value" position="right" formatter={(v: unknown) => `${num(Number(v), digits)}${suffix}`} className="tnum" fill="#ededed" fontSize={12} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
