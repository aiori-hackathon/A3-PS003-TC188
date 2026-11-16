import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { modelName, num } from "../../lib/format";
import { forecastRows, STATUS_TEXT, type Forecast } from "../../lib/replay";
import { Note, Num, NumTh, TableWrap } from "../ui";

export const AXIS = { stroke: "var(--color-line-strong)", tickLine: false, tick: { fill: "var(--color-fg-3)", fontSize: 12 } } as const;
export const hhmm = (t: string) => t.slice(11, 16);

/** Forecast from the frozen model with the inputs it saw and the readings that followed (shared by Replay and the Disruption Explorer). */
export function ForecastView({ f }: { f: Forecast }) {
  const rows = forecastRows(f);
  const shortSha = f.model.checkpoint_sha256.slice(0, 12);
  return (
    <div className="flex flex-col gap-3">
      <div className="-ml-2 select-none" role="figure" aria-label={`Forecast for ${f.sensor} issued at ${hhmm(f.issued_at)}, with later observations for comparison`}>
        <ResponsiveContainer width="100%" height={230}>
          <ComposedChart data={rows} margin={{ top: 18, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--color-grid)" vertical={false} />
            <XAxis dataKey="t" {...AXIS} tickFormatter={hhmm} minTickGap={24} />
            <YAxis {...AXIS} width={40} tickFormatter={(v: number) => num(v, 0)} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-line-strong)", borderRadius: 8, fontSize: 12 }} labelFormatter={(l) => hhmm(String(l))} formatter={(v) => num(Number(v), 1)} isAnimationActive={false} />
            <ReferenceLine x={f.issued_at} stroke="var(--color-cursor)" strokeDasharray="3 3" label={{ value: "Issued", position: "insideTopRight", fill: "var(--color-accent)", fontSize: 12, dy: -16 }} />
            <Line dataKey="observed" name="Observed input" stroke="var(--color-fg)" strokeWidth={1.6} dot={{ r: 2.2, fill: "var(--color-surface)" }} connectNulls={false} isAnimationActive={false} />
            <Line dataKey="filled" name="Filled input (missing reading)" stroke="none" dot={{ r: 3, fill: "var(--color-surface)", stroke: "var(--color-fg-3)", strokeDasharray: "1 1" }} isAnimationActive={false} />
            <Line dataKey="actual" name="Observed afterwards" stroke="var(--color-observed)" strokeDasharray="4 4" strokeWidth={1.5} dot={{ r: 2 }} connectNulls={false} isAnimationActive={false} />
            <Line dataKey="forecast" name="Forecast" stroke="var(--color-accent)" strokeWidth={2.4} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="mt-1 flex flex-wrap gap-x-4 text-xs text-fg-2">
          <span>— observed input</span>
          <span className="text-accent">— forecast</span>
          <span className="text-observed">- - observed afterwards</span>
          {f.input.imputed.some(Boolean) && <span className="text-fg-3">○ filled input</span>}
        </div>
      </div>

      {f.warnings.map((w) => (
        <Note key={w} className="text-fg-2">
          {w}
        </Note>
      ))}

      <dl className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <dt className="text-xs text-fg-3">MAE, this hour</dt>
          <dd className="tnum font-medium">{f.comparison.mae === null ? "—" : num(f.comparison.mae, 1)}</dd>
        </div>
        <div>
          <dt className="text-xs text-fg-3">WAPE, this hour</dt>
          <dd className="tnum font-medium">{f.comparison.wape_pct === null ? "—" : `${num(f.comparison.wape_pct, 1)}%`}</dd>
        </div>
        <div>
          <dt className="text-xs text-fg-3">Steps compared</dt>
          <dd className="tnum font-medium">{f.comparison.n_compared} / 12</dd>
        </div>
      </dl>
      <Note>One sensor and one hour: these errors are not comparable with the study's aggregate WAPE. {f.comparison.note}</Note>

      <details className="rounded-control border border-line">
        <summary className="cursor-pointer px-3 py-2 text-sm text-fg-2">Step-by-step comparison</summary>
        <TableWrap label="Forecast compared with later observations">
          <thead>
            <tr>
              <th>Time</th>
              <NumTh>Forecast</NumTh>
              <NumTh>Observed</NumTh>
              <NumTh>Abs. error</NumTh>
            </tr>
          </thead>
          <tbody>
            {f.forecast.values.map((v, i) => (
              <tr key={i}>
                <td className="tnum">{f.forecast.timestamps[i] ? hhmm(f.forecast.timestamps[i]!) : `+${(i + 1) * 5} min`}</td>
                <Num>{num(v, 1)}</Num>
                <Num className={f.actual.values[i] === null ? "text-fg-3" : undefined}>{f.actual.values[i] === null ? STATUS_TEXT[f.actual.status[i]] : num(f.actual.values[i]!, 0)}</Num>
                <Num>{f.comparison.abs_error[i] === null ? "—" : num(f.comparison.abs_error[i]!, 1)}</Num>
              </tr>
            ))}
          </tbody>
        </TableWrap>
      </details>

      <div className="rounded-control border border-line bg-surface-2/40 p-3 text-xs leading-relaxed text-fg-2">
        <p className="font-medium text-fg">
          {f.model.architecture} · {f.model.algorithm}
        </p>
        <p>
          {f.model.partition_label} · {f.model.training_history} · seed {f.model.seed} ({f.model.seed_policy}) · checkpoint epoch {f.model.checkpoint_epoch}, SHA-256 {shortSha}…
        </p>
        <p>
          {f.period_label}. Study result for this configuration: test WAPE {num(f.model.test_wape_pct_three_seed_mean, 3)}% (mean of three seeds). {f.model.performance_note}
        </p>
        <p className="text-fg-3">
          Model {modelName(f.model.model_id)} · inference {num(f.inference_ms, 1)} ms on {f.device.toUpperCase()} · {f.timezone}
        </p>
      </div>
    </div>
  );
}
