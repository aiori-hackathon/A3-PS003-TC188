import { TrendingDown } from "lucide-react";
import type { ReactNode } from "react";
import { useStudy, type StudyData } from "../lib/data";
import { num, pct } from "../lib/format";
import { ARMS, BASELINE_LABEL, BASELINES, CLIENT_NAME, CLIENTS, SERIES } from "../lib/study";
import type { Metrics, Optimized, Sensor } from "../lib/types";
import { ErrorState, Num, NumTh, Skeleton, Swatch, TableWrap, Tag } from "./ui";

/** Renders the page once the verified study data has loaded; skeleton while loading, a retryable error otherwise. */
export function StudyGate({ children, skeleton = "h-[520px]" }: { children: (d: StudyData) => ReactNode; skeleton?: string }) {
  const { state, retry } = useStudy();
  if (state.status === "ready") return <>{children(state.data)}</>;
  if (state.status === "error") return <ErrorState title="The model service is not available." message={state.message} onRetry={retry} />;
  return (
    <div className="flex flex-col gap-4" aria-busy>
      <Skeleton className="h-10 w-2/3 max-w-[520px]" label="Loading verified results" />
      <Skeleton className={skeleton} />
    </div>
  );
}

/** All four approaches and the reference forecasts on the held-out test; three-seed means. */
export function ResultsTable({ m }: { m: Metrics }) {
  const pe = m.percentage_error;
  return (
    <TableWrap label="Held-out test results">
      <caption className="sr-only">Held-out test results, mean of three seeds; lower is better</caption>
      <thead>
        <tr>
          <th scope="col">Model, mean of 3 seeds</th>
          <NumTh scope="col">MAE</NumTh>
          <NumTh scope="col">RMSE</NumTh>
          <NumTh scope="col">WAPE</NumTh>
          <NumTh scope="col">sMAPE</NumTh>
          <NumTh scope="col">vs persistence</NumTh>
        </tr>
      </thead>
      <tbody>
        {ARMS.map((k) => {
          const a = m.arms[k];
          return (
            <tr key={k} className={k === "local" ? "bg-accent/[0.04]" : undefined}>
              <th scope="row" className="!bg-transparent !text-sm !font-normal !text-fg">
                <span className="flex items-center gap-2.5">
                  <Swatch color={SERIES[k].color} dash={!!SERIES[k].dash} />
                  {a.label}
                  {k === "local" && <Tag>Original GRU</Tag>}
                </span>
              </th>
              <Num>{num(a.mae)}</Num>
              <Num>{num(a.rmse)}</Num>
              <Num>{pe ? pct(pe.arms[k].raw.wape_pct, 2) : "—"}</Num>
              <Num>{pe ? pct(pe.arms[k].raw.smape_pct, 2) : "—"}</Num>
              <Num className="text-accent">−{pct(a.improvement_over_baselines_pct.persistence)}</Num>
            </tr>
          );
        })}
        {BASELINES.map((k, i) => (
          <tr key={k} className={i === 0 ? "[&>*]:!border-line-strong" : undefined}>
            <th scope="row" className="!bg-transparent !text-sm !font-normal !text-fg-2">
              <span className="flex items-center gap-2.5">
                <Swatch color={SERIES[k].color} dash={!!SERIES[k].dash} />
                {BASELINE_LABEL[k]}
              </span>
            </th>
            <Num className="text-fg-2">{num(m.baselines[k].mae)}</Num>
            <Num className="text-fg-2">{num(m.baselines[k].rmse)}</Num>
            <Num className="text-fg-2">{pe ? pct(pe.baselines[k].wape_pct, 2) : "—"}</Num>
            <Num className="text-fg-3">n/a</Num>
            <Num className="text-fg-3">—</Num>
          </tr>
        ))}
      </tbody>
    </TableWrap>
  );
}

/** FedAvg topology: one server exchanging weights with four clients; raw data stays with each client. */
export function FedAvgDiagram({ sensors }: { sensors: Sensor[] }) {
  const counts = CLIENTS.map((c) => sensors.filter((s) => s.client === c).length);
  const label = "FedAvg: a server exchanges model weights with four clients; raw data stays with each client";
  const server = (cx: number, y: number, w: number) => (
    <g>
      <rect x={cx - w / 2} y={y} width={w} height={56} rx={10} fill="#0f1f18" stroke="#3bb27e" strokeOpacity={0.55} />
      <text x={cx} y={y + 24} textAnchor="middle" fill="#ededed" fontSize={14} fontWeight={600}>
        Server
      </text>
      <text x={cx} y={y + 43} textAnchor="middle" fill="#a3a3a3" fontSize={12}>
        averages weights · never sees data
      </text>
    </g>
  );
  const client = (x: number, y: number, w: number, i: number) => (
    <g key={i}>
      <rect x={x - w / 2} y={y} width={w} height={62} rx={10} fill="#1f1f1f" stroke="#353535" />
      <text x={x} y={y + 25} textAnchor="middle" fill="#ededed" fontSize={13} fontWeight={600}>
        {CLIENT_NAME[CLIENTS[i]]}
      </text>
      <text x={x} y={y + 44} textAnchor="middle" fill="#8a8a8a" fontSize={12}>
        {counts[i]} sensors · data stays here
      </text>
    </g>
  );
  const link = (x1: number, y1: number, x2: number, y2: number, i: number) => {
    const my = (y1 + y2) / 2;
    return (
      <g key={`l${i}`}>
        <path d={`M${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`} fill="none" stroke="#353535" strokeWidth={1.4} />
        <path d={`M${x2} ${y2} C ${x2} ${my}, ${x1} ${my}, ${x1} ${y1}`} fill="none" stroke="#3bb27e" strokeWidth={1.6} strokeDasharray="4 10" className="motion-safe:animate-[flow_1.6s_linear_infinite]" />
      </g>
    );
  };
  const xs = [95, 285, 475, 665];
  const pos: [number, number][] = [
    [105, 150],
    [315, 150],
    [105, 240],
    [315, 240],
  ];
  return (
    <>
      <style>{"@keyframes flow{to{stroke-dashoffset:-28}}"}</style>
      <svg className="hidden w-full md:block" viewBox="0 0 760 250" role="img" aria-label={label}>
        {xs.map((x, i) => link(380, 74, x, 166, i))}
        {server(380, 18, 240)}
        {xs.map((x, i) => client(x, 166, 164, i))}
      </svg>
      <svg className="mx-auto block w-full max-w-[460px] md:hidden" viewBox="0 0 420 312" role="img" aria-label={label}>
        {pos.map(([x, y], i) => link(210, 70, x, y, i))}
        {server(210, 14, 250)}
        {pos.map(([x, y], i) => client(x, y, 196, i))}
      </svg>
    </>
  );
}

/** Headline comparison on the independent October-December 2023 period (identical windows and cells for every model). */
export function ForecastingPerformance({ o }: { o: Optimized }) {
  const ind = o.independent;
  const rows = [
    { label: "Optimized MLP", value: ind.optimized.wape_pct, tone: "accent" },
    { label: "Original GRU", value: ind.original_gru.wape_pct, tone: "mid" },
    { label: "Naive persistence", value: ind.persistence?.wape_pct ?? null, tone: "low" },
  ];
  const max = Math.max(...rows.map((r) => r.value ?? 0)) * 1.06;
  return (
    <section className="overflow-hidden rounded-card border border-accent/25 bg-surface bg-[linear-gradient(180deg,rgba(59,178,126,0.07),rgba(59,178,126,0)_60%)]">
      <div className="grid gap-8 p-6 sm:p-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] lg:items-center">
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/35 bg-accent/12 px-2.5 py-1 text-xs font-medium text-accent">
            <TrendingDown className="size-3.5" aria-hidden />
            Independent evaluation · Oct–Dec 2023
          </span>
          <h3 className="mt-4 text-2xl font-semibold tracking-tight sm:text-[32px] sm:leading-tight">{num(ind.relative_reduction_pct, 1)}% Lower Forecasting Error</h3>
          <p className="mt-3 max-w-[52ch] text-md leading-relaxed text-fg-2">
            Optimized temporal forecasting outperforms the original GRU on independent evaluation data.
          </p>
          <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-control border border-line bg-line text-sm">
            <div className="bg-surface px-4 py-3">
              <dt className="text-xs text-fg-3">vs original GRU</dt>
              <dd className="tnum text-xl font-semibold text-accent">−{num(ind.relative_reduction_pct, 1)}%</dd>
            </div>
            <div className="bg-surface px-4 py-3">
              <dt className="text-xs text-fg-3">vs naive persistence</dt>
              <dd className="tnum text-xl font-semibold text-accent">
                {ind.relative_reduction_vs_persistence_pct != null ? `−${num(ind.relative_reduction_vs_persistence_pct, 1)}%` : "not available"}
              </dd>
            </div>
          </dl>
          <p className="mt-3 text-xs text-fg-3">
            Relative WAPE reduction. The optimized MLP is trained independently for each client; it is not a federated model.
          </p>
        </div>
        <div className="flex flex-col gap-5">
          <h4 className="text-sm font-semibold text-fg">Performance Against Naive Forecasting</h4>
          <div
            className="flex flex-col gap-4"
            role="img"
            aria-label={`WAPE on October to December 2023: ${rows.map((r) => `${r.label} ${r.value == null ? "not available" : `${num(r.value, 3)} percent`}`).join(", ")}`}
          >
            {rows.map((r) => (
              <div key={r.label}>
                <div className="mb-2 flex items-baseline justify-between gap-4">
                  <span className={r.tone === "accent" ? "text-sm font-medium text-fg" : "text-sm text-fg-2"}>{r.label}</span>
                  <span className={`tnum text-lg font-semibold ${r.tone === "accent" ? "text-accent" : "text-fg-2"}`}>
                    {r.value == null ? "not available" : `${num(r.value, 3)}%`}
                    {r.value != null && <span className="ml-1.5 text-xs font-normal text-fg-3">WAPE</span>}
                  </span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-surface-3">
                  {r.value != null && (
                    <div
                      className={`h-full rounded-full ${r.tone === "accent" ? "bg-accent" : r.tone === "mid" ? "bg-g2" : "bg-g3"}`}
                      style={{ width: `${(100 * r.value) / max}%` }}
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs leading-relaxed text-fg-3">
            Weighted absolute percentage error, lower is better. All three use the same {(ind.windows / 1e6).toFixed(2)} million windows and observed cells,
            scored once after the optimized model was chosen on validation data.
          </p>
        </div>
      </div>
    </section>
  );
}
