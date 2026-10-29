import { CalendarClock, History } from "lucide-react";
import { useEffect } from "react";
import { ForecastChart } from "../components/charts";
import { ResultsTable, StudyGate } from "../components/study";
import { ButtonLink, Card, CardBody, CardHeader, ErrorState, Legend, SectionHeader, Skeleton, Stat, StatStrip } from "../components/ui";
import { api } from "../lib/api";
import { niceDate } from "../lib/clock";
import { useRequest, type StudyData } from "../lib/data";
import { num, pct, signed } from "../lib/format";
import { displayModels } from "../lib/series";
import { SERIES } from "../lib/study";
import type { ModelKey } from "../lib/types";

function ExampleForecast({ data }: { data: StudyData }) {
  const ex = data.demo.examples[0];
  const req = useRequest(api.predict);
  const { run } = req;
  const models: ModelKey[] = data.health.candidate_enabled ? ["candidate"] : ["final"];
  const key = models[0];
  useEffect(() => {
    void run({ sensor: ex.sensor, time: ex.time, models: [key] });
  }, [run, ex.sensor, ex.time, key]);
  const shown = req.data ? displayModels(req.data) : null;
  return (
    <Card>
      <CardHeader
        title={`${ex.label} · ${ex.client_name}`}
        meta={`Sensor ${ex.sensor} · ${niceDate(ex.time)} · historical, held-out period`}
        action={
          <ButtonLink to="/forecast" variant="ghost" className="-mr-2">
            Open in Forecasting
          </ButtonLink>
        }
      />
      <CardBody>
        {req.error ? (
          <ErrorState title="The example forecast could not run." message={req.error} onRetry={() => void run({ sensor: ex.sensor, time: ex.time, models: [key] })} />
        ) : shown ? (
          <>
            <ForecastChart prediction={shown.pred} models={["final"]} showActual height={260} primaryLabel={shown.primary} />
            <Legend
              items={[
                { label: "Observed", color: SERIES.observed.color },
                { label: `${shown.primary} forecast`, color: SERIES.local.color },
                { label: "Observed afterwards", color: SERIES.actual.color, dash: true },
              ]}
            />
          </>
        ) : (
          <Skeleton className="h-[260px]" label="Running the example forecast" />
        )}
      </CardBody>
    </Card>
  );
}

function Insights({ data }: { data: StudyData }) {
  const m = data.metrics;
  const c = m.contrasts;
  const rows = [
    ["Local-only training was most accurate", "Lowest three-seed mean error of the four approaches", num(m.arms.local.mae)],
    ["Cost of geographic federation", "Geographic FedAvg vs centralised training", signed(c.C1.diff)],
    ["Federation vs training alone", "Geographic FedAvg vs local-only", signed(c.C2.diff)],
    ["Effect of the geographic split", "IID FedAvg vs geographic FedAvg", signed(c.C3.diff)],
    ["Practical threshold", "Smallest difference that counts, fixed before the test", num(m.sesoi_mae)],
  ];
  return (
    <Card>
      <CardHeader title="Federated learning findings" meta="Original GRU approaches · MAE, Aug–Sep 2023 held-out test" />
      <ul>
        {rows.map(([t, d, v], i) => (
          <li key={t} className={`flex items-center justify-between gap-4 px-5 py-3.5 ${i ? "border-t border-line" : ""}`}>
            <div className="min-w-0">
              <p className="text-sm font-medium">{t}</p>
              <p className="text-xs text-fg-3">{d}</p>
            </div>
            <span className={`tnum text-md font-semibold ${i === 0 ? "text-accent" : ""}`}>{v}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

export default function Overview() {
  return (
    <StudyGate>
      {(d) => {
        const m = d.metrics;
        const loc = m.arms.local;
        const pe = m.percentage_error;
        const o = m.optimized;
        return (
          <>
            <section className="mb-10 max-w-[760px] motion-safe:animate-rise">
              <h1 tabIndex={-1} className="text-[32px] font-semibold leading-[1.1] tracking-[-0.03em] outline-none sm:text-display">
                Forecast the next hour of freeway traffic with the study's frozen models.
              </h1>
              <p className="mt-5 max-w-[66ch] text-md leading-relaxed text-fg-2">
                AIORI PS-003 trained one compact GRU four ways (centralised, local-only and two federated variants) on nine months of California
                freeway flow. Every forecast here is computed on request by a frozen checkpoint, and every metric comes from a single held-out test.
              </p>
              <div className="mt-7 flex flex-wrap gap-3">
                <ButtonLink to="/forecast" variant="primary">
                  <History className="size-4" aria-hidden />
                  Replay a historical hour
                </ButtonLink>
                <ButtonLink to="/forecast?mode=future">
                  <CalendarClock className="size-4" aria-hidden />
                  Forecast from your readings
                </ButtonLink>
              </div>
            </section>

            <StatStrip className="grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
              {o ? (
                <>
                  <Stat label="Optimized MLP WAPE" value={pct(o.independent.optimized.wape_pct, 3)} note="Independent evaluation · Oct–Dec 2023" accent />
                  <Stat label="Error reduction" value={`${num(o.independent.relative_reduction_pct, 1)}%`} note={`vs original GRU (${pct(o.independent.original_gru.wape_pct, 3)}), same period`} />
                  <Stat label="Validation WAPE" value={pct(o.validation.wape_pct, 3)} note="Optimized MLP · Jul 11 – Aug 7, 2023" />
                  <Stat label="Evaluation windows" value={(o.independent.windows / 1e6).toFixed(2)} unit="million" note={`${m.n_sensors} sensors · Oct–Dec 2023`} />
                </>
              ) : (
                <>
                  <Stat label="Original GRU MAE" value={num(loc.clipped_mae, 2)} unit="veh / 5 min" note="Local-only GRU, clipped at 0, Aug–Sep test" accent />
                  <Stat label="WAPE" value={pe ? pct(pe.target.final_model_wape_pct, 2) : "—"} note="Original GRU, Aug–Sep 2023 test" />
                  <Stat label="Error vs persistence" value={`−${pct(loc.improvement_over_baselines_pct.persistence)}`} note="Lower MAE than the naive forecast" />
                  <Stat label="Held-out windows" value={(m.test_windows / 1e6).toFixed(2)} unit="million" note={`${m.n_sensors} sensors · 8 Aug – 30 Sep 2023`} />
                </>
              )}
            </StatStrip>

            <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
              <ExampleForecast data={d} />
              <Insights data={d} />
            </div>

            <SectionHeader title="Federated vs Independent Training" meta="Original GRU approaches · Aug 8 – Sep 30, 2023 test · lower is better" />
            <Card>
              <ResultsTable m={m} />
            </Card>
          </>
        );
      }}
    </StudyGate>
  );
}
