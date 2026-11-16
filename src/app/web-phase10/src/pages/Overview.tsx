import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";
import { ForecastChart } from "../components/charts";
import { ResultsTable, StudyGate } from "../components/study";
import { ButtonLink, Card, CardBody, CardHeader, ErrorState, Legend, Note, SectionHeader, Skeleton, Stat, StatStrip } from "../components/ui";
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

const REPLAY = import.meta.env.MODE === "replay";

type Extended = { events: number | null; models: { partition_label: string; partition: string; wape: number; architecture: string; algorithm: string }[] };

/** Figures for the AIORI 2.0 features, read from the replay service (replay builds only; absent in a production build). */
function useExtended(): Extended | null {
  const [x, setX] = useState<Extended | null>(null);
  useEffect(() => {
    if (!REPLAY) return;
    const ac = new AbortController();
    Promise.all([import("../lib/replay"), import("../lib/disruptions")])
      .then(([r, d]) => Promise.allSettled([r.replayApi.meta(ac.signal), d.disruptionsApi.meta(ac.signal)]))
      .then(([meta, dis]) => {
        if (ac.signal.aborted) return;
        setX({
          events: dis.status === "fulfilled" ? dis.value.summary.n_events : null,
          models:
            meta.status === "fulfilled"
              ? meta.value.models.map((m) => ({ partition_label: m.partition_label, partition: m.partition, wape: m.test_wape_pct_three_seed_mean, architecture: m.architecture, algorithm: m.algorithm }))
              : [],
        });
      })
      .catch(() => undefined);
    return () => ac.abort();
  }, []);
  return x;
}

/** The page's way in, drawn as a freeway distance sign: where you can go, and one real figure for each destination. */
function Destinations({ sensors, events }: { sensors: number; events: number | null }) {
  const rows: [string, string, string][] = [
    ...(REPLAY
      ? ([
          ["/replay", "Replay a day on the map", `${sensors} sensors · 2022–2024`],
          ["/disruptions", "Investigate unusual periods", events === null ? "candidate events" : `${num(events, 0)} candidate events`],
        ] as [string, string, string][])
      : []),
    ["/forecast", "Forecast a recorded hour", "12 steps · 60 min"],
    ["/forecast?mode=future", "Forecast from your own readings", "12 readings in"],
    ["/federated", "Compare training approaches", "4 approaches"],
    ["/performance", "Check the evidence", "held-out tests"],
  ];
  return (
    <nav aria-label="Where to start" className="rounded-[12px] bg-sign p-[4px] shadow-float">
      <ul className="rounded-[8px] border-2 border-white/90 px-4 py-1.5 text-white sm:px-5">
        {rows.map(([to, label, figure], i) => (
          <li key={to} className={i ? "border-t border-white/25" : undefined}>
            <Link to={to} className="group flex items-center gap-3 py-3 focus-visible:outline-white">
              <span className="min-w-0 flex-1 text-[17px] font-bold leading-snug">{label}</span>
              <span className="tnum hidden whitespace-nowrap text-sm text-white/80 min-[420px]:inline">{figure}</span>
              <ArrowRight className="size-5 shrink-0 transition-transform duration-200 ease-[var(--ease-out-expo)] group-hover:translate-x-1" aria-hidden />
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function SelectedModel({ x }: { x: Extended }) {
  if (!x.models.length) return null;
  return (
    <>
      <SectionHeader title="The selected federated model" meta="XTraffic 2022–2024 · final test Jul–Dec 2024 · lower is better" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <StatStrip className="grid-cols-1 sm:grid-cols-2">
          {x.models.map((m) => (
            <Stat key={m.partition} label={`Test WAPE · ${m.partition === "iid" ? "IID clients" : "geographic clients"}`} value={pct(m.wape, 3)} note="Mean of three seeds, all sensors" accent={m.partition === "iid"} />
          ))}
        </StatStrip>
        <div className="flex flex-col justify-center gap-2">
          <p className="text-md font-semibold">
            {x.models[0].architecture.split(",")[0]} trained with {x.models[0].algorithm.split(" (")[0]}
          </p>
          <p className="text-sm leading-relaxed text-fg-2">
            Chosen on validation data from five architectures and eight federated algorithms, then tested once. It is the model behind every forecast in Replay and
            the Disruption Explorer.
          </p>
          <Note>
            These are averages over the whole test period. They are not the expected error of any single forecast, and they say nothing about accuracy during
            unusual events.
          </Note>
        </div>
      </div>
    </>
  );
}

export default function Overview() {
  const x = useExtended();
  return (
    <StudyGate>
      {(d) => {
        const m = d.metrics;
        const loc = m.arms.local;
        const pe = m.percentage_error;
        const o = m.optimized;
        return (
          <>
            <section className="mb-4 grid items-center gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:gap-12">
              <div>
                <h1 tabIndex={-1} className="text-[34px] font-extrabold leading-[1.06] outline-none sm:text-[46px]">
                  Freeway traffic, replayed and forecast an hour ahead.
                </h1>
                <p className="mt-5 max-w-[60ch] text-md leading-relaxed text-fg-2">
                  AIORI studies whether traffic forecasting models can be trained across regions without pooling their data. It runs on recorded flow from {m.n_sensors}{" "}
                  California freeway sensors: step through past days, look into unusual periods, or ask a frozen model for the next 60 minutes and see what happened
                  afterwards.
                </p>
                <p className="mt-3 max-w-[60ch] text-sm text-fg-3">Every reading is historical and every forecast is computed on request by a frozen checkpoint.</p>
              </div>
              <Destinations sensors={m.n_sensors} events={x?.events ?? null} />
            </section>

            {x && <SelectedModel x={x} />}

            <SectionHeader title="The original PS-003 study" meta="One compact GRU trained four ways · XTraffic Jan–Sep 2023" />
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
