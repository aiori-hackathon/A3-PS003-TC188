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

/* Where the signal head sits: centred in the 48px gap between the text column (1.15fr) and the action panel (1fr), inside the
   section's 40px side padding. */
const SIGNAL_X = "(40px + (100% - 128px) * 1.15 / 2.15 + 24px)";
const LENS = [
  { y: 31, base: "#3a0d0a", tint: "#6b1712" }, // red, unlit
  { y: 67, base: "#22200a", tint: "#4a4416" }, // yellow, unlit: dark olive like a real amber lens by day
  { y: 103, lit: true }, // green, lit
] as const;

/** A traffic signal over the hero photo's road (desktop and tablet only), drawn after a real US signal: galvanized mast arm
    from the right edge, rigid bracket, wide matte charcoal backplate, three separate black 12" sections bolted down the left
    edge, deep tunnel visors shading the top of each lens, LED-dot lenses with the green lit. The photo itself is unedited. */
function HeroSignal() {
  return (
    <div className="hero-signal absolute inset-x-0 top-[20px] hidden lg:block">
      {/* Mast arm: tapers toward its tip, end cap just past the head. */}
      <div
        className="absolute right-0 h-[9px] shadow-[0_3px_4px_rgba(0,0,0,0.35)] [clip-path:polygon(0_22%,100%_0,100%_100%,0_78%)]"
        style={{ left: `calc(${SIGNAL_X} - 22px)`, background: "linear-gradient(180deg,#c9cfcc 0%,#9aa29e 38%,#6f7773 75%,#545b58 100%)" }}
      />
      <svg viewBox="0 0 54 132" className="absolute top-[4px] w-[38px] -translate-x-1/2 drop-shadow-[0_4px_6px_rgba(0,0,0,0.45)]" style={{ left: `calc(${SIGNAL_X})` }}>
        <defs>
          <linearGradient id="sig-metal" x1="0" x2="1">
            <stop offset="0" stopColor="#5e6662" />
            <stop offset="0.45" stopColor="#b9c0bc" />
            <stop offset="1" stopColor="#59605c" />
          </linearGradient>
          {/* Matte charcoal backplate, a touch lighter toward the top where the sky light falls. */}
          <linearGradient id="sig-plate" x1="0" y1="0" x2="0.35" y2="1">
            <stop offset="0" stopColor="#3b3f3e" />
            <stop offset="1" stopColor="#262928" />
          </linearGradient>
          <linearGradient id="sig-box" x1="0" x2="1">
            <stop offset="0" stopColor="#1c1f1e" />
            <stop offset="0.55" stopColor="#141615" />
            <stop offset="1" stopColor="#0c0d0d" />
          </linearGradient>
          <radialGradient id="sig-green" cx="0.45" cy="0.55" r="0.62">
            <stop offset="0" stopColor="#b8ffe6" />
            <stop offset="0.45" stopColor="#38eab4" />
            <stop offset="1" stopColor="#00a77c" />
          </radialGradient>
          <radialGradient id="sig-glow">
            <stop offset="0" stopColor="#3ff0bb" stopOpacity="0.5" />
            <stop offset="1" stopColor="#3ff0bb" stopOpacity="0" />
          </radialGradient>
          {/* The tunnel visor's shadow over the upper part of each lens. */}
          <linearGradient id="sig-shade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#000" stopOpacity="0.8" />
            <stop offset="0.42" stopColor="#000" stopOpacity="0.25" />
            <stop offset="0.6" stopColor="#000" stopOpacity="0" />
          </linearGradient>
          <pattern id="sig-led" width="1.6" height="1.6" patternUnits="userSpaceOnUse">
            <circle cx="0.8" cy="0.8" r="0.42" fill="#000" opacity="0.3" />
          </pattern>
          <clipPath id="sig-lens">
            <circle cx="0" cy="0" r="11.5" />
          </clipPath>
        </defs>
        {/* Rigid bracket from the arm. */}
        <rect x="24" y="0" width="6" height="12" fill="url(#sig-metal)" />
        <rect x="21" y="1" width="12" height="3.5" rx="1" fill="#7d8581" />
        <rect x="1" y="9" width="52" height="122" rx="3" fill="url(#sig-plate)" />
        {LENS.map((l) => (
          <g key={l.y}>
            {/* One bolted section per lamp. */}
            <rect x="11" y={l.y - 17} width="32" height="34" rx="1.6" fill="url(#sig-box)" stroke="#070808" strokeWidth="0.6" />
            {[l.y - 12, l.y + 12].map((by) => (
              <circle key={by} cx="12.6" cy={by} r="0.75" fill="#555b59" />
            ))}
            <g transform={`translate(27 ${l.y})`}>
              <g clipPath="url(#sig-lens)">
                {"lit" in l ? (
                  <circle r="11.5" fill="url(#sig-green)" />
                ) : (
                  <>
                    <circle r="11.5" fill={l.base} />
                    {/* Daylight reflection on the unlit lens. */}
                    <ellipse cx="-2" cy="4" rx="9" ry="6" fill={l.tint} opacity="0.75" />
                  </>
                )}
                <circle r="11.5" fill="url(#sig-led)" />
                <rect x="-11.5" y="-11.5" width="23" height="23" fill="url(#sig-shade)" />
              </g>
              {"lit" in l && <circle r="20" fill="url(#sig-glow)" />}
              {/* Tunnel visor rim: thick at the top, thinning toward the open bottom. */}
              <path d="M-13.4 3A13.4 13.4 0 0 1 13.4 3L11.6 1A11.6 11.6 0 0 0 -11.6 1Z" fill="#080909" />
              <circle r="11.9" fill="none" stroke="#050606" strokeWidth="0.8" />
            </g>
          </g>
        ))}
      </svg>
    </div>
  );
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
            <section className="relative isolate mb-4 grid items-center gap-8 overflow-hidden rounded-[14px] border border-line bg-raised px-5 py-8 sm:px-8 sm:py-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:gap-12 lg:px-10">
              {/* Backdrop: a palm-lined Los Angeles street (public domain, see public/hero/). The scrim keeps the text side calm; the
                  road's vanishing point falls in the gap between the text and the panel, where a traffic signal hangs on desktop. */}
              <div aria-hidden className="pointer-events-none absolute inset-0 -z-10">
                <img src="/hero/la-palm-street.webp" alt="" width={2048} height={1296} decoding="async" fetchPriority="high" className="hero-photo absolute inset-0 size-full object-cover" />
                <div className="hero-scrim absolute inset-0" />
                <HeroSignal />
              </div>
              <div className="hero-copy">
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
