import type { ReactNode } from "react";
import { StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, PageHeader, Steps } from "../components/ui";

function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(110px,auto)_1fr] text-sm">
      {rows.map(([k, v], i) => (
        <div key={k} className={`col-span-2 grid grid-cols-subgrid py-2.5 ${i ? "border-t border-line" : ""}`}>
          <dt className="text-fg-3">{k}</dt>
          <dd className="text-fg">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

const PROTOCOL = [
  ["Freeze.", "Every completed checkpoint, the data and the code were hashed."],
  ["Pre-check.", "Each checkpoint reproduced its saved validation predictions byte for byte."],
  ["Test once.", "All 12 runs and three reference forecasts on the held-out period."],
  ["Compare.", "Matched seeds, sensor-level bootstrap within clients, freeway-cluster check, Holm correction."],
];

const LIMITS = [
  ["Provisional geography.", "Client labels come from a sensor-ID mapping that is supported but not confirmed."],
  ["Few ramps.", "10 of 200 sensors: few ramps meet the 99% coverage rule over six months."],
  ["Historical only.", "One dataset, evaluated offline; not a real-time deployment validation."],
  ["No validated future accuracy.", "Forecasts after September 2023 use the same frozen model but have not been checked against observed traffic."],
  ["Simulated federation.", "Communication is counted, not measured on a network."],
  ["One configuration.", "The same training settings for every approach, without separate tuning."],
];

export default function Methodology() {
  return (
    <StudyGate>
      {({ metrics: m }) => (
        <>
          <PageHeader title="Methodology">
            A pre-registered, chronological evaluation. Models were chosen on validation data, frozen and hashed, then tested once on a later period none of them had
            seen.
          </PageHeader>

          <Card>
            <CardHeader title="Data and split" meta="XTraffic 2023 · January–September · 5-minute flow" />
            <CardBody className="flex flex-col gap-4">
              <div className="flex h-16 overflow-hidden rounded-control border border-line text-xs" role="img" aria-label="Train 191 days, validation 28 days, test 54 days">
                <div className="flex flex-col justify-center bg-surface-2 px-3" style={{ flex: 191 }}>
                  <b className="font-semibold text-fg">Train · 191 days</b>
                  <span className="text-fg-3">Jan 1 – Jul 10</span>
                </div>
                <div className="flex min-w-[64px] flex-col justify-center border-l border-line bg-surface-3 px-2" style={{ flex: 28 }}>
                  <b className="font-semibold text-fg">Validation</b>
                  <span className="text-fg-3">28 days</span>
                </div>
                <div className="flex flex-col justify-center border-l border-accent/50 bg-accent/12 px-3" style={{ flex: 54 }}>
                  <b className="font-semibold text-accent">Test · 54 days</b>
                  <span className="text-fg-2">Aug 8 – Sep 30</span>
                </div>
              </div>
              <p className="max-w-[80ch] text-sm leading-relaxed text-fg-2">
                200 sensors, 50 per client group, chosen with training-period statistics only. Each forecast window uses 12 past steps (60 minutes) to predict 12
                future steps, and inputs and targets always stay inside one split. October–December were never opened.
              </p>
            </CardBody>
          </Card>

          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Model" />
              <CardBody className="py-2">
                <KV
                  rows={[
                    ["Architecture", "GRU, 64 hidden units, linear head"],
                    ["Input", "12 steps × 4 features"],
                    ["Features", "scaled flow, observed flag, time-of-day sin/cos"],
                    ["Output", "12 steps at once (60 min)"],
                    ["Scaling", "per sensor, training period only"],
                    ["Missing data", "inputs filled from the past; targets masked"],
                  ]}
                />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Training and selection" />
              <CardBody className="py-2">
                <KV
                  rows={[
                    ["Optimiser", "Adam, learning rate 0.001"],
                    ["Batch", "512, gradient clip 1.0"],
                    ["Budget", "≤ 50 epochs or rounds, patience 5"],
                    ["Runs", "4 approaches × 3 seeds = 12, all completed"],
                    ["Selection", "lowest validation MAE, fixed before test"],
                    ["Test windows", <span className="tnum">{m.test_windows.toLocaleString("en-US")}</span>],
                  ]}
                />
              </CardBody>
            </Card>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Evaluation protocol" />
              <CardBody>
                <Steps items={PROTOCOL} />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Limitations" />
              <CardBody>
                <ul className="flex flex-col gap-2.5">
                  {LIMITS.map(([b, t]) => (
                    <li key={b} className="flex gap-3 text-sm leading-relaxed text-fg-2">
                      <span className="mt-2 size-1 shrink-0 rounded-full bg-fg-3" aria-hidden />
                      <span>
                        <b className="font-semibold text-fg">{b}</b> {t}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardBody>
            </Card>
          </div>
        </>
      )}
    </StudyGate>
  );
}
