import type { ReactNode } from "react";
import { StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, PageHeader, SectionHeader, Steps } from "../components/ui";
import { TEST_TARGETS, VAL_TARGETS } from "../lib/federated3";

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

function Bullets({ items }: { items: string[][] }) {
  return (
    <ul className="flex flex-col gap-2.5">
      {items.map(([b, t]) => (
        <li key={b} className="flex gap-3 text-sm leading-relaxed text-fg-2">
          <span className="mt-2 size-1 shrink-0 rounded-full bg-fg-3" aria-hidden />
          <span>
            <b className="font-semibold text-fg">{b}</b> {t}
          </span>
        </li>
      ))}
    </ul>
  );
}

const STAGES = [
  ["Stage 1, architectures.", "5 models × 2 histories × 4 approaches (centralised, local-only, FedAvg IID, FedAvg geographic): 40 runs, 10 epochs, one seed."],
  ["Stage 2, algorithms.", "MLP and N-HiTS × 2 histories × 2 client splits × 8 algorithms: 64 runs, 8 of them reused from Stage 1."],
  ["Stage 3, finalists.", "The 4 best configurations × 2 splits × 3 seeds: 24 runs with a 20-epoch budget, registered before training."],
  ["Freeze.", "The winner was chosen on validation alone and frozen with SHA-256 hashes of every checkpoint, the data, scalers and evaluation code."],
  ["Test once.", "July–December 2024 was evaluated a single time from the frozen manifest. That split is now spent and can't be used for tuning."],
];

const LIMITS = [
  ["Model and history are confounded.", "In Stage 3 the MLP used history A and N-HiTS history B; no 20-epoch N-HiTS run on history A exists."],
  ["Unmatched references.", "The local-only and centralised references had 10 epochs and one seed, half the federated budget."],
  ["Three seeds.", "Standard deviations are rough, especially for the low-variance cells."],
  ["Budget cap.", "Most Stage 3 runs were still improving slowly at epoch 20."],
  ["Provisional geography.", "Client regions and sensor locations come from a supported but unconfirmed sensor-ID mapping."],
  ["Historical and simulated.", "One recorded dataset, evaluated offline; federated communication is counted, not measured on a network."],
];

export default function Methodology() {
  return (
    <StudyGate>
      {({ metrics: m }) => (
        <>
          <PageHeader title="Methodology">
            How the federated forecasting model was built and tested: three years of XTraffic data, a staged search on validation data only, then one evaluation on
            a later half-year none of the models had seen.
          </PageHeader>

          <Card>
            <CardHeader title="Data and split" meta="XTraffic 2022–2024 · 200 sensors · 5-minute flow" />
            <CardBody className="flex flex-col gap-4">
              <div className="flex h-16 overflow-hidden rounded-control border border-line text-xs" role="img" aria-label="Training 2022 to 2023, validation January to June 2024, test July to December 2024">
                <div className="flex flex-col justify-center bg-surface-2 px-3" style={{ flex: 730 }}>
                  <b className="font-semibold text-fg">Training · 2022–2023</b>
                  <span className="text-fg-3">history A: 2023 only · history B: both years</span>
                </div>
                <div className="flex min-w-[72px] flex-col justify-center border-l border-line bg-surface-3 px-3" style={{ flex: 182 }}>
                  <b className="font-semibold text-fg">Validation</b>
                  <span className="text-fg-3">Jan – Jun 2024</span>
                </div>
                <div className="flex min-w-[72px] flex-col justify-center border-l border-accent/50 bg-accent/12 px-3" style={{ flex: 184 }}>
                  <b className="font-semibold text-accent">Test</b>
                  <span className="text-fg-2">Jul – Dec 2024</span>
                </div>
              </div>
              <p className="max-w-[80ch] text-sm leading-relaxed text-fg-2">
                The same 200 sensors and four 50-sensor client groups as the 2023 study. Each window uses 12 past readings (60 minutes) to predict the next 12, and
                targets always stay inside one split. Validation covers {VAL_TARGETS.toLocaleString("en-US")} valid targets and the test{" "}
                {TEST_TARGETS.toLocaleString("en-US")}.
              </p>
            </CardBody>
          </Card>

          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Data preparation" />
              <CardBody className="py-2">
                <KV
                  rows={[
                    ["Input", "flow only, 12 steps × 1 channel"],
                    ["Missing data", "conservative policy: non-finite flow, and readings where flow, occupancy and speed are all 0 in 2022 and 2024, are not targets"],
                    ["Gaps in inputs", "forward-filled up to 6 steps, then the training time-of-day median; targets never filled"],
                    ["Scaling", "training data only; per client group for federated and local runs, pooled for centralised"],
                    ["Metric", "WAPE on raw flow over valid targets: total absolute error ÷ total observed flow"],
                  ]}
                />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Training" />
              <CardBody className="py-2">
                <KV
                  rows={[
                    ["Architectures", "MLP, GRU, TCN, PatchTST, N-HiTS (57k–103k parameters)"],
                    ["Algorithms", "FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, SCAFFOLD, FedNova, FedAvgM"],
                    ["Loss", "MAE; batch 128, gradient clip 1.0"],
                    ["Optimiser", "federated clients: SGD, 100 local steps per round at learning rate 0.01; centralised and local-only: Adam, 0.001"],
                    ["Epoch", "2,048,000 training windows (40 rounds)"],
                    ["Stopping", "patience 3 on a fixed 1M-window validation subset; best epoch re-scored on full validation"],
                  ]}
                />
              </CardBody>
            </Card>
          </div>

          <div className="mt-6 grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Staged search and evaluation" />
              <CardBody>
                <Steps items={STAGES} />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Limitations" />
              <CardBody>
                <Bullets items={LIMITS} />
              </CardBody>
            </Card>
          </div>

          <SectionHeader title="Original 2023 GRU study" meta="the model behind the Forecasting page" />
          <Card>
            <CardBody className="flex flex-col gap-4">
              <div className="flex h-14 overflow-hidden rounded-control border border-line text-xs" role="img" aria-label="Train 191 days, validation 28 days, test 54 days, 2023">
                <div className="flex flex-col justify-center bg-surface-2 px-3" style={{ flex: 191 }}>
                  <b className="font-semibold text-fg">Train · 191 days</b>
                  <span className="text-fg-3">Jan 1 – Jul 10, 2023</span>
                </div>
                <div className="flex min-w-[64px] flex-col justify-center border-l border-line bg-surface-3 px-2" style={{ flex: 28 }}>
                  <b className="font-semibold text-fg">Validation</b>
                  <span className="text-fg-3">28 days</span>
                </div>
                <div className="flex flex-col justify-center border-l border-line-strong bg-surface-3 px-3" style={{ flex: 54 }}>
                  <b className="font-semibold text-fg">Test · 54 days</b>
                  <span className="text-fg-2">Aug 8 – Sep 30</span>
                </div>
              </div>
              <p className="max-w-[80ch] text-sm leading-relaxed text-fg-2">
                A GRU (64 hidden units) with four input features (scaled flow, observed flag, time-of-day sine and cosine), trained four ways (centralised,
                local-only, FedAvg IID, FedAvg geographic) × 3 seeds with Adam at learning rate 0.001. Local-only had the lowest validation MAE and was tested once on{" "}
                <span className="tnum">{m.test_windows.toLocaleString("en-US")}</span> windows. Its results are on the Model performance page.
              </p>
            </CardBody>
          </Card>
        </>
      )}
    </StudyGate>
  );
}
