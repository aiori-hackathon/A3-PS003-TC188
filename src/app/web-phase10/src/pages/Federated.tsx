import { BarCompare } from "../components/charts";
import { FedAvgDiagram, StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Legend, Note, Num, NumTh, PageHeader, SectionHeader, Stat, StatStrip, Steps, Swatch, TableWrap, Tag } from "../components/ui";
import { COMM, STAGE1, STAGE2, STAGE3 } from "../lib/federated3";
import { num, pct, signed } from "../lib/format";
import { ARMS, SERIES } from "../lib/study";
import type { ArmKey, Contrast, RunCost } from "../lib/types";

function Verdict({ c }: { c: Contrast }) {
  if (c.statistical && c.practical) return <Tag tone="accent">Detectable · meaningful</Tag>;
  if (c.statistical) return <Tag>Detectable · below threshold</Tag>;
  return <Tag>Not distinguishable</Tag>;
}

const ROUND = [
  ["Broadcast.", "The server sends the current global model to all four clients."],
  ["Train locally.", "Each client runs 100 SGD steps (batch 128) on its own sensors; raw readings never leave it."],
  ["Upload.", "Clients return their updated weights only."],
  ["Aggregate.", "The server averages the updates, weighted by training windows. FedAvgM adds server momentum 0.9, which smooths the global update across rounds."],
  ["Validate and stop.", "40 rounds make one epoch. Training stops after 3 epochs without validation improvement or at the epoch cap; the best epoch is kept."],
];

export default function Federated() {
  return (
    <StudyGate>
      {({ metrics: m, sensors }) => {
        const c = m.contrasts;
        const rt = m.runtime;
        const runs = (arm: ArmKey) => [0, 1, 2].map((s) => rt[`${arm}_seed${s}`]).filter(Boolean) as RunCost[];
        const avg = (arm: ArmKey, g: (r: RunCost) => number) => runs(arm).reduce((t, r) => t + g(r), 0) / Math.max(1, runs(arm).length);
        return (
          <>
            <PageHeader title="Federated learning">
              Federated learning trains one shared model without moving raw data: each client trains on its own sensors and sends only model weights to a server,
              which combines them. The 2022–2024 study simulated four clients on one machine, compared eight federated algorithms and five architectures in three
              stages, and selected MLP + FedAvgM. The original 2023 GRU study is summarised at the end.
            </PageHeader>

            <Card>
              <CardHeader title="How a federated round works" meta={`about ${num(COMM.mlpMbPerRound, 2)} MB of MLP weights uploaded per round, all four clients`} />
              <CardBody>
                <FedAvgDiagram sensors={sensors} />
              </CardBody>
            </Card>

            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              <Card>
                <CardHeader title="One round" />
                <CardBody>
                  <Steps items={ROUND} />
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Two ways to split sensors into clients" />
                <CardBody className="flex flex-col gap-3 text-sm leading-relaxed text-fg-2">
                  <p>
                    <b className="font-semibold text-fg">Geographic (non-IID).</b> Each client holds one region: LA / Ventura, Bay Area, Coastal / South Central and
                    Inland / Central, 50 sensors each. Traffic differs between regions, which is the realistic and harder case.
                  </p>
                  <p>
                    <b className="font-semibold text-fg">IID.</b> The same 200 sensors are dealt randomly into four clients of 50, so every client sees similar traffic.
                    It is the control that shows how much the regional split costs.
                  </p>
                  <p>Every configuration ran on both splits, with the same seeds, data and training budget.</p>
                </CardBody>
              </Card>
            </div>

            <SectionHeader title="Choosing the algorithm and model" meta="validation only, Jan–Jun 2024 · WAPE, lower is better" />
            <div className="grid gap-6 xl:grid-cols-2">
              <Card>
                <CardHeader title="Stage 2: eight federated algorithms" meta="best run per algorithm · 64 runs, 10 epochs" />
                <CardBody>
                  <BarCompare
                    label="Best geographic validation WAPE per federated algorithm"
                    digits={3}
                    suffix="%"
                    data={STAGE2.map((a) => ({ label: a.algo, value: a.geo, kind: a.algo === "FedAvgM" ? "final" : a.algo === "FedAvg" ? "arm" : "ref" }))}
                  />
                  <Legend
                    items={[
                      { label: "FedAvgM, selected", color: "var(--color-accent)" },
                      { label: "FedAvg, the baseline", color: "var(--color-ref)" },
                    ]}
                  />
                  <Note className="mt-3">
                    Geographic clients shown. Server-side optimisers (FedAvgM, FedAdam, FedYogi) gained 0.12–0.16 points over FedAvg; FedProx, SCAFFOLD and FedNova
                    matched FedAvg.
                  </Note>
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Stage 1: five architectures" meta="best of histories A and B · 40 runs, 10 epochs, plain FedAvg" />
                <TableWrap label="Stage 1 validation WAPE by architecture and training approach">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <NumTh scope="col">Parameters</NumTh>
                      <NumTh scope="col">Centralised</NumTh>
                      <NumTh scope="col">Local-only</NumTh>
                      <NumTh scope="col">FedAvg IID</NumTh>
                      <NumTh scope="col">FedAvg geo</NumTh>
                    </tr>
                  </thead>
                  <tbody>
                    {STAGE1.map((r) => (
                      <tr key={r.model}>
                        <td>{r.model}</td>
                        <Num className="text-fg-3">{r.params.toLocaleString("en-US")}</Num>
                        <Num className="text-fg-2">{pct(r.centralised, 2)}</Num>
                        <Num className="text-fg-2">{pct(r.local, 2)}</Num>
                        <Num>{pct(r.iid, 2)}</Num>
                        <Num>{pct(r.geo, 2)}</Num>
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
                <Note className="px-5 py-4">
                  MLP and N-HiTS federated best, so both went to Stage 2. GRU is strong when trained centrally or locally but loses most under federated averaging.
                </Note>
              </Card>
            </div>

            <Card className="mt-6">
              <CardHeader title="Stage 3: four finalists, three seeds" meta="24 runs, 20 epochs · mean ± SD of full validation WAPE" />
              <TableWrap label="Stage 3 finalists">
                <thead>
                  <tr>
                    <th scope="col">Configuration</th>
                    <NumTh scope="col">IID clients</NumTh>
                    <NumTh scope="col">Geographic clients</NumTh>
                    <NumTh scope="col">Score (mean of both)</NumTh>
                  </tr>
                </thead>
                <tbody>
                  {STAGE3.map((r, i) => (
                    <tr key={r.config} className={i === 0 ? "bg-accent/[0.04]" : undefined}>
                      <td>
                        <span className="flex items-center gap-2">
                          {r.config}
                          {i === 0 && <Tag tone="accent">Selected</Tag>}
                          {i === 3 && <Tag>Baseline</Tag>}
                        </span>
                      </td>
                      <Num>
                        {pct(r.iid[0], 3)} <span className="text-xs text-fg-3">± {num(r.iid[1], 3)}</span>
                      </Num>
                      <Num>
                        {pct(r.geo[0], 3)} <span className="text-xs text-fg-3">± {num(r.geo[1], 3)}</span>
                      </Num>
                      <Num className={i === 0 ? "font-semibold" : undefined}>{pct(r.score, 3)}</Num>
                    </tr>
                  ))}
                </tbody>
              </TableWrap>
              <Note className="px-5 py-4">
                The rule fixed before training: lowest score wins, and the runner-up is also tested only if it is within one pooled seed SD. The gap (0.039) exceeded
                every reading of that SD (0.015–0.022), so only MLP + FedAvgM and the FedAvg baseline were frozen and tested. History A trains on 2023; history B on
                2022–2023.
              </Note>
            </Card>

            <SectionHeader title="Communication and cost" meta="Stage 3, simulated in one process" />
            <StatStrip className="grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
              <Stat label="Rounds per run" value={COMM.roundsPerRun.toLocaleString("en-US")} note="20 epochs × 40 rounds" />
              <Stat label="Upload per MLP run" value={`${num(COMM.mlpGbPerRun, 2)} GB`} note="103,052 parameters × 4 clients × 800 rounds" />
              <Stat label="Upload per N-HiTS run" value={`${num(COMM.nhitsGbPerRun, 2)} GB`} note="60,345 parameters" />
              <Stat label="GPU training time" value={`${num(COMM.gpuHours, 2)} h`} note={`all 24 runs · ${COMM.totalRounds.toLocaleString("en-US")} rounds`} />
            </StatStrip>
            <Note className="mt-3">
              Communication is counted from the weights exchanged, not measured on a network; downloads equal uploads. Clients ran in one process on one GPU, so there is
              no network latency in these figures.
            </Note>

            <SectionHeader title="Original 2023 GRU study" meta="FedAvg only · 3 seeds · held-out test Aug 8 – Sep 30, 2023" />
            <div className="grid gap-6 2xl:grid-cols-2">
              <Card>
                <CardHeader title="Accuracy trade-offs" meta="three matched seeds · MAE difference" />
                <TableWrap label="Pre-registered comparisons">
                  <thead>
                    <tr>
                      <th scope="col">Comparison</th>
                      <NumTh scope="col">Difference</NumTh>
                      <NumTh scope="col">95% interval</NumTh>
                      <th scope="col">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(c).map((x) => (
                      <tr key={`${x.a}-${x.b}`}>
                        <td>
                          {m.arms[x.a].label} vs {m.arms[x.b].label}
                        </td>
                        <Num>{signed(x.diff)}</Num>
                        <Num className="text-fg-2">
                          {num(x.reported_ci95[0])} to {num(x.reported_ci95[1])}
                        </Num>
                        <td>
                          <Verdict c={x} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
                <Note className="px-5 py-4">
                  Positive means the first model has higher error. "Meaningful" uses the threshold fixed before the test, {num(m.sesoi_mae)} MAE. Intervals resample
                  whole sensors within client groups.
                </Note>
              </Card>
              <Card>
                <CardHeader title="Training cost, 2023 GRU" meta="mean per run, CPU" />
                <TableWrap label="Training cost of the 2023 study">
                  <thead>
                    <tr>
                      <th scope="col">Approach</th>
                      <NumTh scope="col">Time</NumTh>
                      <NumTh scope="col">Model traffic</NumTh>
                      <NumTh scope="col">Test MAE</NumTh>
                    </tr>
                  </thead>
                  <tbody>
                    {ARMS.map((k) => (
                      <tr key={k}>
                        <td>
                          <span className="flex items-center gap-2.5">
                            <Swatch color={SERIES[k].color} dash={!!SERIES[k].dash} />
                            {m.arms[k].label}
                          </span>
                        </td>
                        <Num>{num(avg(k, (r) => r.minutes), 0)} min</Num>
                        <Num className="text-fg-2">{k.startsWith("fedavg") ? `${num(avg(k, (r) => r.communication_mb), 1)} MB` : "none"}</Num>
                        <Num>{num(m.arms[k].mae)}</Num>
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
                <Note className="px-5 py-4">
                  In 2023, geographic FedAvg cost {num(c.C1.diff)} MAE against centralised and {num(c.C2.diff)} against local-only training. In the 2022–2024 study
                  federated training still trailed local-only training; see Model performance.
                </Note>
              </Card>
            </div>
          </>
        );
      }}
    </StudyGate>
  );
}
